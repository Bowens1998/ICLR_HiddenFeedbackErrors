"""Summarize all 24 routes only after uniform staged verification."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np


def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser()
    for key in ['runs','reports','manifests','output']:p.add_argument('--'+key,required=True)
    a=p.parse_args();root=Path(a.runs);reports=Path(a.reports);manifests=Path(a.manifests)
    rows=[];values={};identities=None;zero=None;bank=None;provenance=[]
    for i in range(24):
        replica=i//8;checkpoint='best' if (i//4)%2==0 else 'last';model_index=(i%4)//2;algorithm=['random','cem'][i%2]
        sf=root/f'job_{i}'/'summary.json';af=reports/f'job_{i}'/'staged_acceptance.json'
        r=json.loads(sf.read_text());acc=json.loads(af.read_text())
        mf=manifests/f'replica_{replica}_{checkpoint}.json';entry=json.loads(mf.read_text())['models'][model_index]
        assert acc['original_summary_sha256']==digest(sf)
        assert r['hashes']['model_manifest']==digest(mf) and r['hashes']['weights']==entry['weights_sha256']
        assert (r['model_index'],r['arm'],r['algorithm'],r['parameterization'])==(model_index,entry['arm'],algorithm,'full')
        assert entry['checkpoint']==checkpoint and entry['updates']==21000
        assert acc['verification_policy']=='staged_fp32_head_v1' and len(r['cases'])==acc['cases']==128
        assert acc['model_free_simulator_replay'] and acc['max_replay_state_difference']==0
        assert len(acc['stage_archives'])==128 and [v['case'] for v in acc['stage_archives']]==list(range(128))
        ids=[v['seed'] for v in r['cases']];z=np.array([v['zero_cost'] for v in r['cases']])
        if identities is None:identities=ids;zero=z;bank=r['hashes']['bank_manifest']
        assert ids==identities and bank==r['hashes']['bank_manifest'];np.testing.assert_array_equal(z,zero)
        assert all(v['scored_candidates']==9000 for v in r['cases'])
        costs=np.array([v['realized_cost'] for v in r['cases']]);assert np.isfinite(costs).all();values[i]=costs
        rows.append(dict(route=i,replica=replica,checkpoint=checkpoint,arm=r['arm'],algorithm=algorithm,mean_cost=float(costs.mean()),success=float(np.mean([v['success'] for v in r['cases']])),bitwise_score_cases=acc['bitwise_score_cases']))
        provenance.append(dict(route=i,summary_sha256=digest(sf),staged_acceptance_sha256=digest(af),manifest_sha256=digest(mf),original_full_float64_violations=acc['full_float64_score_violations']))
    draws=np.random.default_rng(962001).integers(0,128,(10000,128));contrasts=[]
    for i in range(1,24,2):
        d=values[i]-values[i-1]
        contrasts.append(dict(cem_route=i,random_route=i-1,mean_cost_difference=float(d.mean()),conditional_scenario_95_percentile_interval=np.quantile(d[draws].mean(1),[.025,.975]).tolist()))
    aggregates=[]
    for arm in ['transformer_latent_state','gru_latent_state']:
        for checkpoint in ['best','last']:
            cells=[r for r in rows if r['arm']==arm and r['checkpoint']==checkpoint]
            differences=[c['mean_cost_difference'] for c in contrasts if any(r['route']==c['cem_route'] for r in cells)]
            aggregates.append(dict(arm=arm,checkpoint=checkpoint,random_mean_cost=float(np.mean([r['mean_cost'] for r in cells if r['algorithm']=='random'])),cem_mean_cost=float(np.mean([r['mean_cost'] for r in cells if r['algorithm']=='cem'])),cem_minus_random_mean=float(np.mean(differences)),cem_minus_random_between_data_sd=float(np.std(differences,ddof=1)),replicas=3))
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
    report=dict(rows=rows,contrasts=contrasts,aggregates=aggregates,provenance=provenance,zero_cost=float(zero.mean()),bank_sha256=bank,per_case_costs={str(k):v.tolist() for k,v in values.items()},scope='Three independent training sets, same seed and 128 repeatedly examined development scenarios. Best/last are correlated checkpoints, not independent repetitions. Scenario intervals condition on fitted models and are unadjusted. Remote stage archives remain authoritative; local summary validates recovered report and model identities. No matched narrow/JEPA comparison until those evaluations pass.')
    out.write_text(json.dumps(report,indent=2)+'\n')
    lines=['# Wide-state control: complete staged-verified matrix','','All24 routes retained; each uses9000 candidate scores on the same128 development scenarios. Lower realized cost is better.','','| Arm | Checkpoint | Random mean cost | CEM mean cost | CEM − random | Across-data SD of difference |','|---|---|---:|---:|---:|---:|']
    for r in aggregates:lines.append(f"| {r['arm']} | {r['checkpoint']} | {r['random_mean_cost']:.2f} | {r['cem_mean_cost']:.2f} | {r['cem_minus_random_mean']:.2f} | {r['cem_minus_random_between_data_sd']:.2f} |")
    lines+=['',report['scope'],'','Original whole-float64 failures are retained in the JSON provenance. Staged verification establishes the specified float32 execution; it does not establish float64 planning equivalence or a novel intervention.']
    out.with_suffix('.md').write_text('\n'.join(lines)+'\n');print(json.dumps(aggregates,indent=2))

if __name__=='__main__':main()
