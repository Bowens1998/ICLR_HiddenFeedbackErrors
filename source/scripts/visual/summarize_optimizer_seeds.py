"""Report all three optimizer seeds; scenarios never substitute for training replicates."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
ARMS=['transformer_jepa','gru_jepa','transformer_state','gru_state']
POLICIES=[('random','held'),('random','full'),('cem','held'),('cem','full')]

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser()
    for k in ['original-runs','original-manifest','original-training','seed3073-runs','seed3073-manifest','seed3074-runs','seed3074-manifest','output']:p.add_argument('--'+k,required=True)
    a=p.parse_args();tables=[];values={};seeds=None;bank=None;zero=None;data_hash=None
    for seed,root,mp in [(3072,Path(a.original_runs),Path(a.original_manifest)),(3073,Path(a.seed3073_runs),Path(a.seed3073_manifest)),(3074,Path(a.seed3074_runs),Path(a.seed3074_manifest))]:
        frozen=json.loads(mp.read_text())
        if seed!=3072:assert frozen['layout']=='seed' and len(frozen['routes'])==16 and len(frozen['models'])==4
        for mi,arm in enumerate(ARMS):
            entry=frozen['models'][mi];assert entry['arm']==arm
            if seed==3072:
                tr=Path(a.original_training)/arm/'summary.json';assert digest(tr)==entry['training_summary_sha256'];training=json.loads(tr.read_text());assert training['seed']==seed;dh=training['data_manifest_sha256']
            else:assert entry['seed']==seed;dh=entry['data_manifest_sha256']
            if data_hash is None:data_hash=dh
            assert dh==data_hash
            for pi,(algorithm,parameterization) in enumerate(POLICIES):
                i=mi*4+pi;run=root/f'job_{i}';r=json.loads((run/'summary.json').read_text());acc=json.loads((run/'acceptance.json').read_text());art=json.loads((run/'artifact_manifest.json').read_text())
                assert digest(run/'summary.json')==art['summary.json'] and r['hashes']['model_manifest']==digest(mp) and r['hashes']['weights']==entry['weights_sha256']
                assert (r['arm'],r['algorithm'],r['parameterization'])==(arm,algorithm,parameterization)
                assert len(r['cases'])==acc['cases']==128 and acc['model_free_simulator_replay'] and acc['max_replay_state_difference']==0
                for file,h in json.loads((run/'source/manifest.json').read_text()).items():assert digest(run/'source'/file)==h
                case_seeds=[v['seed'] for v in r['cases']];z=np.array([v['zero_cost'] for v in r['cases']])
                if seeds is None:seeds=case_seeds;bank=r['hashes']['bank_manifest'];zero=z
                assert case_seeds==seeds and r['hashes']['bank_manifest']==bank;np.testing.assert_array_equal(zero,z)
                assert all(v['scored_candidates']==9000 for v in r['cases'])
                v=np.array([c['realized_cost'] for c in r['cases']]);values[seed,arm,algorithm,parameterization]=v
                tables.append({'seed':seed,'arm':arm,'algorithm':algorithm,'parameterization':parameterization,'mean_cost':float(v.mean()),'success':float(np.mean([c['success'] for c in r['cases']]))})
    contrasts=[]
    for arm in ARMS:
        for support in ['held','full']:
            delta=[float((values[seed,arm,'cem',support]-values[seed,arm,'random',support]).mean()) for seed in [3072,3073,3074]]
            contrasts.append({'arm':arm,'parameterization':support,'contrast':'CEM minus random','seed_order':[3072,3073,3074],'per_seed_mean_cost_difference':delta,'mean_across_seeds':float(np.mean(delta)),'sd_across_seeds':float(np.std(delta,ddof=1))})
    result={'scope':'three optimizer seeds on the same training sample;128 shared development scenarios; report all seeds; no independent-data or second-task claim; three seeds do not support a broad population guarantee','training_manifest_sha256':data_hash,'bank_manifest_sha256':bank,'zero_cost':float(zero.mean()),'tables':tables,'planner_contrasts':contrasts}
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True);out.with_suffix('.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['# Optimizer-seed replication','',result['scope'],'','| Model | Search | Action coordinates | Seed3072 cost | Seed3073 cost | Seed3074 cost |','|---|---|---|---:|---:|---:|']
    for arm in ARMS:
        for alg,support in POLICIES:lines.append(f"| {arm} | {alg} | {support} | "+' | '.join(f'{values[s,arm,alg,support].mean():.2f}' for s in [3072,3073,3074])+' |')
    lines.extend(['','Negative CEM-minus-random differences favor CEM. Differences below are individual seed means, not scenario-level training replications.',''])
    for c in contrasts:lines.append(f"- {c['arm']} / {c['parameterization']}: {c['per_seed_mean_cost_difference']}; mean {c['mean_across_seeds']:.2f}, seed SD {c['sd_across_seeds']:.2f}.")
    out.with_suffix('.md').write_text('\n'.join(lines)+'\n');print('SUMMARIZED_ALL_48_ROUTES')
if __name__=='__main__':main()
