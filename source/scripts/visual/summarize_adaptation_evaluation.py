"""Complete prespecified paired fresh-goal comparison; require all72 accepted routes."""
import argparse,json
from pathlib import Path
import numpy as np
from adaptation_streams import sha


def main():
    p=argparse.ArgumentParser()
    for k in ('runs','plan','bank','output'):p.add_argument('--'+k,required=True)
    a=p.parse_args();plan=json.loads(Path(a.plan).read_text());bank=Path(a.bank);bm=json.loads((bank/'manifest.json').read_text())
    assert plan['adaptation_evaluation']['status']=='READY_COMPLETE_FRESH_EVALUATION' and len(plan['routes'])==72 and len(plan['models'])==36
    assert sha(bank/'manifest.json')==plan['bank_manifest_sha256'] and len(bm['cases'])==128 and bm['seed_start']==1321001
    seeds=[c['seed'] for c in bm['cases']];rows=[];data={};bindings=[]
    for i,route in enumerate(plan['routes']):
        d=Path(a.runs)/f'job_{i}';r=json.loads((d/'summary.json').read_text());ac=json.loads((d/'acceptance.json').read_text());am=json.loads((d/'artifact_manifest.json').read_text());e=plan['models'][route['model_index']]
        assert ac['model_free_simulator_replay'] and ac['cases']==128 and len(r['cases'])==128
        assert r['route_index']==i and r['model_index']==route['model_index'] and r['algorithm']==route['algorithm']
        assert r['hashes']['model_manifest']==sha(a.plan) and r['hashes']['bank_manifest']==sha(bank/'manifest.json') and r['hashes']['weights']==e['weights_sha256']
        assert am['summary.json']==sha(d/'summary.json') and [c['seed'] for c in r['cases']]==seeds
        assert [c['index'] for c in r['cases']]==list(range(128)) and all(c['scored_candidates']==9000 for c in r['cases'])
        cost=np.array([c['realized_cost'] for c in r['cases']]);success=np.array([c['success'] for c in r['cases']],dtype=float);assert np.isfinite(cost).all()
        key=(e['original_model_index'],e['adaptation_condition'],route['algorithm']);assert key not in data;data[key]=(cost,success)
        rows.append(dict(route=i,original_model_index=key[0],condition=key[1],algorithm=key[2],recipe=e['score'],arm=e['arm'],replica=e['replica'],mean_cost=float(cost.mean()),success_rate=float(success.mean()),boundary_steps=sum(c['boundary_steps'] for c in r['cases']),costs=cost.tolist(),successes=success.tolist()))
        bindings.append(dict(route=i,summary_sha256=sha(d/'summary.json'),acceptance_sha256=sha(d/'acceptance.json'),artifact_manifest_sha256=sha(d/'artifact_manifest.json')))
    rng=np.random.default_rng(1295001);draws=rng.integers(0,128,size=(10000,128));contrasts=[];groups=[]
    def contrast(ids,left,right,recipe,primary):
        cost=np.stack([data[(i,*left)][0]-data[(i,*right)][0] for i in ids]);succ=np.stack([data[(i,*left)][1]-data[(i,*right)][1] for i in ids]);delta=cost.mean(0);sd=succ.mean(0)
        q=[.00625,.99375] if primary else [.025,.975]
        return dict(recipe=recipe,left=list(left),right=list(right),primary_cost=primary,mean_cost_difference=float(delta.mean()),cost_percentile_interval=np.quantile(delta[draws].mean(1),q).tolist(),cost_interval_level=.9875 if primary else .95,success_difference=float(sd.mean()),secondary_success_95_percentile_interval=np.quantile(sd[draws].mean(1),[.025,.975]).tolist(),per_model=[dict(index=i,cost_difference=float(c.mean()),success_difference=float(s.mean())) for i,c,s in zip(ids,cost,succ)])
    for parity,recipe in [(0,'pose_encoded'),(1,'state')]:
        ids=list(range(parity,12,2))
        for algorithm in ('random','cem'):
            for condition in ('original','expert','planner'):
                groups.append(dict(recipe=recipe,algorithm=algorithm,condition=condition,mean_cost=float(np.mean([data[(i,condition,algorithm)][0].mean() for i in ids])),success_rate=float(np.mean([data[(i,condition,algorithm)][1].mean() for i in ids]))))
            for left,right in [('planner','expert'),('expert','original'),('planner','original')]:contrasts.append(contrast(ids,(left,algorithm),(right,algorithm),recipe,left=='planner' and right=='expert'))
        for condition in ('original','expert','planner'):contrasts.append(contrast(ids,(condition,'cem'),(condition,'random'),recipe,False))
    result=dict(status='COMPLETE72_FRESH_ADAPTATION_ROUTES',rows=rows,groups=groups,contrasts=contrasts,goal_seeds=seeds,bindings=bindings,plan_sha256=sha(a.plan),source_sha256=sha(__file__),bootstrap=dict(seed=1295001,replicates=10000,unit='same goal indices shared across all six models and all contrasts',primary='Four planner-minus-expert cost contrasts, nominal98.75% marginal percentile intervals; approximate Bonferroni-style presentation, not finite-sample certification.',secondary='95% descriptive intervals are not simultaneous guarantees.'),scope='All prespecified final conditions retained. Uncertainty conditional on these fixed fits, over admitted fresh goals; no training-population or cross-task inference. Remote acceptance binds full numerical search and physical replay; raw candidate archives not rechecked by this summary. Imagined and real-image error diagnostics remain separate.')
    with Path(a.output).open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print('COMPLETE72',flush=True)

if __name__=='__main__':main()
