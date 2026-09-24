"""Independent NumPy/SciPy reconstruction from saved model tokens and simulator states."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--bank',required=True);a=p.parse_args();run=Path(a.run);bank=Path(a.bank)
    r=json.loads((run/'summary.json').read_text());b=json.loads((bank/'manifest.json').read_text());assert len(r['cases'])==len(b['cases'])
    assert hashlib.sha256((bank/'manifest.json').read_bytes()).hexdigest()==r['hashes']['bank_manifest']
    for name,h in json.loads((run/'artifact_manifest.json').read_text()).items():assert hashlib.sha256((run/name).read_bytes()).hexdigest()==h,name
    rebuilt=[]
    for row,item in zip(r['cases'],b['cases']):
        i=row['index'];assert i==item['index'] and row['seed']==item['seed'] and row['alignment_verified']
        path=bank/f'case_{i:03d}.npz';assert hashlib.sha256(path.read_bytes()).hexdigest()==item['sha256'];truth=np.load(path);v=np.load(run/f'case_{i:03d}_predictions.npz')
        assert not truth['actions'][0].any();goal=v['goal_tokens'].astype('float64');scores={}
        for method,key in [('native','predicted_tokens'),('oracle_realized','realized_tokens')]:
            token=v[key].astype('float64')
            if r['arm'].endswith('jepa'):
                cost=np.sum((token-goal)**2,axis=1);scores[method]={'block_pose':cost,'agent_and_block_pose':cost}
            else:
                mean,std=np.array(r['target_normalization']['mean']),np.array(r['target_normalization']['std']);s=token*std+mean;g=goal*std+mean
                delta=np.angle(s[:,5]+1j*s[:,4])-np.angle(g[5]+1j*g[4]);delta=(delta+np.pi)%(2*np.pi)-np.pi
                block=np.sum((s[:,2:4]-g[2:4])**2,axis=1)+(30*delta)**2
                scores[method]={'block_pose':block,'agent_and_block_pose':block+np.sum((s[:,:2]-g[:2])**2,axis=1)}
        n=len(truth['actions']);zero=np.ones(n);zero[0]=0;scores['zero_action']={t:zero for t in ['block_pose','agent_and_block_pose']};scores['uniform']={t:np.zeros(n) for t in scores['zero_action']}
        state,target=truth['terminal_states'],truth['goal_state'];angle=(state[:,4]-target[4]+np.pi)%(2*np.pi)-np.pi
        base=np.sum((state[:,2:4]-target[2:4])**2,axis=1)+(30*angle)**2
        costs={'block_pose':base,'agent_and_block_pose':base+np.sum((state[:,:2]-target[:2])**2,axis=1)};new={}
        for task,cost in costs.items():
            np.testing.assert_allclose(cost,v['cost_'+task],rtol=1e-9,atol=1e-7);new[task]={}
            for method,values in scores.items():
                saved=v['score_'+method+'_'+task];np.testing.assert_allclose(values[task],saved,rtol=3e-5,atol=1e-7)
                ties=np.flatnonzero(saved==saved.min());selected=float(np.mean(cost[ties]));random=float(np.mean(cost));best=float(np.min(cost))
                corr=None if np.ptp(saved)==0 or np.ptp(cost)==0 else float(spearmanr(saved,cost).statistic)
                metrics={'spearman':corr,'selected_cost':selected,'random_cost':random,'candidate_best_cost':best,'regret':selected-best,'gain_vs_random':random-selected};new[task][method]=metrics
                assert row['tasks'][task][method]['selected_ties']==ties.tolist()
                for key,value in metrics.items():
                    actual=row['tasks'][task][method][key]
                    if value is None:assert actual is None
                    else:np.testing.assert_allclose(actual,value,rtol=1e-8,atol=1e-7)
        rebuilt.append(new)
    for task,methods in r['aggregate'].items():
        for method,metrics in methods.items():
            for key,value in metrics.items():
                values=[row[task][method][key] for row in rebuilt if row[task][method][key] is not None]
                if values:np.testing.assert_allclose(value,np.mean(values),rtol=1e-8,atol=1e-7)
                else:assert value is None
    result={'accepted_cases':len(rebuilt),'arm':r['arm'],'checkpoint':r['checkpoint'],'scores_costs_ranks_aggregates_and_hashes':'independently reconstructed','scope':r['scope']}
    (run/'acceptance.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
