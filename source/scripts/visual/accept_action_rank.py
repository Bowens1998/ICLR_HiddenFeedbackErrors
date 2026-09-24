"""Independent CPU rescoring from saved embeddings and simulator outcomes."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr

def main():
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--run',required=True);a=p.parse_args()
    data,run=Path(a.data),Path(a.run);summary=json.loads((run/'summary.json').read_text());manifest=json.loads((run/'artifact_manifest.json').read_text())
    for file,expected in manifest.items():assert hashlib.sha256((run/file).read_bytes()).hexdigest()==expected,file
    bank=json.loads((data/'manifest.json').read_text());assert len(summary['cases'])==len(bank['cases'])
    assert hashlib.sha256((data/'manifest.json').read_bytes()).hexdigest()==summary['hashes']['bank_manifest']
    all_rows=[];constant_tasks={k:0 for k in summary['aggregate']};quality=[]
    for row,item in zip(summary['cases'],bank['cases']):
        i=row['index'];assert row['seed']==item['seed'] and i==item['index'] and row['alignment_verified']
        file=data/f'case_{i:03d}.npz';assert hashlib.sha256(file.read_bytes()).hexdigest()==item['sha256']
        z=np.load(file);v=np.load(run/f'case_{i:03d}_predictions.npz');assert z['actions'].shape==(bank['candidates'],25,2)
        assert np.abs(z['actions']).max()<=1 and not z['actions'][0].any()
        goal=v['goal_embedding'];sc={name:np.sum((v[key].astype('float64')-goal)**2,axis=1) for name,key in [('native_latent','predicted_embedding'),('oracle_realized_latent','realized_embedding')]}
        sc['oracle_realized_pixel']=np.mean(((z['terminal_pixels'].astype('float64')-z['goal_pixels'])/255)**2,axis=(1,2,3));sc['uniform_persistence']=np.zeros(bank['candidates'])
        for name,value in sc.items():np.testing.assert_allclose(value,v[name],rtol=3e-5,atol=1e-7)
        s,g=z['terminal_states'],z['goal_state'];angle=(s[:,4]-g[4]+np.pi)%(2*np.pi)-np.pi
        quality.append({'index':i,'seed':row['seed'],'candidate_agent_center_outside_frame':int(np.any((s[:,:2]<0)|(s[:,:2]>512),axis=1).sum()),
                        'candidate_block_center_outside_frame':int(np.any((s[:,2:4]<0)|(s[:,2:4]>512),axis=1).sum()),
                        'goal_agent_center_outside_frame':bool(np.any((g[:2]<0)|(g[:2]>512))),
                        'goal_block_center_outside_frame':bool(np.any((g[2:4]<0)|(g[2:4]>512))),
                        'contact_candidates':int((z['contacts']>0).sum()),'builtin_goal_flag_steps':int(z['terminations'].sum())})
        base=np.sum((s[:,2:4]-g[2:4])**2,axis=1)+(30*angle)**2
        costs={'block_pose':base,'agent_and_block_pose':base+np.sum((s[:,:2]-g[:2])**2,axis=1)}
        rebuilt={}
        for task,cost in costs.items():
            np.testing.assert_allclose(cost,v['cost_'+task],rtol=1e-9,atol=1e-7);rebuilt[task]={};constant_tasks[task]+=int(np.ptp(cost)==0)
            for method in sc:
                # Use saved score precision for exact ties; independently reconstructed scores checked above.
                score=v[method];ties=np.flatnonzero(score==score.min());selected=float(np.mean(cost[ties]));random=float(np.mean(cost));best=float(np.min(cost))
                correlation=None if np.ptp(score)==0 or np.ptp(cost)==0 else float(spearmanr(score,cost).statistic)
                rebuilt[task][method]={'spearman':correlation,'selected_cost':selected,'random_cost':random,'candidate_best_cost':best,'regret':selected-best,'gain_vs_random':random-selected}
                assert row['tasks'][task][method]['selected_ties']==ties.tolist()
                for key,value in rebuilt[task][method].items():
                    actual=row['tasks'][task][method][key]
                    if value is None:assert actual is None
                    else:np.testing.assert_allclose(actual,value,rtol=1e-8,atol=1e-7)
        all_rows.append(rebuilt)
    for task,methods in summary['aggregate'].items():
        for method,metrics in methods.items():
            for key,actual in metrics.items():
                values=[r[task][method][key] for r in all_rows if r[task][method][key] is not None]
                if values:np.testing.assert_allclose(actual,np.mean(values),rtol=1e-8,atol=1e-7)
                else:assert actual is None
    result={'accepted_cases':len(all_rows),'candidates_per_case':bank['candidates'],'constant_cost_cases':constant_tasks,'saved_embeddings_scores_costs_metrics_hashes':'passed','scope':summary['scope'],
            'scenario_quality':quality,'quality_note':'center-outside-frame counts do not detect partially occluded shapes; no cases excluded'}
    (run/'acceptance.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!="scenario_quality"},indent=2))
if __name__=='__main__':main()
