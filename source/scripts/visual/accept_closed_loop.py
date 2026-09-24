"""Independent numeric checks; simulator replay is a separate required acceptance."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--bank',required=True);a=p.parse_args();run=Path(a.run);bank=Path(a.bank)
    summary=json.loads((run/'summary.json').read_text());manifest=json.loads((bank/'manifest.json').read_text());assert hashlib.sha256((bank/'manifest.json').read_bytes()).hexdigest()==summary['hashes']['bank_manifest']
    for name,h in json.loads((run/'artifact_manifest.json').read_text()).items():assert hashlib.sha256((run/name).read_bytes()).hexdigest()==h
    for name,h in json.loads((run/'source/manifest.json').read_text()).items():assert hashlib.sha256((run/'source'/name).read_bytes()).hexdigest()==h
    aggregate=[]
    for row in summary['cases']:
        i=row['index'];item=manifest['cases'][i];assert item['seed']==row['seed'] and row['prefix_replay_verified'];file=bank/f'case_{i:03d}.npz';assert hashlib.sha256(file.read_bytes()).hexdigest()==item['sha256']
        truth=np.load(file);z=np.load(run/f'case_{i:03d}_predictions.npz');seed=int(z['seed']);assert seed==row['seed'];n=summary['replans'];assert z['states'].shape==(n*5+1,7) and z['actions'].shape==(n*5,2)
        np.testing.assert_array_equal(z['frames'][:3],truth['history_pixels']);np.testing.assert_array_equal(z['states'][0],truth['history_states'][-1]);np.testing.assert_array_equal(z['goal_state'],truth['goal_state'])
        assert z['frames'].shape==(n+3,224,224,3) and len(z['boundary'])==n*5
        for j in range(n):
            plans=np.random.default_rng(np.random.SeedSequence([seed,951001,j])).uniform(-.35,.35,(64,5,2)).astype('float32');plans[0]=0;np.testing.assert_array_equal(plans,z['plans'][j]);score=z['scores'][j];assert np.isfinite(score).all()
            if summary['policy']=='model':
                pred=z['predicted_tokens'][j].astype('float64');goal=z['goal_tokens'].astype('float64')
                if summary['model'].endswith('state'):
                    norm=summary['target_normalization'];mean,std=np.array(norm['mean']),np.array(norm['std']);pred=pred*std+mean;goal=goal*std+mean
                    angle=np.angle(pred[:,5]+1j*pred[:,4])-np.angle(goal[5]+1j*goal[4]);angle=(angle+np.pi)%(2*np.pi)-np.pi;reconstructed=np.sum((pred[:,2:4]-goal[2:4])**2,1)+(30*angle)**2
                else:reconstructed=np.sum((pred-goal)**2,1)
                np.testing.assert_allclose(score,reconstructed,rtol=3e-5,atol=1e-7)
                selected=int(np.random.default_rng(np.random.SeedSequence([seed,951003,j])).choice(np.flatnonzero(score==score.min())))
            elif summary['policy']=='zero':selected=0;assert not score.any()
            else:selected=int(np.random.default_rng(np.random.SeedSequence([seed,951002,j])).integers(64));assert not score.any()
            assert z['selected'][j]==selected;np.testing.assert_array_equal(z['actions'][j*5:j*5+5],np.broadcast_to(plans[selected,0],(5,2)))
        s,g=z['states'],z['goal_state'];delta=(s[:,4]-g[4]+np.pi)%(2*np.pi)-np.pi;pos=np.sum((s[:,2:4]-g[2:4])**2,1);allpos=np.sum((s[:,:4]-g[:4])**2,1);cost=pos+(30*delta)**2;success=(pos<400)&(np.abs(delta)<np.pi/9)
        metrics={'terminal_block_cost':float(cost[-1]),'initial_block_cost':float(cost[0]),'terminal_block_success':bool(success[-1]),'initial_block_success':bool(success[0]),'ever_block_success':bool(success.any()),'terminal_agent_block_cost':float(allpos[-1]+(30*delta[-1])**2),'terminal_agent_block_success':bool((allpos[-1]<400)&(abs(delta[-1])<np.pi/9)),'boundary_exit_steps':int(z['boundary'].sum())}
        for key,value in metrics.items():np.testing.assert_allclose(row[key],value,rtol=1e-9,atol=1e-7)
        aggregate.append(metrics)
    for key,value in summary['aggregate'].items():np.testing.assert_allclose(value,np.mean([r[key] for r in aggregate]),rtol=1e-9,atol=1e-7)
    result={'accepted_cases':len(aggregate),'policy':summary['policy'],'model':summary['model'],'replans':summary['replans'],'checks':'hashes, candidate generation, token scores, selection, action execution and outcome aggregates','simulator_replay':'separate audit required'}
    (run/'numeric_acceptance.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
