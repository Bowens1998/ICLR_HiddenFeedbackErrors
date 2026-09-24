"""Fresh-process reconstruction of every original frozen observation/target/action."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from adaptation_streams import expert_windows,planner_windows,sha
from factorial_model import make_model
from evaluation_precision import configure_evaluation_precision


def main():
    p=argparse.ArgumentParser()
    for k in ('cache','data','freeze','train-plan','validation-plan','train-trajectories','validation-trajectories','official','config'):p.add_argument('--'+k,required=True)
    a=p.parse_args();cache=Path(a.cache);report=json.loads((cache/'report.json').read_text());entry=report['entry']
    assert report['status']=='CACHED_REQUIRES_REENCODING_ACCEPTANCE'
    assert report['source_sha256']==sha(Path(__file__).with_name('cache_adaptation_streams.py'))
    for name,digest in report['sources'].items():assert sha(Path(__file__).with_name(name))==digest
    assert sha(Path(a.freeze)/'acceptance.json')==report['expert_acceptance_sha256']
    for split in ('train','validation'):
        pp=Path(getattr(a,split+'_plan'));assert sha(pp)==report['bindings'][split]['plan_sha256']
        pl=json.loads(pp.read_text());assert pl['models'][pl['routes'][report['route']]['model_index']]==entry
    td=Path(entry['training_path']);tr=json.loads((td/'summary.json').read_text())
    assert sha(td/'summary.json')==entry['training_summary_sha256'] and sha(td/'last_weights.pt')==entry['weights_sha256'] and sha(a.config)==tr['config_sha256']
    configure_evaluation_precision();torch.set_num_threads(2)
    model=make_model(a.official,a.config,entry['arm'],tr['seed']);weights=torch.load(td/'last_weights.pt',map_location='cpu',weights_only=True);model.load_state_dict(weights,strict=True);model=model.cuda().eval()
    mean=torch.tensor([.485,.456,.406],device='cuda').reshape(1,1,3,1,1);std=torch.tensor([.229,.224,.225],device='cuda').reshape(1,1,3,1,1)
    action_mean=np.tile(np.asarray(tr['normalization']['mean'],dtype=np.float32),5);action_std=np.tile(np.asarray(tr['normalization']['std'],dtype=np.float32),5)
    rows=[]
    for row in report['rows']:
        fp=cache/row['file'];assert sha(fp)==row['sha256'];z=np.load(fp);kind,split=row['stream'].split('_')
        iterator=expert_windows(a.data,a.freeze,report['actor']//2,split) if kind=='expert' else planner_windows(getattr(a,split+'_trajectories'),getattr(a,split+'_plan'),report['route'])
        assert len(z['observed'])==len(z['identities'])==row['windows'];count=0;max_error=0.
        for i,(identity,(pixels,actions,states)) in enumerate(iterator):
            import hashlib
            assert hashlib.sha256(pixels.tobytes()).hexdigest()==str(z['pixel_sha256'][i])
            np.testing.assert_array_equal(identity,z['identities'][i]);np.testing.assert_array_equal(states,z['raw_states'][i])
            np.testing.assert_allclose((actions.astype(np.float32)-action_mean)/action_std,z['normalized_actions'][i],rtol=2e-6,atol=2e-6)
            images=torch.from_numpy(pixels).to('cuda').permute(0,3,1,2).unsqueeze(0).float().div(255)
            with torch.inference_mode():observed=model.encode({'pixels':(images-mean)/std})['emb'][0].cpu().numpy()
            np.testing.assert_allclose(observed,z['observed'][i],rtol=2e-5,atol=2e-5);max_error=max(max_error,float(np.max(np.abs(observed-z['observed'][i]))))
            if entry['score']=='state':
                raw=states.astype(np.float32);physical=np.concatenate([raw[:,:4],np.sin(raw[:,4:5]),np.cos(raw[:,4:5])],axis=-1)
                norm=entry['target_normalization'];target=((physical-np.asarray(norm['mean'],dtype=np.float32))/np.asarray(norm['std'],dtype=np.float32))[1:]
            else:target=observed[1:]
            np.testing.assert_allclose(target,z['target'][i],rtol=2e-5,atol=2e-5);count+=1
        assert count==row['windows'];rows.append(dict(stream=row['stream'],windows=count,max_abs_reencoding_error=max_error));print('PASS',report['index'],row['stream'],count,flush=True)
    for name,v in model.state_dict().items():torch.testing.assert_close(v.cpu(),weights[name],rtol=0,atol=0)
    out=dict(status='PASS_FULL_FRESH_PROCESS_REENCODING',rows=rows,cache_report_sha256=sha(cache/'report.json'),source_sha256=sha(__file__),scope='Every raw window reread and original checkpoint freshly loaded; all cached observations/targets/actions/identities reconstructed. Tolerance2e-5 for encoder/targets,2e-6 actions. Shared raw iterator identities separately audited. No independent implementation of the neural architecture, and no fitting or utility result.')
    with (cache/'acceptance.json').open('x') as f:json.dump(out,f,indent=2);f.write('\n')

if __name__=='__main__':main()
