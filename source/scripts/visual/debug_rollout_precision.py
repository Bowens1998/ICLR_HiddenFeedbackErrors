"""Locate native/windowed rollout differences without relaxing parity tolerances."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from factorial_model import make_model

def main():
    p=argparse.ArgumentParser()
    for key in ['training','bank','official','config','output']:p.add_argument('--'+key,required=True)
    a=p.parse_args();torch.set_num_threads(4);rows=[]
    for arm in ['gru_jepa','gru_state']:
        run=Path(a.training)/arm;r=json.loads((run/'summary.json').read_text());m=make_model(a.official,a.config,arm,r['seed']);m.load_state_dict(torch.load(run/'best_weights.pt',map_location='cpu',weights_only=True),strict=True);m=m.cuda().eval()
        am=torch.tensor(r['normalization']['mean'],device='cuda').repeat(5);sd=torch.tensor(r['normalization']['std'],device='cuda').repeat(5)
        im=torch.tensor([.485,.456,.406],device='cuda')[None,:,None,None];std=torch.tensor([.229,.224,.225],device='cuda')[None,:,None,None]
        for mode in ['cudnn_tf32_on','strict_fp32']:
            torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=mode=='cudnn_tf32_on'
            for case in range(4):
                z=np.load(Path(a.bank)/f'case_{case:03d}.npz');n=len(z['actions']);pix=(torch.from_numpy(z['history_pixels'].copy()).cuda().permute(0,3,1,2).float()/255-im)/std
                act=np.concatenate([np.broadcast_to(z['prefix'].reshape(1,2,10),(n,2,10)),z['actions'].reshape(n,5,10)],1);act=(torch.from_numpy(act).cuda()-am)/sd
                with torch.inference_mode():
                    native=m.rollout({'pixels':pix[None,None].expand(1,n,-1,-1,-1,-1)},act[None])['predicted_emb'][0]
                    init=m.encode({'pixels':pix[:,None]})['emb'][:,0][None].expand(n,-1,-1).clone();current=init.clone();prefix=init.clone();action_diff=[]
                    for k in range(2,7):
                        full=m.action_encoder(act[:,:k+1])[:,-3:];window=m.action_encoder(act[:,k-2:k+1]);action_diff.append(float((full-window).abs().max()))
                        current=torch.cat([current,m.predict(current[:,-3:],window)[:,-1:]],1)
                        prefix=torch.cat([prefix,m.predict(prefix[:,-3:],full)[:,-1:]],1)
                    passed=True
                    try:torch.testing.assert_close(native,current,rtol=2e-5,atol=2e-5)
                    except AssertionError:passed=False
                    row={'arm':arm,'case':case,'mode':mode,'initial_max_diff':float((native[:,:3]-init).abs().max()),'action_embedding_max_diffs':action_diff,
                         'native_vs_window_max_diff':float((native-current).abs().max()),'native_vs_prefix_max_diff':float((native-prefix).abs().max()),'original_tolerance_passed':passed}
                    rows.append(row);print(json.dumps(row),flush=True)
    Path(a.output).write_text(json.dumps({'gpu':torch.cuda.get_device_name(),'rows':rows},indent=2)+'\n')
if __name__=='__main__':main()
