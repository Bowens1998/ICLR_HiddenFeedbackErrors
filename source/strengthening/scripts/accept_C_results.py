"""Fresh raw six-frame C cache reconstruction and independent endpoint scoring."""
import argparse
import json
import os
from pathlib import Path
import sys
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'strengthening/adapters'),str(ROOT/'scripts/visual')]
from contracts import sha, atomic_json
from factorial_model import make_model
from nonlinear_pose_cost import numpy_pose
from evaluation_precision import configure_evaluation_precision
from adaptation_freeze import configure_dynamics_only, verify_frozen
from run_adaptation_checkpoint_gate import tensor_digest


def main():
    p=argparse.ArgumentParser()
    for k in ['base','heads','cache','runs','A-runs','output']:p.add_argument('--'+k,required=True)
    p.add_argument('--pool',type=int);a=p.parse_args();pool=int(os.environ['SLURM_ARRAY_TASK_ID']) if a.pool is None else a.pool;g=2*pool
    base=Path(a.base);r=base/'releases/planner-data-adaptation-v1/runs';cache=Path(a.cache)/f'group_{g}'
    e=json.loads((r/f'formal_cache/job_{2*g}/report.json').read_text())['entry'];td=Path(e['training_path']);tr=json.loads((td/'summary.json').read_text())
    configure_evaluation_precision();torch.set_num_threads(2)
    model=make_model(base/'releases/visual-v1/official',base/'assets/pusht-v1/models/config.json',e['arm'],tr['seed'])
    model.load_state_dict(torch.load(td/'last_weights.pt',map_location='cpu',weights_only=True),strict=True);model=model.cuda();boundary=configure_dynamics_only(model)
    im=torch.tensor([.485,.456,.406],device='cuda')[None,None,:,None,None];sd=torch.tensor([.229,.224,.225],device='cuda')[None,None,:,None,None]
    am=np.tile(np.asarray(tr['normalization']['mean'],np.float32),5);ast=np.tile(np.asarray(tr['normalization']['std'],np.float32),5)
    cached={};cache_checks=[]
    with torch.inference_mode():
        for split in ['train','validation']:
            fp=cache/f'C_{split}.npz';meta=json.loads((cache/f'C_{split}.json').read_text());assert sha(fp)==meta['file_sha256'];z=dict(np.load(fp));cached[split]=z
            source_reports={route:json.loads((r/f'trajectory_{split}/job_{route}/report.json').read_text()) for route in [8*g+2,8*g+3]}
            for i,(route,index,seed,start) in enumerate(z['identities']):
                case=next(x for x in source_reports[int(route)]['cases'] if x['index']==int(index));assert case['seed']==int(seed)
                path=r/f'trajectory_{split}/job_{route}'/case['file'];assert sha(path)==case['file_sha256']
                with np.load(path) as raw:
                    pixels=raw['pixels'][int(start)//5+np.arange(6)]
                    x=torch.tensor(pixels,device='cuda').permute(0,3,1,2)[None].float()/255
                    encoded=model.encode({'pixels':(x-im)/sd})['emb'][0].cpu().numpy()
                    np.testing.assert_array_equal(encoded,z['observed'][i])
                    np.testing.assert_array_equal(raw['states'][int(start)+np.arange(6)*5],z['raw_states'][i])
                    actions=(raw['actions'][int(start):int(start)+25].astype(np.float32).reshape(5,10)-am)/ast
                    np.testing.assert_array_equal(actions,z['normalized_actions'][i])
            cache_checks.append(dict(split=split,windows=len(z['observed']),report_sha256=sha(cache/f'C_{split}.json'),
                raw_image_encodings_exact=True,physical_times_exact=True,action_blocks_exact=True))
    selection=json.loads((Path(a.heads)/f'group_{g}/selection.json').read_text());hp=Path(selection['head_A']['path']);assert sha(hp)==selection['head_A']['sha256'];head=dict(np.load(hp));rows=[]
    lock=json.loads((ROOT/'strengthening/configs/C_lambda.lock.json').read_text());d=cached['validation'];truth=d['raw_states'][:,3:,2:4]
    for q,obj in enumerate(['decoded_teacher','physical_labels']):
        ar=Path(a.A_runs)/f'job_{2*g+q}';initial=torch.load(ar/'last_weights.pt',map_location='cpu',weights_only=True)
        for condition in ['T0','T1','T2']:
            folder=Path(a.runs)/f'pool{pool}_{obj}_{condition}';assert (folder/'DONE').exists();report=json.loads((folder/'report.json').read_text())
            assert report['updates']==2100 and report['head_A_sha256']==sha(hp) and report['initial_A_checkpoint_sha256']==sha(ar/'last_weights.pt')
            assert report['anchor_lambda']==(lock['selected_lambda'] if condition=='T2' else 0.)
            assert sha(folder/'last_weights.pt')==report['weights_sha256'];model.load_state_dict(torch.load(folder/'last_weights.pt',map_location='cpu',weights_only=True),strict=True)
            frozen=configure_dynamics_only(model)
            for name,value in frozen['frozen'].items():assert torch.equal(value.cpu(),initial[name])
            assert tensor_digest(frozen['frozen'])==report['frozen_sha256'];scores={}
            with torch.inference_mode():
                for mode in ['observed_history','free_running']:
                    outputs=[]
                    for i in range(0,len(truth),128):
                        observed=torch.as_tensor(d['observed'][i:i+128],device='cuda');actions=model.action_encoder(torch.as_tensor(d['normalized_actions'][i:i+128],device='cuda'))
                        history=observed[:,:3];steps=[]
                        for step in range(3):
                            context=observed[:,step:step+3] if mode=='observed_history' else history[:,-3:]
                            token=model.predict(context,actions[:,step:step+3])[:,-1:];steps.append(token[:,0]);history=torch.cat([history,token],1)
                        outputs.append(torch.stack(steps,1).cpu().numpy())
                    decoded=numpy_pose(np.concatenate(outputs),head);errors=((decoded[...,2:4]-truth)**2).sum(-1)
                    scores[mode]=dict(endpoint_mse=float(errors[:,-1].mean()),per_step_mse=errors.mean(0).tolist())
                    np.testing.assert_allclose(scores[mode]['per_step_mse'],report['validation'][mode]['per_step_mse'],rtol=1e-9,atol=1e-7)
            verify_frozen(model,frozen);rows.append(dict(name=folder.name,report_sha256=sha(folder/'report.json'),weights_sha256=report['weights_sha256'],
                frozen_equal_start=True,independent_validation=scores))
    out=Path(a.output)/f'pool_{pool}';out.mkdir(parents=True,exist_ok=False)
    atomic_json(out/'report.json',dict(status='PASS_C_RAW_CACHE_AND_INDEPENDENT_ENDPOINT_RECONSTRUCTION',pool=pool,cache_checks=cache_checks,
        rows=rows,lambda_lock_sha256=sha(ROOT/'strengthening/configs/C_lambda.lock.json'),source_sha256=sha(__file__),
        scope='All 960 raw continuous windows and six final models per pool; optimizer trajectory not rerun. No confirmation or intervention effects.'))
    (out/'DONE').write_text('accepted\n')


if __name__=='__main__':main()
