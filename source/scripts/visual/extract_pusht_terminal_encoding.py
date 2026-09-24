"""Frozen real-terminal image encoding; all inputs inherit accepted physics."""
import argparse,json,time,shutil
from pathlib import Path
import numpy as np
import torch
from factorial_model import make_model
from evaluation_precision import configure_evaluation_precision
from extract_pose_selected_endpoints import sha,decode,pose


def main():
    p=argparse.ArgumentParser()
    for k in ['plan','runs','bank','summary','selected','official','model-config','output']:p.add_argument('--'+k,required=True)
    p.add_argument('--actor',type=int,choices=range(6),required=True);p.add_argument('--engineering',action='store_true');a=p.parse_args()
    plan=json.loads(Path(a.plan).read_text());full=json.loads(Path(a.summary).read_text())
    assert full['plan_sha256']==sha(a.plan) and plan['layout']=='pusht_nonlinear_pose'
    bank=Path(a.bank);bm=json.loads((bank/'manifest.json').read_text());assert sha(bank/'manifest.json')==plan['bank_manifest_sha256']
    sd=Path(a.selected)/f'job_{a.actor}';sr=json.loads((sd/'report.json').read_text());assert sr['plan_sha256']==sha(a.plan)
    precision=configure_evaluation_precision();torch.set_num_threads(2);count=2 if a.engineering else 128
    out=Path(a.output)/f'actor_{a.actor}';out.mkdir(parents=True,exist_ok=False)
    rows=[];cache={};start=time.monotonic()
    with torch.inference_mode():
        for route in range(a.actor*8+2,a.actor*8+8):
            entry=plan['models'][plan['routes'][route]['model_index']]
            d=Path(a.runs)/f'job_{route}';summary=json.loads((d/'summary.json').read_text());acc=json.loads((d/'acceptance.json').read_text());am=json.loads((d/'artifact_manifest.json').read_text())
            binding=next(b for b in full['bindings'] if b['index']==route)
            assert sha(d/'summary.json')==binding['summary_sha256']==am['summary.json'] and sha(d/'acceptance.json')==binding['acceptance_sha256']
            assert acc['cases']==128 and acc['model_free_simulator_replay'] and summary['route_index']==route
            original=next(r for r in sr['rows'] if r['route_index']==route);sp=sd/original['file'];assert sha(sp)==original['file_sha256']
            saved=dict(np.load(sp));td=Path(entry['training_path']);tr=json.loads((td/'summary.json').read_text())
            assert sha(td/'summary.json')==entry['training_summary_sha256'] and sha(td/'last_weights.pt')==entry['weights_sha256']
            assert sha(a.model_config)==tr['config_sha256'] and tr['seed']==3072
            if entry['weights_sha256'] not in cache:
                model=make_model(a.official,a.model_config,entry['arm'],tr['seed']);model.load_state_dict(torch.load(td/'last_weights.pt',map_location='cpu',weights_only=True),strict=True)
                cache[entry['weights_sha256']]=model.cuda().eval()
            model=cache[entry['weights_sha256']]
            head=None
            if entry['score']!='state':
                hp=Path(entry['goal_head']['path']);assert sha(hp)==entry['goal_head']['sha256'];head=dict(np.load(hp))
                headname=f'route_{route}_image_head.npz';shutil.copyfile(hp,out/headname)
            im=torch.tensor([.485,.456,.406],device='cuda')[None,:,None,None];std=torch.tensor([.229,.224,.225],device='cuda')[None,:,None,None]
            def encode(pixels):
                assert pixels.shape==(224,224,3) and pixels.dtype==np.uint8
                x=(torch.as_tensor(pixels.copy(),device='cuda').permute(2,0,1)[None].float()/255-im)/std
                return model.encode({'pixels':x[:,None]})['emb'][0,0]
            tokens=[];poses=[];goals=[];cases=[]
            for i in range(count):
                b=bm['cases'][i];bp=bank/f'case_{i:03d}.npz';assert sha(bp)==b['sha256']
                path=d/f'case_{i:03d}_predictions.npz';assert sha(path)==am[path.name]
                with np.load(path) as z,np.load(bp) as truth:
                    np.testing.assert_array_equal(pose(z['selected_states'][-1]),saved['true_endpoint'][i])
                    np.testing.assert_array_equal(pose(truth['goal_state']),saved['true_goal'][i])
                    token=encode(z['terminal_pixels']);repeat=encode(z['terminal_pixels']);torch.testing.assert_close(token,repeat,rtol=0,atol=0)
                    goal=encode(truth['goal_pixels']).cpu().numpy();np.testing.assert_allclose(goal,z['goal_tokens'],rtol=2e-5,atol=2e-5)
                    assert int(truth['seed'])==b['seed']==int(saved['seed'][i])
                    token=token.cpu().numpy()
                    if head is not None:value=decode(token,head)
                    else:value=token.astype(float)*np.asarray(entry['target_normalization']['std'])+np.asarray(entry['target_normalization']['mean'])
                    tokens.append(token);poses.append(value);goals.append(goal)
                cases.append(dict(index=i,seed=b['seed'],bank_sha256=b['sha256'],archive_sha256=am[path.name]))
            file=f'route_{route}.npz';np.savez_compressed(out/file,real_tokens=np.array(tokens),real_pose=np.array(poses),goal_tokens=np.array(goals),**{k:v[:count] for k,v in saved.items()})
            rows.append(dict(route=route,interface=entry['score'],algorithm=summary['algorithm'],file=file,file_sha256=sha(out/file),cases=cases,
                             entry=entry,summary_sha256=binding['summary_sha256'],acceptance_sha256=binding['acceptance_sha256'],selected_sha256=sha(sp),
                             head_file=headname if head is not None else None,head_sha256=sha(out/headname) if head is not None else None))
    root=Path(__file__).resolve().parents[2]
    r=dict(status='EXTRACTED_REQUIRES_ACCEPTANCE',actor=a.actor,engineering=a.engineering,rows=rows,plan_sha256=sha(a.plan),summary_sha256=sha(a.summary),selected_report_sha256=sha(sd/'report.json'),
           source_sha256=sha(__file__),protocol_sha256=sha(root/'docs/maintrack/PUSHT_TERMINAL_ENCODING_PROTOCOL.md'),precision=precision,gpu=torch.cuda.get_device_name(),elapsed_seconds=time.monotonic()-start,
           sources={s:sha(root/'scripts/visual'/s) for s in ['factorial_model.py','lewm_adapter.py','evaluation_precision.py','extract_pose_selected_endpoints.py']},
           scope='Privileged real-terminal image encoding; frozen image readout, inherited accepted trajectories, no new physics/training or deployable claim.')
    (out/'report.json').write_text(json.dumps(r,indent=2)+'\n');print('EXTRACTED',a.actor,count)


if __name__=='__main__':main()
