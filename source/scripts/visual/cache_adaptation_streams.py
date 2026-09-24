"""Frozen original-model encodings for the prespecified formal continuation streams."""
import argparse,json,time,hashlib
from pathlib import Path
import numpy as np
import torch
from adaptation_streams import expert_windows,planner_windows,sha
from factorial_model import make_model,state_features
from evaluation_precision import configure_evaluation_precision
from adaptation_freeze import configure_dynamics_only,verify_frozen
from run_adaptation_checkpoint_gate import tensor_digest


def main():
    p=argparse.ArgumentParser()
    for k in ('train-plan','validation-plan','train-trajectories','validation-trajectories','train-acceptance','validation-acceptance','data','freeze','official','config','output'):p.add_argument('--'+k,required=True)
    p.add_argument('--index',type=int,choices=range(12),required=True);a=p.parse_args()
    actor=a.index//2;route=actor*8+(2 if a.index%2==0 else 6)
    plan=json.loads(Path(a.train_plan).read_text());vp=json.loads(Path(a.validation_plan).read_text())
    assert plan['models']==vp['models'] and plan['routes']==vp['routes']
    entry=plan['models'][plan['routes'][route]['model_index']];td=Path(entry['training_path']);tr=json.loads((td/'summary.json').read_text())
    assert sha(td/'summary.json')==entry['training_summary_sha256'] and sha(td/'last_weights.pt')==entry['weights_sha256']
    assert sha(a.config)==tr['config_sha256'] and tr['seed']==3072
    assert sha(Path(a.data)/f'replica_{actor//2}'/'n256/manifest.json')==entry['data_manifest_sha256']
    fr=Path(a.freeze);fa=json.loads((fr/'acceptance.json').read_text())
    assert fa['status']=='PASS_ALL_FIXED_EXPERT_TRAIN_AND_VALIDATION_WINDOWS' and fa['freeze_report_sha256']==sha(fr/'report.json')
    assert fa['stream_source_sha256']==sha(Path(__file__).with_name('adaptation_streams.py'))
    bindings={}
    for split in ('train','validation'):
        pp=Path(getattr(a,split+'_plan'));ap=Path(getattr(a,split+'_acceptance'));ac=json.loads(ap.read_text())
        assert ac['status']=='PASS_COMPLETE_FORMAL_TRAJECTORIES_AND_WINDOWS' and ac['split']==split and ac['plan_sha256']==sha(pp)
        for rid in (route,route+1):
            row=next(r for r in ac['rows'] if r['route']==rid)
            assert sha(Path(getattr(a,split+'_trajectories'))/f'job_{rid}'/'report.json')==row['trajectory_report_sha256']
        bindings[split]=dict(plan_sha256=sha(pp),collection_acceptance_sha256=sha(ap))
    precision=configure_evaluation_precision();torch.set_num_threads(2);torch.manual_seed(1292001)
    model=make_model(a.official,a.config,entry['arm'],tr['seed']);model.load_state_dict(torch.load(td/'last_weights.pt',map_location='cpu',weights_only=True),strict=True);model=model.cuda()
    boundary=configure_dynamics_only(model);before=tensor_digest(model.state_dict())
    out=Path(a.output)/f'job_{a.index}';out.mkdir(parents=True,exist_ok=False);rows=[];started=time.monotonic()
    im=torch.tensor([.485,.456,.406],device='cuda')[None,None,:,None,None];sd=torch.tensor([.229,.224,.225],device='cuda')[None,None,:,None,None]
    am=torch.tensor(tr['normalization']['mean'],device='cuda').repeat(5);astd=torch.tensor(tr['normalization']['std'],device='cuda').repeat(5)
    for kind,split in (('expert','train'),('planner','train'),('expert','validation'),('planner','validation')):
        iterator=expert_windows(a.data,a.freeze,actor//2,split) if kind=='expert' else planner_windows(getattr(a,split+'_trajectories'),getattr(a,split+'_plan'),route)
        observed=[];actions=[];states=[];identities=[];targets=[];pixel_hashes=[]
        for identity,(pixels,raw_actions,raw_states) in iterator:
            # Fixed one-window/four-image encoder call, independent of total stream size.
            x=(torch.as_tensor(pixels[None],device='cuda').permute(0,1,4,2,3).float()/255-im)/sd
            with torch.no_grad():
                z=model.encode({'pixels':x})['emb'];u=(torch.as_tensor(raw_actions,device='cuda').float()-am)/astd
                if entry['score']=='state':
                    norm=entry['target_normalization'];target=(state_features(torch.as_tensor(raw_states,device='cuda').float())-torch.tensor(norm['mean'],device='cuda'))/torch.tensor(norm['std'],device='cuda');target=target[1:]
                else:target=z[0,1:]
            assert torch.isfinite(z).all() and torch.isfinite(u).all() and torch.isfinite(target).all()
            observed.append(z[0].cpu().numpy());actions.append(u.cpu().numpy());states.append(raw_states);identities.append(identity);targets.append(target.cpu().numpy());pixel_hashes.append(hashlib.sha256(pixels.tobytes()).hexdigest())
        expected=1280 if split=='train' else (1441 if kind=='expert' else 320)
        assert len(identities)==expected
        name=f'{kind}_{split}.npz'
        np.savez_compressed(out/name,observed=np.stack(observed),normalized_actions=np.stack(actions),raw_states=np.stack(states),identities=np.asarray(identities),target=np.stack(targets),pixel_sha256=np.asarray(pixel_hashes))
        rows.append(dict(stream=kind+'_'+split,windows=expected,file=name,sha256=sha(out/name)));print('CACHED',a.index,kind,split,expected,flush=True)
    verify_frozen(model,boundary);assert tensor_digest(model.state_dict())==before
    report=dict(status='CACHED_REQUIRES_REENCODING_ACCEPTANCE',index=a.index,actor=actor,route=route,entry=entry,rows=rows,bindings=bindings,expert_acceptance_sha256=sha(fr/'acceptance.json'),initial_tensor_sha256=before,trainable_names=boundary['trainable_names'],frozen_tensor_sha256=tensor_digest(boundary['frozen']),precision=precision,gpu=torch.cuda.get_device_name(),elapsed_seconds=time.monotonic()-started,source_sha256=sha(__file__),sources={n:sha(Path(__file__).with_name(n)) for n in ('adaptation_streams.py','adaptation_windows.py','factorial_model.py','lewm_adapter.py','adaptation_freeze.py','evaluation_precision.py')},scope='Original checkpoint, eval FP32, one four-image window per encoder call. All streams cached without updates. Raw states are validation metadata for JEPA; its loss targets are encoded observations only. Requires complete source-image reencoding acceptance before fitting.')
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':main()
