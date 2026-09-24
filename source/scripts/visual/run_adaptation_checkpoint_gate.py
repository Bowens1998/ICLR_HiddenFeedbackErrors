"""Actual-checkpoint engineering; no scientific adaptation or model selection."""
import argparse,json,hashlib,time
from pathlib import Path
import numpy as np
import torch
from factorial_model import make_model,state_features
from adaptation_windows import starts,dense_window,recorded_window
from adaptation_freeze import configure_dynamics_only,verify_frozen,TRAINABLE_ROOTS
from evaluation_precision import configure_evaluation_precision

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def tensor_digest(state):
    h=hashlib.sha256()
    for n,v in sorted(state.items()):
        h.update(n.encode());h.update(str(v.dtype).encode());h.update(str(tuple(v.shape)).encode())
        h.update(v.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser()
    for k in ['plan','trajectories','data','official','config','output']:p.add_argument('--'+k,required=True)
    p.add_argument('--index',type=int,choices=range(12),required=True);a=p.parse_args()
    actor=a.index//2;offset=2 if a.index%2==0 else 6;route=actor*8+offset
    plan=json.loads(Path(a.plan).read_text());entry=plan['models'][plan['routes'][route]['model_index']]
    td=Path(entry['training_path']);tr=json.loads((td/'summary.json').read_text())
    assert sha(td/'summary.json')==entry['training_summary_sha256'] and sha(td/'last_weights.pt')==entry['weights_sha256'] and tr['seed']==3072
    assert sha(a.config)==tr['config_sha256'];data=Path(a.data)/f'replica_{actor//2}'/'n256';dm=json.loads((data/'manifest.json').read_text())
    assert sha(data/'manifest.json')==entry['data_manifest_sha256']==tr['data_manifest_sha256']
    for name in ['pixels.npy','state.npy','action.npy','episodes.npz']:
        assert sha(data/'train'/name)==dm['splits']['train']['files'][name]
    ar={k:np.load(data/'train'/f'{k}.npy',mmap_mode='r') for k in ['pixels','state','action']};ep=np.load(data/'train/episodes.npz')
    expert=[];expert_ids=[]
    for offset0,length,identity in zip(ep['offsets'],ep['lengths'],ep['source_episode_ids']):
        start0=int(offset0);n=int(length)
        for s in starts(n,n):
            expert.append(dense_window(ar['pixels'][start0:start0+n],ar['state'][start0:start0+n],ar['action'][start0:start0+n],s));expert_ids.append([int(identity),s])
            if len(expert)==20:break
        if len(expert)==20:break
    planner=[];planner_ids=[];trajectory_bindings=[]
    for rid in [route,route+1]:
        folder=Path(a.trajectories)/f'job_{rid}';r=json.loads((folder/'report.json').read_text())
        assert r['status']=='PASS_REPLAYED_COMPLETE_SELECTED_TRAJECTORIES' and r['route']==rid and r['plan_sha256']==sha(a.plan)
        trajectory_bindings.append(dict(route=rid,report_sha256=sha(folder/'report.json')))
        for c in r['cases']:
            fp=folder/c['file'];assert sha(fp)==c['file_sha256'];z=np.load(fp)
            for s in starts(len(z['states']),len(z['actions'])):
                planner.append(recorded_window(z['pixels'],z['states'],z['actions'],s));planner_ids.append([rid,c['index'],c['seed'],s])
    assert len(expert)==len(planner)==20
    precision=configure_evaluation_precision();torch.set_num_threads(2);torch.manual_seed(1292001)
    out=Path(a.output)/f'job_{a.index}';out.mkdir(parents=True,exist_ok=False);reports=[];start=time.monotonic()
    for stream,windows,ids in [('expert',expert,expert_ids),('planner',planner,planner_ids)]:
        model=make_model(a.official,a.config,entry['arm'],tr['seed']);original=torch.load(td/'last_weights.pt',map_location='cpu',weights_only=True);model.load_state_dict(original,strict=True);model=model.cuda()
        boundary=configure_dynamics_only(model);initial_hash=tensor_digest(model.state_dict());frozen_hash=tensor_digest(boundary['frozen'])
        pixels=np.stack([w[0] for w in windows]);actions=np.stack([w[1] for w in windows]);states=np.stack([w[2] for w in windows])
        im=torch.tensor([.485,.456,.406],device='cuda')[None,None,:,None,None];sd=torch.tensor([.229,.224,.225],device='cuda')[None,None,:,None,None]
        x=(torch.as_tensor(pixels,device='cuda').permute(0,1,4,2,3).float()/255-im)/sd
        am=torch.tensor(tr['normalization']['mean'],device='cuda').repeat(5);asd=torch.tensor(tr['normalization']['std'],device='cuda').repeat(5)
        actions_t=(torch.as_tensor(actions,device='cuda').float()-am)/asd
        with torch.no_grad():observed=model.encode({'pixels':x})['emb'].detach()
        if entry['score']=='state':
            norm=entry['target_normalization'];target=(state_features(torch.as_tensor(states,device='cuda').float())-torch.tensor(norm['mean'],device='cuda'))/torch.tensor(norm['std'],device='cuda');target=target[:,1:];weight=.5
        else:target=observed[:,1:];weight=1.
        def forward():
            # Native recurrent ops support eval-mode autograd. cuDNN RNN's
            # inference path does not; keep dropout/buffer policy unchanged.
            if entry['arm'].startswith('gru'):
                with torch.backends.cudnn.flags(enabled=False):
                    return model.predict(observed[:,:3],model.action_encoder(actions_t))
            return model.predict(observed[:,:3],model.action_encoder(actions_t))
        with torch.no_grad():before=forward().detach().cpu().numpy()
        opt=torch.optim.AdamW([v for v in model.parameters() if v.requires_grad],lr=1e-5,weight_decay=1e-3);losses=[]
        for _ in range(10):
            opt.zero_grad(set_to_none=True);pred=forward();loss=weight*(pred-target).square().mean();assert torch.isfinite(loss);loss.backward()
            gn=torch.nn.utils.clip_grad_norm_([v for v in model.parameters() if v.requires_grad],1.);assert torch.isfinite(gn);opt.step();verify_frozen(model,boundary);losses.append(float(loss.detach()))
        with torch.no_grad():after=forward().detach().cpu().numpy();encoded_after=model.encode({'pixels':x})['emb']
        torch.testing.assert_close(encoded_after,observed,rtol=0,atol=0);verify_frozen(model,boundary)
        final=model.state_dict();changed=[n for n,v in final.items() if not torch.equal(v.detach().cpu(),original[n])]
        assert set(changed)<=set(boundary['trainable_names'])
        root_changes={root:sum(n.startswith(root+'.') for n in changed) for root in TRAINABLE_ROOTS}
        for root in TRAINABLE_ROOTS:
            if any(n.startswith(root+'.') for n in boundary['trainable_names']):assert root_changes[root]>0
        folder=out/stream;folder.mkdir();torch.save({n:v.detach().cpu() for n,v in final.items()},folder/'weights.pt')
        np.savez_compressed(folder/'forward.npz',observed=observed.cpu().numpy(),observed_after=encoded_after.cpu().numpy(),before=before,after=after,target=target.cpu().numpy(),identities=np.array(ids),normalized_actions=actions_t.cpu().numpy(),raw_states=states)
        reports.append(dict(stream=stream,initial_tensor_sha256=initial_hash,frozen_before_sha256=frozen_hash,frozen_after_sha256=tensor_digest({n:final[n] for n in boundary['frozen']}),trainable_names=boundary['trainable_names'],frozen_names=list(boundary['frozen']),changed_names=changed,changed_per_root=root_changes,loss_weight=weight,losses=losses,final_prediction_loss=float(weight*np.mean((after-target.cpu().numpy())**2)),weights_sha256=sha(folder/'weights.pt'),forward_sha256=sha(folder/'forward.npz')))
        del model,opt,boundary;torch.cuda.empty_cache()
    assert reports[0]['initial_tensor_sha256']==reports[1]['initial_tensor_sha256']
    result=dict(status='UPDATED_REQUIRES_INDEPENDENT_ACCEPTANCE',index=a.index,actor=actor,arm=entry['arm'],route=route,entry=entry,streams=reports,expert_identities=expert_ids,planner_identities=planner_ids,trajectory_bindings=trajectory_bindings,data_manifest_sha256=sha(data/'manifest.json'),plan_sha256=sha(a.plan),precision=precision,gpu=torch.cuda.get_device_name(),elapsed_seconds=time.monotonic()-start,source_sha256=sha(__file__),sources={n:sha(Path(__file__).with_name(n)) for n in ['adaptation_windows.py','adaptation_freeze.py','factorial_model.py','lewm_adapter.py','evaluation_precision.py']},scope='Ten-update real-checkpoint engineering on20 windows per stream; no validation efficacy or formal training claim.')
    (out/'report.json').write_text(json.dumps(result,indent=2)+'\n');print('UPDATED',a.index)

if __name__=='__main__':main()
