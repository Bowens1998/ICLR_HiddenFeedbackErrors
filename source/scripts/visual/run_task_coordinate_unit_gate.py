"""Ten-update real-model engineering for loss-coordinate/label separation."""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from adaptation_streams import sha
from adaptation_gradient_control import unit_global_gradient
from adaptation_task_losses import OBJECTIVES,target_for,loss_for
from nonlinear_pose_cost import NonlinearPoseCost
from factorial_model import make_model
from adaptation_freeze import configure_dynamics_only,verify_frozen,TRAINABLE_ROOTS
from train_adaptation_formal import temporal
from run_adaptation_checkpoint_gate import tensor_digest
from evaluation_precision import configure_evaluation_precision


def main():
    p=argparse.ArgumentParser()
    for key in ('engineering','gru-engineering','official','config','output'):p.add_argument('--'+key,required=True)
    p.add_argument('--index',type=int,choices=range(6),required=True);a=p.parse_args();index=2*a.index
    folder=Path(a.engineering)/f'job_{index}'
    if not (folder/'report.json').exists():folder=Path(a.gru_engineering)/f'job_{index}'
    gr=json.loads((folder/'report.json').read_text());ga=json.loads((folder/'acceptance.json').read_text());assert ga['status']=='PASS_REAL_CHECKPOINT_UPDATE_BOUNDARY' and ga['report_sha256']==sha(folder/'report.json')
    e=gr['entry'];assert e['score']=='pose_encoded';stream=next(s for s in gr['streams'] if s['stream']=='planner');fp=folder/'planner/forward.npz';assert sha(fp)==stream['forward_sha256'];z=np.load(fp)
    td=Path(e['training_path']);tr=json.loads((td/'summary.json').read_text());assert sha(td/'summary.json')==e['training_summary_sha256'] and sha(td/'last_weights.pt')==e['weights_sha256'] and sha(a.config)==tr['config_sha256']
    hp=Path(e['endpoint_head']['path']);assert sha(hp)==e['endpoint_head']['sha256'];head=NonlinearPoseCost.prepare(dict(np.load(hp)),'cuda');assert all(not v.requires_grad for v in head.values())
    precision=configure_evaluation_precision();torch.set_num_threads(2);torch.manual_seed(1342001)
    obs=torch.tensor(z['observed'],device='cuda');actions=torch.tensor(z['normalized_actions'],device='cuda');states=torch.tensor(z['raw_states'],device='cuda');assert len(obs)==20
    original=torch.load(td/'last_weights.pt',map_location='cpu',weights_only=True);out=Path(a.output)/f'job_{a.index}';out.mkdir(parents=True,exist_ok=False);rows=[];initial_gradients=[];start=time.monotonic()
    for objective in OBJECTIVES:
        model=make_model(a.official,a.config,e['arm'],tr['seed']);model.load_state_dict(original,strict=True);model=model.cuda();boundary=configure_dynamics_only(model)
        assert boundary['trainable_names']==stream['trainable_names'];frozen_hash=tensor_digest(boundary['frozen'])
        target=target_for(objective,obs,states,head);assert not target.requires_grad
        with torch.no_grad():before=temporal(model,e['arm'],obs,actions).cpu().numpy()
        np.testing.assert_allclose(before,z['before'],rtol=2e-5,atol=2e-5)
        params=[v for v in model.parameters() if v.requires_grad];optimizer=torch.optim.AdamW(params,lr=1e-5,weight_decay=1e-3);steps=[]
        for step in range(10):
            optimizer.zero_grad(set_to_none=True);pred=temporal(model,e['arm'],obs,actions);loss=loss_for(objective,pred,target,head);assert torch.isfinite(loss);loss.backward()
            root_stats={}
            for root in TRAINABLE_ROOTS:
                gradients=[v.grad for n,v in model.named_parameters() if n.startswith(root+'.') and v.requires_grad and v.grad is not None]
                assert gradients and all(torch.isfinite(g).all() for g in gradients)
                norm=float(torch.sqrt(sum(g.double().square().sum() for g in gradients)));nonzero=sum(int(torch.count_nonzero(g)) for g in gradients);assert norm>0 and nonzero>0
                root_stats[root]=dict(norm=norm,nonzero=nonzero)
            if step==0:initial_gradients.append(torch.cat([(p.grad.detach() if p.grad is not None else torch.zeros_like(p)).flatten().cpu().double() for p in params]).numpy())
            control=unit_global_gradient(params);previous=[p.detach().clone() for p in params];optimizer.step();verify_frozen(model,boundary)
            update_norm=float(torch.sqrt(sum((p.detach().double()-old.double()).square().sum() for p,old in zip(params,previous))));assert np.isfinite(update_norm) and update_norm>0
            steps.append(dict(loss=float(loss.detach()),unclipped_gradient_norm=control['before_norm'],normalized_gradient_norm=control['after_norm'],would_clip_at_one=control['before_norm']>1,optimizer_update_norm=update_norm,roots=root_stats))
        with torch.no_grad():after=temporal(model,e['arm'],obs,actions).cpu().numpy()
        final={n:v.detach().cpu() for n,v in model.state_dict().items()};changed=[n for n,v in final.items() if not torch.equal(v,original[n])];assert set(changed)<=set(boundary['trainable_names'])
        changes={root:sum(n.startswith(root+'.') for n in changed) for root in TRAINABLE_ROOTS};assert all(changes.values())
        d=out/objective;d.mkdir();torch.save(final,d/'weights.pt');np.savez_compressed(d/'forward.npz',observed=z['observed'],normalized_actions=z['normalized_actions'],raw_states=z['raw_states'],identities=z['identities'],target=target.cpu().numpy(),before=before,after=after)
        rows.append(dict(objective=objective,steps=steps,changed_names=changed,changed_per_root=changes,trainable_names=boundary['trainable_names'],frozen_before_sha256=frozen_hash,frozen_after_sha256=tensor_digest({n:final[n] for n in boundary['frozen']}),weights_sha256=sha(d/'weights.pt'),forward_sha256=sha(d/'forward.npz')))
        del model,optimizer,boundary;torch.cuda.empty_cache()
    cosines=[]
    for i in range(3):
        for j in range(i+1,3):
            x,y=initial_gradients[i],initial_gradients[j];cosines.append(dict(left=OBJECTIVES[i],right=OBJECTIVES[j],cosine=float(np.dot(x,y)/(np.linalg.norm(x)*np.linalg.norm(y)))))
    assert sha(hp)==e['endpoint_head']['sha256']
    report=dict(status='UPDATED_REQUIRES_CPU_ACCEPTANCE',gradient_control='unit_global_norm_before_adamw',gradient_control_source_sha256=sha(Path(__file__).with_name('adaptation_gradient_control.py')),index=a.index,original_model_index=index,entry=e,rows=rows,initial_gradient_cosines=cosines,engineering_report_sha256=sha(folder/'report.json'),engineering_forward_sha256=sha(fp),precision=precision,source_sha256=sha(__file__),loss_source_sha256=sha(Path(__file__).with_name('adaptation_task_losses.py')),elapsed_seconds=time.monotonic()-start,scope='Three objectives, ten20-window updates each on two engineering contexts. Finite nonzero gradients and allowed parameter changes required; head/encoder/buffers frozen. All objectives use unit global gradient norm before AdamW; actual optimizer update norms recorded and may differ. No validation/planning result.')
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print('UPDATED',a.index,flush=True)

if __name__=='__main__':main()
