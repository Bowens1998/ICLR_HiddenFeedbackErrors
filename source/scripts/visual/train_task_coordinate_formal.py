"""Prespecified 2100-update continuation; final checkpoint only, no selection."""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from factorial_model import make_model
from adaptation_streams import sha
from adaptation_gradient_control import unit_global_gradient
from adaptation_task_losses import target_for,coordinates
from nonlinear_pose_cost import NonlinearPoseCost,numpy_pose
from adaptation_freeze import configure_dynamics_only,verify_frozen,TRAINABLE_ROOTS
from evaluation_precision import configure_evaluation_precision
from run_adaptation_checkpoint_gate import tensor_digest


def temporal(model,arm,observed,actions):
    if arm.startswith('gru'):
        with torch.backends.cudnn.flags(enabled=False):return model.predict(observed[:,:3],model.action_encoder(actions))
    return model.predict(observed[:,:3],model.action_encoder(actions))


def main():
    p=argparse.ArgumentParser()
    for k in ('cache','freeze','official','config','output','protocol'):p.add_argument('--'+k,required=True)
    p.add_argument('--index',type=int,choices=range(24),required=True);a=p.parse_args()
    protocol_sha=sha(a.protocol)
    objective=('latent','decoded_teacher','physical_labels','native_state')[a.index%4]
    index=2*(a.index//4)+(a.index%4==3);stream='planner';cache=Path(a.cache)/f'job_{index}'
    r=json.loads((cache/'report.json').read_text());ac=json.loads((cache/'acceptance.json').read_text());e=r['entry'];td=Path(e['training_path'])
    assert r['index']==index and ac['status']=='PASS_FULL_FRESH_PROCESS_REENCODING' and ac['cache_report_sha256']==sha(cache/'report.json')
    assert ac['source_sha256']==sha(Path(__file__).with_name('accept_adaptation_cache.py'))
    assert [(x['stream'],x['windows']) for x in ac['rows']]==[(x['stream'],x['windows']) for x in r['rows']]
    tr=json.loads((td/'summary.json').read_text());assert sha(td/'summary.json')==e['training_summary_sha256'] and sha(td/'last_weights.pt')==e['weights_sha256'] and sha(a.config)==tr['config_sha256']
    for name in ('endpoint_head','goal_head'):
        if e.get(name):assert sha(e[name]['path'])==e[name]['sha256']
    freeze=Path(a.freeze);fr=json.loads((freeze/'report.json').read_text());assert sha(freeze/'update_indices.npz')==fr['schedule_sha256']
    assert sha(freeze/'acceptance.json')==r['expert_acceptance_sha256']
    schedule=np.load(freeze/'update_indices.npz')['indices'];assert schedule.shape==(2100,128)
    for epoch in schedule.reshape(210,1280):np.testing.assert_array_equal(np.sort(epoch),np.arange(1280))
    precision=configure_evaluation_precision();torch.set_num_threads(2);torch.manual_seed(1292001)
    original=torch.load(td/'last_weights.pt',map_location='cpu',weights_only=True)
    model=make_model(a.official,a.config,e['arm'],tr['seed']);model.load_state_dict(original,strict=True);model=model.cuda();boundary=configure_dynamics_only(model)
    assert boundary['trainable_names']==r['trainable_names'] and tensor_digest(model.state_dict())==r['initial_tensor_sha256']
    assert tensor_digest(boundary['frozen'])==r['frozen_tensor_sha256']
    head=None;raw_head=None
    if e['score']!='state':
        raw_head=dict(np.load(e['endpoint_head']['path']));head=NonlinearPoseCost.prepare(raw_head,'cuda')
    assert (objective=='native_state')==(e['score']=='state')
    datasets={}
    for row in r['rows']:
        fp=cache/row['file'];assert sha(fp)==row['sha256'];z=np.load(fp)
        datasets[row['stream']]={k:torch.as_tensor(z[k],device='cuda') for k in ('observed','normalized_actions','target','raw_states')}
        assert len(z['observed'])==row['windows']
        item=datasets[row['stream']];item['fit_target']=item['target'] if objective=='native_state' else target_for(objective,item['observed'],item['raw_states'],head)
    weight=.5 if e['score']=='state' else 1.
    def evaluate():
        result={}
        with torch.no_grad():
            for name,z in datasets.items():
                values=[temporal(model,e['arm'],z['observed'][i:i+128],z['normalized_actions'][i:i+128]).cpu().numpy() for i in range(0,len(z['observed']),128)]
                result[name]=np.concatenate(values)
        return result
    out=Path(a.output)/f'job_{a.index}';out.mkdir(parents=True,exist_ok=False);started=time.monotonic();before=evaluate()
    parameters=[v for v in model.parameters() if v.requires_grad];optimizer=torch.optim.AdamW(parameters,lr=1e-5,weight_decay=1e-3)
    z=datasets[stream+'_train'];assert len(z['observed'])==1280;losses=[];gradients=[];unit_norms=[];update_norms=[]
    for step,indices in enumerate(schedule):
        ix=torch.as_tensor(indices.astype(np.int64),device='cuda');optimizer.zero_grad(set_to_none=True)
        pred=temporal(model,e['arm'],z['observed'][ix],z['normalized_actions'][ix]);value=coordinates(pred,head) if objective in ('decoded_teacher','physical_labels') else pred;loss=weight*(value-z['fit_target'][ix]).square().mean();assert torch.isfinite(loss)
        loss.backward();control=unit_global_gradient(parameters);previous=[p.detach().clone() for p in parameters];optimizer.step();verify_frozen(model,boundary)
        update=float(torch.sqrt(sum((p.detach().double()-old.double()).square().sum() for p,old in zip(parameters,previous))));assert np.isfinite(update) and update>0
        losses.append(float(loss.detach()));gradients.append(control['before_norm']);unit_norms.append(control['after_norm']);update_norms.append(update)
        if (step+1)%500==0:print('UPDATED',a.index,step+1,flush=True)
    after=evaluate();verify_frozen(model,boundary);final={n:v.detach().cpu() for n,v in model.state_dict().items()}
    changed=[n for n,v in final.items() if not torch.equal(v,original[n])];assert set(changed)<=set(boundary['trainable_names'])
    root_changes={root:sum(n.startswith(root+'.') for n in changed) for root in TRAINABLE_ROOTS};assert all(root_changes.values())
    torch.save(final,out/'last_weights.pt');np.savez_compressed(out/'predictions.npz',**{phase+'__'+name:v for phase,rows in [('before',before),('after',after)] for name,v in rows.items()})
    metrics={name:{phase:float(weight*np.mean((values[name].astype(np.float64)-datasets[name]['target'].cpu().numpy().astype(np.float64))**2)) for phase,values in [('before',before),('after',after)]} for name in datasets}
    objective_metrics={}
    for name in datasets:
        objective_metrics[name]={}
        for phase,values in [('before',before),('after',after)]:
            value=values[name].astype(np.float64)
            if objective in ('decoded_teacher','physical_labels'):value=(numpy_pose(value,raw_head)-raw_head['target_mean'])/raw_head['target_scale']
            objective_metrics[name][phase]=float(weight*np.mean((value-datasets[name]['fit_target'].cpu().numpy())**2))
    for name in ('endpoint_head','goal_head'):
        if e.get(name):assert sha(e[name]['path'])==e[name]['sha256']
    assert sha(a.protocol)==protocol_sha
    report=dict(status='TRAINED_REQUIRES_INDEPENDENT_ACCEPTANCE',index=a.index,model_index=index,stream=stream,objective=objective,gradient_control='unit_global_norm_before_adamw',unit_gradient_norms=unit_norms,optimizer_update_norms=update_norms,objective_metrics=objective_metrics,protocol_sha256=protocol_sha,control_source_sha256=sha(Path(__file__).with_name('adaptation_gradient_control.py')),loss_source_sha256=sha(Path(__file__).with_name('adaptation_task_losses.py')),entry=e,updates=2100,batch_size=128,learning_rate=1e-5,weight_decay=1e-3,clip_norm=None,loss_weight=weight,losses=losses,unclipped_gradient_norms=gradients,metrics=metrics,trainable_names=boundary['trainable_names'],changed_names=changed,changed_per_root=root_changes,initial_tensor_sha256=r['initial_tensor_sha256'],frozen_before_sha256=r['frozen_tensor_sha256'],frozen_after_sha256=tensor_digest({n:final[n] for n in boundary['frozen']}),cache_report_sha256=sha(cache/'report.json'),cache_acceptance_sha256=sha(cache/'acceptance.json'),schedule_sha256=sha(freeze/'update_indices.npz'),weights_sha256=sha(out/'last_weights.pt'),predictions_sha256=sha(out/'predictions.npz'),precision=precision,elapsed_seconds=time.monotonic()-started,gpu=torch.cuda.get_device_name(),source_sha256=sha(__file__),scope='Single fixed final checkpoint after2100 unit-gradient updates. Common prediction metrics differ from the fitted objective for head-coordinate conditions; no condition selection. Four-stream before/after teacher-forced predictions retained for independent acceptance and endpoint diagnostics. No actual planning utility or evaluation result.')
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print('TRAINED',a.index,flush=True)

if __name__=='__main__':main()
