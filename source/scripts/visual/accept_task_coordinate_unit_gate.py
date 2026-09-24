"""Fresh CPU checkpoints and independent NumPy reconstruction of target semantics."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from adaptation_streams import sha
from factorial_model import make_model
from adaptation_freeze import configure_dynamics_only,verify_frozen
from run_adaptation_checkpoint_gate import tensor_digest
from extract_pose_selected_endpoints import decode


def main():
    p=argparse.ArgumentParser()
    for k in ('run','official','config'):p.add_argument('--'+k,required=True)
    a=p.parse_args();d=Path(a.run);r=json.loads((d/'report.json').read_text());e=r['entry'];td=Path(e['training_path']);tr=json.loads((td/'summary.json').read_text())
    assert r['status']=='UPDATED_REQUIRES_CPU_ACCEPTANCE' and r['source_sha256']==sha(Path(__file__).with_name('run_task_coordinate_unit_gate.py')) and r['loss_source_sha256']==sha(Path(__file__).with_name('adaptation_task_losses.py'))
    assert sha(td/'last_weights.pt')==e['weights_sha256'] and sha(td/'summary.json')==e['training_summary_sha256'] and sha(a.config)==tr['config_sha256']
    hp=Path(e['endpoint_head']['path']);assert sha(hp)==e['endpoint_head']['sha256'];head=dict(np.load(hp));torch.set_num_threads(2)
    original=torch.load(td/'last_weights.pt',map_location='cpu',weights_only=True);rows=[]
    assert r['gradient_control']=='unit_global_norm_before_adamw' and r['gradient_control_source_sha256']==sha(Path(__file__).with_name('adaptation_gradient_control.py'))
    assert [x['objective'] for x in r['rows']]==['latent','decoded_teacher','physical_labels']
    for row in r['rows']:
        folder=d/row['objective'];assert sha(folder/'weights.pt')==row['weights_sha256'] and sha(folder/'forward.npz')==row['forward_sha256'];z=np.load(folder/'forward.npz')
        model=make_model(a.official,a.config,e['arm'],tr['seed']);model.load_state_dict(original,strict=True);boundary=configure_dynamics_only(model)
        assert boundary['trainable_names']==row['trainable_names'] and tensor_digest(boundary['frozen'])==row['frozen_before_sha256']==row['frozen_after_sha256']
        obs=torch.from_numpy(z['observed']);actions=torch.from_numpy(z['normalized_actions']);assert len(obs)==20
        if row['objective']=='latent':target=z['observed'][:,1:]
        elif row['objective']=='decoded_teacher':target=(decode(z['observed'][:,1:],head)-head['target_mean'])/head['target_scale']
        else:
            states=z['raw_states'].astype(float);physical=np.concatenate([states[...,:4],np.sin(states[...,4:5]),np.cos(states[...,4:5])],axis=-1);target=((physical-head['target_mean'])/head['target_scale'])[:,1:]
        np.testing.assert_allclose(target,z['target'],rtol=1e-10,atol=1e-10)
        with torch.no_grad():before=model.predict(obs[:,:3],model.action_encoder(actions)).numpy()
        np.testing.assert_allclose(before,z['before'],rtol=2e-5,atol=2e-5)
        updated=torch.load(folder/'weights.pt',map_location='cpu',weights_only=True);model.load_state_dict(updated,strict=True);verify_frozen(model,boundary)
        changed=[n for n,v in updated.items() if not torch.equal(v,original[n])];assert changed==row['changed_names'] and set(changed)<=set(row['trainable_names'])
        for root,n in row['changed_per_root'].items():assert n>0 and n==sum(x.startswith(root+'.') for x in changed)
        with torch.no_grad():after=model.predict(obs[:,:3],model.action_encoder(actions)).numpy()
        np.testing.assert_allclose(after,z['after'],rtol=2e-5,atol=2e-5)
        assert all(abs(s['normalized_gradient_norm']-1)<1e-6 and np.isfinite(s['optimizer_update_norm']) and s['optimizer_update_norm']>0 for s in row['steps'])
        assert len(row['steps'])==10 and all(np.isfinite(s['loss']) and np.isfinite(s['unclipped_gradient_norm']) and s['unclipped_gradient_norm']>0 for s in row['steps'])
        pred=z['before'].astype(float)
        if row['objective']!='latent':pred=(decode(pred,head)-head['target_mean'])/head['target_scale']
        np.testing.assert_allclose(np.mean((pred-target)**2),row['steps'][0]['loss'],rtol=2e-5,atol=2e-5)
        rows.append(dict(objective=row['objective'],max_cpu_before_difference=float(abs(before-z['before']).max()),max_cpu_after_difference=float(abs(after-z['after']).max())))
    report=dict(status='PASS_UNIT_GRADIENT_TASK_COORDINATE_ENGINEERING',index=r['index'],rows=rows,report_sha256=sha(d/'report.json'),source_sha256=sha(__file__),scope='CPU original/final prediction reconstruction, bytewise frozen state and exact allowed changes; independent NumPy targets/initial losses. Does not rerun gradient/update trajectory or establish utility.')
    with (d/'acceptance.json').open('x') as f:json.dump(report,f,indent=2);f.write('\n')
    print('PASS',r['index'],flush=True)

if __name__=='__main__':main()
