"""CPU reconstruction of original/final full-stream predictions and frozen state."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from adaptation_streams import sha
from factorial_model import make_model
from adaptation_freeze import configure_dynamics_only,verify_frozen
from run_adaptation_checkpoint_gate import tensor_digest


def main():
    p=argparse.ArgumentParser()
    for k in ('run','cache','freeze','official','config'):p.add_argument('--'+k,required=True)
    a=p.parse_args();d=Path(a.run);r=json.loads((d/'report.json').read_text());e=r['entry'];cache=Path(a.cache)/f"job_{r['model_index']}"
    assert r['status']=='TRAINED_REQUIRES_INDEPENDENT_ACCEPTANCE' and r['source_sha256']==sha(Path(__file__).with_name('train_adaptation_formal.py'))
    assert sha(cache/'report.json')==r['cache_report_sha256'] and sha(cache/'acceptance.json')==r['cache_acceptance_sha256']
    cr=json.loads((cache/'report.json').read_text());assert cr['entry']==e
    assert sha(Path(a.freeze)/'update_indices.npz')==r['schedule_sha256']
    assert r['updates']==2100 and r['batch_size']==128 and r['learning_rate']==1e-5 and r['weight_decay']==1e-3 and r['clip_norm']==1.
    assert len(r['losses'])==len(r['unclipped_gradient_norms'])==2100 and np.isfinite(r['losses']).all() and np.isfinite(r['unclipped_gradient_norms']).all()
    assert sha(d/'last_weights.pt')==r['weights_sha256'] and sha(d/'predictions.npz')==r['predictions_sha256'];saved=np.load(d/'predictions.npz')
    td=Path(e['training_path']);tr=json.loads((td/'summary.json').read_text());assert sha(td/'last_weights.pt')==e['weights_sha256'] and sha(td/'summary.json')==e['training_summary_sha256'] and sha(a.config)==tr['config_sha256']
    torch.set_num_threads(2);original=torch.load(td/'last_weights.pt',map_location='cpu',weights_only=True);updated=torch.load(d/'last_weights.pt',map_location='cpu',weights_only=True)
    model=make_model(a.official,a.config,e['arm'],tr['seed']);model.load_state_dict(original,strict=True);boundary=configure_dynamics_only(model)
    assert boundary['trainable_names']==r['trainable_names'] and tensor_digest(original)==r['initial_tensor_sha256']
    assert tensor_digest(boundary['frozen'])==r['frozen_before_sha256']==r['frozen_after_sha256']
    changed=[n for n,v in updated.items() if not torch.equal(v,original[n])];assert changed==r['changed_names'] and set(changed)<=set(r['trainable_names'])
    for root,count in r['changed_per_root'].items():assert count==sum(n.startswith(root+'.') for n in changed) and count>0
    rows=[];weight=.5 if e['score']=='state' else 1.;assert weight==r['loss_weight']
    for phase,state in [('before',original),('after',updated)]:
        model.load_state_dict(state,strict=True);verify_frozen(model,boundary)
        for row in cr['rows']:
            fp=cache/row['file'];assert sha(fp)==row['sha256'];z=np.load(fp);values=[]
            with torch.no_grad():
                for i in range(0,len(z['observed']),128):
                    obs=torch.from_numpy(z['observed'][i:i+128]);actions=torch.from_numpy(z['normalized_actions'][i:i+128]);values.append(model.predict(obs[:,:3],model.action_encoder(actions)).numpy())
            pred=np.concatenate(values);reference=saved[phase+'__'+row['stream']]
            np.testing.assert_allclose(pred,reference,rtol=2e-5,atol=2e-5)
            loss=weight*np.mean((reference.astype(float)-z['target'].astype(float))**2)
            np.testing.assert_allclose(loss,r['metrics'][row['stream']][phase],rtol=2e-5,atol=2e-5)
            rows.append(dict(phase=phase,stream=row['stream'],windows=len(pred),max_abs_cpu_difference=float(np.max(abs(pred-reference)))))
    for name in ('endpoint_head','goal_head'):
        if e.get(name):assert sha(e[name]['path'])==e[name]['sha256']
    result=dict(status='PASS_FORMAL_FIT_FROZEN_STATE_AND_CPU_PREDICTIONS',index=r['index'],rows=rows,report_sha256=sha(d/'report.json'),source_sha256=sha(__file__),scope='Every before/after cached-input temporal prediction independently reconstructed on CPU; original/final checkpoints reloaded, all frozen tensors/buffers and unchanged head hashes checked. Optimizer trajectory not independently rerun. Full source-image cache reconstruction is separately accepted; no actual planning utility.')
    with (d/'acceptance.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print('PASS',r['index'],flush=True)

if __name__=='__main__':main()
