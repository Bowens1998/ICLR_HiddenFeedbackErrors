"""Reload both actual checkpoints; independently replay temporal predictions on CPU."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from factorial_model import make_model,state_features
from adaptation_freeze import configure_dynamics_only,verify_frozen,TRAINABLE_ROOTS
from run_adaptation_checkpoint_gate import sha,tensor_digest

def main():
    p=argparse.ArgumentParser()
    for k in ['input','official','config']:p.add_argument('--'+k,required=True)
    a=p.parse_args();d=Path(a.input);r=json.loads((d/'report.json').read_text());assert r['status']=='UPDATED_REQUIRES_INDEPENDENT_ACCEPTANCE'
    e=r['entry'];td=Path(e['training_path']);assert sha(td/'last_weights.pt')==e['weights_sha256'];tr=json.loads((td/'summary.json').read_text());assert sha(td/'summary.json')==e['training_summary_sha256'] and sha(a.config)==tr['config_sha256']
    torch.set_num_threads(2);original=torch.load(td/'last_weights.pt',map_location='cpu',weights_only=True);rows=[]
    assert [s['stream'] for s in r['streams']]==['expert','planner'] and len(r['expert_identities'])==len(r['planner_identities'])==20
    for s in r['streams']:
        folder=d/s['stream'];assert sha(folder/'weights.pt')==s['weights_sha256'] and sha(folder/'forward.npz')==s['forward_sha256']
        model=make_model(a.official,a.config,e['arm'],tr['seed']);model.load_state_dict(original,strict=True);boundary=configure_dynamics_only(model)
        assert boundary['trainable_names']==s['trainable_names'] and list(boundary['frozen'])==s['frozen_names']
        assert tensor_digest(original)==s['initial_tensor_sha256'] and tensor_digest(boundary['frozen'])==s['frozen_before_sha256']==s['frozen_after_sha256']
        z=np.load(folder/'forward.npz');np.testing.assert_array_equal(z['identities'],r[s['stream']+'_identities']);np.testing.assert_array_equal(z['observed'],z['observed_after'])
        obs=torch.from_numpy(z['observed']);act=torch.from_numpy(z['normalized_actions'])
        if e['score']=='state':
            norm=e['target_normalization'];truth=(state_features(torch.from_numpy(z['raw_states']).float())-torch.tensor(norm['mean']))/torch.tensor(norm['std']);np.testing.assert_allclose(z['target'],truth[:,1:].numpy(),rtol=2e-5,atol=2e-5)
        else:np.testing.assert_array_equal(z['target'],z['observed'][:,1:])
        with torch.no_grad():before=model.predict(obs[:,:3],model.action_encoder(act)).numpy()
        np.testing.assert_allclose(before,z['before'],rtol=2e-5,atol=2e-5)
        updated=torch.load(folder/'weights.pt',map_location='cpu',weights_only=True);model.load_state_dict(updated,strict=True);verify_frozen(model,boundary)
        changed=[n for n,v in updated.items() if not torch.equal(v,original[n])];assert changed==s['changed_names'] and set(changed)<=set(s['trainable_names'])
        for root,count in s['changed_per_root'].items():assert count==sum(n.startswith(root+'.') for n in changed) and count>0
        with torch.no_grad():after=model.predict(obs[:,:3],model.action_encoder(act)).numpy()
        np.testing.assert_allclose(after,z['after'],rtol=2e-5,atol=2e-5);assert len(s['losses'])==10 and np.isfinite(s['losses']).all()
        loss=s['loss_weight']*np.mean((z['after'].astype(float)-z['target'].astype(float))**2);np.testing.assert_allclose(loss,s['final_prediction_loss'],rtol=2e-5,atol=2e-5)
        rows.append(dict(stream=s['stream'],changed_parameters=len(changed),max_before_cpu_difference=float(abs(before-z['before']).max()),max_after_cpu_difference=float(abs(after-z['after']).max())))
    result=dict(status='PASS_REAL_CHECKPOINT_UPDATE_BOUNDARY',index=r['index'],rows=rows,report_sha256=sha(d/'report.json'),source_sha256=sha(__file__),scope='Independent checkpoint reload, bytewise frozen state, CPU temporal prediction/target/loss reconstruction. Cached image encodings checked unchanged, not independently recomputed on CPU; engineering only.')
    with (d/'acceptance.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print('PASS',r['index'])

if __name__=='__main__':main()
