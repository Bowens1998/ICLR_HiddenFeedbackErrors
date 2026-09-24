"""Independent NumPy reconstruction and frozen-state audit of the head pilot."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def decode(z,h):
    value=(np.asarray(z,dtype=np.float64)-h['mean'])/h['scale']
    for layer in [0,2,4]:
        value=np.einsum('...d,od->...o',value,h[f'{layer}.weight'])+h[f'{layer}.bias']
        if layer!=4:value=np.clip(value,0,None)
    return value*h['target_scale']+h['target_mean']
def target(state):
    state=np.asarray(state,dtype=np.float64)
    return np.concatenate([state[...,:4],np.sin(state[...,4:5]),np.cos(state[...,4:5])],axis=-1)
def metrics(x,y,scale):
    error=x-y
    return {'normalized_mse':float(np.mean(np.square(error/scale))),
        'block_position_mse':float(np.mean(np.square(error[...,2:4]).sum(-1))),
        'per_output_mse':np.mean(np.square(error.reshape(-1,6)),axis=0).tolist()}

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--cache',required=True);p.add_argument('--protocol',required=True);a=p.parse_args()
    d=Path(a.run);r=json.loads((d/'report.json').read_text());cache=Path(a.cache)/f"job_{2*r['group']}"
    cr=json.loads((cache/'report.json').read_text());assert sha(cache/'report.json')==r['cache_report_sha256']
    assert sha(a.protocol)==r['protocol_sha256'];assert r['source_sha256']==sha(Path(__file__).with_name('run_readout_calibration_pilot.py'))
    old=dict(np.load(d/'original_head.npz'));new=dict(np.load(d/'fitted_head.npz'))
    original_head=dict(np.load(cr['entry']['endpoint_head']['path']))
    for key in old:np.testing.assert_array_equal(old[key],original_head[key])
    assert sha(d/'fitted_head.npz')==r['fitted_head_sha256'] and sha(d/'head_predictions.npz')==r['head_predictions_sha256']
    for key in ['mean','scale','target_mean','target_scale']:np.testing.assert_array_equal(old[key],new[key])
    assert r['head_updates']==len(r['head_losses'])==3000 and np.isfinite(r['head_losses']).all()
    arrays={};rebuilt={};maximum=0.
    with np.load(d/'head_predictions.npz') as saved:
        for name,binding in r['data_bindings'].items():
            assert sha(binding['file'])==binding['sha256'];z=dict(np.load(binding['file']));arrays[name]=z;truth=target(z['raw_states'])
            np.testing.assert_allclose(truth,saved[name+'__truth'],rtol=0,atol=1e-12)
            rebuilt[name]={}
            for tag,head in [('old',old),('new',new)]:
                prediction=decode(z['observed'],head)
                maximum=max(maximum,float(np.max(abs(prediction-saved[name+'__'+tag]))))
                np.testing.assert_allclose(prediction,saved[name+'__'+tag],rtol=1e-11,atol=1e-8)
                mm=metrics(prediction,truth,head['target_scale']);rebuilt[name][tag]=mm
                for key in mm:np.testing.assert_allclose(mm[key],r['head_metrics'][name][tag][key],rtol=1e-11,atol=1e-8)
    gate=all(rebuilt['planner_validation']['new'][key]<=.9*rebuilt['planner_validation']['old'][key] for key in ['normalized_mse','block_position_mse'])
    assert gate==r['head_gate_passed']
    training_path=Path(cr['entry']['training_path']);assert sha(training_path/'last_weights.pt')==r['original_model_sha256']
    original=torch.load(training_path/'last_weights.pt',map_location='cpu',weights_only=True)
    fixed=set(original)-set(cr['trainable_names']);model_checks=[]
    assert len(r['continuations'])==(2 if gate else 0)
    for row in r['continuations']:
        sub=d/row['objective'];assert sha(sub/'last_weights.pt')==row['weights_sha256'] and sha(sub/'predictions.npz')==row['predictions_sha256']
        final=torch.load(sub/'last_weights.pt',map_location='cpu',weights_only=True)
        assert set(final)==set(original) and all(torch.equal(final[k],original[k]) for k in fixed)
        assert any(not torch.equal(final[k],original[k]) for k in cr['trainable_names'])
        assert row['updates']==len(row['losses'])==len(row['unit_gradient_norms'])==2100
        assert np.isfinite(row['losses']).all();np.testing.assert_allclose(row['unit_gradient_norms'],1.,rtol=0,atol=1e-6)
        with np.load(sub/'predictions.npz') as z:
            for name,data in arrays.items():
                for phase in ['before','after']:
                    pred=z[phase+'__'+name];assert pred.shape==(len(data['observed']),3,192) and np.isfinite(pred).all()
                    mm=metrics(decode(pred,new),target(data['raw_states'])[:,1:],new['target_scale'])
                    for key in mm:np.testing.assert_allclose(mm[key],row['metrics'][name][phase][key],rtol=1e-10,atol=1e-7)
        model_checks.append({'objective':row['objective'],'weights_sha256':row['weights_sha256'],'all_frozen_tensors_unchanged':True})
    receipt={'status':'PASS_CALIBRATION_AND_CONTINUATION_ARRAYS' if gate else 'PASS_CALIBRATION_GATE_FAILURE_RETAINED',
        'group':r['group'],'head_gate_passed':gate,'reconstructed_head_metrics':rebuilt,'model_checks':model_checks,
        'maximum_head_prediction_reconstruction_error':maximum,'report_sha256':sha(d/'report.json'),'source_sha256':sha(__file__),
        'protocol_sha256':sha(a.protocol),'scope':'Independent NumPy head/metric reconstruction and exact frozen-state checks; no claim of a second full optimizer replay or improved downstream rollout.'}
    (d/'acceptance.json').write_text(json.dumps(receipt,indent=2)+'\n');print(receipt['status'],r['group'],flush=True)

if __name__=='__main__':main()
