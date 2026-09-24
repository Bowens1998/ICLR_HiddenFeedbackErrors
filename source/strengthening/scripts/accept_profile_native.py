"""Full NumPy native-head acceptance, then training-only full-space QP timing."""
import argparse
import json
from pathlib import Path
import resource
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'strengthening/adapters'),str(ROOT/'strengthening/scripts')]
from contracts import sha, atomic_json, require_role
from accept_heads import values
from projection_shard import run_shard


def main():
    p=argparse.ArgumentParser()
    for k in ['cache','head','output']:p.add_argument('--'+k,required=True)
    a=p.parse_args();cache=Path(a.cache);hr=Path(a.head);r=json.loads((hr/'report.json').read_text())
    assert (hr/'DONE').exists() and sha(hr/'weights.npz')==r['weights_sha256']
    assert sha(cache/'report.json')==r['cache_report_sha256'] and sha(hr/'validation_predictions.npz')==r['validation_predictions_sha256']
    head=dict(np.load(hr/'weights.npz'));norm=dict(np.load(cache/'normalizers.npz'))
    for k,v in norm.items():np.testing.assert_array_equal(head[k],v)
    meta=json.loads((cache/'head_validation.json').read_text());require_role(meta,{'head_validation'})
    assert sha(meta['file'])==meta['file_sha256'];assert sha(cache/'head_validation_labels.npy')==meta['labels_sha256']
    data=np.memmap(meta['file'],mode='r',dtype=np.float32,shape=tuple(meta['shape']));truth=np.load(cache/'head_validation_labels.npy');outputs=[]
    for i in range(0,len(data),32):
        value=(data[i:i+32].astype(float)-head['mean'])/head['scale']
        for layer in [0,2,4]:
            value=np.dot(value,head[f'{layer}.weight'].astype(float).T)+head[f'{layer}.bias']
            if layer<4:value=np.maximum(value,0)
        outputs.append(value*head['target_scale']+head['target_mean'])
    pred=np.concatenate(outputs);saved=dict(np.load(hr/'validation_predictions.npz'))
    np.testing.assert_array_equal(truth,saved['truth']);np.testing.assert_allclose(pred,saved['predicted'],rtol=1e-10,atol=1e-7)
    scores=values(pred,truth,head['target_scale'])
    for key,value in scores.items():np.testing.assert_allclose(value,r['best']['validation'][key],rtol=1e-10,atol=1e-7)
    assert r['best']['step']==min(r['history'],key=lambda row:(row['validation']['six_normalized_mse'],row['step']))['step']
    atomic_json(hr/'acceptance.json',dict(status='PASS_FULL_NATIVE_HEAD_NUMPY_RECONSTRUCTION',report_sha256=sha(hr/'report.json'),
        head_sha256=sha(hr/'weights.npz'),normalizers_unchanged=True,validation_rows=len(pred),metrics=scores,source_sha256=sha(__file__)))
    basis=json.loads((cache/'basis_train.json').read_text());require_role(basis,{'basis_train'})
    assert sha(cache/'QP_profile_inputs.npz')==basis['QP_inputs_sha256'];data=dict(np.load(cache/'QP_profile_inputs.npz'))
    policy=ROOT/'strengthening/configs/B_projection_profile.json';limits=json.loads(policy.read_text())['full_space_engineering_acceptance']
    families=[]
    for i in range(4):
        families.append(dict(tokens=[data['predicted'][i]]*2,guidance=[data['observed'][i],data['observed'][(i+1)%4]],
            head_sets=[[head]]*2,reference_heads=[head]*2,member_ids=['actual','cyclic_training_source']))
    out=Path(a.output);start=time.monotonic()
    report=run_shard(out/'full_space',dict(head_acceptance_sha256=sha(hr/'acceptance.json'),basis_sha256=sha(cache/'basis_train.json'),
        input_sha256=sha(cache/'QP_profile_inputs.npz'),role='native_projection_numerical_development',policy_sha256=sha(policy)),families)
    elapsed=time.monotonic()-start;peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2
    feasible=elapsed/8<=limits['mean_wall_seconds_per_QP_including_geometry_and_verification_max'] and peak<=limits['peak_resident_memory_gib_max']
    atomic_json(out/'report.json',dict(status='PASS_FULL_NATIVE_QP_ENGINEERING_BUDGET' if feasible else 'FULL_NATIVE_QP_EXCEEDS_ENGINEERING_BUDGET',
        qp_count=8,families=4,dimension=75264,elapsed_seconds=elapsed,mean_wall_seconds_per_qp=elapsed/8,
        peak_resident_memory_gib=peak,projection_report_sha256=sha(out/'full_space/report.json'),
        policy_sha256=sha(policy),source_sha256=sha(__file__),
        scope='Native training-only inputs, full nonlinear head/regions and FP32 insertion verified; no rollout efficacy evaluated.'))
    (out/'DONE').write_text('profile_complete\n')


if __name__=='__main__':main()
