"""Local head calibration reconstruction against hash-bound cached observations."""
import argparse
import json
from pathlib import Path
import numpy as np
from accept_readout_calibration_evaluation import decode, sha

p=argparse.ArgumentParser();p.add_argument('--pilot',required=True);p.add_argument('--cache',required=True);a=p.parse_args()
rows=[]
for group in (0,1):
    root=Path(a.pilot)/f'group_{group}';r=json.loads((root/'report.json').read_text());accepted=json.loads((root/'acceptance.json').read_text())
    assert accepted['report_sha256']==sha(root/'report.json')
    assert r['head_predictions_sha256']==sha(root/'head_predictions.npz')
    assert r['fitted_head_sha256']==sha(root/'fitted_head.npz')
    predictions=dict(np.load(root/'head_predictions.npz'));metrics={};maximum=0.
    for split,binding in r['data_bindings'].items():
        path=Path(a.cache)/f'job_{group*2}'/Path(binding['file']).name
        assert sha(path)==binding['sha256']
        cache=dict(np.load(path));state=cache['raw_states'].astype(np.float64)
        truth=np.concatenate((state[...,:4],np.sin(state[...,4:5]),np.cos(state[...,4:5])),axis=-1)
        np.testing.assert_allclose(truth,predictions[split+'__truth'],rtol=0,atol=1e-12)
        metrics[split]={}
        for tag,file in [('old','original_head.npz'),('new','fitted_head.npz')]:
            head=dict(np.load(root/file));decoded=decode(cache['observed'],head)
            np.testing.assert_allclose(decoded,predictions[split+'__'+tag],rtol=1e-11,atol=1e-8)
            maximum=max(maximum,float(np.max(np.abs(decoded-predictions[split+'__'+tag]))))
            delta=decoded-truth
            metrics[split][tag]={'normalized_mse':float(np.square(delta/head['target_scale']).mean()),'block_position_mse':float(np.square(delta[...,2:4]).sum(-1).mean()),'per_output_mse':np.square(delta.reshape(-1,6)).mean(0).tolist()}
            for key,value in metrics[split][tag].items():
                np.testing.assert_allclose(value,r['head_metrics'][split][tag][key],rtol=1e-11,atol=1e-8)
    gate=all(metrics['planner_validation']['new'][k]<=.9*metrics['planner_validation']['old'][k] for k in ['normalized_mse','block_position_mse'])
    assert gate==r['head_gate_passed']==accepted['head_gate_passed']
    assert len(r['continuations'])==(2 if gate else 0)
    rows.append({'group':group,'gate_passed':gate,'report_sha256':sha(root/'report.json'),'cluster_acceptance_sha256':sha(root/'acceptance.json'),'metrics':metrics,'maximum_decode_discrepancy':maximum})
out={'status':'PASS_BOTH_CALIBRATION_GATES_AND_ALL_SAVED_HEAD_PREDICTIONS','rows':rows,'source_sha256':sha(__file__),'scope':'Independent local NumPy reconstruction of calibration, all four data splits, and gates. Frozen continuation tensor checks are in the hash-bound cluster acceptance receipts.'}
path=Path(a.pilot)/'local_acceptance.json';path.write_text(json.dumps(out,indent=2)+'\n');print(out['status'],sha(path))
