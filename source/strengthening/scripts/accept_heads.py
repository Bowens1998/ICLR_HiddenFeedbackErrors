"""CPU NumPy reconstruction of both selected heads and unchanged normalizers."""
import argparse
import json
import os
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'scripts/visual'),str(ROOT/'strengthening/adapters')]
from contracts import atomic_json, sha


def numpy_pose(tokens, head):
    # Independent CPU evaluator: no Torch or training evaluator import.
    value=(np.asarray(tokens,dtype=np.float64)-head['mean'])/head['scale']
    for i in [0,2,4]:
        value=np.einsum('bi,ji->bj',value,head[f'{i}.weight'])+head[f'{i}.bias']
        if i<4:value=np.clip(value,0,None)
    return value*head['target_scale']+head['target_mean']


def values(pred,truth,scale):
    diff=pred-truth;ang=np.angle(np.exp(1j*(np.arctan2(pred[:,4],pred[:,5])-np.arctan2(truth[:,4],truth[:,5]))))
    return {'six_normalized_mse':float(np.mean((diff/scale)**2)),
            'block_position_mse':float(np.mean(diff[:,2]**2+diff[:,3]**2)),
            'agent_position_mse':float(np.mean(diff[:,0]**2+diff[:,1]**2)),
            'wrapped_angle_mse':float(np.mean(ang**2))}


def main():
    p=argparse.ArgumentParser()
    for k in ['base','heads','cache']:p.add_argument('--'+k,required=True)
    p.add_argument('--group',type=int);a=p.parse_args();g=int(os.environ['SLURM_ARRAY_TASK_ID']) if a.group is None else a.group
    root=Path(a.heads)/f'group_{g}';cache=Path(a.cache)/f'group_{g}'
    if not (root/'DONE').exists():raise ValueError('Missing complete head grid')
    report=json.loads((root/'report.json').read_text());selection=json.loads((root/'selection.json').read_text())
    assert report['selection_sha256']==sha(root/'selection.json') and report['fit_count']==12
    assert len(report['fits'])==12 and all(x['updates']==3000 for x in report['fits'])
    with np.load(cache/'head_validation.npz') as z:data={k:z[k] for k in z.files}
    oldpath=Path(a.base)/f'releases/pusht-nonlinear-transfer-v1/runs/pose_fit/job_{g}/encoded/weights.npz'
    assert sha(oldpath)==selection['old_head_sha256'];old=dict(np.load(oldpath))
    oldpred=numpy_pose(data['observed'],old);baseline={}
    for dom in ['expert','planner']:
        mask=data['domain']==dom;baseline[dom]=values(oldpred[mask],data['labels'][mask],old['target_scale'])
        for k,v in baseline[dom].items():np.testing.assert_allclose(v,report['baseline'][dom][k],rtol=1e-10,atol=1e-8)
    rows=[]
    for role in ['head_A','head_B']:
        chosen=selection[role];hp=Path(chosen['path']);assert sha(hp)==chosen['sha256'];h=dict(np.load(hp))
        for key in ['mean','scale','target_mean','target_scale']:np.testing.assert_array_equal(h[key],old[key])
        pred=numpy_pose(data['observed'],h);checks={};passed=True
        for dom in ['expert','planner']:
            mask=data['domain']==dom;scores=values(pred[mask],data['labels'][mask],old['target_scale']);checks[dom]=scores
            for k,v in scores.items():np.testing.assert_allclose(v,chosen['metrics'][dom][k],rtol=1e-10,atol=1e-8)
            for k,factor in [('six_normalized_mse',.9),('block_position_mse',.9),('agent_position_mse',1.05),('wrapped_angle_mse',1.05)]:
                passed=passed and baseline[dom][k]>0 and scores[k]<=factor*baseline[dom][k]
        assert bool(passed)==chosen['quality_gate']['passed']
        rows.append(dict(role=role,path=str(hp),sha256=sha(hp),quality_gate_pass=bool(passed),metrics=checks))
    candidates=[(f,f['best']['best_qualified']) for f in report['fits'] if f['seed_role']=='A' and 'best_qualified' in f['best']]
    if not candidates:candidates=[(f,f['best']['best_validation']) for f in report['fits'] if f['seed_role']=='A']
    chosen=min(candidates,key=lambda p:(p[1]['J_validation'],p[0]['width'],p[0]['size'],p[1]['step']))
    assert chosen[1]['sha256']==selection['head_A']['sha256']
    atomic_json(root/'acceptance.json',dict(status='PASS_INDEPENDENT_SELECTED_HEAD_RECONSTRUCTION',group=g,
        report_sha256=sha(root/'report.json'),selection_sha256=sha(root/'selection.json'),rows=rows,
        validation_cache_sha256=sha(cache/'head_validation.npz'),source_sha256=sha(__file__),
        scope='Both selected heads independently evaluated with NumPy FP64; fit roster and validation-only selection checked'))


if __name__=='__main__':main()
