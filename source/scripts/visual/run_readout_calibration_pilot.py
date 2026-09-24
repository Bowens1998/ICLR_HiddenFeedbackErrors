"""Validation-gated head refit and matched continuation; no test-goal fitting."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from adaptation_streams import sha
from adaptation_freeze import configure_dynamics_only, verify_frozen
from adaptation_gradient_control import unit_global_gradient
from adaptation_task_losses import target_for, coordinates
from factorial_model import make_model, state_features
from nonlinear_pose_cost import numpy_pose, NonlinearPoseCost
from evaluation_precision import configure_evaluation_precision
from run_adaptation_checkpoint_gate import tensor_digest
from train_task_coordinate_formal import temporal


def metrics(pred, truth, scale):
    error=pred-truth
    return {'normalized_mse':float(np.mean((error/scale)**2)),
            'block_position_mse':float(np.mean(np.sum(error[...,2:4]**2,-1))),
            'per_output_mse':np.mean(error.reshape(-1,6)**2,axis=0).tolist()}


def main():
    p=argparse.ArgumentParser()
    for key in ['cache','freeze','official','config','output','protocol']:
        p.add_argument('--'+key,required=True)
    p.add_argument('--group',type=int,choices=[0,1],required=True)
    a=p.parse_args();started=time.monotonic();protocol_sha=sha(a.protocol)
    cache=Path(a.cache)/f'job_{2*a.group}';r=json.loads((cache/'report.json').read_text())
    ac=json.loads((cache/'acceptance.json').read_text())
    assert ac['status']=='PASS_FULL_FRESH_PROCESS_REENCODING' and ac['cache_report_sha256']==sha(cache/'report.json')
    e=r['entry'];td=Path(e['training_path']);tr=json.loads((td/'summary.json').read_text())
    assert sha(td/'summary.json')==e['training_summary_sha256'] and sha(td/'last_weights.pt')==e['weights_sha256']
    assert sha(a.config)==tr['config_sha256'] and e['score']=='pose_encoded'
    hp=Path(e['endpoint_head']['path']);assert sha(hp)==e['endpoint_head']['sha256']
    original_head=dict(np.load(hp));out=Path(a.output)/f'group_{a.group}';out.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(out/'original_head.npz',**original_head)
    datasets={};truth={};data_bindings={}
    for row in r['rows']:
        fp=cache/row['file'];assert sha(fp)==row['sha256']
        datasets[row['stream']]=dict(np.load(fp));z=datasets[row['stream']]
        truth[row['stream']]=state_features(torch.as_tensor(z['raw_states'],dtype=torch.float64)).numpy()
        assert z['observed'].shape==(row['windows'],4,192)
        data_bindings[row['stream']]={'file':str(fp),'sha256':sha(fp),'windows':row['windows']}
    configure_evaluation_precision();torch.set_num_threads(2);torch.manual_seed(1392001+a.group)
    net=nn.Sequential(nn.Linear(192,256),nn.ReLU(),nn.Linear(256,256),nn.ReLU(),nn.Linear(256,6)).cuda()
    net.load_state_dict({k:torch.as_tensor(original_head[k]) for k in net.state_dict()},strict=True)
    z=datasets['planner_train']['observed'].reshape(-1,192)
    x=torch.as_tensor((z.astype(float)-original_head['mean'])/original_head['scale'],device='cuda',dtype=torch.float32)
    y=torch.as_tensor(((truth['planner_train']-original_head['target_mean'])/original_head['target_scale']).reshape(-1,6),device='cuda',dtype=torch.float32)
    opt=torch.optim.Adam(net.parameters(),lr=1e-4,weight_decay=0)
    rng=np.random.default_rng(1392001+a.group);losses=[]
    for step in range(3000):
        ids=rng.integers(0,len(x),256);opt.zero_grad(set_to_none=True)
        loss=(net(x[ids])-y[ids]).square().mean();assert torch.isfinite(loss)
        loss.backward();opt.step();losses.append(float(loss.detach()))
    fitted={k:v.detach().cpu().numpy().copy() for k,v in net.state_dict().items()}
    fitted.update({k:original_head[k] for k in ['mean','scale','target_mean','target_scale']})
    np.savez_compressed(out/'fitted_head.npz',**fitted)
    head_metrics={};saved={}
    for name,d in datasets.items():
        old=numpy_pose(d['observed'],original_head);new=numpy_pose(d['observed'],fitted)
        head_metrics[name]={'old':metrics(old,truth[name],fitted['target_scale']),
                            'new':metrics(new,truth[name],fitted['target_scale'])}
        saved.update({name+'__old':old,name+'__new':new,name+'__truth':truth[name]})
    np.savez_compressed(out/'head_predictions.npz',**saved)
    validation=head_metrics['planner_validation']
    gate=all(validation['new'][k] <= .9*validation['old'][k] for k in ['normalized_mse','block_position_mse'])
    report={'status':'HEAD_GATE_PASSED' if gate else 'HEAD_GATE_FAILED','group':a.group,'arm':e['arm'],
        'head_gate_passed':gate,'head_updates':3000,'head_seed':1392001+a.group,'head_losses':losses,
        'head_metrics':head_metrics,'data_bindings':data_bindings,'cache_report_sha256':sha(cache/'report.json'),
        'original_head_sha256':sha(hp),'fitted_head_sha256':sha(out/'fitted_head.npz'),
        'head_predictions_sha256':sha(out/'head_predictions.npz'),'protocol_sha256':protocol_sha,
        'original_model_sha256':e['weights_sha256'],'source_sha256':sha(__file__),'continuations':[],
        'gpu':torch.cuda.get_device_name()}
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print('HEAD_GATE',a.group,gate,validation,flush=True)
    if not gate:return
    del net,opt,x,y;torch.cuda.empty_cache()
    fr=Path(a.freeze);freeze=json.loads((fr/'report.json').read_text());assert sha(fr/'update_indices.npz')==freeze['schedule_sha256']
    schedule=np.load(fr/'update_indices.npz')['indices'];assert schedule.shape==(2100,128)
    head=NonlinearPoseCost.prepare(fitted,'cuda')
    tensors={name:{k:torch.as_tensor(d[k],device='cuda') for k in ['observed','normalized_actions','raw_states']} for name,d in datasets.items()}
    for objective in ['decoded_teacher','physical_labels']:
        torch.manual_seed(1292001)
        model=make_model(a.official,a.config,e['arm'],tr['seed'])
        original=torch.load(td/'last_weights.pt',map_location='cpu',weights_only=True)
        model.load_state_dict(original,strict=True);model=model.cuda();boundary=configure_dynamics_only(model)
        assert tensor_digest(model.state_dict())==r['initial_tensor_sha256']
        assert tensor_digest(boundary['frozen'])==r['frozen_tensor_sha256']
        target={name:target_for(objective,d['observed'],d['raw_states'],head) for name,d in tensors.items()}
        def evaluate():
            with torch.no_grad():
                return {name:np.concatenate([temporal(model,e['arm'],d['observed'][i:i+128],d['normalized_actions'][i:i+128]).cpu().numpy() for i in range(0,len(d['observed']),128)]) for name,d in tensors.items()}
        before=evaluate();parameters=[v for v in model.parameters() if v.requires_grad]
        opt=torch.optim.AdamW(parameters,lr=1e-5,weight_decay=1e-3);losses=[];norms=[]
        d=tensors['planner_train']
        for ids in schedule:
            ix=torch.as_tensor(ids.astype(np.int64),device='cuda');opt.zero_grad(set_to_none=True)
            prediction=temporal(model,e['arm'],d['observed'][ix],d['normalized_actions'][ix])
            loss=(coordinates(prediction,head)-target['planner_train'][ix]).square().mean();assert torch.isfinite(loss)
            loss.backward();control=unit_global_gradient(parameters);opt.step();verify_frozen(model,boundary)
            losses.append(float(loss.detach()));norms.append(control['after_norm'])
        after=evaluate();verify_frozen(model,boundary)
        dest=out/objective;dest.mkdir();torch.save({k:v.detach().cpu() for k,v in model.state_dict().items()},dest/'last_weights.pt')
        np.savez_compressed(dest/'predictions.npz',**{phase+'__'+name:z for phase,results in [('before',before),('after',after)] for name,z in results.items()})
        ms={}
        for name in datasets:
            ms[name]={phase:metrics(numpy_pose(results[name],fitted),truth[name][:,1:],fitted['target_scale']) for phase,results in [('before',before),('after',after)]}
        row={'objective':objective,'updates':2100,'metrics':ms,'losses':losses,'unit_gradient_norms':norms,
            'weights_sha256':sha(dest/'last_weights.pt'),'predictions_sha256':sha(dest/'predictions.npz'),
            'frozen_before_sha256':r['frozen_tensor_sha256'],'frozen_after_sha256':tensor_digest(boundary['frozen']),
            'schedule_sha256':freeze['schedule_sha256'],'initial_tensor_sha256':r['initial_tensor_sha256']}
        report['continuations'].append(row)
        (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        print('CONTINUATION_COMPLETE',a.group,objective,flush=True)
        del model,opt,boundary;torch.cuda.empty_cache()
    assert sha(a.protocol)==protocol_sha
    report.update(status='CALIBRATION_AND_CONTINUATION_REQUIRE_ACCEPTANCE',elapsed_seconds=time.monotonic()-started)
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
