"""Complete four-stream window diagnostics, without selecting fitted conditions."""
import argparse,json
from pathlib import Path
import numpy as np
from adaptation_streams import sha
from nonlinear_pose_cost import numpy_pose


def main():
    p=argparse.ArgumentParser()
    for k in ('fits','cache','output'):p.add_argument('--'+k,required=True)
    a=p.parse_args();rows=[];bindings=[];originals={}
    for task in range(24):
        d=Path(a.fits)/f'job_{task}';r=json.loads((d/'report.json').read_text());ac=json.loads((d/'acceptance.json').read_text())
        assert r['index']==task and r['model_index']==2*(task//4)+(task%4==3) and r['objective']==('latent','decoded_teacher','physical_labels','native_state')[task%4]
        assert ac['status']=='PASS_TASK_COORDINATE_FORMAL_FIT_AND_CPU_PREDICTIONS' and ac['report_sha256']==sha(d/'report.json')
        assert ac['source_sha256']==sha(Path(__file__).with_name('accept_task_coordinate_formal_fit.py'))
        cache=Path(a.cache)/f"job_{r['model_index']}";cr=json.loads((cache/'report.json').read_text());assert r['cache_report_sha256']==sha(cache/'report.json')
        assert sha(d/'predictions.npz')==r['predictions_sha256'];predictions=np.load(d/'predictions.npz');e=r['entry'];head=None
        if e['score']!='state':
            assert sha(e['endpoint_head']['path'])==e['endpoint_head']['sha256'];head=dict(np.load(e['endpoint_head']['path']))
        for item in cr['rows']:
            fp=cache/item['file'];assert sha(fp)==item['sha256'];z=np.load(fp);name=item['stream'];truth=z['raw_states'][:,-1]
            for phase in ('before','after'):
                pred=predictions[phase+'__'+name];key=(r['model_index'],name)
                if phase=='before':
                    if key in originals:np.testing.assert_allclose(pred,originals[key],rtol=2e-5,atol=2e-5)
                    else:originals[key]=pred.copy()
                endpoint=pred[:,-1]
                if e['score']=='state':
                    norm=e['target_normalization'];pose=endpoint.astype(float)*np.asarray(norm['std'])+np.asarray(norm['mean'])
                else:pose=numpy_pose(endpoint,head)
                error=pose[:,2:4]-truth[:,2:4];angle=np.arctan2(pose[:,4],pose[:,5])-truth[:,4];angle=np.arctan2(np.sin(angle),np.cos(angle))
                distance=np.linalg.norm(error,axis=1)
                rows.append(dict(task=task,model_index=r['model_index'],replica=e['replica'],arm=e['arm'],recipe=e['score'],condition=r['objective'],phase=phase,data_stream=name,windows=len(pred),prediction_loss=r['metrics'][name][phase],objective_loss=r['objective_metrics'][name][phase],block_position_mse=float(np.mean(np.sum(error**2,axis=1))),block_angle_mse=float(np.mean(angle**2)),block_position_rmse=float(np.sqrt(np.mean(np.sum(error**2,axis=1)))),endpoint_precision=float(np.mean((distance<20)&(abs(angle)<np.pi/9)))))
        bindings.append(dict(task=task,report_sha256=sha(d/'report.json'),acceptance_sha256=sha(d/'acceptance.json'),cache_report_sha256=sha(cache/'report.json')))
    assert len(rows)==192
    deltas=[]
    for task in range(24):
        for stream in ('expert_train','planner_train','expert_validation','planner_validation'):
            before=next(r for r in rows if r['task']==task and r['data_stream']==stream and r['phase']=='before');after=next(r for r in rows if r['task']==task and r['data_stream']==stream and r['phase']=='after')
            deltas.append(dict(task=task,recipe=after['recipe'],condition=after['condition'],data_stream=stream,**{k:after[k]-before[k] for k in ('prediction_loss','objective_loss','block_position_mse','block_angle_mse','endpoint_precision')}))
    report=dict(status='COMPLETE24_TASK_COORDINATE_WINDOW_DIAGNOSTICS',rows=rows,deltas=deltas,bindings=bindings,source_sha256=sha(__file__),scope='All24 fixed unit-gradient fits and all four cached streams. Objective loss and common prediction loss are distinct and not interchangeable across recipes. Teacher-forced next-token predictions at the last window position, not full25-action autoregressive planner forecasts. Endpoint precision measures reconstruction of the actual window endpoint, not goal-reaching success. Overlapping windows/shared expert validation are not independent samples; descriptive diagnostics only, no checkpoint/condition selection or planning utility claim.')
    with Path(a.output).open('x') as f:json.dump(report,f,indent=2);f.write('\n')
    print('COMPLETE24',len(rows),flush=True)

if __name__=='__main__':main()
