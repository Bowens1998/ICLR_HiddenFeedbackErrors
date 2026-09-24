"""Complete imagined/real-image/goal reconstruction diagnostics on selected outcomes."""
import argparse,json
from pathlib import Path
import numpy as np
from adaptation_streams import sha


def errors(pred,truth):
    squared=np.sum((pred[:,2:4]-truth[:,2:4])**2,axis=1)
    angle=np.arctan2(pred[:,4],pred[:,5])-np.arctan2(truth[:,4],truth[:,5]);angle=np.arctan2(np.sin(angle),np.cos(angle))
    return dict(position_squared_error=squared,angle_squared_error=angle**2,precise=(squared<400)&(abs(angle)<np.pi/9))


def main():
    p=argparse.ArgumentParser()
    for key in ('diagnostics','evaluation','plan','output'):p.add_argument('--'+key,required=True)
    a=p.parse_args();plan=json.loads(Path(a.plan).read_text());evaluation=json.loads(Path(a.evaluation).read_text())
    assert evaluation['status']=='COMPLETE96_FRESH_TASK_COORDINATE_ROUTES' and evaluation['plan_sha256']==sha(a.plan)
    rows=[];bindings=[];all_errors={};goal_reference={}
    for index in range(12):
        d=Path(a.diagnostics)/f'job_{index}';r=json.loads((d/'report.json').read_text());ac=json.loads((d/'acceptance.json').read_text())
        assert ac['status']=='PASS_ALL_ENDPOINT_DECODE_AND_COST_RECONSTRUCTIONS' and ac['report_sha256']==sha(d/'report.json')
        assert ac['source_sha256']==sha(Path(__file__).with_name('accept_task_coordinate_endpoint_diagnostics.py'))
        route_indices=[i for i,route in enumerate(plan['routes']) if plan['models'][route['model_index']]['original_model_index']==index]
        assert len(route_indices)==(10 if index%2==0 else 6)
        assert r['index']==index and r['plan_sha256']==sha(a.plan) and r['same_image_checks']==256*len(route_indices) and r['max_same_image_difference']==0 and len(r['rows'])==len(route_indices)
        bindings.append(dict(index=index,report_sha256=sha(d/'report.json'),acceptance_sha256=sha(d/'acceptance.json')))
        for j,item in enumerate(r['rows']):
            route=route_indices[j];assert item['route']==route;e=plan['models'][plan['routes'][route]['model_index']];assert item['entry']==e
            binding=next(b for b in evaluation['bindings'] if b['route']==route)
            assert binding['summary_sha256']==item['summary_sha256'] and binding['acceptance_sha256']==item['acceptance_sha256']
            fp=d/item['file'];assert sha(fp)==item['sha256'];z=np.load(fp);assert z['seed'].tolist()==evaluation['goal_seeds']
            erow=evaluation['rows'][route];assert erow['route']==route;np.testing.assert_array_equal(z['realized_cost'],erow['costs']);np.testing.assert_array_equal(z['success'],erow['successes'])
            if index in goal_reference:np.testing.assert_allclose(z['goal_pose'],goal_reference[index],rtol=2e-5,atol=2e-5)
            else:goal_reference[index]=z['goal_pose'].copy()
            record=dict(route=route,original_model_index=index,condition=e['adaptation_condition'],algorithm=plan['routes'][route]['algorithm'],recipe=e['score'],arm=e['arm'])
            for kind,pred,truth in [('imagined',z['imagined_pose'],z['true_endpoint']),('real_image',z['real_pose'],z['true_endpoint']),('goal',z['goal_pose'],z['true_goal'])]:
                values=errors(pred,truth);all_errors[(route,kind)]=values
                record[kind]=dict(position_mse=float(values['position_squared_error'].mean()),angle_mse=float(values['angle_squared_error'].mean()),precision=float(values['precise'].mean()),per_goal={k:v.tolist() for k,v in values.items()})
            rows.append(record)
    assert len(rows)==96
    groups=[]
    for recipe in ('pose_encoded','state'):
        for condition in (('original','clipped_latent','unit_latent','unit_decoded_teacher','unit_physical_labels') if recipe=='pose_encoded' else ('original','clipped_state','unit_state')):
            for algorithm in ('random','cem'):
                subset=[r for r in rows if r['recipe']==recipe and r['condition']==condition and r['algorithm']==algorithm];assert len(subset)==6
                groups.append(dict(recipe=recipe,condition=condition,algorithm=algorithm,**{kind:{metric:float(np.mean([r[kind][metric] for r in subset])) for metric in ('position_mse','angle_mse','precision')} for kind in ('imagined','real_image','goal')}))
    result=dict(status='COMPLETE96_ROUTES_SELECTED_ENDPOINT_DIAGNOSTICS',rows=rows,groups=groups,bindings=bindings,same_image_checks=24576,max_same_image_difference=0.,evaluation_sha256=sha(a.evaluation),plan_sha256=sha(a.plan),source_sha256=sha(__file__),scope='All12288 selected outcomes; 128 shared independent admitted goals conditional on fixed fits. Imagined/real-image errors compare estimates to each condition’s own visited terminal state; goal error compares to fixed goal state. Exact same-image neural checks establish unchanged perception in bound extractions, not equal errors across different states. Precision is reconstruction within20pixels/20degrees, not goal-reaching success. Privileged descriptive error diagnostics, no additional causal identification or deployable method.')
    with Path(a.output).open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print('COMPLETE96_ENDPOINTS',flush=True)

if __name__=='__main__':main()
