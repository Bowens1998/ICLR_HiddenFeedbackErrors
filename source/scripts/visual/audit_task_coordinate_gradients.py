"""Real-checkpoint readout decoding and directional finite-difference checks."""
import json
from pathlib import Path
import numpy as np
import torch
from adaptation_task_losses import coordinates,target_for,loss_for,OBJECTIVES
from nonlinear_pose_cost import NonlinearPoseCost,numpy_pose
from adaptation_streams import sha


def main():
    torch.set_num_threads(2);rows=[]
    for index in range(0,12,2):
        root=Path('runs/adaptation_checkpoint_engineering_v1')/f'job_{index}'
        if not (root/'report.json').exists():root=Path('runs/adaptation_checkpoint_engineering_native_gru_v1')/f'job_{index}'
        r=json.loads((root/'report.json').read_text());stream=next(s for s in r['streams'] if s['stream']=='planner');fp=root/'planner/forward.npz';assert sha(fp)==stream['forward_sha256'];z=np.load(fp)
        head_path=Path('runs/adaptation_endpoint_diagnostics_v1')/f'job_{index}'/f'head_{index*6}.npz'
        assert sha(head_path)==r['entry']['endpoint_head']['sha256'];raw=dict(np.load(head_path));head=NonlinearPoseCost.prepare(raw,'cpu')
        x=torch.tensor(z['before'],dtype=torch.float64,requires_grad=True);obs=torch.tensor(z['observed'],dtype=torch.float64);states=torch.tensor(z['raw_states'],dtype=torch.float64)
        expected=(numpy_pose(z['before'],raw)-raw['target_mean'])/raw['target_scale']
        np.testing.assert_allclose(coordinates(x,head).detach().numpy(),expected,rtol=1e-10,atol=1e-10)
        rng=np.random.default_rng(1342001+index);directions=rng.normal(size=(4,*x.shape));directions/=np.linalg.norm(directions.reshape(4,-1),axis=1).reshape(4,1,1,1)
        checks=[]
        for objective in OBJECTIVES:
            target=target_for(objective,obs,states,head);loss=loss_for(objective,x,target,head);g=torch.autograd.grad(loss,x)[0].detach().numpy();assert np.isfinite(g).all() and np.linalg.norm(g)>0
            for direction in directions:
                step=torch.tensor(direction*1e-6);upper=float(loss_for(objective,x.detach()+step,target,head));lower=float(loss_for(objective,x.detach()-step,target,head));finite=(upper-lower)/2e-6;analytic=float(np.sum(g*direction));np.testing.assert_allclose(finite,analytic,rtol=1e-4,atol=1e-8)
                checks.append(dict(objective=objective,analytic=analytic,finite_difference=finite))
        rows.append(dict(index=index,engineering_forward_sha256=sha(fp),head_sha256=sha(head_path),checks=checks))
    out=Path('runs/hpg/planner_data_adaptation_v1/task_coordinate_gradient_audit.json')
    with out.open('x') as f:json.dump(dict(status='PASS6_REAL_READOUTS72_DIRECTIONAL_CHECKS',rows=rows,source_sha256=sha(__file__),loss_source_sha256=sha(Path(__file__).with_name('adaptation_task_losses.py')),scope='Fixed engineering predictions; independent NumPy decoding and central differences with epsilon1e-6. Does not establish full-model gradient reachability, successful fitting or utility.'),f,indent=2);f.write('\n')
    print('PASS6heads72directions')

if __name__=='__main__':main()
