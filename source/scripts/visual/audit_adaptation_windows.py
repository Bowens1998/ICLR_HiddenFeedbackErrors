"""Bind every engineering window to accepted trajectory frames/actions/states."""
import json,hashlib
from pathlib import Path
import numpy as np
from adaptation_windows import starts,recorded_window

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    root=Path('data/planner_adaptation_trajectory_engineering_v1')
    recovery=Path('runs/hpg/planner_data_adaptation_v1/trajectory_engineering_recovery.json')
    accepted=json.loads(recovery.read_text());assert accepted['status']=='PASS48_RECOVERED_TRAJECTORIES'
    rows=[];counts={}
    for binding in accepted['rows']:
        route=binding['route'];d=root/f'job_{route}';r=json.loads((d/'report.json').read_text())
        assert sha(d/'report.json')==binding['report_sha256'] and len(r['cases'])==2
        for c in r['cases']:
            path=d/c['file'];assert sha(path)==c['file_sha256'];z=np.load(path)
            indices=starts(len(z['states']),len(z['actions']));assert indices==[0,5,10,15,20]
            for start in indices:
                pixels,actions,states=recorded_window(z['pixels'],z['states'],z['actions'],start)
                assert pixels.shape==(4,224,224,3) and actions.shape==(3,10) and states.shape==(4,7)
                for j in range(4):
                    np.testing.assert_array_equal(pixels[j],z['pixels'][start//5+j])
                    np.testing.assert_array_equal(states[j],z['states'][start+5*j])
                for j in range(3):np.testing.assert_array_equal(actions[j].reshape(5,2),z['actions'][start+5*j:start+5*(j+1)])
                rows.append(dict(route=route,case=c['index'],seed=c['seed'],start=start,endpoint=start+15,action_stop_exclusive=start+15,source_archive_sha256=sha(path)))
            counts[str(route)]=counts.get(str(route),0)+len(indices)
    assert len(rows)==240 and len(counts)==24 and set(counts.values())=={10}
    result=dict(status='PASS240_ENGINEERING_TRAINING_WINDOWS',rows=rows,per_route=counts,recovery_sha256=sha(recovery),source_sha256=sha(__file__),window_source_sha256=sha(Path(__file__).with_name('adaptation_windows.py')),
                scope='Alignment on two engineering contexts,48 retained trajectories. Repeated overlapping windows are not independent samples. Dense expert adapter shares semantics; no real expert stream selected or training run yet. No future action at/after target image enters prediction inputs.')
    out=Path('runs/hpg/planner_data_adaptation_v1/window_engineering_acceptance.json')
    with out.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print('PASS240 windows /48 retained trajectories /2 shared contexts')

if __name__=='__main__':main()
