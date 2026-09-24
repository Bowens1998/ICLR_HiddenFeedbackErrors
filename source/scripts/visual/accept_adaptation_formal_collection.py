"""Require complete accepted routes and every unfiltered formal trajectory/window."""
import argparse, json, hashlib
from pathlib import Path
import numpy as np
from adaptation_windows import starts, recorded_window


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser()
    for k in ['plan','bank','planners','trajectories','output']:p.add_argument('--'+k,required=True)
    a=p.parse_args();plan=json.loads(Path(a.plan).read_text());split=plan['adaptation_formal']['split']
    count={'train':128,'validation':32}[split];bank=Path(a.bank);bm=json.loads((bank/'manifest.json').read_text())
    assert len(bm['cases'])==count and sha(bank/'manifest.json')==plan['bank_manifest_sha256']
    routes=plan['adaptation_formal']['execute_routes'];assert routes==[i*8+j for i in range(6) for j in (2,3,6,7)]
    rows=[];windows=0;failures=0;boundary_trajectories=0
    for route in routes:
        pd=Path(a.planners)/f'job_{route}';td=Path(a.trajectories)/f'job_{route}'
        s=json.loads((pd/'summary.json').read_text());ac=json.loads((pd/'acceptance.json').read_text());am=json.loads((pd/'artifact_manifest.json').read_text());tr=json.loads((td/'report.json').read_text())
        assert ac['model_free_simulator_replay'] and ac['cases']==count
        assert s['route_index']==tr['route']==route and len(s['cases'])==len(tr['cases'])==count
        assert s['hashes']['model_manifest']==tr['plan_sha256']==sha(a.plan)
        assert s['hashes']['bank_manifest']==sha(bank/'manifest.json')
        assert tr['status']=='PASS_REPLAYED_COMPLETE_SELECTED_TRAJECTORIES' and tr['split']==split
        assert am['summary.json']==tr['summary_sha256']==sha(pd/'summary.json')
        assert tr['acceptance_sha256']==sha(pd/'acceptance.json')
        assert tr['source_sha256']==sha(Path(__file__).with_name('extract_adaptation_trajectories.py'))
        for i,(c,sr,br) in enumerate(zip(tr['cases'],s['cases'],bm['cases'])):
            assert c['index']==sr['index']==br['index']==i and c['seed']==sr['seed']==br['seed']
            assert c['source_archive_sha256']==am[f'case_{i:03d}_predictions.npz']
            assert c['bank_archive_sha256']==br['sha256']==sha(bank/f'case_{i:03d}.npz')
            fp=td/c['file'];assert sha(fp)==c['file_sha256']
            assert c['source_success']==sr['success'] and c['source_boundary_steps']==sr['boundary_steps']
            failures+=not c['source_success'];boundary_trajectories+=c['source_boundary_steps']>0
            with np.load(fp) as z,np.load(bank/f'case_{i:03d}.npz') as b:
                assert int(z['seed'])==c['seed'] and z['pixels'].shape==(8,224,224,3) and z['pixels'].dtype==np.uint8
                assert z['states'].shape==(36,7) and z['actions'].shape==(35,2)
                assert np.isfinite(z['states']).all() and np.isfinite(z['actions']).all()
                np.testing.assert_array_equal(z['pixels'][:3],b['history_pixels'])
                np.testing.assert_array_equal(z['states'][[0,5,10]],b['history_states'])
                np.testing.assert_array_equal(z['actions'][:10],b['prefix'])
                assert starts(36,35)==[0,5,10,15,20]
                for t in starts(36,35):
                    pixels,actions,states=recorded_window(z['pixels'],z['states'],z['actions'],t)
                    np.testing.assert_array_equal(pixels,z['pixels'][t//5:t//5+4])
                    np.testing.assert_array_equal(states,z['states'][t+np.arange(4)*5])
                    np.testing.assert_array_equal(actions.reshape(15,2),z['actions'][t:t+15]);windows+=1
        rows.append(dict(route=route,trajectory_report_sha256=sha(td/'report.json'),planner_summary_sha256=sha(pd/'summary.json'),planner_acceptance_sha256=sha(pd/'acceptance.json'),cases=count))
    assert windows==24*count*5
    result=dict(status='PASS_COMPLETE_FORMAL_TRAJECTORIES_AND_WINDOWS',split=split,contexts=count,trajectories=24*count,windows=windows,retained_unsuccessful=failures,retained_boundary_trajectories=boundary_trajectories,rows=rows,plan_sha256=sha(a.plan),source_sha256=sha(__file__),window_source_sha256=sha(Path(__file__).with_name('adaptation_windows.py')),scope='Local full archive/hash and window alignment acceptance. Physical/search replays are bound to remote planner and trajectory acceptance; not rerun here. Overlapping windows and shared contexts are not independent samples. Retention counts are not scientific utility estimates.')
    with Path(a.output).open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print('PASS',split,result['trajectories'],windows)

if __name__=='__main__':main()
