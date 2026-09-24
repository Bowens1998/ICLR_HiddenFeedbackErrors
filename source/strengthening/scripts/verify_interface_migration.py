"""New generic interfaces must reproduce the existing development arrays exactly."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'strengthening/adapters'))
from contracts import sha,atomic_json
from confirmation_results import physical_pose


def main():
    p=argparse.ArgumentParser();p.add_argument('--base',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    root=Path(a.base)/'releases/feedback-strengthening-v1/artifacts';reports={};comparisons=[]
    def report(path):
        assert (path/'DONE').exists();r=json.loads((path/'report.json').read_text());reports[str(path/'report.json')]=sha(path/'report.json');return r
    def compare(left,right):
        with np.load(left) as x,np.load(right) as y:
            assert set(x.files)==set(y.files)
            for k in x.files:np.testing.assert_array_equal(x[k],y[k])
        comparisons.append(dict(old=str(left),new=str(right),old_sha256=sha(left),new_sha256=sha(right)))
    for g in [0,1]:
        old=root/f'AC_development_cache_v1/group_{g}';new=root/f'interface_migration_AC_cache_v1/group_{g}'
        ro=report(old);rn=report(new);assert len(ro['models'])==len(rn['models'])
        compare(old/'observed.npz',new/'observed.npz')
        head={k:v.astype(np.float64) for k,v in dict(np.load(new/'head_A.npz')).items()}
        for x,y in zip(ro['models'],rn['models'],strict=True):
            assert x['model']==y['model'];compare(old/x['file'],new/y['file'])
            with np.load(new/y['file']) as z:np.testing.assert_allclose(physical_pose(z['free_tokens'],head),z['free_pose'],rtol=1e-10,atol=1e-7)
        old=root/f'AC_development_rollout_v1/group_{g}';new=root/f'interface_migration_AC_rollout_v1/group_{g}'
        ro=report(old);rn=report(new);assert len(ro['families'])==len(rn['families'])
        for x,y in zip(ro['families'],rn['families'],strict=True):
            assert x['descriptor']==y['descriptor'];compare(old/x['file'],new/y['file'])
    old=root/'B_development_cache_v1';new=root/'interface_migration_B_cache_v1'
    ro=report(old);rn=report(new);assert len(ro['rows'])==len(rn['rows'])==64
    compare(old/'baseline_metrics.npz',new/'baseline_metrics.npz')
    for x,y in zip(ro['rows'],rn['rows'],strict=True):
        assert (x['index'],x['seed'])==(y['index'],y['seed']);compare(old/x['file'],new/y['file'])
    # Isolate the rollout interface with exactly the same FP32 insertion. A
    # separately re-solved QP may cross an FP32 rounding boundary while both
    # solutions pass the unchanged 1e-6 full-head/region/norm constraints.
    audit=root/'native_QP_migration_audit_v1';ar=report(audit)
    assert ar['status']=='PASS_SAME_INPUTS_BOTH_NATIVE_QP_SOLVES_VALID'
    for row in ar['rows']:
        assert all(check['accepted'] for side in row['old_and_new_full_head_region_norm_checks'] for check in side)
    solved=root/'interface_migration_B_rollout_v1';sr=report(solved)
    assert sr['projection_report_sha256']==ar['new_QP_report_sha256']
    old=root/'B_development_rollout_v1';new=root/'interface_migration_B_fixed_insertion_v1'
    ro=report(old);rn=report(new);assert len(ro['rows'])==len(rn['rows'])==2
    for x,y in zip(ro['rows'],rn['rows'],strict=True):
        assert (x['case'],x['seed'])==(y['case'],y['seed']);compare(old/x['file'],new/y['file'])
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    atomic_json(out/'report.json',dict(status='PASS_GENERIC_INTERFACE_DEVELOPMENT_PARITY',
        source_sha256=sha(__file__),reports=reports,comparisons=comparisons,
        native_independent_solve_audit_sha256=sha(audit/'report.json'),
        native_repeated_QP_bitwise_identical=False,numerical_tolerances_unchanged=True,
        scope='TF and GRU full64 baselines plus fixed first2 corrected cases; native full64 baseline and exact fixed-insertion replay of first2 cases. Repeated native QP solutions are separately verified under unchanged constraints; not claimed bitwise identical. Frozen scientific design and unseen confirmation unchanged.'))
    (out/'DONE').write_text('development_parity_accepted\n')


if __name__=='__main__':main()
