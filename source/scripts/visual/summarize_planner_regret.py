"""Paired goal-resampling diagnostic of frozen cross-fit decisions, not fit uncertainty."""
import argparse
import json
from pathlib import Path
import numpy as np
from analyze_coverage_goals import sha


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data',required=True)
    p.add_argument('--crossfit',required=True)
    p.add_argument('--output',required=True)
    a=p.parse_args();data=Path(a.data);crossfit=Path(a.crossfit)
    r=json.loads((data/'report.json').read_text());f=json.loads((crossfit/'report.json').read_text())
    assert f['source_data_report_sha256']==sha(data/'report.json')
    assert r['data_sha256']==sha(data/'paired_data.npz')
    assert f['predictions_sha256']==sha(crossfit/'predictions.npz')
    z=dict(np.load(data/'paired_data.npz'));q=dict(np.load(crossfit/'predictions.npz'));groups=[]
    for g in f['groups']:
        task,interface=g['task'],g['interface'];key=task+'_'+interface
        rows=[v for v in r['rows'] if v['task']==task and v['interface']==interface]
        y=np.stack([z[f"{task}_{v['provenance'][0]['route']}_labels"] for v in rows])
        n=y.shape[1];choice=q[key+'_choice_cem'];fold=q[key+'_fold']
        constant=np.empty((6,n),dtype=bool)
        for m in g['fold_models']:
            constant[:,fold==m['fold']]=m['parameters']['target_mean']<0
        costs=np.where(choice,y[:,:,3],y[:,:,2]);base=np.where(constant,y[:,:,3],y[:,:,2])
        success=np.where(choice,y[:,:,5],y[:,:,4]);base_success=np.where(constant,y[:,:,5],y[:,:,4])
        np.testing.assert_allclose(costs.mean(),g['costs']['ridge'])
        np.testing.assert_allclose(base.mean(),g['costs']['train_constant'])
        # Same goal resample applies to all six backbones. Models and decisions stay fixed.
        sample=np.random.default_rng(1272901).integers(0,n,size=(10000,n))
        effects={}
        for name,value in [('cost',costs-base),('success',success-base_success)]:
            by_goal=value.mean(0)
            effects[name]=dict(mean=float(value.mean()),frozen_decision_goal_resampling_interval=np.quantile(by_goal[sample].mean(1),[.025,.975]).tolist(),by_backbone=value.mean(1).tolist())
        headroom=g['costs']['train_constant']-g['costs']['oracle_pair']
        groups.append(dict(task=task,interface=interface,effects=effects,cem_selection_fraction=g['cem_selection_fraction'],decision_changes_from_train_constant=int((choice!=constant).sum()),total_decisions=int(choice.size),unavailable_oracle_headroom=headroom,headroom_fraction_recovered=-effects['cost']['mean']/headroom if headroom>0 else None))
    result=dict(status='COMPLETE8_FROZEN_DECISION_DIAGNOSTICS',groups=groups,bootstrap=dict(seed=1272901,replicates=10000,unit='shared goal across six fixed backbones',scope='Frozen OOF decisions; does not refit selectors or account for overlapping training-fold dependence, selection of research hypotheses, backbone population uncertainty, or consumed-bank reuse. Descriptive intervals, not valid independent-confirmation confidence intervals or significance tests.'),source_sha256=sha(__file__),data_report_sha256=sha(data/'report.json'),crossfit_report_sha256=sha(crossfit/'report.json'))
    Path(a.output).write_text(json.dumps(result,indent=2)+'\n')
    for g in groups:print(g['task'],g['interface'],g['effects'],'switches',g['decision_changes_from_train_constant'],'headroom',g['headroom_fraction_recovered'])


if __name__=='__main__':main()
