"""D: exploratory complete-pool cost decomposition; no model/simulator calls."""
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'scripts/visual'),str(ROOT/'strengthening/adapters')]
from contracts import atomic_json, sha, namespace_seed
from feedback_ranking_metrics import physical_goal_cost, ranking_metrics

METRICS=['cost_mse','bias_squared','centered_variance','pair_difference_mse','cost_mae',
         'spearman','kendall_tau_b','pair_sign_agreement','top3_incident_pair_sign_agreement',
         'selected_cost','free_top_two_margin','bias']


def measures(pred,truth):
    error=pred-truth;bias=error.mean(-1);var=((error-bias[:,None])**2).mean(-1);mse=(error**2).mean(-1)
    ii,jj=np.triu_indices(32,1);pd=pred[:,ii]-pred[:,jj];td=truth[ii]-truth[jj]
    pairs=((pd-td)**2).mean(-1)
    np.testing.assert_allclose(mse,bias*bias+var,rtol=2e-13,atol=1e-7)
    np.testing.assert_allclose(pairs,64/31*var,rtol=2e-13,atol=1e-7)
    top=truth<=np.partition(truth,2)[2];local=top[ii]|top[jj]
    margin=np.sort(pred[0])[1]-np.sort(pred[0])[0]
    result=[]
    for b in range(4):
        rank=ranking_metrics(pred[b],truth)
        agreement=np.sign(pd[b])==np.sign(td)
        result.append([mse[b],bias[b]**2,var[b],pairs[b],abs(error[b]).mean(),
                       rank['spearman'] if rank['spearman'] is not None else np.nan,
                       rank['kendall_tau_b'] if rank['kendall_tau_b'] is not None else np.nan,
                       agreement.mean(),agreement[local].mean(),rank['selected_cost'],margin,bias[b]])
    return np.asarray(result)


def main():
    p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    src=Path(a.input);out=Path(a.output);out.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    spec=dict(scope='EXPLORATORY_PREVIOUSLY_EVALUATED_GOALS',goals=512,groups=6,objectives=3,candidates=32,
              branches=['free','act','don','reset'],metrics=METRICS,bootstrap_draws=20000,
              bootstrap_seed=namespace_seed(20260919,'D_exploratory_bootstrap'),
              pair_rule='Exact sign agreement, ties retained as sign zero; no tie tolerance',
              local_pair_rule='At least one endpoint in truth top-3 including ties at third cost',
              source_sha256=sha(__file__),primary_claims='Original four selected-cost primary contrasts unchanged')
    atomic_json(out/'analysis_spec.json',spec)
    values=np.empty((6,3,4,512,len(METRICS)),float);truths=np.empty((6,512,32));predictions=np.empty((6,3,4,512,32));seeds=np.empty((6,512),np.int64);bindings=[]
    for g in range(6):
        bank=src/f'bank_{g}';bm=json.loads((bank/'manifest.json').read_text());sd=src/f'scores_{g}';report=json.loads((sd/'report.json').read_text())
        assert len(bm['cases'])==512 and report['expected_cases']==512 and report['bank_manifest_sha256']==sha(bank/'manifest.json')
        for c in bm['cases']:
            i=c['index'];fp=bank/c['outcome_file'];assert sha(fp)==c['outcome_sha256']
            with np.load(fp) as z:
                assert int(z['seed'])==c['seed'];seeds[g,i]=c['seed'];truths[g,i]=physical_goal_cost(z['terminal_states'],z['goal_state'])
        for q,slot in enumerate([2,3,4]):
            mr=next(x for x in report['models'] if x['model_index']==8*g+slot);assert len(mr['cases'])==512
            for c in mr['cases']:
                i=c['index'];fp=sd/c['file'];assert sha(fp)==c['sha256'];assert c['seed']==int(seeds[g,i])
                with np.load(fp) as z:
                    np.testing.assert_array_equal(z['branches'],['free','act','don','reset']);pred=z['pose_cost']
                assert pred.shape==(4,32) and np.isfinite(pred).all()
                values[g,q,:,i]=measures(pred,truths[g,i]);predictions[g,q,:,i]=pred
        bindings.append(dict(group=g,bank_manifest_sha256=sha(bank/'manifest.json'),score_report_sha256=sha(sd/'report.json'),full_score_files_hashed=1536,full_outcome_files_hashed=512))
        print('DECOMPOSED_GROUP',g,flush=True)
    assert np.all(seeds==seeds[:1])
    # Verify unchanged original metrics against the pre-existing goal-level evidence.
    legacy=ROOT/'output/repro/feedback_diagnostic_evidence/data/decision_goal_metrics.npz'
    if legacy.exists():
        with np.load(legacy) as z:
            old=z['values']
        for new_m,old_m in [(9,0),(5,2),(6,3),(4,5)]:
            np.testing.assert_allclose(values[...,new_m],old[:,:,0,:,:,old_m],rtol=1e-12,atol=1e-8,equal_nan=True)
    goal=np.nanmean(values,axis=0);delta=goal[:,1]-goal[:,0]
    rng=np.random.default_rng(spec['bootstrap_seed']);samples=[]
    for _ in range(20):
        ix=rng.integers(0,512,size=(1000,512));samples.append(np.nanmean(delta[:,ix],axis=2))
    draws=np.concatenate(samples,axis=1);summary=[]
    for q,name in enumerate(['latent','decoded_teacher','physical_labels']):
        fields={}
        for m,metric in enumerate(METRICS):
            fields[metric]=dict(free=float(np.nanmean(goal[q,0,:,m])),actual=float(np.nanmean(goal[q,1,:,m])),
                actual_minus_free=float(np.nanmean(delta[q,:,m])),ci95=np.nanquantile(draws[q,:,m],[.025,.975]).tolist(),
                undefined_pool_values=int(np.isnan(values[:,q,:,:,m]).sum()))
        summary.append(dict(objective=name,metrics=fields))
    np.savez_compressed(out/'pool_arrays.npz',metrics=values,predicted_cost=predictions,physical_cost=truths,seeds=seeds,metric_names=np.array(METRICS))
    np.savez_compressed(out/'goal_arrays.npz',metrics=goal,actual_minus_free=delta,seeds=seeds[0])
    atomic_json(out/'decision_error_decomposition.json',dict(status='PASS_COMPLETE_EXPLORATORY_D',spec=spec,
        summaries=summary,bindings=bindings,full_pool_count=9216,branch_pool_checks=36864,
        missing_cost_pools=0,legacy_metric_parity=legacy.exists(),elapsed_seconds=time.monotonic()-start,
        array_sha256=sha(out/'pool_arrays.npz'),goal_array_sha256=sha(out/'goal_arrays.npz')))
    (out/'DONE').write_text('accepted\n')


if __name__=='__main__':main()
