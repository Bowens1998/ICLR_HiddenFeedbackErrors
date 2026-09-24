"""Independent arithmetic checks for the review-driven saved-array analyses."""
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr, kendalltau

ROOT=Path(__file__).resolve().parents[2]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    out=ROOT/'runs/review_followup_v2';r=json.loads((out/'summary.json').read_text())
    primary=json.loads((ROOT/'runs/hpg/fiber_confirmation_v1/summary.json').read_text())['primary_contrasts']
    teacher=np.asarray(next(x for x in primary if x['condition']=='unit_decoded_teacher' and x['right']=='free')['paired_goal_differences'])
    labels=np.asarray(next(x for x in primary if x['condition']=='unit_physical_labels' and x['right']=='free')['paired_goal_differences'])
    delta=labels-teacher;draws=np.random.default_rng(1368001).integers(0,128,(20000,128))
    ordered=np.sort(np.add.reduce(delta[draws],axis=1)/128)
    ci=[]
    for q in [.025,.975]:
        ix=q*(len(ordered)-1);lo=int(np.floor(ix));f=ix-lo
        ci.append(float(ordered[lo]*(1-f)+ordered[lo+1]*f))
    row=r['prediction']['change_in_gap_actual_minus_free']
    np.testing.assert_allclose(row['mean'],delta.mean(),rtol=0,atol=1e-10)
    np.testing.assert_allclose(row['exploratory_ci95'],ci,rtol=0,atol=1e-10)
    compact=json.loads((ROOT/'runs/hpg/feedback_ranking_v1/confirmation/summary_compact.json').read_text())
    objectives=['unit_latent','unit_decoded_teacher','unit_physical_labels']
    for obj in objectives:
        for score in ['pose','latent']:
            for b in ['free','act','don','reset']:
                for metric in ['selected_cost','spearman','kendall_tau_b']:
                    a=r['decision'][obj][score]['branches'][b][metric]['mean']
                    expected=compact['objectives'][obj]['score_spaces'][score]['branches'][b][metric]['mean']
                    np.testing.assert_allclose(a,expected,rtol=0,atol=1e-9)
    saved=np.load(out/'decision_goal_metrics.npz');v=saved['values'];assert v.shape==(6,3,2,4,512,6)
    checks=0;maximum=0.
    for group in range(6):
        for case in [0,1,17,63,127,255,383,511]:
            root=ROOT/'runs/feedback_ranking_confirmation_v1'
            with np.load(root/f'bank_{group}/outcomes/case_{case:03d}.npz') as z:
                states=z['terminal_states'];goal=z['goal_state']
                theta=np.angle(np.exp(1j*(states[:,4]-goal[4])))
                real=np.sum((states[:,2:4]-goal[2:4])**2,axis=1)+900*theta**2
            for c in range(3):
                with np.load(root/f'scores_{group}/model_{8*group+c+2}/case_{case:03d}.npz') as z:
                    for si,space in enumerate(['pose','latent']):
                        scores=z[space+'_cost'];fixed=np.flatnonzero(scores[0]==np.min(scores[0]))
                        for b in range(4):
                            ids=np.flatnonzero(scores[b]==np.min(scores[b]))
                            independent=[float(np.mean(real[ids])),
                                float(np.mean(np.abs(scores[b,ids]-real[ids]))) if si==0 else 0.,
                                float(spearmanr(scores[b],real).statistic),float(kendalltau(scores[b],real,variant='b').statistic),
                                float(np.mean(np.abs(scores[b,fixed]-real[fixed]))) if si==0 else 0.,
                                float(np.mean(np.abs(scores[b]-real))) if si==0 else 0.]
                            actual=v[group,c,si,b,case]
                            np.testing.assert_allclose(actual,independent,rtol=0,atol=1e-9,equal_nan=True)
                            maximum=max(maximum,float(np.nanmax(abs(actual-np.asarray(independent)))));checks+=1
    receipt={'status':'PASS_REVIEW_FOLLOWUP_ARRAYS_AND_STATISTICS','summary_sha256':sha(out/'summary.json'),
        'analysis_source_sha256':sha(ROOT/'scripts/visual/analyze_review_followup_v2.py'),'verifier_source_sha256':sha(__file__),
        'paired_gap_estimate':float(delta.mean()),'paired_gap_ci95':ci,'all_72_existing_branch_metric_means_match':True,
        'independently_reconstructed_branch_score_cases':checks,'maximum_sampled_metric_error':maximum,
        'decision_model_goal_rows':9216,'scope':'All original branch means and paired objective-gap statistics reconciled. Independent physical-cost and SciPy rank reconstruction on 144 fixed model-goal rows (1152 branch-score cases). New arrays retain all 9216 rows. Exploratory, not a new confirmation.'}
    (out/'independent_acceptance.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2))

if __name__=='__main__':main()
