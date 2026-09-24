"""Exploratory review analyses from accepted arrays; never changes primary tests."""
import hashlib
import json
from pathlib import Path

import numpy as np

from feedback_ranking_metrics import physical_goal_cost, ranking_metrics

ROOT = Path(__file__).resolve().parents[2]
OBJECTIVES = ['unit_latent', 'unit_decoded_teacher', 'unit_physical_labels']
BRANCHES = ['free', 'act', 'don', 'reset']


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def estimate(x, draws):
    x = np.asarray(x, dtype=float)
    assert x.ndim == 1 and len(x) == draws.shape[1] and not np.isinf(x).any()
    boot = np.nanmean(x[draws],axis=1)
    assert np.isfinite(boot).all()
    return {'mean': float(np.nanmean(x)), 'exploratory_ci95': np.quantile(boot, [.025, .975]).tolist(),
            'valid_goals':int(np.isfinite(x).sum()),'total_goals':len(x)}


def association(x, y, draws):
    x, y = np.asarray(x), np.asarray(y)
    def corr(a, b):
        a, b = a-a.mean(-1, keepdims=True), b-b.mean(-1, keepdims=True)
        return (a*b).sum(-1)/np.sqrt((a*a).sum(-1)*(b*b).sum(-1))
    boot = corr(x[draws], y[draws])
    assert np.isfinite(boot).all()
    return {'pearson': float(corr(x, y)), 'exploratory_ci95': np.quantile(boot, [.025, .975]).tolist()}


def prediction_analysis():
    draws = np.random.default_rng(1368001).integers(0, 128, (20000, 128))
    paths = ['free', 'fiber', 'shuffled', 'full', 'reset', 'observed', 'teacher']
    position = np.empty((6, 3, len(paths), 4, 128, 5))
    calibration = np.empty((6, 4, 128, 5))
    bindings = []
    for group in range(6):
        base = ROOT/f'runs/fiber_confirmation_rollout_v1/job_{group}'
        horizon = ROOT/f'runs/fiber_confirmation_horizon_v1/job_{group}'
        report = json.loads((base/'report.json').read_text())
        accepted = json.loads((base/'acceptance.json').read_text())
        hreport = json.loads((horizon/'report.json').read_text())
        haccepted = json.loads((horizon/'acceptance.json').read_text())
        assert accepted['report_sha256'] == sha(base/'report.json')
        assert haccepted['report_sha256'] == sha(horizon/'report.json')
        for c in range(3):
            model = group*8+c+2
            row = next(r for r in report['rows'] if r['model_index'] == model)
            fp = base/row['file']
            assert sha(fp) == row['sha256'] == accepted['rows'][c]['file_sha256']
            with np.load(fp) as z:
                truth = z['true_pose']
                for k, path in enumerate(paths[:-1]):
                    position[group, c, k] = ((z[path+'_pose'][..., 2:4]-truth[..., 2:4])**2).sum(-1)
                    np.testing.assert_allclose(position[group,c,k], accepted['rows'][c]['metrics'][path]['position_mse'],rtol=1e-10,atol=1e-8)
                if c == 0:
                    head = dict(np.load(base/row['head_file']))
                    calibration[group] = (((z['observed_pose']-truth)/head['target_scale'])**2).mean(-1)
                else:
                    np.testing.assert_allclose(position[group,c,5], position[group,0,5],rtol=0,atol=0)
            hrow = next(r for r in hreport['rows'] if r['model_index'] == model)
            hp = horizon/hrow['file']; assert sha(hp) == hrow['sha256']
            with np.load(hp) as z:
                position[group,c,6] = ((z['teacher_pose'][...,2:4]-z['true_pose'][...,2:4])**2).sum(-1)
        bindings.append({'group':group,'fiber_report_sha256':sha(base/'report.json'),'horizon_report_sha256':sha(horizon/'report.json')})
    endpoint = position.mean(3)[..., -1]  # groups, objectives, paths, goals
    table = {}
    for c,obj in enumerate(OBJECTIVES):
        table[obj] = {p:estimate(endpoint[:,c,i].mean(0),draws) for i,p in enumerate(paths)}
    gaps = endpoint[:,2]-endpoint[:,1]
    gap_table = {p:{**estimate(gaps[:,i].mean(0),draws),'per_model':gaps[:,i].mean(1).tolist()} for i,p in enumerate(paths)}
    change = gaps[:,1]-gaps[:,0]
    error5 = calibration[...,0].mean((0,1))
    associations = {'readout_error_vs_free_objective_gap':association(error5,gaps[:,0].mean(0),draws)}
    for c,obj in enumerate(OBJECTIVES):
        delta = (endpoint[:,c,1]-endpoint[:,c,0]).mean(0)
        associations['readout_error_vs_'+obj+'_actual_minus_free'] = association(error5,delta,draws)
    return {'absolute_endpoint_position_mse':table,'labels_minus_teacher':gap_table,
            'change_in_gap_actual_minus_free':{**estimate(change.mean(0),draws),'per_model':change.mean(1).tolist()},
            'observed_encoding_calibration':{'horizons':[5,10,15,20,25],
                'block_position_mse':[estimate(position[:,0,5,:,:,i].mean((0,1)),draws) for i in range(5)],
                'six_output_standardized_mse':[estimate(calibration[...,i].mean((0,1)),draws) for i in range(5)]},
            'goal_level_associations':associations,'bindings':bindings}


def decision_analysis():
    root = ROOT/'runs/feedback_ranking_confirmation_v1'
    compact = json.loads((ROOT/'runs/hpg/feedback_ranking_v1/confirmation/summary_compact.json').read_text())
    receipt = json.loads((ROOT/'runs/hpg/feedback_ranking_v1/confirmation/independent_full_revalidation.json').read_text())
    assert receipt['status'] == 'PASS_INDEPENDENT_FRESH512_LOCAL_REVALIDATION'
    # group, objective, score, branch, goal, metric
    names = ['selected_cost','selected_cost_absolute_prediction_error','spearman','kendall_tau_b',
             'free_selected_absolute_prediction_error','pool_absolute_prediction_error']
    values = np.empty((6,3,2,4,512,len(names)))
    margin_rows=[]
    for group in range(6):
        sr = json.loads((root/f'scores_{group}/report.json').read_text())
        bm = json.loads((root/f'bank_{group}/manifest.json').read_text())
        assert sha(root/f'bank_{group}/manifest.json') == sr['bank_manifest_sha256']
        for goal,case in enumerate(bm['cases']):
            op=root/f'bank_{group}'/case['outcome_file'];assert sha(op)==case['outcome_sha256']
            with np.load(op) as z: actual=physical_goal_cost(z['terminal_states'],z['goal_state'])
            for c in range(3):
                row=sr['models'][c]['cases'][goal];fp=root/f'scores_{group}'/row['file']
                assert sha(fp)==row['sha256']
                with np.load(fp) as z:
                    for si,space in enumerate(['pose','latent']):
                        scores=z[space+'_cost']
                        metrics=[]
                        for b in range(4):
                            m=ranking_metrics(scores[b],actual);metrics.append(m)
                            ids=m['argmin_indices']
                            # Native latent distance is not a physical cost prediction.
                            ae=float(np.abs(scores[b,ids]-actual[ids]).mean()) if si==0 else 0.
                            free_ids=np.flatnonzero(scores[0]==scores[0].min())
                            fixed_ae=float(np.abs(scores[b,free_ids]-actual[free_ids]).mean()) if si==0 else 0.
                            pool_ae=float(np.abs(scores[b]-actual).mean()) if si==0 else 0.
                            values[group,c,si,b,goal]=[m['selected_cost'],ae,
                                m['spearman'] if m['spearman'] is not None else np.nan,
                                m['kendall_tau_b'] if m['kendall_tau_b'] is not None else np.nan,
                                fixed_ae,pool_ae]
                        if si==0:
                            free=scores[0];order=np.sort(free);iqr=float(np.quantile(free,.75)-np.quantile(free,.25))
                            changed=metrics[0]['argmin_indices']!=metrics[1]['argmin_indices']
                            margin_rows.append([group,c,goal,changed,order[1]-order[0],iqr,
                                float(free[metrics[1]['argmin_indices']].mean()-free.min())])
        print('decision group complete',group,flush=True)
    draws=np.random.default_rng(1391001).integers(0,512,(20000,512))
    result={}
    for c,obj in enumerate(OBJECTIVES):
        result[obj]={}
        for si,space in enumerate(['pose','latent']):
            x=np.nanmean(values[:,c,si],axis=0)  # branches, goals, metrics
            # Bind the independent recomputation to the accepted primary summary.
            for b,branch in enumerate(BRANCHES):
                expected=compact['objectives'][obj]['score_spaces'][space]['branches'][branch]['selected_cost']['mean']
                np.testing.assert_allclose(x[b,:,0].mean(),expected,rtol=1e-12,atol=1e-8)
            result[obj][space]={'branches':{},'contrasts':{},
                'undefined_rank_rows':{branch:int(np.isnan(values[:,c,si,b,:,2]).sum()) for b,branch in enumerate(BRANCHES)}}
            for b,branch in enumerate(BRANCHES):
                result[obj][space]['branches'][branch]={n:estimate(x[b,:,j],draws) for j,n in enumerate(names) if si==0 or j in (0,2,3)}
            for left,right in [(1,0),(2,0),(3,0),(1,3)]:
                paired=np.nanmean(values[:,c,si,left]-values[:,c,si,right],axis=0)
                result[obj][space]['contrasts'][BRANCHES[left]+'_minus_'+BRANCHES[right]]={n:estimate(paired[:,j],draws) for j,n in enumerate(names) if si==0 or j in (0,2,3)}
        rows=np.asarray([r for r in margin_rows if r[1]==c],float)
        changed=rows[:,3].astype(bool);valid=rows[:,5]>0
        norm=rows[valid,4]/rows[valid,5];penalty=rows[valid,6]/rows[valid,5]
        result[obj]['pose_near_ties']={'rows':len(rows),'changed_rows':int(changed.sum()),'zero_iqr_rows':int((~valid).sum()),
          'exact_minimum_tie_fraction':float((rows[:,4]==0).mean()),
          'normalized_margin_quantiles':np.quantile(norm,[0,.25,.5,.75,.9,1]).tolist(),
          'normalized_margin_changed_quantiles':np.quantile(norm[changed[valid]],[0,.25,.5,.75,.9,1]).tolist(),
          'free_score_penalty_changed_quantiles':np.quantile(penalty[changed[valid]],[0,.25,.5,.75,.9,1]).tolist(),
          'thresholds':[{ 'threshold':t,'all_fraction':float((norm<=t).mean()),'changed_fraction':float((norm[changed[valid]]<=t).mean()),'unchanged_fraction':float((norm[~changed[valid]]<=t).mean())} for t in [.01,.05,.1]],
          'scope':'Descriptive fixed-model/goal rows, not independent replicates. Margins normalized by within-pool IQR; zero-IQR pools excluded only from normalized-margin summaries and counted.'}
    dest=ROOT/'runs/review_followup_v2';dest.mkdir(exist_ok=True)
    np.savez_compressed(dest/'decision_goal_metrics.npz',values=values,margin_rows=np.asarray(margin_rows))
    return result


def main():
    out=ROOT/'runs/review_followup_v2';out.mkdir(exist_ok=True)
    prediction=prediction_analysis()
    (out/'prediction.json').write_text(json.dumps(prediction,indent=2)+'\n')
    result={'status':'EXPLORATORY_SAVED_ARRAY_RECONSTRUCTION','prediction':prediction,'decision':decision_analysis(),
            'source_sha256':sha(__file__),'protocol_sha256':sha(ROOT/'docs/maintrack/REVIEW_FOLLOWUP_V2_PROTOCOL.md'),
            'scope':'Review-driven exploratory analyses; all original primary estimands unchanged. Goal bootstrap conditions on fixed models, donors and candidate pool.'}
    (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print('COMPLETE',out,flush=True)


if __name__=='__main__':main()
