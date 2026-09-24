"""Render accepted review-driven analyses without altering primary results."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
source=ROOT/'runs/review_followup_v2/summary.json'
r=json.loads(source.read_text())
receipt=json.loads((source.parent/'independent_acceptance.json').read_text())
assert receipt['summary_sha256']==hashlib.sha256(source.read_bytes()).hexdigest()
objs=['unit_latent','unit_decoded_teacher','unit_physical_labels']
labels=['Latent','Teacher','Labels']

def effect(x,d=2):
    a,b=x['exploratory_ci95']
    return f"${x['mean']:.{d}f}\\;[{a:.{d}f},{b:.{d}f}]$"

def table(columns,header,rows,caption,label):
    return ('\\begin{table}[!htbp]\n\\centering\\small\n\\setlength{\\tabcolsep}{3pt}\n'
            +'\\begin{tabular}{@{}'+columns+'@{}}\n\\toprule\n'+header+' \\\\\n\\midrule\n'
            +'\n'.join(' & '.join(row)+' \\\\' for row in rows)
            +'\n\\bottomrule\n\\end{tabular}\n\\caption{'+caption+'}\n\\label{'+label+'}\n\\end{table}\n')

text=r'''\section{Objective gaps, readout error, and decision diagnostics}
\label{app:review-analysis}
These additional analyses use previously evaluated goals and accepted arrays,
without new model selection or exclusions. They were specified after the original
results were known. All new intervals are exploratory 95\% paired-goal
percentile intervals; they do not expand either primary confirmation family.
Prediction analyses average six fixed groups and four streams within each of 128
goals, then use 20000 shared draws with seed 1368001. Decision analyses average
the six fixed groups within each of 512 goals and use 20000 shared draws with seed
1391001. Undefined rank correlations are retained as missing, with their counts
recorded; no corresponding cost outcome is removed.

\paragraph{The supervision gap and the within-model intervention.}
Table~\ref{tab:absolute-prediction} reports endpoint error levels for every saved
branch. ``Observed encoding'' evaluates the frozen head directly on the actual
endpoint image; ``observed history'' still predicts the endpoint from earlier
observations. They are different diagnostics. Matched actual guidance improves
both coordinate objectives but does not restore their observed-history ordering.
The labels-minus-teacher gap increases by 275.19, with interval [48.62,501.03]
(Table~\ref{tab:objective-gap}). Four of six group point estimates increase;
their changes are -95.74, 374.24, 205.41, 578.11, -110.15, and 699.27.
Thus the experiment identifies correctable feedback error within each model,
without supplying a causal decomposition of the ordering between objectives.
'''
paths=[('observed','Observed encoding'),('teacher','Observed history'),('free','Free'),
       ('fiber','Matched actual'),('shuffled','Matched donor'),('full','Full projection'),('reset','Observed-token reset')]
rows=[[name]+[f"{r['prediction']['absolute_endpoint_position_mse'][obj][path]['mean']:.2f}" for obj in objs] for path,name in paths]
text+=table('lrrr','Branch & Latent & Teacher & Labels',rows,
 'Absolute block-position MSE at action 25, in squared pixels. Full projection is the unscaled readout-preserving correction; reset replaces the token with its observed encoding. All branches use the same physical targets.', 'tab:absolute-prediction')
gap=r['prediction']['labels_minus_teacher']
rows=[[name,effect(gap[path])] for path,name in [('teacher','Observed history'),('free','Free'),('fiber','Matched actual')]]
rows.append(['Change: actual minus free',effect(r['prediction']['change_in_gap_actual_minus_free'])])
text+=table('lr',r'Condition & Labels minus teacher: mean [95\% interval]',rows,
 'Paired supervision gaps. The last row is a difference of paired intervention effects, not subtraction of confidence-interval endpoints.', 'tab:objective-gap')
text+=r'''
\paragraph{Observed-encoding readout error.}
The frozen readout is not exact on the diagnostic trajectories
(Table~\ref{tab:observed-readout}). Teacher targets $g(E(o))$ are compatible with
the observed token by construction; physical-label targets need not be.
Predictor compensation for readout error is therefore a concrete alternative
source of the supervision gap. We measure association using each goal's action-5
six-output normalized head MSE, averaged over groups and streams, and that goal's
endpoint gap or actual-minus-free intervention effect. Pearson correlations are
'''
ass=r['prediction']['goal_level_associations']
names=[('readout_error_vs_free_objective_gap','free labels-minus-teacher gap'),
       ('readout_error_vs_unit_latent_actual_minus_free','latent intervention'),
       ('readout_error_vs_unit_decoded_teacher_actual_minus_free','teacher intervention'),
       ('readout_error_vs_unit_physical_labels_actual_minus_free','labels intervention')]
pieces=[]
for k,name in names:
    x=ass[k];a,b=x['exploratory_ci95'];pieces.append(f"${x['pearson']:.3f}\\;[{a:.3f},{b:.3f}]$ ({name})")
text+=', '.join(pieces)+'.\n'
text+=r'''Negative intervention differences denote improvement, so a positive
correlation denotes a less favorable effect at higher readout error. These
associations do not establish that calibration causes the gap or mediates the
intervention. Goal difficulty and distribution shift can affect both quantities.
'''
cal=r['prediction']['observed_encoding_calibration']
rows=[[str(t),effect(cal['block_position_mse'][i]),effect(cal['six_output_standardized_mse'][i],3)] for i,t in enumerate(cal['horizons'])]
text+=table('lrr','Action & Observed block-position MSE & Six-output normalized MSE',rows,
 'Frozen-head error on actual image encodings. The six-output metric averages squared errors after the original per-output normalization; it is not an error in complete physical state.', 'tab:observed-readout')
text+=r'''
\paragraph{Full reset and absolute decision costs.}
The full observed-token reset uses the observation already available after the
executed prefix. It is evaluated over exactly the same candidate pool as every
other branch. Table~\ref{tab:reset-decision} reports all branches under both scores.
None of the pose-score reset-minus-free intervals excludes zero. Native latent
scoring gives secondary reset benefits for the latent and labels objectives.
This score dependence is consistent with the secondary actual-guidance results
in Appendix~\ref{app:decisions}; it does not replace the four primary pose-score
tests or establish that no scoring rule can benefit from feedback correction.
'''
rows=[]
for obj,label in zip(objs,labels):
    for score in ['pose','latent']:
        d=r['decision'][obj][score]
        rows.append([label+' / '+score]+[f"{d['branches'][b]['selected_cost']['mean']:.2f}" for b in ['free','act','don','reset']]+[effect(d['contrasts']['reset_minus_free']['selected_cost'])])
text+=table('lrrrrr','Objective / score & Free & Actual & Donor & Reset & Reset minus free',rows,
 r'Absolute mean realized selected cost and exploratory reset-minus-free 95\% intervals. Scores select the action; every column reports the same physical cost with exact-tie expectation. Native latent scores themselves are not interpreted as physical cost predictions.', 'tab:reset-decision')
text+=r'''
\paragraph{Ranking and cost prediction.}
Actual guidance does not resolve the weak pose-score ordering of physical
candidate costs (Table~\ref{tab:candidate-ranks}). All actual-minus-free rank
intervals cross zero. At the same time, mean absolute physical-cost prediction
error decreases over the entire candidate pool for all objectives
(Table~\ref{tab:cost-calibration}). At the candidates selected by the free branch,
both coordinate objectives improve, whereas the latent objective worsens.
The error at each branch's own selected candidates also decreases, but that
comparison changes both the predicted scores and the candidates being evaluated.
Keeping the free-selected candidates fixed separates those two changes.
These distinctions explain how better cost estimates can coexist with unresolved
ranking and selected-cost gains; they do not identify a unique cause of the latter.
'''
rows=[]
for obj,label in zip(objs,labels):
    d=r['decision'][obj]['pose'];c=d['contrasts']['act_minus_free']
    rows.append([label]+[f"{d['branches'][b]['spearman']['mean']:.4f}" for b in ['free','act','reset']]+[effect(c['spearman'],4),effect(c['kendall_tau_b'],4)])
text+=table('lrrrrr',r'Objective & Free $\rho$ & Actual $\rho$ & Reset $\rho$ & $\Delta\rho$ & $\Delta\tau_b$',rows,
 r'Within-pool Spearman and Kendall rank diagnostics under the pose score. Differences are actual minus free, with exploratory 95\% intervals. Correlations are averaged within goal over defined fixed-model comparisons.', 'tab:candidate-ranks')
rows=[]
for obj,label in zip(objs,labels):
    c=r['decision'][obj]['pose']['contrasts']['act_minus_free']
    rows.append([label]+[effect(c[k]) for k in ['pool_absolute_prediction_error','free_selected_absolute_prediction_error','selected_cost_absolute_prediction_error']])
text+=table('lrrr','Objective & All candidates & Fixed free selection & Each branch\'s selection',rows,
 r'Actual-minus-free change in mean absolute cost prediction error, in physical cost units, with exploratory 95\% intervals. The three columns define different candidate distributions; only the first two hold the evaluated candidates fixed.', 'tab:cost-calibration')
text+=r'''
\paragraph{How close are the choices to a tie?}
Let $s_{(1)}\le s_{(2)}$ be the two lowest free-branch pose scores and let
$I=Q_{.75}(s)-Q_{.25}(s)$ be the pool's interquartile range. We summarize
$(s_{(2)}-s_{(1)})/I$ at three fixed thresholds
(Table~\ref{tab:near-ties}). All 9216 pools have positive $I$.
Changed choices have smaller margins more often, but are not restricted to
near ties: at the 5\%-of-IQR threshold, only 30.8--32.6\% of changed choices
qualify, versus 12.2--15.5\% of unchanged choices. Exact minimum-score ties occur
in only 0.52\%, 0.26\%, and 0.65\% of latent, teacher and labels cases.
These are descriptive proportions of fixed model-goal rows; the rows are not
treated as independent evidence for an additional hypothesis test.
'''
rows=[]
for obj,label in zip(objs,labels):
    d=r['decision'][obj]['pose_near_ties'];rows.append([label]+[f"{100*x['all_fraction']:.1f} / {100*x['changed_fraction']:.1f}" for x in d['thresholds']])
text+=table('lrrr',r'Objective & $\le1\%$ IQR & $\le5\%$ IQR & $\le10\%$ IQR',rows,
 'Near-minimum margins: percentage of all choices / percentage of changed choices. Thresholds scale by the within-pool score IQR, not by the minimum score or physical outcome.', 'tab:near-ties')
(ROOT/'paper/iclr2027/review_followup.tex').write_text(text)
print('Wrote accepted review follow-up appendix')
