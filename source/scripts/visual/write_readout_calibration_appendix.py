"""Render accepted pilot results into the manuscript appendix."""
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[2]
s = json.loads((ROOT/'runs/readout_calibration_evaluation_v1/group_0/local_summary.json').read_text())
assert s['status'] == 'PASS_FIXED_ACTION_ARRAYS_AND_PAIRED_GOAL_RECONSTRUCTION'

def interval(v):
    return f"{v['mean'][-1]:.2f} [{v['ci95_low'][-1]:.2f}, {v['ci95_high'][-1]:.2f}]"

lines = [r'''\clearpage
\section{Readout dependence of the supervision ordering}
\label{app:calibration}
We select the first training pool's Transformer and GRU by roster order, before
fitting a new readout. Starting from each original 192--256--256--6 ReLU head,
we fit its weights on the 1280 planner-training windows' four observed tokens,
retaining the original input and target normalization. We use 3000 Adam steps
at learning rate $10^{-4}$, no weight decay, and batches of 256 sampled with
replacement (seed 1392001 plus group index). The final checkpoint is used;
there is no validation checkpoint selection. A predeclared gate requires at
least 10\% lower six-output normalized MSE \emph{and} block-position MSE on
320 separate planner-validation windows. Table~\ref{tab:calibration-gate}
retains both groups. The GRU fails the six-output criterion despite its lower
position error; no GRU continuation follows.
\begin{table}[H]
\centering\small
\begin{tabular}{llrrrrl}
\toprule
& & \multicolumn{2}{c}{Six-output MSE} & \multicolumn{2}{c}{Position MSE} & \\
Validation & Predictor & Old & New & Old & New & Gate\\
\midrule''']
for group, label in [(0,'Transformer'),(1,'GRU')]:
    r=json.loads((ROOT/f'runs/readout_calibration_pilot_v1/group_{group}/report.json').read_text())
    m=r['head_metrics']['planner_validation']
    lines.append(f"Planner & {label} & {m['old']['normalized_mse']:.3f} & {m['new']['normalized_mse']:.3f} & {m['old']['block_position_mse']:.2f} & {m['new']['block_position_mse']:.2f} & {'Pass' if r['head_gate_passed'] else 'Fail'}"+r'\\')
for group, label in [(0,'Transformer'),(1,'GRU')]:
    r=json.loads((ROOT/f'runs/readout_calibration_pilot_v1/group_{group}/report.json').read_text())
    m=r['head_metrics']['expert_validation']
    lines.append(f"Expert & {label} & {m['old']['normalized_mse']:.3f} & {m['new']['normalized_mse']:.3f} & {m['old']['block_position_mse']:.2f} & {m['new']['block_position_mse']:.2f} & ---"+r'\\')
lines.append(r'''\bottomrule
\end{tabular}
\caption{Readout calibration on separate validation windows. Position error is
in squared pixels. The gate uses planner validation only; error increases on
the separate 1441-window expert validation set for both models.}
\label{tab:calibration-gate}
\end{table}
For the passing Transformer, we freeze the new head and repeat both coordinate
continuations from the original 21000-update checkpoint: 2100 updates with the
same training windows, index schedule, optimizer, gradient normalization and
frozen-module boundary as Appendix E.
Evaluation uses all 128 previously evaluated goals, all four original action
streams, and horizons 5, 10, 15, 20 and 25. Actions are unchanged; there is no
new search. The old-head runs reproduce all 2048 complete free/observed-history
token trajectories bit for bit. Independent NumPy reconstruction verifies
all saved decoded predictions. Table~\ref{tab:calibration-outcomes} reports
endpoint results; all five horizons are retained in the experiment records.
Uncertainty uses 20000 shared-goal draws (seed 1393001), after averaging the
four streams within each goal. All intervals are exploratory 95\%.
\begin{table}[H]
\centering\small
\begin{tabular}{llrr}
\toprule
Head & History & Teacher MSE & Labels MSE\\
\midrule''')
for head in ['old','new']:
    for branch,label in [('free','Free'),('teacher','Observed')]:
        m=s['absolute_position_mse']
        lines.append(f"{head.capitalize()} & {label} & {m[head+'/decoded_teacher/'+branch]['mean'][-1]:.2f} & {m[head+'/physical_labels/'+branch]['mean'][-1]:.2f}"+r'\\')
lines.append(r'''\midrule
\multicolumn{2}{l}{Labels minus teacher} & \multicolumn{2}{c}{Mean [95\% interval]}\\''')
for key,label in [('old/free','Old / free'),('new/free','New / free'),('old/teacher','Old / observed'),('new/teacher','New / observed')]:
    lines.append(r'\multicolumn{2}{l}{'+label+r'} & \multicolumn{2}{c}{'+interval(s['labels_minus_teacher'][key])+r'}\\')
for key,label in [('free','Gap change / free'),('teacher','Gap change / observed')]:
    lines.append(r'\multicolumn{2}{l}{'+label+r'} & \multicolumn{2}{c}{'+interval(s['new_minus_old_objective_gap'][key])+r'}\\')
lines.append(r'''\bottomrule
\end{tabular}
\caption{Fixed-action endpoint position MSE and paired objective gaps for the
single passing Transformer. Gap change is new minus old, with shared goals.}
\label{tab:calibration-outcomes}
\end{table}
Decoding the actual endpoint image gives MSE 5404.46 under the old head and
4071.19 under the new one: a paired change of $-1333.26$ with interval
$[-2129.53,-587.83]$. The previously separated objective orderings become
unresolved under the new head. This establishes readout dependence in the tested
Transformer. The improvement is specific to the planner distribution.
Fitting the head changes calibration and geometry together; it does
not isolate calibration as a mediator. Appendix~\ref{app:second-head} separately
tests the complete readout-preserving feedback intervention under the fitted heads.
''')
(ROOT/'paper/iclr2027/readout_calibration.tex').write_text('\n'.join(lines))
