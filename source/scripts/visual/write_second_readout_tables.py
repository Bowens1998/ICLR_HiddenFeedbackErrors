"""Make manuscript tables from independently accepted second-head statistics."""
from pathlib import Path
import json
ROOT=Path(__file__).resolve().parents[2]
s=json.loads((ROOT/'runs/second_readout_analysis_v1/summary.json').read_text());assert s['status']=='PASS_ALL_THREE_SECOND_READOUT_CONFIGURATIONS'
labels=['Transformer: fixed','GRU: fixed','Transformer: adapted'];obj={'latent':'Latent','decoded_teacher':'Teacher','physical_labels':'Labels'}
def sci(v):
    if v == 0:return '$0$'
    base,power=f'{v:.2e}'.split('e')
    return '$'+base+r'\times10^{'+str(int(power))+'}$'
def ci(v):return f"${v['mean'][-1]:.2f}\\;[{v['ci95_low'][-1]:.2f},{v['ci95_high'][-1]:.2f}]$"
rows=[r'''\begin{table}[htbp]
\centering\small\setlength{\tabcolsep}{2.5pt}
\begin{tabular}{llrr}
\toprule
Configuration & Objective & Actual minus free & Actual minus donor\\
\midrule''']
for profile,label in zip(s['profiles'],labels):
    for j,row in enumerate(profile['rows']):rows.append(f"{label if j==0 else ''} & {obj[row['objective']]} & {ci(row['actual_minus']['free'])} & {ci(row['actual_minus']['donor'])}"+r'\\')
    if profile['index']!=2:rows.append(r'\addlinespace')
rows.append(r'''\bottomrule
\end{tabular}
\caption{The complete feedback comparison under a second pose head. Each configuration retains all 128 reused goals, four fixed action streams and three objectives; brackets are exploratory paired-goal 95\% intervals. The first two configurations change the head and preserve model weights. The third uses the already adapted Transformer continuations; its supervision ordering was unresolved. Displacements match across all six objective/source directions within each configuration. Negative values favor actual guidance.}
\label{tab:second-head}
\end{table}''')
(ROOT/'paper/iclr2027/second_readout_main.tex').write_text('\n'.join(rows)+'\n')
lines=[r'''\section{Second-readout feedback sensitivity}
\label{app:second-head}
This focused experiment fixes its three configurations before constructing new
feedback corrections. All heads, model weights, recipient goals, donor goals and
actions already exist. The first training pool's Transformer and GRU are selected
by roster order. Configurations 0 and 1 use the three original continuation models
with the fitted head from Appendix~\ref{app:calibration}. This holds the predictive
model fixed while changing the readout, its admissible correction set and the
measurement of future pose. Configuration 2 uses the same Transformer head with
the two previously adapted coordinate continuations and the original latent
continuation; latent training does not use the pose head. No new training occurs.

The GRU's earlier calibration gate failure remains a failure. Its fixed-model
head sensitivity is not a calibrated-continuation study, and no GRU continuation
is added. Both fitted heads improve planner position error but worsen expert
validation error (Table~\ref{tab:calibration-gate}); this is not a comparison to
universally better physical-state measurements.

We retain all 128 previously evaluated goals and all four original action streams.
The donor bank is the independent 128-goal bank, with permutation seed 1368001.
For every objective and source, we solve the same fixed-region convex projection
under the second head, then rescale all six directions to their minimum
standardized norm. The complete six-output readout is rechecked after FP32
conversion; activation-region constraints and norm matching use the original
$10^{-6}$ tolerances. No goal, zero-norm case or adverse outcome is removed.
Free, matched actual, matched donor and full observed-token reset branches use
the original batch-of-300 arithmetic and the same selected actions. Every complete
free trajectory must exactly reproduce its original or calibrated-head archive.

All 9216 projections are accepted, and all 4608 complete free-token trajectories
match bit for bit. Independent local reconstruction checks source bindings,
readouts, regions, displacements and all saved horizon predictions. Table~\ref{tab:second-geometry}
reports constraint residuals and scales. Because each head induces a different
feasible region, its common displacement is recomputed: an across-head effect
change is not a pure calibration effect at identical perturbation directions.
\begin{table}[H]
\centering\small
\begin{tabular}{lrrr}
\toprule
Configuration & Mean shared norm & Max. readout deviation & Max. region violation\\
\midrule''']
for profile,label in zip(s['profiles'],labels):
    geo=profile['geometry'];error=max(x['matched_readout_error'] for x in geo['rows']);region=max(x['region_violation'] for x in geo['rows']);lines.append(f"{label} & {geo['mean_target_norm']:.3f} & {sci(error)} & {sci(region)}"+r'\\')
lines.append(r'''\bottomrule
\end{tabular}
\caption{All three-objective configurations pass the original nonlinear constraint and norm checks. Norms use the fixed head-input standardization; readout deviations use its six-output standardization.}
\label{tab:second-geometry}
\end{table}
\begin{table}[H]
\centering\small
\begin{tabular}{llrrrr}
\toprule
Configuration & Objective & Free & Actual & Donor & Reset\\
\midrule''')
for profile,label in zip(s['profiles'],labels):
    for j,row in enumerate(profile['rows']):lines.append(f"{label if j==0 else ''} & {obj[row['objective']]} & "+' & '.join(f"{row['absolute_position_mse'][b]['mean'][-1]:.2f}" for b in ['free','actual','donor','reset'])+r'\\')
lines.append(r'''\bottomrule
\end{tabular}
\caption{Absolute endpoint block-position MSE under the second head. The reset changes the current readout and is a separate reference.}
\label{tab:second-absolute}
\end{table}
The main-text contrasts in Table~\ref{tab:second-head} average four streams within
each goal before 20000 shared-goal bootstrap draws (seed 1394001). Intervals are
exploratory 95\%; these consumed goals do not constitute a new confirmation.
All five horizons are retained in the evidence package. Configurations are not
pooled as independent replicates. Table~\ref{tab:original-head-context} gives the
original-head effects for the same fixed models, avoiding substitution of
six-group aggregate estimates for these individual configurations.
\begin{table}[H]
\centering\small\setlength{\tabcolsep}{2.5pt}
\begin{tabular}{llrr}
\toprule
Original head & Objective & Actual minus free & Actual minus donor\\
\midrule''')
for profile,label in zip(s['profiles'][:2],['Transformer','GRU']):
    for j,row in enumerate(profile['rows']):lines.append(f"{label if j==0 else ''} & {obj[row['objective']]} & {ci(row['original_head_actual_minus']['free'])} & {ci(row['original_head_actual_minus']['donor'])}"+r'\\')
lines.append(r'''\bottomrule
\end{tabular}
\caption{Original-head context on the identical first-pool models and goals, with the same exploratory bootstrap draws as the second-head sensitivity. Full paired changes of intervention effects are retained in the evidence package.}
\label{tab:original-head-context}
\end{table}''')
(ROOT/'paper/iclr2027/second_readout_appendix.tex').write_text('\n'.join(lines)+'\n')
