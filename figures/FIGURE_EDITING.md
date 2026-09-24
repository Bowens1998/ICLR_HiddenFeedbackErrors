# Editable figures for the integrated manuscript

All active figures have editable SVG text and vector objects, a vector PDF used by LaTeX, and a generator. Install DejaVu Sans before editing SVGs in Inkscape or Illustrator. The scientific numbers are read from accepted result records; redrawing performs no fitting, inference, or new statistical analysis.

| Location | Vector basename | Generator and data |
| --- | --- | --- |
| Figure 1: intervention and extensions | `feedback_design` | `draw_feedback_design.py` |
| Figure 2: core and evaluation-only measurement | `confirmation_measurement_audit` | `draw_confirmation_measurement_audit.py`, matching JSON |
| Figure 3: training under both readouts | `training_readout_sensitivity` | `draw_training_readout_sensitivity.py`, matching JSON |
| Figure 4: prospective tests | `prospective_results` | `draw_prospective_results.py`, matching JSON |
| Appendix: motion constraints | `motion_constraint_results` | `draw_result_figures.py`, `result_figure_data.json` |
| Appendix: readout learning curves | `head_learning_curves` | `draw_appendix_figures_56.py`, `appendix_figure_56_data.json` |
| Appendix: earlier decision decomposition | `decision_decomposition` | `draw_appendix_figures_56.py`, `appendix_figure_56_data.json` |
| Appendix: historical confirmation and ordering | `confirmation`, `reversal` | `draw_legacy_figures.py`, `legacy_figure_data.json` |
| Appendix: Reacher | `reacher_transfer` | `draw_legacy_figures.py`, `legacy_figure_data.json` |

Run the generators from the extracted `figures` directory. Figure 1 requires ReportLab and accepts `--font-dir /usr/share/fonts/truetype/dejavu`. The quantitative plots require NumPy and Matplotlib. All accept `--output-dir DIR`; their default JSON inputs are supplied. No model or repository is needed for redrawing or for compiling the paper. Python is not needed to compile the already supplied figure PDFs.

Figure 1 uses a 7.5-inch source canvas. At manuscript width, its supporting text is approximately 6 points; mathematical scripts are smaller. Inspect the printed size after any edits. PDF and SVG share the same primitives. To refresh its preview, run `pdftoppm -singlefile -scale-to 1800 -png feedback_design.pdf feedback_design`.

## Figure 1 semantics

A–D depict the original 256-recipient core. E depicts three separate prospective extensions, not more branches of the same core population.

- `E` includes image preprocessing, the encoder and observation projector. `F_theta` includes the action encoder, temporal predictor and prediction projector. A frozen `F_theta` is shared between branches of one model, not across independently trained models.
- Three observed tokens at times −10, −5 and 0 produce the insertion token at action 5. Each transition receives all three aligned five-action blocks. Actual and donor guidance come from their respective action-5 observations.
- The source-specific projection preserves all six nonlinear `g_A` outputs and the original ReLU activation region. `x` and `d` use the fixed input-standardized metric. The common norm is the minimum across three objectives and two sources; unchanged free feedback is outside this six-direction minimum.
- The original `z_5` reaches both projections and the match/unscale/add reconstruction. Unscaling multiplies the standardized displacement by the fixed latent standard deviations before addition. Projection directions are not averaged.
- Three branches continue independently. History cells display times −5, 0 and 5; the explicit definition below them gives token values. Four transitions reach action 25. Each GRU restarts on its three-token window. No later observations enter the rollout.
- `g_pos` selects block x/y from `g_A`. Terminal `L_25` is squared Euclidean error against the same recipient endpoint, with no division by two. Ground truth and readout outputs enter scoring, not the transition. Equalities apply only to insertion-time outputs within a model, at the stated floating-point acceptance tolerance.
- The reserved-evaluator extension compares `{g_A}` and `{g_A,g_C}` at one eight-direction norm (two constraints × two objectives × two sources). `g_D` only measures the predictions. Its 256 recipients are fresh; the readout fits use disjoint C/D measurement parents.
- The diagnosis uses a four-direction T0 probe (two objectives × two sources), 256 calibration and 512 separate test recipients. It scores with the prior `g_eval`, not new `g_D`. Its feature `s_probe = donor loss − actual loss` differs from population remaining correctability `G = E[free loss − actual loss]`.
- The matched decision experiment uses another 512 recipients, 32 suffixes after a common five-action prefix, and eight directions (T0/T1 × two objectives × two sources). Every model retains its own insertion anchor. The same candidate set is used by all models in a pool. The endpoint remains action 25.

## Quantitative plot semantics

Figure 2 retains all nine original primary contrasts and complete fixed-group points. Evaluation-only insertion and terminal means are descriptive. Its native model has a separate bank and scale.

Figure 3 scores the same training branches with both readouts; every pool and training condition remains. Points are descriptive, not joint confidence regions. Complete paired intervals remain in the appendix.

Figure 4 retains all four reserved-evaluator primary contrasts, the single incremental-diagnosis primary contrast and both matched-decision primary contrasts. The panels use their own populations, units and confidence families: 98.75%, 95% and 97.5%, respectively. Positive favors actual guidance in A and the augmented predictor in B; negative favors actual guidance in C. Crossing intervals remain visible. No contrast is selected for display according to its outcome.

The motion figure keeps its six primary and separate six secondary 99.1667% intervals. The earlier decision decomposition keeps its complete three-objective contrasts and squared cost units. Historical plots retain their original populations, readouts and inferential identities; they are not additional replications of the prospective experiments.

The objective names are **Latent**, **Coordinate teacher**, and **Physical labels**. Training conditions are **T0: observed history**, **T1: recursive**, and **T2: + latent anchoring**. PushT prediction uses terminal block-position MSE; response-prediction MSE has units pixels to the fourth power; decision plots use physical pose cost. These quantities are not interchangeable.

## Design references

The requested Top-Conf Figure Gallery (https://github.com/qwdwqfwq/topconf-paper-figure-gallery) informed the prior visual hierarchy and spacing. No third-party artwork was copied. There is no official ICLR figure style. This revision retains that visual system and updates the experiment map and accepted quantitative results; the manuscript keeps its 14 verified scientific references.
