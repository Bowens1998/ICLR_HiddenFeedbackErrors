# When Predictions Become Inputs

Anonymous code and reproducibility materials for **When Predictions Become Inputs: Hidden Feedback Errors in Latent World Models**.

## Quick start: the three prospective experiments

```bash
python -m pip install -r requirements.txt
OPENBLAS_NUM_THREADS=2 python verify_prospective.py --output PROSPECTIVE_VERIFICATION.json
```

The supplied complete arrays reconstruct all eight reserved-evaluator contrasts (four primary and four secondary), the incremental-diagnosis contrast, and both matched-decision primary contrasts. This command also checks the fixed regression prediction arithmetic and all 12 decision-model branch means. It regenerates the original fixed bootstrap draws, with recipient counts 256, 512 and 512. It performs no new fitting, model selection or experimental data generation. Expected status is `PASS_COMPLETE_PROSPECTIVE_SAVED_CONTRAST_RECONSTRUCTION`.

The four measurement primary intervals are positive. Incremental diagnostic gain and both matched-decision primary comparisons are unresolved. All results remain included. The original numerical/readout/population acceptance records are retained; the compact command verifies saved contrast arithmetic, not the complete upstream computation. The frozen S2 numerical verifier retains its original draft-schema metadata; the actual deployed protocol and outer acceptance are separately bound. A draft-schema label is not a claim that scientific choices were selected after outcomes.

## Earlier evidence and raw-input demonstration

The nine evidence archives in `ARCHIVE_MANIFEST.json` retain the earlier evidence packages, with anonymous text metadata where required. This code repository and the code-only ZIP exclude those large archives. Place the separately supplied archives under `archives/`; the complete reproduction ZIP includes them. See `ARTIFACTS.md` for the file inventory and access status. Then:

```bash
python verify_saved_evidence.py --list
python verify_saved_evidence.py --package core --work-dir replay_core
```

The optional `--package all` route reconstructs all earlier saved-evidence packages. It was already validated in the retained receipts; the present revision does not rerun all historical experiments. `RUNTIMES.md` records those earlier measured checks. Do not confuse saved-evidence verification with new model training or simulator regeneration.

`feedback_diagnostic_end_to_end_v1.zip` separately includes three study-trained checkpoints, readouts, normalizers and two fixed raw-image development examples. Extract it and follow its README to generate new projections and GPU rollouts. Its earlier clean GPU acceptance receipts remain in `validation/`. This is a limited demonstration, not reproduction of all training or all populations.

## Code and paper map

| Paper evidence | Included implementation / evidence |
|---|---|
| Figure 1, readout-preserving projection | `source/scripts/visual/readout_fiber.py`; prospective `scripts/s1_projection.py`, `stage2/projection_four.py`, `stage3/projection_eight.py` |
| Figure 2, stronger task measurements | Core archive; independent evaluation-readout archive |
| Figure 3, recursive training | Core archive; `source/strengthening/scripts/train_rolling.py` |
| Figure 4A, reserved evaluator | `prospective/data/s1_*`; prospective fitting, projection, scoring and independent-reference source |
| Figure 4B, incremental diagnosis | `prospective/data/s2_*`; prospective `stage2/regression.py` and `independent_regression.py` |
| Figure 4C, matched decision | `prospective/data/s3_*`; prospective `stage3/population.py` and `decision_statistics.py` |
| Motion constraints and rotated control | Motion archive, including original frozen evaluator sources and complete saved-token evidence |
| Native PointMaze and failed measurement gates | PointMaze and velocity-qualification archives; original negatives retained |
| Earlier cost decomposition and historical tests | Core/legacy/reporting archives and `historical_boundaries/` |

Prospective source paths above are relative to `source/strengthening/prospective_mechanism_v9_20260923/`. The source tree retains the actual scientific functions and original CLI parsers. Infrastructure launchers are excluded. The original frozen-model production entry points require the bound external image corpora, encoder/predictor checkpoints, simulator and correctly remapped asset manifests; they are supplied for inspection, not advertised as portable one-command training. The runnable CPU entry points above and the archived two-case demonstration are the verified portable interfaces.

## Resources and external assets

The new tests reuse frozen predictors; only four new pose readouts and two ridge regressions are fitted. Successful measured allocations were 0.78 / 1.86 / 1.72 GPU-hours and 7.57 / 10.61 / 8.38 CPU-core-hours for measurement / diagnosis / decision, respectively. These exclude queue time, original model training and failed attempts. The initial missing-solver decision attempt used another 0.61 GPU-hours and 2.45 CPU-core-hours. GPU tasks used one RTX PRO 6000, with at most three concurrent decision jobs. The repaired solver was OSQP 0.6.7.post3 with QDLDL 0.1.9.post1; the repair did not alter scientific settings.

Complete historical training images, every study-trained checkpoint and external DINO-WM weights are not redistributed here. Exact model/readout/normalizer identities and fixed data roles are retained in protocol and acceptance records. External DINO-WM acquisition follows the official source identified in the core archive. A full raw-input regeneration of all historical experiments was deferred; this release makes no claim that it was executed.

## Figures, provenance and anonymity

`figures/FIGURE_EDITING.md` lists the editable SVG/PDF assets, generators and supplied plotting inputs. All seven Figure 4 primary contrasts are shown; no result was selected for display by significance. `TERMINOLOGY.md` maps historical identifiers to the manuscript terms.

`MANIFEST.json` authenticates supplied files; `SOURCE_DERIVATION.json` separately records original and anonymous hashes. Scientific source edits only anonymize machine/account metadata; numerical arrays are byte-identical. Reviewer documentation and Figure 1 sampling labels are synchronized with the manuscript. Original provenance hashes remain original identities, not hashes of redacted records. `/external-assets/` denotes intentionally unbundled assets, not a working path. No author identities, private host configuration, repository history, access tokens or review conversations are included. Public upstream author attributions remain intact.

This repository publishes the anonymized code-only v9 payload. Large evidence archives are separate artifacts; they are not part of this commit. The complete reproduction ZIP contains both code and those archives. `ARTIFACTS.md` documents their scope and expected hashes. File-content anonymization does not anonymize a hosting account: reviewer access must use a separately anonymized endpoint or anonymized supplementary files. No reviewer URL is asserted here.
