# When Predictions Become Inputs

Code and experiment materials for **When Predictions Become Inputs: Hidden Feedback Errors in Latent World Models**.

The code follows the paper's experimental workflow: construct latent-model predictions, apply feedback interventions that preserve the current task readout, continue the rollout, and evaluate future predictions and action selection. It also includes readout fitting, recursive training, and figure generation.

## Download materials

**[Download the complete reproduction package (1.35 GB)](https://github.com/Bowens1998/ICLR_HiddenFeeadbackErrors/releases/download/v9/when_predictions_become_inputs_anonymous_reproduction_v9.zip)**

The complete package contains the code and all nine experiment archives.

| Download | Contents |
|---|---|
| [Code-only package (4.5 MB)](https://github.com/Bowens1998/ICLR_HiddenFeeadbackErrors/releases/download/v9/when_predictions_become_inputs_anonymous_code_v9.zip) | Source code, configurations, plotting inputs, and saved analysis arrays |
| [SHA256SUMS.txt](https://github.com/Bowens1998/ICLR_HiddenFeeadbackErrors/releases/download/v9/SHA256SUMS.txt) | Checksums for the two ZIP files |
| [ARCHIVE_MANIFEST.json](https://github.com/Bowens1998/ICLR_HiddenFeeadbackErrors/releases/download/v9/ARCHIVE_MANIFEST.json) | Archive filenames, sizes, and checksums |

All files are available on the **[v9 Release page](https://github.com/Bowens1998/ICLR_HiddenFeeadbackErrors/releases/tag/v9)**. Extract the complete package into its own directory, or copy its `archives/` directory into an existing clone.

## Repository structure

```text
source/
  scripts/visual/                 # Models, readouts, interventions, and evaluation
  strengthening/
    adapters/                     # Shared model and training utilities
    scripts/                      # Readout fitting and model training
    prospective_mechanism_v9_20260923/
      scripts/                    # Measurement and reserved-readout evaluation
      stage2/                     # Training-response diagnosis
      stage3/                     # Candidate-action evaluation and selection
      verification/               # Reference implementations and checks
prospective/
  protocols/                      # Experiment configurations and data assignments
  data/                           # Saved arrays and analysis reports
figures/                          # Plotting data, generators, and vector figures
archives/                         # Additional experiment packages
historical_boundaries/            # Supporting analyses and cross-task records
validation/                       # Numerical checks and execution records
verify_prospective.py             # Reconstruct prospective comparisons
verify_saved_evidence.py          # Run analysis for archived experiments
```

## Main components

- **Models and task readouts:** [`factorial_model.py`](source/scripts/visual/factorial_model.py) and [`nonlinear_pose_cost.py`](source/scripts/visual/nonlinear_pose_cost.py) define model construction and nonlinear pose evaluation.
- **Feedback interventions:** [`readout_fiber.py`](source/scripts/visual/readout_fiber.py) implements the readout-preserving projection. Related scripts in `source/scripts/visual/` prepare inputs, run rollout branches, and compute comparison statistics.
- **Recursive training:** [`train_rolling.py`](source/strengthening/scripts/train_rolling.py) runs the training conditions, using the prediction and loss functions in [`rolling_training.py`](source/strengthening/adapters/rolling_training.py).
- **Measurement, diagnosis, and decisions:** [`prospective_mechanism_v9_20260923/`](source/strengthening/prospective_mechanism_v9_20260923/) contains the three experiment pipelines. Their configurations and saved analysis inputs are in `prospective/`.

## Run the saved-result analysis

From the repository root:

```bash
python -m pip install -r requirements.txt
OPENBLAS_NUM_THREADS=2 python verify_prospective.py --output PROSPECTIVE_VERIFICATION.json
```

This reconstructs the measurement, training-response diagnosis, and action-selection comparisons from `prospective/data/` and writes a JSON report. Use a new output filename for each run.

For archived experiments, place the corresponding ZIP files in `archives/`, then list the available analyses or run one package:

```bash
python verify_saved_evidence.py --list
python verify_saved_evidence.py --package core --work-dir replay_core
```

The runner extracts the selected package and writes its outputs under `--work-dir`. Use a new directory for each run. [`RUNTIMES.md`](RUNTIMES.md) lists the execution environments and recorded runtimes.

## Model experiments

Training and rollout scripts are under `source/`. GPU experiments use the dataset, checkpoint, readout, and normalizer paths specified in their experiment manifests.

For a raw-image example, extract `archives/feedback_diagnostic_end_to_end_v1.zip` and follow its README. That package contains checkpoints, readouts, normalizers, input images, and commands for projection and rollout.

## Generate figures

```bash
python -m pip install -r requirements-figures.txt
python figures/draw_training_readout_sensitivity.py --output-dir figure_output
```

Figure generators read the supplied plotting data and export PDF, SVG, and PNG files. [`figures/FIGURE_EDITING.md`](figures/FIGURE_EDITING.md) maps the paper figures to their generators and inputs.

See [`TERMINOLOGY.md`](TERMINOLOGY.md) for the mapping between code identifiers and paper terminology.
