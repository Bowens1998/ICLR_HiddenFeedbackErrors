# Measured reproduction times

These are measured timings, not estimates for retraining. CPU commands ran in separate processes with two BLAS/OpenMP threads on an AMD Ryzen 9 5900HS, Linux x86-64, Python 3.13.9 and NumPy 2.3.5. They used fresh isolated archive extractions and the existing local Python environment. Archives were authenticated before execution. Dependency installation, archive extraction and the outer archive-hash check are excluded; verification performed inside each command is included.

| Saved-evidence verifier | Wall time (s) | Peak resident memory (MiB) | Result |
|---|---:|---:|---|
| confirmation | 0.64 | 272.0 | PASS |
| decision | 0.39 | 392.7 | PASS |
| legacy | 1.36 | 75.6 | PASS |
| motion | 219.18 | 661.1 | PASS |
| qualification | 0.77 | 83.3 | PASS |
| reporting | 0.88 | 143.9 | PASS |
| second_readout | 0.30 | 151.0 | PASS |
| independent_evaluation_readout | 4.63 | 362.8 | PASS |
| pointmaze_confirmation | 45.30 | 522.6 | PASS |

Exact commands, archive hashes, process measurements and output logs are in `validation/cpu_replay/`, `validation/independent_evaluation_readout/` and `validation/pointmaze_confirmation/`. The evaluation-readout verifier additionally used SciPy 1.16.3. Other hardware and numerical-library builds can differ. The earlier archive READMEs retain their original tested dependency versions; the table above records this additional replay environment.

## Previously recorded GPU demonstration

The raw-input package was previously validated in a clean Python 3.11 environment using PyTorch 2.7.0+cu128, NumPy 1.26.4 and an NVIDIA RTX PRO 6000 Blackwell Server Edition. The following are the program-reported diagnostic-computation durations from those original bound receipts; they exclude environment installation and a separately timed independent verifier. They are not newly measured in this CPU packaging audit.

| Fixed development input | Constraint mode | Recorded diagnostic time (s) |
|---|---|---:|
| case_000 | dual-comparison | 3.870453 |
| case_000 | single | 3.391216 |
| case_001 | dual-comparison | 3.560474 |
| case_001 | single | 1.678658 |

The four original reports and independent verification receipts are bound by `validation/gpu_demo_original/VALIDATION.json`, copied from the raw-input archive. Full training runtime, full simulator replication runtime and a verified CPU raw-input runtime are **not measured by this supplement**.

## Current prospective compact check

Verified with Python 3.13.9 and NumPy 2.3.5, two BLAS threads. The measured complete saved-contrast check took 0.364 seconds on the packaging host, including manifest checks. This is a compact CPU check, not new training, projection or simulator timing. The exact receipt is validation/prospective_saved_contrasts.json.
