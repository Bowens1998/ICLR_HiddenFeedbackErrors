# Frozen identifiers and manuscript terminology

The manuscript uses one displayed term for each concept. Original array keys, Python identifiers, filenames and frozen protocol text retain their historical spelling to preserve their numerical and hash bindings.

| Frozen identifier or older display term | Manuscript meaning |
|---|---|
| `decoded_teacher`, `unit_decoded_teacher` | Coordinate-teacher objective |
| `physical_labels`, `unit_physical_labels` | Physical-label objective |
| `latent`, `unit_latent` | Latent objective |
| `head`, `probe` in the readout-fitting code | Task readout; distinct from attention heads |
| `fiber` in original prediction arrays | Actual-guided, readout-preserving correction |
| `shuffled` in original prediction arrays | Donor-guided, readout-preserving correction |
| `teacher` branch in original prediction arrays | Observed-history prediction; distinct from the coordinate-teacher training objective |
| `observed` branch | Observed encoding scored through the readout |
| `joint` in the motion condition list | Pose + velocity constraints |
| `rotated` in the motion condition list | Pose + rotated velocity constraints |
| `pose` in the motion condition list | Pose-only constraints |
| `G_free_minus_actual` | Remaining correctability, the free–actual loss difference |
| `donor` minus `actual` in motion contrasts | Source benefit, not remaining correctability |
| `dose`, `budget` in displacement records | Standardized displacement norm or matching-family convention, as defined in that record; full resets are not norm-matched corrections |

Historical pool indices are preserved. An identifier in an earlier experiment must not be remapped to another experiment merely because both use the same displayed architecture name.

The manuscript uses **recipient** for the resampled evaluation case. Historical `goal`, `goal_id` and `goal_contrasts` identifiers retain their frozen spelling; they are not additional independent physical targets. A goal state remains the desired physical target, distinct from the recipient case. New symbols are `g_C` (additional pose constraint), `g_D` (reserved evaluator), and `s_probe` (donor minus actual loss); `s_probe` is distinct from remaining correctability `G`.
