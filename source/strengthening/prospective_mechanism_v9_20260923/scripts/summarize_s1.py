"""Frozen S1 paired-recipient estimands; no case, readout or horizon selection."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def checked_json(path, expected):
    path = Path(path)
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError(f"Hash mismatch: {path}")
    return json.loads(path.read_text())


def compute_goal_contrasts(predictions, truth):
    """Axes: group, objective, constraint, recipient, stream, branch, step, pose."""
    p = np.asarray(predictions, np.float64)
    t = np.asarray(truth, np.float64)
    if p.ndim != 8 or p.shape[:3] != (2, 2, 2) or p.shape[4:] != (4, 4, 5, 6):
        raise ValueError("Unexpected complete-population prediction shape")
    n = p.shape[3]
    if t.shape != (2, n, 4, 5, 6) or not np.isfinite(p).all() or not np.isfinite(t).all():
        raise ValueError("Invalid truth or nonfinite evidence")
    for branch in (0, 3):
        if not np.array_equal(p[:, :, 0, :, :, branch], p[:, :, 1, :, :, branch]):
            raise ValueError("Free/reset copies must be identical across constraint sets")
    loss = np.sum((p[..., 2:4] - t[:, None, None, :, :, None, :, 2:4]) ** 2, axis=-1)
    vectors = {}
    for o, objective in enumerate(("decoded_teacher", "physical_labels")):
        # Each expression below is [group, recipient, stream]; only recipient is resampled.
        free = loss[:, o, 0, :, :, 0, -1]
        a = loss[:, o, 0, :, :, 1, -1]
        ac = loss[:, o, 1, :, :, 1, -1]
        donor = loss[:, o, 1, :, :, 2, -1]
        delta = p[:, o, :, :, :, 1, 0, 2:4] - p[:, o, :, :, :, 0, 0, 2:4]
        displacement = np.sum(delta * delta, axis=-1)
        vectors[f"{objective}/U_AC"] = (free - ac).mean(axis=(0, 2))
        vectors[f"{objective}/S_AC"] = (donor - ac).mean(axis=(0, 2))
        vectors[f"{objective}/U_A_minus_U_AC"] = (ac - a).mean(axis=(0, 2))
        vectors[f"{objective}/insertion_displacement_suppression"] = (
            displacement[:, 0] - displacement[:, 1]
        ).mean(axis=(0, 2))
    return vectors, loss


def bootstrap_summary(vectors, indices, specification):
    indices = np.asarray(indices)
    n = len(next(iter(vectors.values())))
    if indices.dtype != np.int64 or indices.shape != (specification['bootstrap_draws'], n):
        raise ValueError("Unexpected bootstrap schema")
    if indices.min() < 0 or indices.max() >= n:
        raise ValueError("Bootstrap index outside population")
    result = {}
    for family in ('primary', 'secondary'):
        slots = specification[family]
        if len(slots) != 4:
            raise ValueError("S1 family requires exactly four frozen slots")
        tail = specification['family_alpha'] / (2 * len(slots))
        rows = []
        for slot in slots:
            v = vectors[slot['id']]
            # Independent verifier uses count-weight multiplication, not this gather path.
            resampled = np.empty(len(indices), np.float64)
            for begin in range(0, len(indices), 1000):
                sample = indices[begin:begin + 1000]
                resampled[begin:begin + len(sample)] = v[sample].mean(axis=1)
            ci = np.quantile(resampled, [tail, 1 - tail], method='linear')
            rows.append({**slot, 'estimate': float(v.mean()), 'ci': ci.tolist(),
                         'coverage': float(1 - 2 * tail), 'positive_support': bool(ci[0] > 0),
                         'recipient_count': n})
        result[family] = rows
    return result


def main():
    parser = argparse.ArgumentParser()
    for key in ('protocol', 'binding', 'acceptance'):
        parser.add_argument('--' + key, required=True)
        parser.add_argument('--' + key + '-sha256', required=True)
    parser.add_argument('--output', required=True)
    a = parser.parse_args()
    protocol = checked_json(a.protocol, a.protocol_sha256)
    binding = checked_json(a.binding, a.binding_sha256)
    acceptance = checked_json(a.acceptance, a.acceptance_sha256)
    if protocol['study_id'] != 'prospective_mechanism_v9_s1_20260923':
        raise ValueError('Wrong scientific protocol')
    if acceptance['status'] != 'FULL_S1_CONFIRMATION_ACCEPTED_BEFORE_D_SCORING':
        raise ValueError('Formal acceptance is required before outcome analysis')
    if acceptance['protocol_sha256'] != a.protocol_sha256 or binding['protocol_sha256'] != a.protocol_sha256:
        raise ValueError('Cross-protocol evidence')
    if binding['acceptance_sha256'] != a.acceptance_sha256:
        raise ValueError('Evidence not bound to this accepted population')
    datafile = Path(binding['arrays']['path'])
    if hashlib.sha256(datafile.read_bytes()).hexdigest() != binding['arrays']['sha256']:
        raise ValueError('Scored arrays hash mismatch')
    with np.load(datafile, allow_pickle=False) as archive:
        p, t, indices = [archive[k] for k in ('predictions_D', 'truth', 'bootstrap_indices')]
    if list(p.shape) != protocol['scores']['prediction_shape'] or list(t.shape) != protocol['scores']['truth_shape']:
        raise ValueError('Incomplete formal population')
    expected = np.random.default_rng(protocol['statistics']['bootstrap_seed']).integers(
        0, 256, size=(20000, 256), dtype=np.int64)
    if not np.array_equal(indices, expected):
        raise ValueError('Bootstrap draws differ from frozen seed and algorithm')
    vectors, loss = compute_goal_contrasts(p, t)
    summary = bootstrap_summary(vectors, indices, protocol['statistics'])
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(out / 'goal_contrasts.npz', **vectors)
    # Complete architecture/objective/constraint/branch/horizon descriptive means.
    summary.update(status='COMPLETE_PRODUCTION_SUMMARY_PENDING_INDEPENDENT_VERIFICATION',
                   protocol_sha256=a.protocol_sha256, binding_sha256=a.binding_sha256,
                   acceptance_sha256=a.acceptance_sha256,
                   descriptive_block_mse_axes=['group', 'objective', 'constraint', 'branch', 'horizon'],
                   descriptive_block_mse=loss.mean(axis=(3, 4)).tolist(),
                   caveat='Fixed model roster; recipients are bootstrap units. Unresolved is not equivalence.')
    (out / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
