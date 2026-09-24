"""Score every frozen S1 raw token after complete-population scientific opening.

The admission receipt is reauthenticated before deserializing D. No model,
projection, dose, population, horizon or readout is selected using these scores.
"""
import argparse
from pathlib import Path

import numpy as np

from s1_common import ROOT, atomic_json, atomic_npz, load_protocol, sha
from s1_readout import numpy_forward, relu_forward, validate_relu
from accept_s1_population import (ACCEPTED, STUDY, Inventory, axis_labels, array_check,
                                  bound_argument, check_source_closure)

SHARD_GOALS = 32


def validate_opening(receipt, cfg, protocol_sha, selected_sha, a_sha, sources_sha):
    if (receipt.get('status') != ACCEPTED or receipt.get('study_id') != STUDY or
            receipt.get('protocol_sha256') != protocol_sha or receipt.get('selected_lock_sha256') != selected_sha or
            receipt.get('a_bindings_sha256') != a_sha or receipt.get('sources_sha256') != sources_sha or
            receipt.get('axis_labels') != axis_labels(cfg) or receipt.get('population_count') != 8 or
            receipt.get('goal_count') != 256 or receipt.get('complete_population_accepted') is not True or
            receipt.get('D_weights_deserialized') is not False or receipt.get('D_predictions_opened') is not False or
            receipt.get('effects_computed') is not False or not receipt.get('input_files_sha256')):
        raise ValueError('Complete immutable pre-D scientific opening is required')
    rows = receipt.get('populations', [])
    if len(rows) != 8 or {(r['group'], r['stream']) for r in rows} != {(g, s) for g in [0, 1] for s in range(4)}:
        raise ValueError('Missing/duplicate accepted group/stream')
    for row in rows:
        if row.get('count') != 256 or any(row.get(k) is not True for k in (
                'complete_finite_shapes', 'native_identity_checks', 'insertion_and_full_function_checks')):
            raise ValueError('Incomplete per-population numerical acceptance')
        for key in ('rollout', 'arrays', 'qp_lock', 'recipient_cache', 'donor_cache'):
            entry = row[key]
            if receipt['input_files_sha256'].get(str(Path(entry['path']).resolve())) != entry['sha256']:
                raise ValueError('Accepted population omitted from bound input inventory')
    if len(receipt.get('heads', [])) != 6 or {(r['group'], r['head_role']) for r in receipt['heads']} != {
            (g, h) for g in [0, 1] for h in ['A', 'C', 'D']}:
        raise ValueError('Accepted six-head identity roster incomplete')
    return rows


def validate_head(head, role):
    if role in ('A', 'C'):
        validate_relu(head)
        if head['0.weight'].shape[1] != 192 or head['4.weight'].shape[0] != 6:
            raise ValueError('Wrong construction readout dimensions')
        if role == 'C' and (head['0.weight'].shape != (512, 192) or head['2.weight'].shape != (512, 512)):
            raise ValueError('New C is not the frozen 512-wide ReLU head')
        return
    shapes = dict(mean=(192,), scale=(192,), target_mean=(6,), target_scale=(6,))
    shapes.update({'input.weight': (256, 192), 'input.bias': (256,),
                   'output.weight': (6, 256), 'output.bias': (6,), 'skip.weight': (6, 192)})
    for i in range(2):
        for name in ('fc1', 'fc2'):
            shapes[f'blocks.{i}.{name}.weight'] = (256, 256)
            shapes[f'blocks.{i}.{name}.bias'] = (256,)
    if set(head) != set(shapes) or any(np.shape(head[k]) != shape or not np.isfinite(head[k]).all()
                                     for k, shape in shapes.items()):
        raise ValueError('Wrong complete reserved GELU architecture/parameters')
    if (head['scale'] <= 0).any() or (head['target_scale'] <= 0).any():
        raise ValueError('Invalid reserved readout normalization')


def assemble_group_shard(stream_arrays, start, stop):
    """Explicit axis placement: no implicit reshape may mix goal and stream."""
    if set(stream_arrays) != set(range(4)) or not 0 <= start < stop:
        raise ValueError('All four streams and a nonempty shard are required')
    n = stream_arrays[0]['tokens'].shape[2]
    if stop > n:
        raise ValueError('Goal shard outside accepted population')
    tokens = np.empty((2, 2, stop-start, 4, 4, 5, 192), np.float32)
    truth = np.empty((stop-start, 4, 5, 6), np.float64)
    for stream in range(4):
        arr = stream_arrays[stream]
        array_check(arr['tokens'], (2, 2, n, 4, 5, 192), np.float32, 'saved tokens')
        array_check(arr['truth'], (n, 5, 6), np.float64, 'saved truth')
        tokens[:, :, :, stream] = arr['tokens'][:, :, start:stop]
        truth[:, stream] = arr['truth'][start:stop]
    return tokens, truth


def decode_all(tokens, head, role):
    validate_head(head, role)
    flat = tokens.reshape(-1, 192); out = np.empty((len(flat), 6), np.float64)
    for start in range(0, len(flat), 1024):
        stop = min(start + 1024, len(flat))
        out[start:stop] = (numpy_forward(flat[start:stop], head, 'D') if role == 'D'
                           else relu_forward(head, flat[start:stop], physical=True))
    if not np.isfinite(out).all():
        raise ValueError('Nonfinite complete readout prediction; no rows dropped')
    return out.reshape(*tokens.shape[:-1], 6)


def frozen_bootstrap(cfg):
    stats = cfg['statistics']; n = cfg['scores']['goal_count']
    if n != 256 or stats['bootstrap_draws'] != 20000 or stats['bootstrap_seed'] != 3350567987:
        raise ValueError('Changed frozen bootstrap design')
    return np.random.default_rng(stats['bootstrap_seed']).integers(0, n, size=(20000, n), dtype=np.int64)


def score(a):
    out = Path(a.output)
    if out.exists():
        raise ValueError('Never overwrite score or partial-score outputs')
    cfg = load_protocol(a.protocol, a.protocol_sha256); inv = Inventory()
    inv.add(a.protocol, a.protocol_sha256)
    receipt = inv.read(bound_argument(a, 'acceptance'))
    rows = validate_opening(receipt, cfg, a.protocol_sha256, a.selected_lock_sha256, a.a_bindings_sha256, a.sources_sha256)
    # This complete byte-level recheck precedes every D deserialization.
    for path, checksum in receipt['input_files_sha256'].items():
        inv.add(path, checksum)
    selected = inv.read(bound_argument(a, 'selected_lock')); references = inv.read(bound_argument(a, 'a_bindings'))
    sources = inv.read(bound_argument(a, 'sources'))
    required = [Path(__file__), Path(__file__).with_name('accept_s1_population.py'),
        Path(__file__).with_name('s1_readout.py'), Path(__file__).with_name('s1_common.py'),
        ROOT / 'strengthening/presubmission_v8_20260922/independent_readout/scripts/readout.py']
    check_source_closure(sources, inv, a.protocol_sha256, required)
    if receipt.get('source_sha256') != sha(Path(__file__).with_name('accept_s1_population.py')):
        raise ValueError('Admission was produced by another implementation')
    expected = []
    for group in [0, 1]:
        ref = [r['head_A'] for r in references['groups'] if r['group'] == group]
        if len(ref) != 1:
            raise ValueError('Ambiguous construction head')
        expected.append(dict(group=group, head_role='A', **ref[0]))
    expected += [dict(group=r['group'], head_role=r['head_role'], path=r['checkpoint'], sha256=r['checkpoint_sha256'])
                 for r in selected['heads']]
    key = lambda r: (r['group'], r['head_role'])
    if sorted(expected, key=key) != sorted(receipt['heads'], key=key):
        raise ValueError('Scoring readouts differ from independently accepted frozen heads')
    heads = {}
    for entry in expected:
        path = inv.record(entry)
        # This is the first point where D weights can be decoded.
        with np.load(path, allow_pickle=False) as z:
            head = dict(z)
        validate_head(head, entry['head_role'])
        heads[(entry['group'], entry['head_role'])] = head
    out.mkdir(parents=True, exist_ok=False)
    shape = tuple(cfg['scores']['prediction_shape'])
    predictions = {h: np.empty(shape, np.float64) for h in ['A', 'C', 'D']}
    truth = np.empty(cfg['scores']['truth_shape'], np.float64)
    common_norm = np.empty((2, 256, 4), np.float64); shards = []
    for group in [0, 1]:
        streams = {}
        for row in rows:
            if row['group'] != group:
                continue
            stream = row['stream']
            with np.load(inv.record(row['arrays']), allow_pickle=False) as z:
                streams[stream] = dict(z)
            qp = inv.read(row['qp_lock']); coverage = np.zeros(256, np.int64)
            for shard in qp['shards']:
                with np.load(inv.add(shard['arrays'], shard['arrays_sha256']), allow_pickle=False) as z:
                    ids, norm = z['goal_indices'], z['common_norm']
                lo, hi = shard['start'], shard['stop']
                array_check(ids, (hi-lo,), np.int64, 'norm goal IDs')
                np.testing.assert_array_equal(ids, np.arange(lo, hi))
                array_check(norm, (hi-lo,), np.float64, 'accepted common norm')
                if (norm < 0).any() or not 0 <= lo < hi <= 256:
                    raise ValueError('Invalid accepted common norm/range')
                common_norm[group, lo:hi, stream] = norm; coverage[lo:hi] += 1
            if not np.array_equal(coverage, np.ones(256, np.int64)):
                raise ValueError('Incomplete common-norm population')
        for lo in range(0, 256, SHARD_GOALS):
            hi = min(256, lo + SHARD_GOALS)
            tokens, physical_truth = assemble_group_shard(streams, lo, hi)
            truth[group, lo:hi] = physical_truth
            for role in ['A', 'C', 'D']:
                predictions[role][group, :, :, lo:hi] = decode_all(tokens, heads[(group, role)], role)
            path = out / 'raw_token_shards' / f'group_{group}_goals_{lo:03d}_{hi:03d}.npz'
            atomic_npz(path, tokens=tokens, goal_indices=np.arange(lo, hi, dtype=np.int64))
            shards.append(dict(group=group, path=str(path.resolve()), sha256=sha(path),
                               tokens_key='tokens', goal_indices_key='goal_indices'))
    for role, p in predictions.items():
        array_check(p, shape, np.float64, role + ' predictions')
        for branch in [0, 3]:
            np.testing.assert_array_equal(p[:, :, 0, :, :, branch], p[:, :, 1, :, :, branch])
    array_check(truth, cfg['scores']['truth_shape'], np.float64, 'all truth')
    array_check(common_norm, (2, 256, 4), np.float64, 'all common norms')
    archive = out / 'scores.npz'
    atomic_npz(archive, **{'predictions_' + h: p for h, p in predictions.items()}, truth=truth,
               common_norm=common_norm, bootstrap_indices=frozen_bootstrap(cfg))
    record = dict(path=str(archive.resolve()), sha256=sha(archive))
    common = dict(study_id=STUDY, protocol_sha256=a.protocol_sha256, acceptance_sha256=a.acceptance_sha256,
        selected_lock_sha256=a.selected_lock_sha256, a_bindings_sha256=a.a_bindings_sha256,
        sources_sha256=a.sources_sha256, axis_labels=axis_labels(cfg), arrays=record,
        source_sha256=sha(__file__), head_bindings=expected)
    atomic_json(out / 'production_binding.json', common)
    independent = dict(study_id=STUDY, scientific_opening_gate='ACCEPTED_ALL_FROZEN_INPUTS_AND_NUMERICAL_FAMILIES',
        design=bound_argument(a, 'protocol'), acceptance=bound_argument(a, 'acceptance'), axis_labels=axis_labels(cfg),
        scores_archive=dict(**record, predictions_key='predictions_D', truth_key='truth', norm_key='common_norm'),
        bootstrap_archive=dict(**record, indices_key='bootstrap_indices'),
        reserved_readouts=[dict(group=g, path=heads_record['path'], sha256=heads_record['sha256'])
            for g in [0, 1] for heads_record in expected if heads_record['group'] == g and heads_record['head_role'] == 'D'],
        raw_token_shards=shards)
    atomic_json(out / 'independent_binding.json', independent)
    atomic_json(out / 'report.json', dict(status='COMPLETE_S1_RESERVED_SCORING_PENDING_INDEPENDENT_VERIFICATION',
        **common, prediction_axes=cfg['scores']['prediction_axes'], prediction_shape=list(shape),
        truth_shape=list(truth.shape), common_norm_shape=list(common_norm.shape), raw_token_shards=shards,
        production_binding_sha256=sha(out / 'production_binding.json'),
        independent_binding_sha256=sha(out / 'independent_binding.json'),
        inferential_statistics_computed=False, readout_forward_computed=True,
        scope='All raw tokens decoded by fixed A/C/D and each head own physical '
        'normalizers. D is primary; A/C descriptive. No effects-based exclusions; all eight inferential '
        'contrasts are computed downstream from the complete saved arrays.'))
    (out / 'DONE').write_text('Complete fixed-population reserved readout scoring\n')
    return common


def main():
    p = argparse.ArgumentParser()
    for key in ('protocol', 'acceptance', 'selected-lock', 'a-bindings', 'sources'):
        p.add_argument('--' + key, required=True); p.add_argument('--' + key + '-sha256', required=True)
    p.add_argument('--output', required=True)
    score(p.parse_args())


if __name__ == '__main__':
    main()
