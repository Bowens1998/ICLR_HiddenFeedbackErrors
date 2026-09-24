"""Open the complete S1 population only after numerical and provenance checks.

No reserved D checkpoint is deserialized and no readout prediction or effect is
computed here. D files are authenticated as opaque bytes. Output is a new receipt;
partial/rejected populations cannot produce the scientific-opening status.
"""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from s1_common import ROOT, atomic_json, checked_json, load_protocol, namespace_seed, sha
from s1_projection import MEMBERS, verify_token
from project_s1 import validate_qualification
from fresh_inputs import s1_context, s1_resolve
from cache_inputs import s1_stream, s1_case
from run_rollout import s1_read_qp

ACCEPTED = 'FULL_S1_CONFIRMATION_ACCEPTED_BEFORE_D_SCORING'
STUDY = 'prospective_mechanism_v9_s1_20260923'


def axis_labels(cfg):
    s = cfg['scores']
    return dict(groups=cfg['groups'], objectives=cfg['objectives'],
                constraints=s['constraints'], streams=s['streams'],
                branches=s['branches'], horizons=s['horizons_primitive'], pose=s['pose_order'])


def validate_roster(binding, cfg, protocol_sha):
    rows = binding.get('populations', [])
    expected = {(g, s) for g in cfg['groups'] for s in cfg['scores']['streams']}
    if (binding.get('study_id') != STUDY or binding.get('protocol_sha256') != protocol_sha or
            binding.get('kind') != 'confirmation' or binding.get('axis_labels') != axis_labels(cfg) or
            len(rows) != 8 or {(r['group'], r['stream']) for r in rows} != expected):
        raise ValueError('Missing/duplicate population, wrong axes or non-confirmation binding')
    return sorted(rows, key=lambda r: (r['group'], r['stream']))


class Inventory:
    """Authenticate every dependency once and retain its identity for reopening."""
    def __init__(self):
        self.files = {}

    def add(self, path, expected):
        path = str(Path(path).resolve())
        if not isinstance(expected, str) or len(expected) != 64:
            raise ValueError('Every input needs an externally propagated SHA256')
        if path in self.files and self.files[path] != expected:
            raise ValueError('Conflicting immutable file identity')
        if path not in self.files and sha(path) != expected:
            raise ValueError(f'Changed bound input: {path}')
        self.files[path] = expected
        return Path(path)

    def record(self, entry):
        return self.add(entry['path'], entry['sha256'])

    def read(self, entry):
        return json.loads(self.record(entry).read_text())


def bound_argument(a, key):
    return dict(path=str(Path(getattr(a, key)).resolve()), sha256=getattr(a, key + '_sha256'))


def check_source_closure(lock, inventory, protocol_sha, required, *, head=False):
    status = ('S1_HEAD_IMPLEMENTATION_FROZEN_BEFORE_INPUT_AUDIT_OR_FITS' if head
              else 'S1_INPUT_SOURCES_FROZEN')
    if lock.get('status') != status or lock.get('protocol_sha256') != protocol_sha or not lock.get('files'):
        raise ValueError('Missing frozen implementation/source closure')
    root = Path(lock['source_root']).resolve()
    if root != ROOT.resolve():
        raise ValueError('Source closure belongs to another execution tree')
    resolved = set()
    for name, checksum in lock['files'].items():
        p = Path(name); p = p if p.is_absolute() else root / p
        resolved.add(inventory.add(p, checksum))
    if any(Path(p).resolve() not in resolved for p in required):
        raise ValueError('Required admission/scoring dependency not source-frozen')
    if not head and 'confirmation' not in lock.get('authorized_kinds', []):
        raise ValueError('Source lock does not authorize confirmation')


def array_check(array, shape, dtype, name):
    if array.shape != tuple(shape) or array.dtype != np.dtype(dtype) or not np.isfinite(array).all():
        raise ValueError(f'Wrong shape/dtype or nonfinite complete {name}')


def validate_rollout_arrays(arr, cache, replacements, count):
    array_check(arr['tokens'], (2, 2, count, 4, 5, 192), np.float32, 'tokens')
    array_check(arr['truth'], (count, 5, 6), np.float64, 'truth')
    array_check(arr['goal_indices'], (count,), np.int64, 'goal IDs')
    array_check(arr['seeds'], (count,), np.int64, 'parent seeds')
    if not np.array_equal(arr['goal_indices'], np.arange(count, dtype=np.int64)):
        raise ValueError('Missing or reordered recipient')
    if len(np.unique(arr['seeds'])) != count:
        raise ValueError('Repeated recipient parent')
    for key in ('truth', 'seeds'):
        np.testing.assert_array_equal(arr[key], cache[key])
    for branch in (0, 3):
        np.testing.assert_array_equal(arr['tokens'][:, 0, :, branch], arr['tokens'][:, 1, :, branch])
    np.testing.assert_array_equal(arr['tokens'][:, 0, :, 0], cache['free'])
    for oi in range(2):
        np.testing.assert_array_equal(arr['tokens'][oi, 0, :, 3, 0], cache['observed'][:, 0])
    np.testing.assert_array_equal(arr['tokens'][:, :, :, 1:3, 0], replacements)


def validate_precision(report):
    p = report.get('precision', {})
    if (p.get('matmul_precision') != 'highest' or p.get('matmul_allow_tf32') is not False or
            p.get('cudnn_allow_tf32') is not False or p.get('cudnn_benchmark') is not False):
        raise ValueError('Native FP32 inference arithmetic not certified')


def validate_checks(report, count, objectives):
    rows = report.get('insertion_checks', [])
    if (len(rows) != 2 * count or {(r['objective'], r['case']) for r in rows} !=
            {(o, i) for o in objectives for i in range(count)}):
        raise ValueError('Missing native/identity/insertion check')
    flags = ('free_exact', 'identity_exact', 'inserted_fp32_exact', 'inputs_unchanged')
    if any(r.get(k) is not True for r in rows for k in flags):
        raise ValueError('Native arithmetic or insertion check failed')
    if (report.get('head_weights_opened') is not False or
            report.get('free_and_reset_bitwise_equal_across_constraints') is not True):
        raise ValueError('Rollout admission violated head exclusion or copy contract')
    validate_precision(report)


def source_digest(inventory, name):
    path = str(Path(__file__).with_name(name).resolve())
    if path not in inventory.files:
        raise ValueError(f'Unbound producer: {name}')
    return inventory.files[path]


def authenticate_models(models, ctx, inventory, group, objectives):
    entry = ctx['sources']['model_bindings']
    model_path = s1_resolve(ctx, entry['path'])
    lock = inventory.read(dict(path=str(model_path), sha256=entry['sha256']))
    groups = [r for r in lock['groups'] if r['group'] == group]
    if len(groups) != 1:
        raise ValueError('Missing or duplicate frozen model group')
    gr = groups[0]; expected = []
    for objective in objectives:
        rows = [m for m in gr['models'] if m['objective'] == objective]
        if len(rows) != 1:
            raise ValueError('Missing or duplicate frozen objective model')
        selected = rows[0]; original = gr['original']
        inventory.add(original['checkpoint'], original['checkpoint_sha256'])
        inventory.add(selected['checkpoint'], selected['checkpoint_sha256'])
        inventory.add(Path(selected['checkpoint']).parent / 'report.json', selected['report_sha256'])
        summary = inventory.read(dict(path=original['summary_path'], sha256=original['summary_sha256']))
        inventory.add(ctx['base'] / 'assets/pusht-v1/models/config.json', summary['config_sha256'])
        expected.append(dict(group=group, architecture=gr['architecture'], original=original,
                             selected=selected, model_bindings_sha256=entry['sha256']))
    if models != expected:
        raise ValueError('Changed world-model/normalizer/objective binding')


def authenticate_raw(cache, ctx, inventory, group, stream):
    """Replay file/physical identities only; no encoder, model or evaluator runs."""
    refs = s1_stream(ctx, group, stream)
    if cache['input_reports_sha256'] != refs['hashes']:
        raise ValueError('Cache/raw source ancestry changed')
    for p, checksum in refs['hashes'].items():
        inventory.add(p, checksum)
    role = json.loads((refs['bank'] / 'role.json').read_text())
    inventory.add(refs['bank'] / 'acceptance.json', role['acceptance_sha256'])
    lineage = inventory.read(dict(path=str(refs['bank'] / 'lineage.json'), sha256=role['lineage_sha256']))
    if lineage.get('status') != 'PASS_ALL_FIXED_SEED_RANGES_DISJOINT':
        raise ValueError('Missing accepted historical seed isolation')
    content = inventory.read(cache['content_lineage'])
    if (content.get('status') != 'PASS_S1_CONTENT_ISOLATION' or
            content.get('protocol_sha256') != ctx['protocol_sha256'] or
            content.get('sources_sha256') != ctx['sources_sha256'] or content.get('kind') != 'confirmation'):
        raise ValueError('Missing complete confirmation content isolation')
    expected_roles = {'development/recipient', 'development/donor', 'confirmation/recipient', 'confirmation/donor'}
    if (set(content.get('populations', {})) != expected_roles or
            set(content.get('head_overlap_hashes', {})) != expected_roles or
            any(content['head_overlap_hashes'].values())):
        raise ValueError('Incomplete or failed cross-role pixel isolation')
    pairs = content.get('fresh_population_pairs', [])
    expected_pairs = {frozenset((a, b)) for a in expected_roles for b in expected_roles if a != b}
    if (len(pairs) != 6 or {frozenset((p['left'], p['right'])) for p in pairs} != expected_pairs or
            any(p['overlap_count'] != 0 or p['pixel_sha256'] for p in pairs)):
        raise ValueError('Incomplete or failed fresh population pixel isolation')
    for p, checksum in content['input_files_sha256'].items():
        inventory.add(p, checksum)
    for entry in content['head_caches']:
        inventory.record(entry)
    for p, checksum in refs['hashes'].items():
        if '/actions/' not in p and content['input_files_sha256'].get(p) != checksum:
            raise ValueError('Raw input absent from full content-isolation audit')
    rows = []
    for i in range(ctx['spec']['count']):
        z, az, pz = s1_case(ctx, refs, i)
        for root, r, filekey, hashkey in [
                (refs['bank'], refs['bm']['cases'][i], None, 'sha256'),
                (refs['actions'], refs['ar']['cases'][i], 'file', 'file_sha256'),
                (refs['physics'], refs['pr']['cases'][i], 'file', 'file_sha256')]:
            inventory.add(root / (r[filekey] if filekey else f'case_{i:03d}.npz'), r[hashkey])
        for value in (z['history_states'], az['population_actions'], pz['states'], pz['actions']):
            if not np.isfinite(value).all():
                raise ValueError('Nonfinite raw input')
        state = pz['states'][[15, 20, 25, 30, 35]]
        rows.append(dict(seed=int(z['seed']), truth=np.c_[state[:, :4], np.sin(state[:, 4]), np.cos(state[:, 4])],
                         known_actions=az['selected_actions'][:5].reshape(10)))
    return rows


def read_cache(record, ctx, inventory, group, stream):
    r = inventory.read(record); n = ctx['spec']['count']; recipient = ctx['role'] == 'recipient'
    expected = dict(status='PASS_S1_OBSERVED_AND_FREE_CACHE', **ctx['binding'], group=group, stream=stream, count=n)
    objectives = ctx['cfg']['objectives'] if recipient else ctx['cfg']['objectives'][:1]
    if (any(r.get(k) != v for k, v in expected.items()) or r.get('cases') != list(range(n)) or
            r.get('objectives') != objectives or r.get('frozen_tensors_unchanged') is not True or
            r.get('donor_model_predictions_generated') is not False or
            r.get('native_endpoint_and_identity_replacement_exact') is not recipient or
            r.get('source_sha256') != source_digest(inventory, 'cache_inputs.py')):
        raise ValueError('Changed/unaccepted complete cache')
    validate_precision(r); authenticate_models(r['models'], ctx, inventory, group, objectives)
    raw = authenticate_raw(r, ctx, inventory, group, stream)
    with np.load(inventory.record(r['arrays']), allow_pickle=False) as z:
        arr = dict(z)
    expected_keys = {'initial', 'observed', 'truth', 'known_actions', 'seeds'} | ({'free'} if recipient else set())
    if set(arr) != expected_keys:
        raise ValueError('Unexpected cache arrays, including possible donor predictions')
    for key, shape, dtype in [('initial', (n, 3, 192), np.float32), ('observed', (n, 5, 192), np.float32),
            ('truth', (n, 5, 6), np.float64), ('known_actions', (n, 10), np.float32), ('seeds', (n,), np.int64)]:
        array_check(arr[key], shape, dtype, key)
    if recipient:
        array_check(arr['free'], (2, n, 5, 192), np.float32, 'free')
    np.testing.assert_array_equal(arr['truth'], np.stack([r['truth'] for r in raw]))
    np.testing.assert_array_equal(arr['known_actions'], np.asarray([r['known_actions'] for r in raw], np.float32))
    np.testing.assert_array_equal(arr['seeds'], [r['seed'] for r in raw])
    if len(np.unique(arr['seeds'])) != n:
        raise ValueError('Repeated raw parents')
    return r, arr


def authenticate_heads(a, inventory, protocol_sha):
    selected = inventory.read(bound_argument(a, 'selected_lock'))
    qualification = inventory.read(bound_argument(a, 'qualification'))
    references = inventory.read(bound_argument(a, 'a_bindings'))
    validate_qualification(selected, qualification, protocol_sha, a.selected_lock_sha256, a.a_bindings_sha256)
    if qualification.get('formal_qualification_gate_passed') is not True:
        raise ValueError('Formal qualification gate not passed')
    if qualification.get('source_sha256') != source_digest(inventory, 'qualify_heads.py'):
        raise ValueError('Unbound qualification producer')
    for row in qualification['rows']:
        refs = [r['head_A'] for r in references['groups'] if r['group'] == row['group']]
        if (len(refs) != 1 or row.get('reference_head_sha256') != refs[0]['sha256'] or
                set(row.get('files', {})) != {'expert', 'planner'}):
            raise ValueError('Qualification reference head or complete domain binding changed')
        for value in row['files'].values():
            # Opaque byte hash only: no qualification prediction arrays opened.
            inventory.add(Path(a.qualification).parent / value['file'], value['sha256'])
    construction = {}; head_records = []
    for g in [0, 1]:
        refs = [r['head_A'] for r in references['groups'] if r['group'] == g]
        if len(refs) != 1:
            raise ValueError('Missing/duplicate original construction head')
        with np.load(inventory.record(refs[0]), allow_pickle=False) as z:
            construction[g] = {'A': dict(z)}
        head_records.append(dict(group=g, head_role='A', **refs[0]))
    for row in selected['heads']:
        inventory.add(row['checkpoint'], row['checkpoint_sha256'])
        fit = inventory.read(dict(path=row['fit_report'], sha256=row['fit_report_sha256']))
        inventory.add(row['forward_fixture'], row['forward_fixture_sha256'])
        inventory.add(Path(row['views']) / 'report.json', row['views_report_sha256'])
        if (fit.get('status') != 'PASS_S1_FIXED_HEAD_FIT' or fit.get('protocol_sha256') != protocol_sha or
                fit.get('group') != row['group'] or fit.get('head_role') != row['head_role'] or
                fit['selected']['sha256'] != row['checkpoint_sha256'] or
                fit['views_report_sha256'] != row['views_report_sha256'] or fit['views'] != row['views']):
            raise ValueError('Changed selected fit/view/checkpoint chain')
        if row['head_role'] == 'C':
            with np.load(row['checkpoint'], allow_pickle=False) as z:
                construction[row['group']]['C'] = dict(z)
        head_records.append(dict(group=row['group'], head_role=row['head_role'],
                                 path=row['checkpoint'], sha256=row['checkpoint_sha256']))
    return construction, head_records


def validate_numerical_family(lock, inventory, ctx, group, stream, heads, recipient, donor, selected_sha,
                              qual_sha, a_sha, head_hashes):
    n = ctx['spec']['count']; cfg = ctx['cfg']
    if (lock.get('all_goals_covered') is not True or lock.get('selected_lock_sha256') != selected_sha or
            lock.get('qualification_sha256') != qual_sha or
            lock.get('source_sha256') != source_digest(inventory, 'freeze_qp_shards.py')):
        raise ValueError('Missing independent numerical-family acceptance')
    norm = np.empty(n, np.float64); covered = np.zeros(n, np.int64); max_deviation = 0.
    perm = np.random.default_rng(namespace_seed(cfg['root_seed'], cfg['donor_assignment']['seed_namespace'].format(
        kind='confirmation', stream=stream))).permutation(n)
    for shard in lock['shards']:
        r = inventory.read(dict(path=shard['report'], sha256=shard['report_sha256']))
        lo, hi = shard['start'], shard['stop']; k = hi - lo
        if not 0 <= lo < hi <= n:
            raise ValueError('Invalid numerical shard range')
        if (r.get('a_bindings_sha256') != a_sha or r.get('source_sha256') != source_digest(inventory, 'project_s1.py') or
                r.get('head_A_sha256') != head_hashes['A'] or r.get('head_C_sha256') != head_hashes['C'] or
                len(r.get('matching_reports', [])) != k):
            raise ValueError('Unbound/missing projection receipts')
        with np.load(inventory.add(shard['arrays'], shard['arrays_sha256']), allow_pickle=False) as z:
            values, full, norms, ids, donors = [z[x] for x in ('replacements', 'directions', 'common_norm', 'goal_indices', 'donor_indices')]
        array_check(full, (2, 2, k, 2, 192), np.float64, 'full directions')
        array_check(values, (2, 2, k, 2, 192), np.float32, 'full insertions')
        array_check(norms, (k,), np.float64, 'norms')
        array_check(ids, (k,), np.int64, 'QP recipient IDs')
        array_check(donors, (k,), np.int64, 'donor IDs')
        np.testing.assert_array_equal(ids, np.arange(lo, hi)); np.testing.assert_array_equal(donors, perm[lo:hi])
        if (norms < 0).any():
            raise ValueError('Negative family norm')
        for j, goal in enumerate(range(lo, hi)):
            receipt = r['matching_reports'][j]
            axes = [(cfg['objectives'].index(m['objective']), cfg['scores']['constraints'].index(m['condition']),
                     cfg['projection']['sources'].index(m['source'])) for m in MEMBERS]
            native = np.array([np.linalg.norm(full[oi, ci, j, si]) for oi, ci, si in axes])
            shrink = receipt['shrink_factor']; target = float(native.min())
            if (receipt.get('status') != 'ACCEPTED_S1_EIGHT_MEMBER_FAMILY' or receipt.get('goal') != goal or
                    receipt.get('members') != MEMBERS or receipt.get('family_size') != 8 or
                    receipt.get('recipient_seed') != int(recipient['seeds'][goal]) or
                    receipt.get('donor_seed') != int(donor['seeds'][perm[goal]]) or receipt.get('donor_index') != int(perm[goal]) or
                    shrink not in cfg['projection']['shrink_factors'] or receipt.get('unshrunk_common_norm') != target or
                    receipt.get('effective_norm') != norms[j] or norms[j] != target * shrink or
                    receipt.get('legitimate_zero_norm') != (target == 0) or len(receipt.get('solvers', [])) != 8 or
                    any(s['status'].lower() != 'solved' or s['head_count'] != (1 if q < 4 else 2)
                        for q, s in enumerate(receipt['solvers']))):
                raise ValueError('Changed eight-member/common-norm family')
            np.testing.assert_allclose(native, receipt['native_norms'], rtol=1e-12, atol=1e-12)
            factors = cfg['projection']['shrink_factors'][:cfg['projection']['shrink_factors'].index(shrink) + 1]
            if [x['shrink'] for x in receipt['attempts']] != factors:
                raise ValueError('Missing numerical backoff attempts')
            for factor in factors:
                checks = []
                for magnitude, (oi, ci, si) in zip(native, axes):
                    original = recipient['free'][oi, goal, 0]
                    alpha = 0. if magnitude == 0 else target * factor / magnitude
                    candidate = (original.astype(np.float64) + alpha * full[oi, ci, j, si] * heads['A']['scale']).astype(np.float32)
                    hs = [heads['A']] if ci == 0 else [heads['A'], heads['C']]
                    check = verify_token(original, candidate, hs, heads['A'], expected_norm=target * factor)
                    checks.append(check['accepted'])
                    if factor == shrink:
                        np.testing.assert_array_equal(candidate, values[oi, ci, j, si])
                        max_deviation = max(max_deviation, *(h['normalized_output_deviation'] for h in check['heads']))
                if all(checks) != (factor == shrink):
                    raise ValueError('Failed final insertion or skipped passing earlier dose')
        norm[lo:hi] = norms; covered[lo:hi] += 1
    if not np.array_equal(covered, np.ones(n, np.int64)):
        raise ValueError('Missing or repeated full numerical family')
    return norm, max_deviation


def accept(a):
    if Path(a.output).exists():
        raise ValueError('Never overwrite an admission receipt')
    cfg = load_protocol(a.protocol, a.protocol_sha256); inv = Inventory()
    inv.add(a.protocol, a.protocol_sha256)
    binding = inv.read(bound_argument(a, 'binding')); rows = validate_roster(binding, cfg, a.protocol_sha256)
    sources = inv.read(bound_argument(a, 'sources')); head_source = inv.read(bound_argument(a, 'head_implementation'))
    required = [Path(__file__).with_name(n) for n in ('accept_s1_population.py', 'score_reserved_readouts.py',
        'fresh_inputs.py', 'cache_inputs.py', 'run_rollout.py', 'project_s1.py', 'freeze_qp_shards.py',
        's1_common.py', 's1_readout.py', 's1_projection.py', 'qualify_heads.py')]
    required += [ROOT / 'strengthening/adapters/verifier.py',
                 ROOT / 'strengthening/presubmission_v8_20260922/independent_readout/scripts/readout.py']
    check_source_closure(sources, inv, a.protocol_sha256, required)
    check_source_closure(head_source, inv, a.protocol_sha256, [], head=True)
    heads, head_records = authenticate_heads(a, inv, a.protocol_sha256)
    contexts = {}
    for role in ('recipient', 'donor'):
        contexts[role] = s1_context(SimpleNamespace(protocol=a.protocol, protocol_sha256=a.protocol_sha256,
            sources=a.sources, sources_sha256=a.sources_sha256, kind='confirmation', role=role))
    completed = []; all_seeds = {}; n = cfg['scores']['goal_count']
    for row in rows:
        g, s = row['group'], row['stream']; ctx = contexts['recipient']
        rollout = inv.read(row['rollout'])
        expected = dict(status='PASS_S1_COMPLETE_ACCEPTED_ROLLOUT', **ctx['binding'], group=g, stream=s, count=n,
            objectives=cfg['objectives'], constraints=cfg['scores']['constraints'], branches=cfg['scores']['branches'],
            horizons=cfg['scores']['horizons_primitive'], selected_lock_sha256=a.selected_lock_sha256,
            qualification_sha256=a.qualification_sha256, source_sha256=source_digest(inv, 'run_rollout.py'))
        if any(rollout.get(k) != v for k, v in expected.items()):
            raise ValueError('Rollout identity, axes or immutable source mismatch')
        validate_checks(rollout, n, cfg['objectives'])
        rc, recipient = read_cache(rollout['recipient_cache'], ctx, inv, g, s)
        dc, donor = read_cache(row['donor_cache'], contexts['donor'], inv, g, s)
        if set(recipient['seeds']) & set(donor['seeds']):
            raise ValueError('Recipient/donor parent overlap')
        for role, data in [('recipient', recipient), ('donor', donor)]:
            if role in all_seeds:
                np.testing.assert_array_equal(data['seeds'], all_seeds[role])
            all_seeds[role] = data['seeds'].copy()
        if rollout['models'] != rc['models'] or rollout['input_reports_sha256'] != rc['input_reports_sha256']:
            raise ValueError('Changed raw/model ancestry during rollout')
        inv.record(rollout['qp_lock'])
        replacements, qp = s1_read_qp(rollout['qp_lock']['path'], rollout['qp_lock']['sha256'],
                                     ctx=ctx, group=g, stream=s, cache_sha=rollout['recipient_cache']['sha256'])
        if qp['donor_cache_sha256'] != row['donor_cache']['sha256']:
            raise ValueError('Different donor population in QP')
        norms, residual = validate_numerical_family(qp, inv, ctx, g, s, heads[g], recipient, donor,
            a.selected_lock_sha256, a.qualification_sha256, a.a_bindings_sha256,
            {r['head_role']: r['sha256'] for r in head_records if r['group'] == g})
        with np.load(inv.record(rollout['arrays']), allow_pickle=False) as z:
            arr = dict(z)
        if set(arr) != {'tokens', 'truth', 'seeds', 'goal_indices'}:
            raise ValueError('Unexpected rollout keys; evaluator predictions forbidden')
        validate_rollout_arrays(arr, recipient, replacements, n)
        completed.append(dict(group=g, stream=s, count=n, rollout=row['rollout'], arrays=rollout['arrays'],
            qp_lock=rollout['qp_lock'], recipient_cache=rollout['recipient_cache'], donor_cache=row['donor_cache'],
            zero_norm_families=int((norms == 0).sum()), maximum_rechecked_output_deviation=residual,
            complete_finite_shapes=True, native_identity_checks=True, insertion_and_full_function_checks=True))
    receipt = dict(status=ACCEPTED, study_id=STUDY, protocol_sha256=a.protocol_sha256,
        binding_sha256=a.binding_sha256, selected_lock_sha256=a.selected_lock_sha256,
        qualification_sha256=a.qualification_sha256, a_bindings_sha256=a.a_bindings_sha256,
        sources_sha256=a.sources_sha256, head_implementation_sha256=a.head_implementation_sha256,
        axis_labels=axis_labels(cfg), population_count=8, goal_count=n, populations=completed,
        heads=head_records, input_files_sha256=inv.files, source_sha256=sha(__file__),
        complete_population_accepted=True, D_weights_deserialized=False, D_predictions_opened=False,
        effects_computed=False, scope='Complete immutable raw-input ancestry and saved FP32 numerical/rollout '
        'identities, including full A/AC functions. Native encoder/rollout parity is authenticated from producer '
        'checks; this gate does not rerun world models. D checkpoints/qualification archives hashed as opaque bytes.')
    atomic_json(a.output, receipt)
    return receipt


def main():
    p = argparse.ArgumentParser()
    for key in ('protocol', 'binding', 'selected-lock', 'qualification', 'a-bindings', 'sources', 'head-implementation'):
        p.add_argument('--' + key, required=True); p.add_argument('--' + key + '-sha256', required=True)
    p.add_argument('--output', required=True)
    accept(p.parse_args())


if __name__ == '__main__':
    main()
