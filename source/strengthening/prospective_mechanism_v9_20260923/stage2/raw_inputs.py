"""DRAFT S2 raw-input adapter; no seeds, source lock or execution authorization.

The S1 planner/replay bodies are reused unchanged, with exactly three isolated
role helpers replaced. No S1 context validator or protocol identity is reused.
All future actual paths, seeds and receipt hashes must come from a frozen S2
protocol/source contract. Importing this module loads no Torch or simulator.
"""
import argparse
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import types

import numpy as np

PHASE = Path(__file__).resolve().parents[1]
ROOT = PHASE.parents[1]
KERNEL = PHASE / 'scripts/fresh_inputs.py'
ROLES = ('calibration_recipient', 'calibration_donor', 'test_recipient', 'test_donor')
COUNTS = dict(zip(ROLES, (256, 256, 512, 512)))
RECIPIENT_ROUTES = (0, 1, 8, 9, 16, 17)
DONOR_ROUTES = (0, 8, 16)
OLD_ROLES = ('fit', 'validation', 'qualification')
HEAD_GROUPS = (0, 1, 2, 4)  # S2 coordinates0/2/4 plus S1 C/D source coordinates0/1.
OBSERVED_LOCK = 'strengthening/prospective_mechanism_v9_20260923/manifests/v8_OBSERVED_CACHES.lock.json'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 << 20), b''): h.update(chunk)
    return h.hexdigest()


def checked_json(path, checksum):
    if (not isinstance(checksum, str) or len(checksum) != 64 or
            any(c not in '0123456789abcdef' for c in checksum) or sha(path) != checksum):
        raise ValueError('Missing or changed actual hash binding: ' + str(path))
    return json.loads(Path(path).read_text())


def write_exclusive(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False); f.write('\n')
        f.flush(); os.fsync(f.fileno())


def resolve(ctx, path):
    p = Path(path)
    return p if p.is_absolute() else ctx['root'] / p


def routes(role):
    if role not in ROLES: raise ValueError('Unknown S2 raw-input bank role')
    return DONOR_ROUTES if role.endswith('_donor') else RECIPIENT_ROUTES


def _integer(value, minimum=0):
    return isinstance(value, int) and not isinstance(value, bool) and value >= minimum


def validate_design(protocol):
    """Validate a future schema; no default seed or inferred admission budget."""
    if protocol.get('status') != 'S2_SCIENTIFIC_PROTOCOL_FROZEN':
        raise ValueError('S2 is not frozen; draft code is not authorization')
    raw = protocol['raw_inputs']
    if (raw.get('native_population') != 300 or raw.get('relative_horizons') != [5, 10, 15, 20, 25] or
            raw.get('reference_policy_indices') != dict(recipient=list(RECIPIENT_ROUTES), donor=list(DONOR_ROUTES))):
        raise ValueError('Changed native arithmetic, horizons or fixed policy streams')
    banks = raw['banks']
    if len(banks) != 4 or [r.get('role') for r in banks] != list(ROLES):
        raise ValueError('All four fixed S2 banks must occur exactly once in order')
    for row in banks:
        if (row.get('count') != COUNTS[row['role']] or not _integer(row.get('seed_start')) or
                not _integer(row.get('max_seeds'), row['count']) or row['seed_start'] + row['max_seeds'] > 2**32):
            raise ValueError('Missing/invalid frozen seed range or changed256/512 count')
    for i, a in enumerate(banks):
        for b in banks[i + 1:]:
            if max(a['seed_start'], b['seed_start']) < min(a['seed_start'] + a['max_seeds'], b['seed_start'] + b['max_seeds']):
                raise ValueError('Fixed S2 attempt ranges overlap')
    for key in ('source_base', 'output_root'):
        if not Path(raw.get(key, '')).is_absolute(): raise ValueError('Actual absolute path remains unresolved: ' + key)
    return raw


def _read_pair(ctx, row):
    return checked_json(resolve(ctx, row['path']), row['sha256'])


def _prior_sources(ctx):
    rows = ctx['sources']['prior_s1_sources']
    if len(rows) != 2 or len({r['sha256'] for r in rows}) != 2:
        raise ValueError('Both actual S1 development/confirmation source gates are required')
    docs = [_read_pair(ctx, r) for r in rows]
    if ({tuple(d.get('authorized_kinds', [])) for d in docs} != {('development',), ('confirmation',)} or
            any(d.get('status') != 'S1_INPUT_SOURCES_FROZEN' or d['protocol_sha256'] != ctx['sources']['prior_s1_protocol']['sha256'] for d in docs)):
        raise ValueError('Wrong prior S1 source identity')
    return docs


def context(protocol_path, protocol_sha256, source_path, source_sha256, role):
    """Only actual S2 contracts authorize execution; never call s1_context."""
    if role not in ROLES: raise ValueError('Unknown S2 role')
    protocol = checked_json(protocol_path, protocol_sha256); raw = validate_design(protocol)
    sources = checked_json(source_path, source_sha256)
    if (sources.get('status') != 'S2_RAW_INPUT_SOURCES_FROZEN' or sources.get('protocol_sha256') != protocol_sha256 or
            sources.get('authorized_roles') != list(ROLES) or Path(sources['source_root']).resolve() != ROOT.resolve()):
        raise ValueError('Wrong source role, protocol or deployed source root')
    ctx = dict(root=ROOT, cfg=protocol, raw=raw, sources=sources, role=role,
        kind=role.split('_')[0], spec=next(r for r in raw['banks'] if r['role'] == role),
        base=Path(raw['source_base']), artifacts=Path(raw['output_root']),
        protocol_sha256=protocol_sha256, sources_sha256=source_sha256)
    required = [Path(__file__), KERNEL, ROOT / 'scripts/visual/prepare_prospective_pusht.py',
        ROOT / 'scripts/visual/accept_adaptation_contexts.py', ROOT / 'scripts/visual/controlled_search.py',
        ROOT / 'scripts/visual/image_planner_cost.py', ROOT / 'scripts/visual/nonlinear_pose_cost.py',
        ROOT / 'scripts/visual/factorial_model.py', ROOT / 'scripts/visual/lewm_adapter.py',
        ROOT / 'scripts/visual/evaluation_precision.py', ROOT / 'scripts/visual/score_feedback_ranking.py',
        ROOT / 'strengthening/adapters/artifact_io.py']
    files = sources['files']; actual = {}
    for name, digest in files.items():
        path = resolve(ctx, name)
        if sha(path) != digest: raise ValueError('Changed frozen source/runtime closure: ' + str(path))
        actual[str(path.resolve())] = digest
    if any(str(p.resolve()) not in actual for p in required): raise ValueError('Required direct source absent')
    old = _prior_sources(ctx)
    prior_protocol = _read_pair(ctx, sources['prior_s1_protocol'])
    if prior_protocol.get('study_id') != 'prospective_mechanism_v9_s1_20260923': raise ValueError('Wrong S1 protocol ancestry')
    kernel_key = str(KERNEL.relative_to(ROOT))
    if any(d['files'].get(kernel_key) != sha(KERNEL) for d in old): raise ValueError('Legacy planner/replay body changed after S1 freeze')
    expected_registry = {(r['path'], r['sha256']) for d in old for r in d['legacy_bank_manifests']}
    registry = sources['legacy_bank_manifests']
    if (len({r['path'] for r in registry}) != len(registry) or
            not expected_registry <= {(r['path'], r['sha256']) for r in registry}):
        raise ValueError('Incomplete/duplicate owned historical manifest registry')
    if any(d['reference_policies'] != sources['reference_policies'] for d in old): raise ValueError('Original policy roster changed')
    obs = sources['observed_cache_lock']
    if any(d['files'].get(OBSERVED_LOCK) != obs['sha256'] for d in old): raise ValueError('Historical observed-head lock changed')
    # External model/simulator/config/runtime closure is copied without revision.
    for name, digest in old[0]['files'].items():
        if Path(name).is_absolute() and files.get(name) != digest:
            raise ValueError('Missing immutable external runtime/source dependency: ' + name)
    ctx['binding'] = dict(stage='S2', protocol_sha256=protocol_sha256, sources_sha256=source_sha256,
        kind=ctx['kind'], role=role, bank_role=role, wrapper_source_sha256=sha(__file__),
        reused_kernel_source_sha256=sha(KERNEL), bridge_helpers=bridge_identity())
    sys.path[:0] = [str(ROOT / 'strengthening/adapters'), str(ROOT / 'scripts/visual')]
    return ctx


def historical_attempt_seeds(manifest, manifests_by_sha=None, seen=()):
    """Reconstruct recorded actual attempts; never treat unused budget as exposure.

    Sequential generators retain accepted+rejected records. Derived ranking
    banks name their original context manifest; follow that actual hash within
    the bound registry. An unresolvable provenance branch blocks admission.
    """
    admitted = [r.get('seed') for r in manifest.get('cases', [])]
    if not admitted or any(not _integer(x) or x >= 2**32 for x in admitted) or len(set(admitted)) != len(admitted):
        raise ValueError('Malformed historical admitted seed roster')
    if 'rejected_seeds' not in manifest:
        parent = manifest.get('bindings', {}).get('contexts_manifest_sha256')
        if parent is None or parent in seen or parent not in (manifests_by_sha or {}):
            raise ValueError('UNRESOLVED historical attempts: no complete rejected log or bound parent manifest')
        source = manifests_by_sha[parent]
        if not set(admitted) <= {r['seed'] for r in source['cases']}:
            raise ValueError('Derived bank contains parents absent from its bound source')
        return historical_attempt_seeds(source, manifests_by_sha, (*seen, parent))
    rejected = [r.get('seed') for r in manifest['rejected_seeds']]
    values = admitted + rejected
    if any(not _integer(x) or x >= 2**32 for x in values) or len(set(values)) != len(values):
        raise ValueError('Malformed or repeated actual admission/rejection record')
    start = manifest.get('seed_start'); start = min(values) if start is None else start
    stop = max(values) + 1
    if not _integer(start) or sorted(values) != list(range(start, stop)):
        raise ValueError('UNRESOLVED missing historical actual-attempt records')
    budget = manifest.get('max_seeds')
    if budget is not None and (not _integer(budget, 1) or stop > start + budget):
        raise ValueError('Historical attempts exceed their declared maximum budget')
    return set(values)


def _s1_records(ctx):
    rows = ctx['sources']['s1_banks']
    expected = {(k, r) for k in ('development', 'confirmation') for r in ('recipient', 'donor')}
    if len(rows) != 4 or {(r['kind'], r['role']) for r in rows} != expected:
        raise ValueError('All four historical S1 banks are required exactly once')
    source_hashes = {r['sha256'] for r in ctx['sources']['prior_s1_sources']}
    records = []
    for row in rows:
        root = Path(row['path']); role = checked_json(root / 'role.json', row['role_sha256'])
        if (role.get('protocol_sha256') != ctx['sources']['prior_s1_protocol']['sha256'] or
                role.get('sources_sha256') not in source_hashes or role.get('kind') != row['kind'] or role.get('role') != row['role'] or
                role.get('count') != (64 if row['kind'] == 'development' else 256) or not (root / 'DONE').is_file()):
            raise ValueError('Unaccepted or changed prior S1 bank')
        manifest = checked_json(root / 'manifest.json', role['parent_manifest_sha256'])
        acceptance = checked_json(root / 'acceptance.json', role['acceptance_sha256'])
        checked_json(root / 'lineage.json', role['lineage_sha256'])
        if (manifest_hash := sha(root / 'manifest.json')) != row['manifest_sha256'] or len(manifest['cases']) != role['count']:
            raise ValueError('S1 manifest identity/count changed')
        if acceptance.get('status') != 'PASS_FULL_REFERENCE_AND_GOAL_REPLAY' or acceptance['manifest_sha256'] != manifest_hash:
            raise ValueError('Unaccepted prior full bank replay')
        records.append((row, root, role, manifest))
    return records


def lineage(ctx):
    old = []; old_seeds = set()
    manifests = {r['sha256']: _read_pair(ctx, r) for r in ctx['sources']['legacy_bank_manifests']}
    for record in ctx['sources']['legacy_bank_manifests']:
        values = historical_attempt_seeds(manifests[record['sha256']], manifests)
        old_seeds.update(values)
        old.append(dict(**record, actual_attempts=len(values), minimum_seed=min(values), maximum_seed=max(values),
                        scope='recorded accepted/rejected attempts, following explicit derived-bank parent bindings'))
    for record, root, role, manifest in _s1_records(ctx):
        values = historical_attempt_seeds(manifest); old_seeds.update(values)
        old.append(dict(path=str(root / 'manifest.json'), sha256=record['manifest_sha256'], actual_attempts=len(values),
                        minimum_seed=min(values), maximum_seed=max(values), scope='actual S1 accepted and rejected attempts'))
    observed = _read_pair(ctx, ctx['sources']['observed_cache_lock'])
    groups = [g for g in observed['groups'] if g['group'] in HEAD_GROUPS]
    if len(groups) != 4 or {g['group'] for g in groups} != set(HEAD_GROUPS): raise ValueError('Missing S1/S2 head source coordinates')
    head_parents = set(); head_reports = []
    for group in groups:
        path = Path(group['cache']) / 'report.json'; report = checked_json(path, group['cache_report_sha256'])
        for role in OLD_ROLES:
            values = report['parent_ids'][role + '/planner']
            if any(not _integer(x) for x in values): raise ValueError('Malformed historical head parent IDs')
            head_parents.update(values)
        head_reports.append(dict(group=group['group'], path=str(path), sha256=group['cache_report_sha256']))
    fresh = []
    for bank in ctx['raw']['banks']:
        start, stop = bank['seed_start'], bank['seed_start'] + bank['max_seeds']
        overlaps = sorted(s for s in old_seeds if start <= s < stop)
        parents = sorted(p for p in head_parents if start <= p < stop)
        if overlaps or parents: raise ValueError('Frozen S2 attempt range overlaps prior use; block, never skip or resample')
        fresh.append(dict(role=bank['role'], start=start, stop=stop))
    return dict(status='PASS_S2_ALL_RECORDED_ATTEMPTS_DISJOINT', **ctx['binding'],
        old_manifest_attempts=old, old_unique_actual_attempts=len(old_seeds), head_reports=head_reports,
        old_head_unique_planner_parents=len(head_parents), fresh_ranges=fresh,
        scope='Complete recorded actual attempts in the bound owned registry and four S1 banks, plus observed planner parents. '
              'Unused historical seed budgets are not classified as observed data. '
              'No seed-isolation claim for expert data, encoder pretraining, or unregistered projects.')


def _path(ctx, stage, role=None):
    return ctx['artifacts'] / stage / (role or ctx['role'])


def _bank(ctx):
    root = _path(ctx, 'banks'); role = json.loads((root / 'role.json').read_text())
    if (role.get('status') != 'PASS_S2_RAW_BANK' or any(role.get(k) != v for k, v in ctx['binding'].items()) or
            role.get('count') != ctx['spec']['count'] or not (root / 'DONE').is_file()):
        raise ValueError('Wrong S2 bank identity or incomplete role')
    manifest = checked_json(root / 'manifest.json', role['parent_manifest_sha256'])
    acceptance = checked_json(root / 'acceptance.json', role['acceptance_sha256'])
    checked_json(root / 'lineage.json', role['lineage_sha256'])
    _validate_generated_manifest(manifest, ctx['spec'])
    if (acceptance.get('status') != 'PASS_FULL_REFERENCE_AND_GOAL_REPLAY' or
            acceptance.get('manifest_sha256') != role['parent_manifest_sha256'] or len(acceptance['rows']) != ctx['spec']['count']):
        raise ValueError('Missing complete independent generated-bank replay')
    return root, role, manifest


def _validate_generated_manifest(manifest, spec):
    if (len(manifest['cases']) != spec['count'] or manifest.get('seed_start') != spec['seed_start'] or
            manifest.get('max_seeds') != spec['max_seeds'] or [r['index'] for r in manifest['cases']] != list(range(spec['count']))):
        raise ValueError('Incomplete fixed generated bank')
    admitted = [r['seed'] for r in manifest['cases']]; rejected = [r['seed'] for r in manifest['rejected_seeds']]
    if any(not isinstance(r.get('reason'), str) or not r['reason'].strip() for r in manifest['rejected_seeds']):
        raise ValueError('Missing complete rejection reasons')
    historical_attempt_seeds(manifest)
    if (admitted != sorted(admitted) or len(set(admitted + rejected)) != len(admitted + rejected) or
            sorted(admitted + rejected) != list(range(spec['seed_start'], max(admitted) + 1))):
        raise ValueError('Missing/repeated/reordered admission attempts; complete rejection history required')


def _reference(ctx, index):
    if isinstance(index, bool) or index not in routes(ctx['role']): raise ValueError('Disallowed reference stream for this S2 role')
    entry = ctx['sources']['reference_policies']; policies = _read_pair(ctx, entry)
    row = policies['rows'][index]
    if row['index'] != index or row['entry']['adaptation_condition'] != 'original' or row['entry']['arm'] != 'transformer_jepa':
        raise ValueError('Changed fixed original Transformer reference policy')
    return row


def bridge_identity():
    return {old: dict(replacement=new.__name__, wrapper_source_sha256=sha(__file__),
                     function_source_sha256=hashlib.sha256(inspect.getsource(new).encode()).hexdigest())
            for old, new in (('s1_bank', _bank), ('s1_path', _path), ('s1_reference', _reference))}


def bridge_function(kernel, name):
    """Clone the authentic function with only three role helpers substituted.

    No source transformation, module mutation, copied scientific function body,
    __file__ substitution, or call through s1_context is permitted.
    """
    if name not in ('s1_plan', 's1_replay'): raise ValueError('Only the two audited bodies may be bridged')
    original = getattr(kernel, name); environment = original.__globals__.copy()
    environment.update(s1_bank=_bank, s1_path=_path, s1_reference=_reference)
    result = types.FunctionType(original.__code__, environment, original.__name__, original.__defaults__, original.__closure__)
    result.__kwdefaults__ = original.__kwdefaults__
    return result


def _kernel(ctx):
    name = str(KERNEL.relative_to(ROOT))
    if sha(KERNEL) != ctx['sources']['files'][name]: raise ValueError('Changed legacy function body before use')
    spec = importlib.util.spec_from_file_location('_s2_immutable_legacy_input_kernel', KERNEL)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def _intent(ctx, stage, output, index=None):
    if output.exists(): raise ValueError('Existing output retained; partial recovery requires separate engineering adjudication')
    name = stage + ('_' + str(index) if index is not None else '') + '.json'
    target = ctx['artifacts'] / 'intents' / ctx['role'] / name
    write_exclusive(target, dict(**ctx['binding'], stage_operation=stage, policy_index=index,
                                output=str(output), began_unix_seconds=time.time()))
    return target


def prepare(ctx):
    evidence = lineage(ctx)  # All historical attempts audited before any generation.
    out = _path(ctx, 'banks'); _intent(ctx, 'prepare', out)
    simulator = ctx['base'] / 'releases/visual-v1/stable-worldmodel'; spec = ctx['spec']
    subprocess.run([sys.executable, str(ctx['root'] / 'scripts/visual/prepare_prospective_pusht.py'),
        '--source', str(simulator), '--output', str(out), '--cases', str(spec['count']),
        '--seed-start', str(spec['seed_start']), '--max-seeds', str(spec['max_seeds'])], check=True)
    subprocess.run([sys.executable, str(ctx['root'] / 'scripts/visual/accept_adaptation_contexts.py'),
        '--simulator', str(simulator), '--bank', str(out), '--count', str(spec['count']),
        '--seed-start', str(spec['seed_start'])], check=True)
    manifest = json.loads((out / 'manifest.json').read_text()); _validate_generated_manifest(manifest, spec)
    accepted = json.loads((out / 'acceptance.json').read_text())
    if accepted.get('status') != 'PASS_FULL_REFERENCE_AND_GOAL_REPLAY' or accepted['manifest_sha256'] != sha(out / 'manifest.json'):
        raise ValueError('Independent raw bank replay failed')
    write_exclusive(out / 'lineage.json', evidence)
    write_exclusive(out / 'role.json', dict(status='PASS_S2_RAW_BANK', **ctx['binding'], count=spec['count'],
        seed_start=spec['seed_start'], max_seeds=spec['max_seeds'], parent_manifest_sha256=sha(out / 'manifest.json'),
        acceptance_sha256=sha(out / 'acceptance.json'), lineage_sha256=sha(out / 'lineage.json'),
        head_fit_role=False, purpose='S2 fixed raw context only; no model effect or response opened'))
    with (out / 'DONE').open('x') as f: f.write('S2 fresh bank independently replayed\n')


def run_reference(ctx, index, stage):
    if stage not in ('plan', 'replay'): raise ValueError('Unknown immutable input kernel operation')
    _reference(ctx, index); _bank(ctx)
    out = _path(ctx, 'actions' if stage == 'plan' else 'physics') / f'route_{index}'
    _intent(ctx, stage, out, index)
    module = _kernel(ctx); bridge_function(module, 's1_' + stage)(ctx, index)
    report = json.loads((out / 'report.json').read_text())
    expected = 'PASS_COMPACT_FIXED_REFERENCE_SEARCH' if stage == 'plan' else 'PASS_COMPLETE_SELECTED_PHYSICS'
    if (report.get('status') != expected or len(report['cases']) != ctx['spec']['count'] or
            any(report['binding'].get(k) != v for k, v in ctx['binding'].items()) or
            report['binding']['source_sha256'] != sha(KERNEL) or not (out / 'DONE').is_file()):
        raise ValueError('Incomplete unchanged-kernel output or wrong S2 wrapper/kernel provenance')
    write_exclusive(out / 's2_admission.json', dict(status='PASS_S2_RAW_' + stage.upper(), **ctx['binding'],
        policy_index=index, count=ctx['spec']['count'], report_sha256=sha(out / 'report.json'),
        kernel_function='s1_' + stage, scientific_response_opened=False))


def _pixel_hashes(path, checksum, keys):
    if sha(path) != checksum: raise ValueError('Changed raw pixel input')
    result = set(); count = 0
    with np.load(path, allow_pickle=False) as data:
        for key, shape in keys:
            pixels = data[key]
            if pixels.dtype != np.uint8 or pixels.shape != shape: raise ValueError('Incomplete rendered uint8 pixel population')
            frames = pixels[None] if len(shape) == 3 else pixels
            for frame in frames: result.add(hashlib.sha256(np.ascontiguousarray(frame).tobytes()).hexdigest()); count += 1
    return result, count


def _population_pixels(bank, manifest, physics_records, count, record_binding, files):
    if len(manifest['cases']) != count or [r['index'] for r in manifest['cases']] != list(range(count)):
        raise ValueError('Incomplete pixel case population')
    pixels = set(); total = 0
    for row in manifest['cases']:
        path = bank / f"case_{row['index']:03d}.npz"
        hashes, n = _pixel_hashes(path, row['sha256'], [('history_pixels', (3, 224, 224, 3)),
            ('goal_pixels', (224, 224, 3)), ('terminal_pixels', (32, 224, 224, 3))])
        pixels.update(hashes); total += n; files[str(path)] = row['sha256']
    for path, checksum in physics_records:
        report = checked_json(path, checksum); files[str(path)] = checksum
        if (report.get('status') != 'PASS_COMPLETE_SELECTED_PHYSICS' or len(report['cases']) != count or
                not (path.parent / 'DONE').is_file() or any(report['binding'].get(k) != v for k, v in record_binding.items()) or
                report['binding']['parent_manifest_sha256'] != sha(bank / 'manifest.json') or
                [r['index'] for r in report['cases']] != list(range(count))):
            raise ValueError('Changed complete role/route physics population')
        for row in report['cases']:
            file = path.parent / row['file']; hashes, n = _pixel_hashes(file, row['file_sha256'], [('pixels', (8, 224, 224, 3))])
            if row['seed'] != manifest['cases'][row['index']]['seed'] or row.get('all_two_replays_exact') is not True:
                raise ValueError('Wrong parent or incomplete independent selected replay')
            pixels.update(hashes); total += n; files[str(file)] = row['file_sha256']
    return pixels, dict(total_frames=total, unique_pixels=len(pixels))


def compare_content(new, historical, head_pixels):
    if set(new) != set(ROLES): raise ValueError('Incomplete four-role content population')
    expected = {k + '/' + r for k in ('development', 'confirmation') for r in ('recipient', 'donor')}
    if set(historical) != expected: raise ValueError('Incomplete four-bank S1 content population')
    head_overlaps = {role: sorted(pixels & head_pixels) for role, pixels in new.items()}
    old_overlaps = {role: {key: sorted(pixels & old) for key, old in historical.items()} for role, pixels in new.items()}
    pairs = [dict(left=a, right=b, overlap=sorted(new[a] & new[b]))
             for i, a in enumerate(ROLES) for b in ROLES[i + 1:]]
    passed = not (any(head_overlaps.values()) or any(values for row in old_overlaps.values() for values in row.values()) or any(p['overlap'] for p in pairs))
    return dict(status='PASS_S2_RETAINED_PIXEL_ISOLATION' if passed else 'BLOCKED_S2_PIXEL_ALIAS',
        head_overlap_hashes=head_overlaps, s1_overlap_hashes=old_overlaps, new_role_pairs=pairs)


def _prior_physics_paths(bank, records):
    if len(records) != 8 or {r['index'] for r in records} != set(range(8)):
        raise ValueError('All eight S1 selected reference routes required')
    expected = {i: bank.parent.parent / 'physics' / bank.name / f'route_{i}' / 'report.json' for i in range(8)}
    if (len({r['path'] for r in records}) != 8 or
            any(Path(r['path']).resolve() != expected[r['index']].resolve() for r in records)):
        raise ValueError('S1 physics paths must uniquely match the actual bank/role/eight routes')
    return [(Path(r['path']), r['sha256']) for r in sorted(records, key=lambda r: r['index'])]


def content_audit(ctx):
    out = ctx['artifacts'] / 'content_lineage.json'; _intent(ctx, 'content-audit', out)
    seed_receipt = lineage(ctx); files = {}; observed = _read_pair(ctx, ctx['sources']['observed_cache_lock'])
    head_pixels = set(); caches = []
    for group in [g for g in observed['groups'] if g['group'] in HEAD_GROUPS]:
        expected_names = {role + '_' + domain + '.npz' for role in OLD_ROLES for domain in ('expert', 'planner')}
        if set(group['arrays']) != expected_names: raise ValueError('Incomplete selected observed-head source roles')
        for name, checksum in group['arrays'].items():
            path = Path(group['cache']) / name
            if sha(path) != checksum: raise ValueError('Changed historical head pixel cache')
            with np.load(path, allow_pickle=False) as data: hashes = data['pixel_sha256'].astype(str)
            if hashes.ndim != 1 or any(len(x) != 64 or any(c not in '0123456789abcdef' for c in x) for x in hashes):
                raise ValueError('Invalid historical rendered-pixel identity')
            head_pixels.update(hashes.tolist()); caches.append(dict(group=group['group'], path=str(path), sha256=checksum, rows=len(hashes)))
    if len(caches) != 24: raise ValueError('All24 S1/S2 source-head caches required')
    old_pixels = {}; new_pixels = {}; populations = {}
    for row, bank, role, manifest in _s1_records(ctx):
        entries = _prior_physics_paths(bank, row['physics_reports'])
        key = row['kind'] + '/' + row['role']
        pixels, summary = _population_pixels(bank, manifest, entries, role['count'],
            dict(protocol_sha256=ctx['sources']['prior_s1_protocol']['sha256'], sources_sha256=role['sources_sha256'], kind=row['kind'], role=row['role']), files)
        old_pixels[key] = pixels; populations['S1/' + key] = summary
    for role in ROLES:
        local = dict(ctx, role=role, kind=role.split('_')[0], spec=next(r for r in ctx['raw']['banks'] if r['role'] == role))
        local['binding'] = dict(ctx['binding'], role=role, bank_role=role, kind=local['kind'])
        bank, bank_role, manifest = _bank(local); entries = []
        for index in routes(role):
            path = _path(local, 'physics') / f'route_{index}'
            admitted = json.loads((path / 's2_admission.json').read_text())
            if (admitted.get('status') != 'PASS_S2_RAW_REPLAY' or admitted.get('policy_index') != index or
                    admitted.get('count') != COUNTS[role] or any(admitted.get(k) != v for k, v in local['binding'].items())):
                raise ValueError('Incomplete S2 selected-future admission')
            entries.append((path / 'report.json', admitted['report_sha256']))
            files[str(path / 's2_admission.json')] = sha(path / 's2_admission.json')
        pixels, summary = _population_pixels(bank, manifest, entries, COUNTS[role], local['binding'], files)
        new_pixels[role] = pixels; populations['S2/' + role] = summary
    result = compare_content(new_pixels, old_pixels, head_pixels)
    result.update(ctx['binding']); result.update(head_caches=caches, populations=populations, input_files_sha256=files,
        seed_lineage=seed_receipt, head_unique_pixels=len(head_pixels),
        scope='Exact retained uint8 history, goal, all32 generator reference terminal frames and all selected rendered future frames. '
              'New four roles mutually disjoint and compared against all four S1 banks/eight routes and all24 corresponding0/1/2/4 expert/planner head caches. '
              'Unretained rejected-branch pixels, older105-registry raw pixel contents, expert parents outside these caches and encoder pretraining pixels are not compared.',
        failure_policy='Block complete population. No row removal, resampling, appended seeds or outcome selection.')
    write_exclusive(out, result)
    if result['status'] != 'PASS_S2_RETAINED_PIXEL_ISOLATION': raise ValueError('Complete S2 input population blocked by content alias')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('protocol', 'protocol-sha256', 'sources', 'sources-sha256'): parser.add_argument('--' + name, required=True)
    parser.add_argument('--role', choices=ROLES, required=True)
    parser.add_argument('--stage', choices=('audit', 'prepare', 'plan', 'replay', 'content-audit'), required=True)
    parser.add_argument('--index', type=int)
    args = parser.parse_args(); ctx = context(args.protocol, args.protocol_sha256, args.sources, args.sources_sha256, args.role)
    if args.stage == 'audit': print(json.dumps(lineage(ctx), indent=2)); return
    if args.stage == 'prepare': prepare(ctx); return
    if args.stage == 'content-audit': content_audit(ctx); return
    if args.index is None: raise ValueError('Explicit actual policy index is required; no implicit array reindexing')
    run_reference(ctx, args.index, args.stage)


if __name__ == '__main__': main()
