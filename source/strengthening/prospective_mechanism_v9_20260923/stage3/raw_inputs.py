"""Draft genuine S3 fresh-bank / compact common-prefix adapter.

No execution is authorized by this source. A future frozen S3 protocol and raw
source lock, and actual complete S2 acceptance, are mandatory. The two original
planner/physics function bodies are unchanged; exactly three copied-globals
helpers replace their S1 role plumbing. No S1/S2 protocol is impersonated.
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

import common_prefix

PHASE = Path(__file__).resolve().parents[1]
ROOT = PHASE.parents[1]
KERNEL = PHASE / 'scripts/fresh_inputs.py'
ROLES = ('recipient', 'donor')
POLICIES = (0, 8, 16)


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


# Pure metadata validators; no S2 context, admission, model or public loader.
prior = _load(PHASE / 'stage2/raw_inputs.py', '_s3_prior_attempt_validators')
sha, checked_json, write_exclusive = prior.sha, prior.checked_json, prior.write_exclusive
require = common_prefix.require


def pair(path):
    return dict(path=str(Path(path).resolve()), sha256=sha(path))


def resolve(ctx, name):
    path = Path(name)
    return path if path.is_absolute() else ctx['root'] / path


def document(ctx, entry):
    return checked_json(resolve(ctx, entry['path']), entry['sha256'])


def validate_design(protocol):
    require(protocol.get('status') == 'S3_SCIENTIFIC_PROTOCOL_FROZEN' and
            protocol.get('scientific_protocol_frozen') is True, 'S3 draft is not execution authority')
    p, d = protocol['population'], protocol['shared_prefix_decision']
    for key, expected in dict(recipient_count=512, donor_count=512, max_attempts_per_role=100000,
            pools=[0, 1, 2], groups=[0, 2, 4], objectives=['decoded_teacher', 'physical_labels'],
            conditions=['T0', 'T1']).items():
        require(p.get(key) == expected, 'Changed fixed S3 population: ' + key)
    for key, expected in dict(candidate_count=32, candidate_subsample_namespace=1378001,
            common_executed_actions=5, historical_actions=10, history_tokens=3,
            native_root_candidate_count=300, reference_policy_rows=[0, 8, 16],
            legacy_reference_route_ids=[0, 32, 64], suffix_actions=20,
            same_observed_image_for_all_suffixes=True).items():
        require(d.get(key) == expected, 'Changed common-prefix arithmetic: ' + key)
    specs = []
    for role in ROLES:
        seed = p[role + '_seed_start']; budget = p['max_attempts_per_role']
        require(type(seed) is int and 0 <= seed < seed + budget <= 2**32, 'Unresolved/invalid fixed seed range')
        specs.append(dict(role=role, count=512, seed_start=seed, max_seeds=budget))
    require(max(s['seed_start'] for s in specs) >= min(s['seed_start'] + s['max_seeds'] for s in specs),
            'S3 role attempt ranges overlap')
    raw = protocol['raw_inputs']  # Operational paths to be bound by root, not guessed here.
    for key in ('source_base', 'output_root'):
        require(Path(raw.get(key, '')).is_absolute(), 'Unresolved actual path: ' + key)
    return raw, specs


def context(protocol_path, protocol_sha256, source_path, source_sha256, role):
    require(role in ROLES, 'Unknown genuine S3 role')
    protocol = checked_json(protocol_path, protocol_sha256); raw, specs = validate_design(protocol)
    sources = checked_json(source_path, source_sha256)
    require(sources.get('status') == 'S3_RAW_INPUT_SOURCES_FROZEN' and
            sources.get('protocol_sha256') == protocol_sha256 and sources.get('authorized_roles') == list(ROLES) and
            Path(sources['source_root']).resolve() == ROOT.resolve(), 'Wrong S3 source/protocol/role/root')
    ctx = dict(root=ROOT, cfg=protocol, raw=raw, sources=sources, role=role,
        spec=specs[ROLES.index(role)], specs=specs, base=Path(raw['source_base']), artifacts=Path(raw['output_root']),
        protocol_sha256=protocol_sha256, sources_sha256=source_sha256)
    # Completion is the actual finalizer receipt, not an invented extra gate.
    completion_pair = protocol['prior_stage_gate']['actual_receipt']
    require(completion_pair == sources['s2_completion'], 'Changed sequential S2 completion binding')
    completion = document(ctx, completion_pair)
    require(completion.get('status') == 'PASS_COMPLETE_S2_REGRESSION_AND_INDEPENDENT_STATISTICS' and
            completion.get('count') == 512, 'S2 has not completely finished')
    completed_inputs = document(ctx, completion['input_binding'])
    require(completed_inputs['protocol'] == sources['prior_s2_protocol'], 'Wrong completed S2 protocol')
    old = document(ctx, sources['prior_s2_sources'])
    require(old.get('status') == 'S2_RAW_INPUT_SOURCES_FROZEN' and
            old['protocol_sha256'] == sources['prior_s2_protocol']['sha256'], 'Wrong immutable raw ancestry')
    document(ctx, sources['prior_s2_protocol'])
    require(old['reference_policies'] == sources['reference_policies'], 'Changed original reference policy roster')
    files = sources['files']; actual = {}
    for name, digest in files.items():
        path = resolve(ctx, name); require(sha(path) == digest, 'Changed source/runtime dependency: ' + str(path))
        actual[str(path.resolve())] = digest
    required = [Path(__file__), Path(common_prefix.__file__), common_prefix.LEGACY, KERNEL,
        PHASE / 'stage2/raw_inputs.py', ROOT / 'scripts/visual/prepare_prospective_pusht.py',
        ROOT / 'scripts/visual/accept_adaptation_contexts.py']
    require(all(str(p.resolve()) in actual for p in required), 'Missing direct S3/source validator closure')
    require(all(files.get(k) == v for k, v in old['files'].items()), 'Changed or omitted inherited original closure')
    require(protocol['shared_prefix_decision']['candidate_source']['sha256'] == sha(common_prefix.LEGACY),
            'Changed legacy candidate chooser/replay body')
    require(sources['seed_registry'] == protocol['population']['isolation_registry'], 'Wrong predeclared seed registry')
    ctx['old_sources'] = old
    ctx['binding'] = dict(stage='S3', role=role, bank_role=role, protocol_sha256=protocol_sha256,
        sources_sha256=source_sha256, s2_completion_sha256=completion_pair['sha256'],
        wrapper_source_sha256=sha(__file__), compact_adapter_sha256=sha(common_prefix.__file__),
        reused_kernel_source_sha256=sha(KERNEL), bridge_helpers=bridge_identity())
    sys.path[:0] = [str(ROOT / 'strengthening/adapters'), str(ROOT / 'scripts/visual')]
    return ctx


def lineage(ctx):
    registry = document(ctx, ctx['sources']['seed_registry'])
    require(registry.get('status') == 'PASS_S3_DRAFT_NAMESPACE_RANGES_DISJOINT_FROM_BOUND_RECORDED_ATTEMPTS' and
            registry.get('scientific_effects_accessed') is False and
            registry['raw_sources'] == ctx['sources']['prior_s2_sources'], 'Wrong pre-effect namespace evidence')
    old = ctx['old_sources']; prior_acceptance = document(ctx, registry['raw_acceptance'])
    require(prior_acceptance.get('status') == 'PASS_S2_COMPLETE_RAW_INPUT_POPULATION' and
            prior_acceptance['sources_sha256'] == ctx['sources']['prior_s2_sources']['sha256'], 'Wrong complete S2 raw ancestry')
    required = {(r['path'], r['sha256']) for r in old['legacy_bank_manifests']}
    required.update((str(Path(r['path']) / 'manifest.json'), r['manifest_sha256']) for r in old['s1_banks'])
    required.update((r['manifest']['path'], r['manifest']['sha256']) for r in prior_acceptance['banks'])
    records = registry['historical_registry']
    require(len(records) == len(required) and {(r['path'], r['sha256']) for r in records} == required,
            'Missing, duplicate or replaced historical/S1/S2 manifest')
    manifests = {r['sha256']: document(ctx, r) for r in records}; seeds = set()
    for record in records:
        values = prior.historical_attempt_seeds(manifests[record['sha256']], manifests)
        require((len(values), min(values), max(values)) ==
                (record['actual_attempts'], record['minimum_seed'], record['maximum_seed']), 'Changed actual attempt history')
        seeds.update(values)
    head_parents = set(); rows = registry['head_reports']
    require(len(rows) == 6 and {r['group'] for r in rows} == set(range(6)), 'Incomplete historical head parent reports')
    observed = document(ctx, old['observed_cache_lock'])
    expected = {(g['group'], str(Path(g['cache']) / 'report.json'), g['cache_report_sha256']) for g in observed['groups']}
    require({(r['group'], r['path'], r['sha256']) for r in rows} == expected, 'Changed observed-cache ancestry')
    for record in rows:
        report = document(ctx, record)
        for old_role in ('fit', 'validation', 'qualification'):
            values = report['parent_ids'][old_role + '/planner']
            require(all(type(v) is int and v >= 0 for v in values), 'Invalid old planner parents')
            head_parents.update(values)
    require(len(seeds) == registry['unique_actual_attempts'] and len(head_parents) == registry['head_unique_planner_parents'],
            'Changed complete namespace exposure totals')
    ranges = [dict(role=s['role'], count=512, max_attempts=s['max_seeds'], start=s['seed_start'],
                   stop=s['seed_start'] + s['max_seeds']) for s in ctx['specs']]
    require(ranges == registry['ranges'], 'Frozen ranges differ from pre-effect metadata choice')
    for r in ranges:
        require(not any(r['start'] <= v < r['stop'] for v in seeds | head_parents), 'Prior exposure; never skip or reseed')
    return dict(status='PASS_S3_RECORDED_ATTEMPT_ISOLATION', **ctx['binding'], seed_registry=ctx['sources']['seed_registry'],
        historical_manifest_count=len(records), unique_actual_attempts=len(seeds), head_unique_planner_parents=len(head_parents),
        ranges=ranges, retained_pixel_isolation='PENDING_AFTER_GENERATION',
        scope='Recorded actual attempts and observed planner parents only; unused old budgets are not exposure. '
              'No expert/pretraining/unregistered-project seed isolation or old raw-pixel isolation claim.')


def _path(ctx, stage, role=None):
    return ctx['artifacts'] / stage / (role or ctx['role'])


def validate_bank(manifest, acceptance, spec, manifest_sha):
    prior._validate_generated_manifest(manifest, spec)
    require(acceptance.get('status') == 'PASS_FULL_REFERENCE_AND_GOAL_REPLAY' and
            acceptance.get('manifest_sha256') == manifest_sha and len(acceptance['rows']) == spec['count'],
            'Incomplete independent generated-bank replay')
    for row, saved in zip(manifest['cases'], acceptance['rows'], strict=True):
        require(all(saved.get(k) == row[k] for k in ('index', 'seed', 'sha256')) and
                saved.get('independently_replayed_branches') == 33, 'Changed/partial independent33-branch acceptance')


def _bank(ctx):
    out = _path(ctx, 'banks'); role = json.loads((out / 'role.json').read_text())
    require(role.get('status') == 'PASS_S3_RAW_BANK' and role.get('count') == 512 and (out / 'DONE').is_file() and
            all(role.get(k) == v for k, v in ctx['binding'].items()), 'Wrong/incomplete genuine S3 bank')
    manifest = checked_json(out / 'manifest.json', role['parent_manifest_sha256'])
    acceptance = checked_json(out / 'acceptance.json', role['acceptance_sha256'])
    line = checked_json(out / 'lineage.json', role['lineage_sha256'])
    require(line.get('status') == 'PASS_S3_RECORDED_ATTEMPT_ISOLATION' and
            all(line.get(k) == v for k, v in ctx['binding'].items()), 'Wrong bank lineage')
    validate_bank(manifest, acceptance, ctx['spec'], role['parent_manifest_sha256'])
    return out, role, manifest


def _reference(ctx, index):
    require(type(index) is int and index in POLICIES, 'Wrong S3 reference policy')
    row = document(ctx, ctx['sources']['reference_policies'])['rows'][index]; pool = index // 8
    require((row['index'], row['group'], row['legacy_route_index']) == (index, 2 * pool, 32 * pool) and
            row['entry']['adaptation_condition'] == 'original' and row['entry']['arm'] == 'transformer_jepa' and
            row['entry']['score'] == 'pose_encoded' and row['route']['algorithm'] == 'random' and
            row['route']['parameterization'] == 'full', 'Changed original random/full pose reference')
    return row


def bridge_identity():
    return {old: dict(replacement=new.__name__, wrapper_source_sha256=sha(__file__),
        function_source_sha256=hashlib.sha256(inspect.getsource(new).encode()).hexdigest())
        for old, new in (('s1_bank', _bank), ('s1_path', _path), ('s1_reference', _reference))}


def bridge_function(kernel, name):
    require(name in ('s1_plan', 's1_replay'), 'Only unchanged planner/replay bodies may be bridged')
    original = getattr(kernel, name); namespace = original.__globals__.copy()
    namespace.update(s1_bank=_bank, s1_path=_path, s1_reference=_reference)
    result = types.FunctionType(original.__code__, namespace, original.__name__, original.__defaults__, original.__closure__)
    result.__kwdefaults__ = original.__kwdefaults__
    return result


def _intent(ctx, stage, output, index=None):
    require(not output.exists(), 'Existing/partial output retained; no overwrite or automatic retry')
    name = stage + ('' if index is None else '_' + str(index)) + '.json'
    write_exclusive(ctx['artifacts'] / 'intents' / ctx['role'] / name,
        dict(**ctx['binding'], operation=stage, policy_index=index, output=str(output), began_unix_seconds=time.time()))


def prepare(ctx):
    evidence = lineage(ctx); out = _path(ctx, 'banks'); _intent(ctx, 'prepare', out)
    simulator = ctx['base'] / 'releases/visual-v1/stable-worldmodel'; s = ctx['spec']
    subprocess.run([sys.executable, str(ROOT / 'scripts/visual/prepare_prospective_pusht.py'), '--source', str(simulator),
        '--output', str(out), '--cases', str(s['count']), '--seed-start', str(s['seed_start']), '--max-seeds', str(s['max_seeds'])], check=True)
    subprocess.run([sys.executable, str(ROOT / 'scripts/visual/accept_adaptation_contexts.py'), '--simulator', str(simulator),
        '--bank', str(out), '--count', str(s['count']), '--seed-start', str(s['seed_start'])], check=True)
    manifest = json.loads((out / 'manifest.json').read_text()); accepted = json.loads((out / 'acceptance.json').read_text())
    validate_bank(manifest, accepted, s, sha(out / 'manifest.json'))
    require(manifest['script_sha256'] == sha(ROOT / 'scripts/visual/prepare_prospective_pusht.py') and
            accepted['source_sha256'] == sha(ROOT / 'scripts/visual/accept_adaptation_contexts.py'), 'Changed generator/acceptor')
    write_exclusive(out / 'lineage.json', evidence)
    write_exclusive(out / 'role.json', dict(status='PASS_S3_RAW_BANK', **ctx['binding'], count=512,
        seed_start=s['seed_start'], max_seeds=s['max_seeds'], parent_manifest_sha256=sha(out / 'manifest.json'),
        acceptance_sha256=sha(out / 'acceptance.json'), lineage_sha256=sha(out / 'lineage.json'),
        retained_pixel_isolation='PENDING_AFTER_GENERATION', model_input_admission=False))
    with (out / 'DONE').open('x') as f: f.write('S3 full generated bank and33branches accepted\n')


def run_reference(ctx, index, stage):
    require(stage in ('plan', 'replay'), 'Wrong reference operation')
    _reference(ctx, index); _bank(ctx)
    output = _path(ctx, 'actions' if stage == 'plan' else 'physics') / f'route_{index}'
    _intent(ctx, stage, output, index)
    require(sha(KERNEL) == ctx['sources']['files'][str(KERNEL.relative_to(ROOT))], 'Changed immutable planner/replay source')
    kernel = _load(KERNEL, '_s3_unchanged_input_kernel')
    bridge_function(kernel, 's1_' + stage)(ctx, index)
    report = json.loads((output / 'report.json').read_text())
    expected = 'PASS_COMPACT_FIXED_REFERENCE_SEARCH' if stage == 'plan' else 'PASS_COMPLETE_SELECTED_PHYSICS'
    require(report.get('status') == expected and len(report['cases']) == 512 and (output / 'DONE').is_file() and
            all(report['binding'].get(k) == v for k, v in ctx['binding'].items()) and
            report['binding']['source_sha256'] == sha(KERNEL), 'Incomplete output/wrong wrapper/kernel identity')
    write_exclusive(output / 's3_admission.json', dict(status='PASS_S3_RAW_' + stage.upper(), **ctx['binding'],
        policy_index=index, count=512, report_sha256=sha(output / 'report.json'), kernel_function='s1_' + stage))


def reference_report(ctx, index, stage):
    root = _path(ctx, stage) / f'route_{index}'
    admission = json.loads((root / 's3_admission.json').read_text())
    operation = 'PLAN' if stage == 'actions' else 'REPLAY'
    require(admission.get('status') == 'PASS_S3_RAW_' + operation and admission.get('policy_index') == index and
            admission.get('count') == 512 and all(admission.get(k) == v for k, v in ctx['binding'].items()), 'Wrong reference admission')
    report = checked_json(root / 'report.json', admission['report_sha256'])
    expected = 'PASS_COMPACT_FIXED_REFERENCE_SEARCH' if stage == 'actions' else 'PASS_COMPLETE_SELECTED_PHYSICS'
    require(report.get('status') == expected and len(report['cases']) == 512 and
            [r['index'] for r in report['cases']] == list(range(512)) and (root / 'DONE').is_file() and
            all(report['binding'].get(k) == v for k, v in ctx['binding'].items()) and
            report['binding']['source_sha256'] == sha(KERNEL), 'Wrong complete reference report')
    return root, report


def _npz(path, checksum):
    require(sha(path) == checksum, 'Changed compact source payload: ' + str(path))
    with np.load(path, allow_pickle=False) as data: return {key: data[key].copy() for key in data.files}


def _save(path, arrays):
    with path.open('xb') as stream:
        np.savez_compressed(stream, **arrays); stream.flush(); os.fsync(stream.fileno())
    return pair(path)


def convert(ctx, pool):
    require(type(pool) is int and pool in range(3), 'Wrong pool')
    index, route = 8 * pool, 32 * pool; _reference(ctx, index)
    bank, role, manifest = _bank(ctx)
    action_root, actions = reference_report(ctx, index, 'actions')
    physics_root, physics = reference_report(ctx, index, 'physics')
    require(actions['binding']['bank_manifest_sha256'] == role['parent_manifest_sha256'] and
            physics['binding']['parent_manifest_sha256'] == role['parent_manifest_sha256'] and
            physics['binding']['action_report_sha256'] == sha(action_root / 'report.json') and
            physics.get('independent_replays') == 1024, 'Wrong complete selected physics ancestry')
    out = _path(ctx, 'common_prefix') / f'pool_{pool}'; _intent(ctx, 'convert', out, index)
    out.mkdir(parents=True, exist_ok=False); (out / 'scorer_inputs').mkdir()
    env_factory = None
    if ctx['role'] == 'recipient':
        (out / 'outcomes').mkdir()
        simulator = ctx['base'] / 'releases/visual-v1/stable-worldmodel'; sys.path.insert(0, str(simulator))
        from stable_worldmodel.envs.pusht.env import PushT
        env_factory = lambda: PushT(resolution=224)
    records = []
    try:
        for item, a, p in zip(manifest['cases'], actions['cases'], physics['cases'], strict=True):
            i = item['index']; cp = bank / f'case_{i:03d}.npz'; ap = action_root / a['file']; pp = physics_root / p['file']
            require(all(r['index'] == i and r['seed'] == item['seed'] for r in (a, p)) and
                    a['input_sha256'] == p['input_sha256'] == item['sha256'] and
                    p['action_file_sha256'] == a['file_sha256'] and p.get('all_two_replays_exact') is True and
                    a.get('population_replay_exact') is True and a.get('scored_candidates') == 9000,
                    'Changed case/action/physics lineage; no skipping')
            context_arrays, action_arrays, physics_arrays = (_npz(cp, item['sha256']), _npz(ap, a['file_sha256']), _npz(pp, p['file_sha256']))
            if ctx['role'] == 'recipient':
                inputs, outcomes, checks = common_prefix.build_compact_case(env_factory, context_arrays, action_arrays, physics_arrays, a, route)
            else:
                inputs, checks = common_prefix.build_donor_case(context_arrays, action_arrays, physics_arrays, a, route)
            record = dict(index=i, seed=item['seed'], parent_id=item['seed'], bank=pair(cp), actions=pair(ap), physics=pair(pp),
                selected_index=a['selected_index'], selected_iteration=a['selected_iteration'], checks=checks,
                scorer_input=_save(out / 'scorer_inputs' / f'case_{i:03d}.npz', inputs))
            if ctx['role'] == 'recipient': record['outcomes'] = _save(out / 'outcomes' / f'case_{i:03d}.npz', outcomes)
            write_exclusive(out / f'case_{i:03d}.json', record); records.append(record)
    except Exception as error:
        write_exclusive(out / 'FAILURE.json', dict(status='BLOCKED_S3_COMMON_PREFIX_POPULATION', **ctx['binding'],
            pool=pool, completed_indices=[r['index'] for r in records], failed_index=len(records), error=repr(error),
            failure_policy='Keep partial artifacts; no replacement, skipping, retry or accepted manifest.'))
        raise
    report = dict(status='PASS_S3_COMPLETE_COMMON_PREFIX_POOL', **ctx['binding'], pool=pool, group=2 * pool,
        policy_index=index, legacy_reference_route=route, count=512, parent_ids=[r['seed'] for r in manifest['cases']],
        bank_manifest=pair(bank / 'manifest.json'), bank_role_metadata=pair(bank / 'role.json'),
        action_report=pair(action_root / 'report.json'), physics_report=pair(physics_root / 'report.json'),
        cases=records, physical_suffix_outcome_count=512 * 32 if ctx['role'] == 'recipient' else 0,
        candidate_source=pair(common_prefix.LEGACY), model_input_admission=False,
        scope='Creation-time common-prefix replay and saved source checks. Retained pixel isolation and independent '
              'whole-population model admission remain required. Inputs/outcomes are separate by data flow, not OS permissions.')
    write_exclusive(out / 'report.json', report)
    with (out / 'DONE').open('x') as stream: stream.write('S3 complete common-prefix pool\n')
    return report


def role_manifest(ctx):
    """Assemble input-only manifest; never issue the model admission itself."""
    bank, role, manifest = _bank(ctx); rows = []
    target = _path(ctx, 'common_prefix') / 'input_manifest.json'; _intent(ctx, 'manifest', target)
    for pool in range(3):
        root = _path(ctx, 'common_prefix') / f'pool_{pool}'; report = json.loads((root / 'report.json').read_text())
        require(report.get('status') == 'PASS_S3_COMPLETE_COMMON_PREFIX_POOL' and report.get('count') == 512 and
                (report['pool'], report['group'], report['policy_index'], report['legacy_reference_route']) == (pool, 2 * pool, 8 * pool, 32 * pool) and
                all(report.get(k) == v for k, v in ctx['binding'].items()) and
                report['bank_manifest'] == pair(bank / 'manifest.json') and (root / 'DONE').is_file() and
                len(report['cases']) == 512, 'Incomplete common-prefix role population')
        cases = []
        for original, case in zip(manifest['cases'], report['cases'], strict=True):
            require((case['index'], case['seed']) == (original['index'], original['seed']) and
                    sha(case['scorer_input']['path']) == case['scorer_input']['sha256'], 'Changed ordered scorer inputs')
            cases.append({k: case[k] for k in ('index', 'seed', 'scorer_input')})
        rows.append(dict(pool=pool, group=2 * pool, policy_index=8 * pool, legacy_reference_route=32 * pool, cases=cases))
    result = dict(status='S3_COMPLETE_COMMON_PREFIX_ROLE_INPUT_MANIFEST', protocol_sha256=ctx['protocol_sha256'],
        sources_sha256=ctx['sources_sha256'], role=ctx['role'], count=512,
        parent_ids=[r['seed'] for r in manifest['cases']], routes=rows,
        model_input_admission=False, scope='Input-only manifest. A separate S3 accepted-context receipt must bind complete content and saved-array acceptance.')
    write_exclusive(target, result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('protocol', 'protocol-sha256', 'sources', 'sources-sha256'): p.add_argument('--' + key, required=True)
    p.add_argument('--role', choices=ROLES, required=True)
    p.add_argument('--stage', choices=('audit', 'prepare', 'plan', 'replay', 'convert', 'manifest'), required=True)
    p.add_argument('--index', type=int); p.add_argument('--pool', type=int)
    a = p.parse_args(); ctx = context(a.protocol, a.protocol_sha256, a.sources, a.sources_sha256, a.role)
    if a.stage == 'audit': print(json.dumps(lineage(ctx), indent=2)); return
    if a.stage == 'prepare': prepare(ctx); return
    if a.stage == 'convert': convert(ctx, a.pool); return
    if a.stage == 'manifest': role_manifest(ctx); return
    run_reference(ctx, a.index, a.stage)


if __name__ == '__main__': main()
