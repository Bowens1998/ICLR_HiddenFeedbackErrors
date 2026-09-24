"""Draft complete S2 probe admission before q_g scoring.

The JSON schemas below are target interfaces. No actual S2 protocol, population
or authorization is supplied by this module. This gate reads only authenticated
raw inputs, T0 latent caches/rollouts and construction g_A; it never deserializes q_g,
T1, response predictions or diagnostic errors. It verifies saved-data linkage
and independently checks every four-ray family, not QP optimality or image
encoding/dynamics by a second implementation.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
PHASE = HERE.parent
ROOT = PHASE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
from projection_four import (OBJECTIVES, SOURCES, SHRINK_FACTORS, TOLERANCE,
                             RECEIPT_ATOL, accept_t0_four_family)
import model_runtime

HORIZONS = [5, 10, 15, 20, 25]
BRANCHES = ['free', 'actual', 'donor', 'reset']
CACHE_AXES = dict(objectives=list(OBJECTIVES), horizons=HORIZONS, latent_dim=192)
ROLLOUT_AXES = dict(**CACHE_AXES, branches=BRANCHES, insertion_sources=list(SOURCES))
PROJECTION_AXES = dict(objectives=list(OBJECTIVES), sources=list(SOURCES), latent_dim=192)
STATUS = 'S2_COMPLETE_PROBE_POPULATION_ACCEPTED_BEFORE_QG_SCORING'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _pair(value):
    if (not isinstance(value, dict) or set(value) != {'path', 'sha256'} or
            not Path(value.get('path', '')).is_absolute() or
            not isinstance(value.get('sha256'), str) or len(value['sha256']) != 64 or
            any(c not in '0123456789abcdef' for c in value['sha256'])):
        raise ValueError('An exact actual absolute path/SHA256 pair is required')
    if sha(value['path']) != value['sha256']:
        raise ValueError('Changed hash-bound input: ' + value['path'])
    return Path(value['path'])


def _json(pair):
    path = _pair(pair)
    value = json.loads(path.read_text())
    _pair(pair)
    return value


def _arrays(pair):
    path = _pair(pair)
    with np.load(path, allow_pickle=False) as data:
        if len(data.files) != len(set(data.files)):
            raise ValueError('Duplicate archive key')
        value = {key: data[key] for key in data.files}
    _pair(pair)
    return value


def _exact(value, expected, label):
    if value != expected:
        raise ValueError('Changed or incomplete ' + label)


def _array(value, shape, dtype, label):
    value = np.asarray(value)
    if value.shape != shape or value.dtype != dtype or not np.isfinite(value).all():
        raise ValueError('Wrong complete finite shape/dtype: ' + label)
    return value


def _equal(left, right, label):
    if not np.array_equal(left, right):
        raise ValueError('Saved values disagree: ' + label)


def _size(split):
    if split not in ('calibration', 'test'):
        raise ValueError('Only complete calibration/test splits are supported')
    return 256 if split == 'calibration' else 512


def _write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())


def validate_assignment(assignment, *, split, seed):
    n = _size(split)
    _array(assignment, (n,), np.dtype('int64'), 'donor permutation')
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise ValueError('Frozen donor permutation seed is unresolved')
    _equal(np.sort(assignment), np.arange(n, dtype=np.int64), 'global donor permutation coverage')
    expected = np.random.Generator(np.random.PCG64(seed)).permutation(n).astype(np.int64)
    _equal(assignment, expected, 'single frozen PCG64 donor permutation')


def validate_pool(recipient, donor, projection, families, rollout, head_A,
                  assignment, *, split):
    """Complete per-pool value gate; no file identity is asserted by this helper."""
    n = _size(split)
    _exact(set(recipient), {'free', 'observed_history', 'reset', 'observed', 'truth',
                           'prefix_actions', 'selected_actions', 'selected_index', 'seeds'}, 'recipient cache keys')
    _exact(set(donor), {'observed5', 'seeds'}, 'donor cache keys')
    for key in ('free', 'observed_history', 'reset'):
        _array(recipient[key], (2, n, 5, 192), np.dtype('float32'), 'cache/' + key)
    _array(recipient['observed'], (n, 5, 192), np.dtype('float32'), 'observed')
    _array(recipient['truth'], (n, 5, 6), np.dtype('float64'), 'physical truth')
    _array(recipient['prefix_actions'], (n, 10, 2), np.dtype('float32'), 'prefix actions')
    _array(recipient['selected_actions'], (n, 25, 2), np.dtype('float32'), 'selected actions')
    _array(recipient['selected_index'], (n,), np.dtype('int64'), 'selected index')
    if np.any(recipient['selected_index'] < 0) or np.any(recipient['selected_index'] >= 300):
        raise ValueError('Selected action outside native 300 population')
    _array(donor['observed5'], (n, 192), np.dtype('float32'), 'donor observed5')
    for name, values in (('recipient', recipient['seeds']), ('donor', donor['seeds'])):
        _array(values, (n,), np.dtype('int64'), name + ' seeds')
        if len(np.unique(values)) != n or np.any(values < 0) or np.any(values >= 2**32):
            raise ValueError('Incomplete unique ' + name + ' seed roster')
    if np.intersect1d(recipient['seeds'], donor['seeds']).size:
        raise ValueError('Recipient/donor actual parents overlap')
    _array(assignment, (n,), np.dtype('int64'), 'donor permutation')
    _equal(np.sort(assignment), np.arange(n, dtype=np.int64), 'permutation coverage')
    _exact(set(projection), {'replacements', 'directions'}, 'projection keys')
    _array(projection['replacements'], (n, 2, 2, 192), np.dtype('float32'), 'FP32 replacement tokens')
    _array(projection['directions'], (n, 2, 2, 192), np.dtype('float64'), 'FP64 complete rays')
    if len(families) != n:
        raise ValueError('Every recipient needs its indivisible four-member report')
    _exact(set(rollout), {'tokens', 'identity', 'observed_history', 'native_endpoint', 'inserted', 'seeds'}, 'rollout keys')
    tokens = _array(rollout['tokens'], (2, n, 4, 5, 192), np.dtype('float32'), 'all rollout branches/horizons')
    for key in ('identity', 'observed_history'):
        _array(rollout[key], (2, n, 5, 192), np.dtype('float32'), key)
    _array(rollout['native_endpoint'], (2, n, 192), np.dtype('float32'), 'native endpoint')
    _array(rollout['inserted'], (2, n, 2, 192), np.dtype('float32'), 'actually inserted tokens')
    _array(rollout['seeds'], (n,), np.dtype('int64'), 'rollout seeds')
    _equal(rollout['seeds'], recipient['seeds'], 'rollout parent order')
    _equal(tokens[:, :, 0], recipient['free'], 'all free horizons')
    _equal(tokens[:, :, 3], recipient['reset'], 'all reset horizons')
    _equal(rollout['observed_history'], recipient['observed_history'], 'all observed-history horizons')
    _equal(rollout['identity'], recipient['free'], 'identity reroll')
    _equal(rollout['native_endpoint'], tokens[:, :, 0, -1], 'native300 endpoint')
    _equal(rollout['observed_history'][:, :, 0], tokens[:, :, 0, 0], 'observed-history first prediction')
    _equal(tokens[:, :, 3, 0], np.broadcast_to(recipient['observed'][None, :, 0], (2, n, 192)), 'reset insertion')
    replacements = np.transpose(projection['replacements'], (1, 0, 2, 3))
    _equal(rollout['inserted'], replacements, 'actual FP32 projected insertion')
    _equal(tokens[:, :, 1:3, 0], replacements, 'corrected first saved tokens')
    accepted = []
    for goal in range(n):
        predicted = {objective: recipient['free'][i, goal, 0] for i, objective in enumerate(OBJECTIVES)}
        guides = dict(actual=recipient['observed'][goal, 0], donor=donor['observed5'][assignment[goal]])
        # Independent dense evaluation/ray reconstruction; never call the QP producer.
        check = accept_t0_four_family(predicted, guides, head_A,
            projection['replacements'][goal], projection['directions'][goal], families[goal],
            goal_index=goal, donor_assignment=assignment)
        if check['effective_norm'] == 0:
            for branch in (1, 2):
                _equal(tokens[:, goal, branch], recipient['free'][:, goal], 'zero-dose full reroll')
        accepted.append(check)
    return dict(families=n, directions=4 * n,
        effective_common_norm=[x['effective_norm'] for x in accepted],
        shrink_factor=[x['shrink_factor'] for x in accepted],
        zero_families=sum(x['effective_norm'] == 0 for x in accepted),
        maximum_normalized_output_deviation=max(c['heads'][0]['normalized_output_deviation'] for a in accepted for c in a['checks']),
        maximum_region_violation=max(c['heads'][0]['region_violation'] for a in accepted for c in a['checks']),
        maximum_norm_residual=max(abs(c['actual_norm'] - c['expected_norm']) for a in accepted for c in a['checks']))


def _configuration(binding):
    _exact(set(binding), {'status', 'split', 'protocol', 'sources', 'raw_acceptance',
                         'caches', 'projections', 'rollouts', 'donor_assignment'}, 'population binding fields')
    _exact(binding['status'], 'S2_PROBE_POPULATION_BINDING_FROZEN', 'population binding status')
    split = binding['split']; n = _size(split)
    protocol = _json(binding['protocol'])
    _exact(protocol.get('status'), 'S2_SCIENTIFIC_PROTOCOL_FROZEN', 'scientific freeze')
    cfg = protocol['probe_population']
    required = dict(axes=ROLLOUT_AXES, native_population=300, history_tokens=3,
                    prefix_steps=10, insertion_relative_step=5,
                    shrink_factors=list(SHRINK_FACTORS), tolerance=TOLERANCE,
                    receipt_atol=RECEIPT_ATOL)
    for key, expected in required.items():
        _exact(cfg.get(key), expected, 'frozen probe arithmetic/' + key)
    _exact(set(cfg['donor_assignment_seeds']), {'calibration', 'test'}, 'both donor permutation seeds')
    sources = _json(binding['sources'])
    _exact(sources.get('status'), 'S2_PROBE_POPULATION_SOURCES_FROZEN', 'probe source freeze')
    _exact(sources.get('protocol_sha256'), binding['protocol']['sha256'], 'source/protocol binding')
    closure = sources['files']
    required_paths = [Path(__file__), HERE / 'projection_four.py', HERE / 'model_runtime.py',
                      HERE / 'cache_population.py', HERE / 'raw_inputs.py', HERE / 'accept_raw_inputs.py',
                      *model_runtime.METADATA.values(),
                      PHASE / 'scripts/s1_projection.py', ROOT / 'strengthening/adapters/verifier.py']
    _exact(set(sources['producers']), {'projection', 'rollout'}, 'both actual producing wrappers')
    for pair in sources['producers'].values():
        path = _pair(pair)
        if path.suffix != '.py':
            raise ValueError('Actual projection/rollout Python wrapper source required')
        required_paths.append(path)
    actual = set()
    for name, digest in closure.items():
        path = Path(name); path = path if path.is_absolute() else ROOT / path
        # Complete inherited raw closure may include historical heads/checkpoints.
        # Authenticate bytes only here; never deserialize these closure entries.
        _pair(dict(path=str(path), sha256=digest)); actual.add(path.resolve())
    if not {x.resolve() for x in required_paths} <= actual:
        raise ValueError('Incomplete direct gate/kernel/runtime source closure')
    raw_sources = _json(sources['raw_sources']); cache_sources = _json(sources['cache_sources'])
    _exact(raw_sources.get('status'), 'S2_RAW_INPUT_SOURCES_FROZEN', 'raw source freeze')
    _exact(raw_sources.get('protocol_sha256'), binding['protocol']['sha256'], 'raw source/protocol')
    _exact(cache_sources.get('protocol_sha256'), binding['protocol']['sha256'], 'cache source/protocol')
    _exact(cache_sources.get('status'), 'S2_LATENT_CACHE_SOURCES_FROZEN', 'cache source freeze')
    _exact(cache_sources.get('raw_input_sources'), sources['raw_sources'], 'cache/raw source chain')
    _exact(cache_sources.get('raw_population_acceptance'), binding['raw_acceptance'], 'cache/raw population chain')
    for source in (raw_sources, cache_sources):
        for name, digest in source['files'].items():
            _exact(closure.get(name), digest, 'complete input/cache source closure')
    heads = sources['heads_A']
    if len(heads) != 3 or {(h['pool'], h['group']) for h in heads} != {(i, 2*i) for i in range(3)}:
        raise ValueError('All three fixed construction heads are required exactly once')
    return split, n, protocol, cfg, sources


def _raw_admission(binding, sources, split, n):
    raw = _json(binding['raw_acceptance'])
    _exact(raw.get('status'), 'PASS_S2_COMPLETE_RAW_INPUT_POPULATION', 'complete raw acceptance')
    _exact(raw.get('protocol_sha256'), binding['protocol']['sha256'], 'raw protocol')
    for key, value in dict(bank_count=4, route_count=18, saved_triples_verified=6912,
                           models_or_readouts_loaded=False).items():
        _exact(raw.get(key), value, 'raw whole-population/' + key)
    _exact(raw.get('source_sha256'), sha(HERE / 'accept_raw_inputs.py'), 'raw acceptance source')
    for key in ('sources_sha256', 'raw_sources_sha256'):
        _exact(raw.get(key), sources['raw_sources']['sha256'], key)
    content = _json(raw['content_lineage'])
    _exact(content.get('status'), 'PASS_S2_RETAINED_PIXEL_ISOLATION', 'complete content isolation')
    _exact(content.get('protocol_sha256'), binding['protocol']['sha256'], 'content protocol')
    _exact(content.get('sources_sha256'), sources['raw_sources']['sha256'], 'content source')
    banks = raw['banks']
    roles = {'calibration_recipient': 256, 'calibration_donor': 256, 'test_recipient': 512, 'test_donor': 512}
    if len(banks) != 4 or {b['bank_role'] for b in banks} != set(roles):
        raise ValueError('All four accepted raw banks required')
    parents = {}
    for bank in banks:
        role = bank['bank_role']; ids = bank['parent_ids']
        _exact(bank['count'], roles[role], 'raw bank count')
        if len(ids) != roles[role] or len(set(ids)) != roles[role] or any(type(x) is not int or not 0 <= x < 2**32 for x in ids):
            raise ValueError('Incomplete actual raw parent roster')
        parents[role] = ids; _pair(bank['manifest']); _pair(bank['role_metadata'])
    for i, role in enumerate(roles):
        for other in list(roles)[i+1:]:
            if set(parents[role]) & set(parents[other]):
                raise ValueError('Accepted raw role parent populations overlap')
    admissions = raw['admissions']; expected_roles = {
        'probe_calibration', 'probe_test', 'probe_donor_calibration', 'probe_donor_test',
        'calibration_response', 'test_response'}
    if len(admissions) != 6 or {r['role'] for r in admissions} != expected_roles:
        raise ValueError('All six independently accepted runtime roles required')
    accepted = {}
    for row in admissions:
        pair = {k: row[k] for k in ('path', 'sha256')}; doc = _json(pair)
        role = row['role']; part = 'calibration' if 'calibration' in role else 'test'
        bank_role = part + ('_donor' if 'donor' in role else '_recipient')
        stream = 'donor_observed' if 'donor' in role else 'response' if 'response' in role else 'probe'
        bank = next(b for b in banks if b['bank_role'] == bank_role)
        for key, value in dict(status='S2_FIXED_CONTEXTS_ACCEPTED', protocol_sha256=binding['protocol']['sha256'],
                sources_sha256=sources['raw_sources']['sha256'], bank_role=bank_role,
                count=roles[bank_role], parent_ids=parents[bank_role], stream_role=stream, bank_manifest=bank['manifest']).items():
            _exact(doc.get(key), value, 'runtime raw admission/' + key)
        manifest = _json(doc['input_manifest'])
        for key, value in dict(status='S2_COMPLETE_ROLE_STREAM_INPUT_MANIFEST',
                protocol_sha256=binding['protocol']['sha256'], sources_sha256=sources['raw_sources']['sha256'],
                bank_role=bank_role, stream_role=stream, count=roles[bank_role], parent_ids=parents[bank_role]).items():
            _exact(manifest.get(key), value, 'complete raw input manifest/' + key)
        accepted[role] = (pair, doc, manifest)
    return raw, parents, accepted


def _pose(states):
    value = np.asarray(states, dtype=np.float64)
    # Preserve the existing scalar-per-frame FP64 trig operation order exactly.
    return np.array([[s[0], s[1], s[2], s[3], np.sin(s[4]), np.cos(s[4])]
                     for s in value], dtype=np.float64)


def _raw_cases(report, admission, input_manifest, arrays, *, donor):
    """Rehash original files and independently reconstruct physical truth/actions."""
    n = report['count']; pool = report['pool']; cases = report['cases']
    if len(cases) != n or [c['index'] for c in cases] != list(range(n)):
        raise ValueError('Complete cache case order required')
    routes = input_manifest['routes']
    if len(routes) != 3 or {x['pool'] for x in routes} != {0, 1, 2}:
        raise ValueError('All three input manifest pools required')
    route = next(x for x in routes if x['pool'] == pool)
    _exact(route['policy_index'], 8 * pool, 'probe raw policy index')
    if len(route['cases']) != n:
        raise ValueError('Incomplete independently accepted input route')
    for index, (case, original) in enumerate(zip(cases, route['cases'], strict=True)):
        for key in ('index', 'seed', 'bank', 'actions', 'physics', 'selected_index', 'selected_iteration'):
            _exact(case.get(key), original.get(key), 'raw/cache case identity/' + key)
        _exact(case['seed'], admission['parent_ids'][index], 'admitted raw seed order')
        _exact(case['parent_id'], case['seed'], 'raw parent ID/seed')
        for key in ('bank', 'actions', 'physics'):
            _pair(case[key])
        for key in ('bank_manifest', 'actions_report', 'physics_report'):
            if key in case:
                _pair(case[key])
        with np.load(case['bank']['path'], allow_pickle=False) as data:
            prefix = data['prefix']; history = data['history_states']; seed = data['seed']
        with np.load(case['actions']['path'], allow_pickle=False) as data:
            population = data['population_actions']; selected = data['selected_actions']
            chosen = data['selected_index']; action_seed = data['seed']
            iteration = data['selected_iteration']
        with np.load(case['physics']['path'], allow_pickle=False) as data:
            states = data['states']; actions = data['actions']; physics_seed = data['seed']
        _array(prefix, (10, 2), np.dtype('float32'), 'raw prefix')
        _array(population, (300, 25, 2), np.dtype('float32'), 'raw native300 controls')
        _array(selected, (25, 2), np.dtype('float32'), 'raw selected controls')
        if states.shape != (36, 7) or history.shape != (3, 7) or not np.isfinite(states).all() or not np.isfinite(history).all():
            raise ValueError('Incomplete finite raw physical states')
        _array(actions, (35, 2), np.dtype('float32'), 'full raw controls')
        for value in (seed, action_seed, physics_seed, chosen, iteration):
            if np.asarray(value).shape != () or np.asarray(value).dtype.kind not in 'iu':
                raise ValueError('Raw identity/index must be an integer scalar')
        for value in (seed, action_seed, physics_seed):
            _exact(int(value), case['seed'], 'raw file seed')
        j = int(chosen)
        if not 0 <= j < 300 or not 0 <= int(iteration) < 30:
            raise ValueError('Raw selected search index outside frozen population')
        _exact(case['selected_index'], j, 'cache selected member')
        _exact(case['selected_iteration'], int(iteration), 'cache search iteration')
        _equal(population[j], selected, 'selected member of native population')
        _equal(actions, np.concatenate([prefix, selected]), 'physics controls')
        _equal(states[[0, 5, 10]], history, 'physical history identity')
        _exact(int(arrays['seeds'][index]), case['seed'], 'cached seed')
        if not donor:
            _equal(arrays['prefix_actions'][index], prefix, 'cached prefix')
            _equal(arrays['selected_actions'][index], selected, 'cached selected action')
            _exact(int(arrays['selected_index'][index]), j, 'cached chosen candidate')
            _equal(arrays['truth'][index], _pose(states[[15, 20, 25, 30, 35]]), 'physical truth at all horizons')


def _cache(pair, binding, sources, raw, admissions, *, split, pool, donor):
    report = _json(pair); n = _size(split)
    role = split + ('_donor' if donor else '_recipient')
    runtime_role = 'probe_donor_' + split if donor else 'probe_' + split
    admission_pair, admission, manifest = admissions[runtime_role]
    expected = dict(status='PASS_COMPLETE_S2_LATENT_CACHE', role=role, split=split,
        pool=pool, group=2*pool, policy_index=8*pool, count=n,
        protocol_sha256=binding['protocol']['sha256'], cache_sources_sha256=sources['cache_sources']['sha256'],
        raw_sources_sha256=sources['raw_sources']['sha256'], raw_population_acceptance=binding['raw_acceptance'],
        input_admission=admission_pair, content_lineage=raw['content_lineage'], axes=CACHE_AXES)
    for key, value in expected.items():
        _exact(report.get(key), value, 'cache report/' + key)
    for key, value in dict(whole_population_retained=True, response_models_deserialized=False,
            readout_weights_deserialized=False, readout_outputs_computed=False, donor_model_predictions_generated=False,
            native300_identity_and_endpoint_exact=not donor, observed_shared_encoder_exact=not donor,
            frozen_model_state_unchanged=True, action_normalization_unchanged=True,
            donor_encoder_only=donor).items():
        if report.get(key) is not value:
            raise ValueError('Incomplete cache computation guard: ' + key)
    _exact(report.get('source_sha256'), sha(HERE / 'cache_population.py'), 'cache producer source')
    _exact(report.get('input_manifest'), admission['input_manifest'], 'cache input manifest')
    cache_sources = _json(sources['cache_sources'])
    _exact(report['runtime_contract'], cache_sources['runtime_contracts'][runtime_role], 'frozen role runtime contract')
    contract = _json(report['runtime_contract'])
    _exact(contract.get('status'), 'S2_MODEL_RUNTIME_CONTRACT_FROZEN', 'cache runtime status')
    _exact(contract.get('role'), runtime_role, 'cache runtime role')
    _exact(contract.get('protocol'), binding['protocol'], 'cache runtime protocol')
    _exact(contract.get('input_admission'), admission_pair, 'cache runtime input admission')
    allowed = contract['models']; models = report['models']
    expected_models = [model_runtime.describe_fixed_model(p, objective, 'T0')
                       for p in range(3) for objective in OBJECTIVES]
    _exact(allowed, expected_models, 'entire fixed T0 runtime model allowlist')
    expected_objectives = [OBJECTIVES[0]] if donor else list(OBJECTIVES)
    if len(models) != len(expected_objectives) or [m['spec']['objective'] for m in models] != expected_objectives:
        raise ValueError('Complete fixed cache model/encoder roster required')
    for model in models:
        spec = model['spec']
        if spec not in allowed or spec['pool'] != pool or spec['group'] != 2*pool or spec['condition'] != 'T0':
            raise ValueError('Cache model is not the admitted T0 coordinate system')
        _exact(model.get('objective'), spec['objective'], 'saved model objective')
        _exact(model.get('spec_sha256'), model_runtime.value_sha(spec), 'saved full model specification digest')
        summary = _json(spec['summary'])
        normalization = model_runtime._normalization(summary['normalization'])
        _exact(model.get('normalization_sha256'), model_runtime.value_sha(normalization), 'actual action-normalizer digest')
        _exact(model['access_receipt']['role'], runtime_role, 'cache model access role')
        _exact(model['access_receipt']['runtime_contract_sha256'], report['runtime_contract']['sha256'], 'model runtime hash')
        _exact(model['access_receipt'].get('input_admission'), admission_pair, 'model input admission hash')
        if donor:
            _exact(model['access_receipt'].get('permitted_operation'), 'encode_donor_observations_only', 'donor encoder-only access')
    arrays = _arrays(report['arrays'])
    _exact(report.get('array_schema'), {k: dict(shape=list(v.shape), dtype=str(v.dtype)) for k, v in arrays.items()},
           'complete cache array schema')
    _equal(arrays['seeds'], np.asarray(admission['parent_ids'], np.int64), 'cache admitted parent order')
    _raw_cases(report, admission, manifest, arrays, donor=donor)
    return report, arrays


def accept_population(binding_path, binding_sha256, output):
    """Future file admission; exclusive output, no q_g deserialization/output."""
    output = Path(output)
    intent = output.with_name(output.name + '.intent.json')
    if output.exists() or intent.exists():
        raise ValueError('Existing acceptance output is retained, never overwritten')
    binding_pair = dict(path=str(Path(binding_path).resolve()), sha256=binding_sha256)
    binding = _json(binding_pair)
    split, n, protocol, cfg, sources = _configuration(binding)
    raw, parents, admissions = _raw_admission(binding, sources, split, n)
    expected_cache = {(p, r) for p in range(3) for r in ('recipient', 'donor')}
    if len(binding['caches']) != 6 or {(r['pool'], r['role']) for r in binding['caches']} != expected_cache:
        raise ValueError('All six pool/recipient-donor caches required exactly once')
    for kind in ('projections', 'rollouts'):
        if len(binding[kind]) != 3 or {r['pool'] for r in binding[kind]} != {0, 1, 2}:
            raise ValueError('All three complete ' + kind + ' required exactly once')
    assignment_arrays = _arrays(binding['donor_assignment'])
    _exact(set(assignment_arrays), {'donor_assignment', 'recipient_ids', 'donor_ids'}, 'assignment archive keys')
    assignment = assignment_arrays['donor_assignment']
    validate_assignment(assignment, split=split, seed=cfg['donor_assignment_seeds'][split])
    for key, role in (('recipient_ids', '_recipient'), ('donor_ids', '_donor')):
        _array(assignment_arrays[key], (n,), np.dtype('int64'), key)
        _equal(assignment_arrays[key], np.asarray(parents[split+role], np.int64), 'assignment source parent order')
    _write(intent, dict(status='S2_PROBE_ADMISSION_STARTED', population_binding=binding_pair, split=split,
                        q_g_weights_deserialized=False, response_model_outputs_opened=False))
    pools = []
    for pool in range(3):
        cache_pairs = {r['role']: r['report'] for r in binding['caches'] if r['pool'] == pool}
        recipient_report, recipient = _cache(cache_pairs['recipient'], binding, sources, raw, admissions,
            split=split, pool=pool, donor=False)
        donor_report, donor = _cache(cache_pairs['donor'], binding, sources, raw, admissions,
            split=split, pool=pool, donor=True)
        _exact(recipient_report['models'][0]['spec'], donor_report['models'][0]['spec'],
               'recipient/donor frozen encoder construction model identity')
        head = next(h for h in sources['heads_A'] if h['pool'] == pool)
        head_pair = head['file']; head_A = _arrays(head_pair)
        # The model spec binds its historical construction head; no newly fitted C/D or q_g is substituted.
        for model in recipient_report['models']:
            _exact(model['spec']['head_A_sha256'], head_pair['sha256'], 'fixed construction g_A')
        projection_pair = next(r['report'] for r in binding['projections'] if r['pool'] == pool)
        projected = _json(projection_pair)
        common = dict(split=split, pool=pool, group=2*pool, count=n,
            protocol_sha256=binding['protocol']['sha256'], probe_sources_sha256=binding['sources']['sha256'],
            recipient_cache=cache_pairs['recipient'], donor_cache=cache_pairs['donor'],
            donor_assignment=binding['donor_assignment'], head_A=head_pair, model_role='T0')
        for key, value in dict(**common, status='PASS_COMPLETE_DRAFT_S2_T0_FOUR_FAMILY_PROJECTION', axes=PROJECTION_AXES,
                source_sha256=sources['producers']['projection']['sha256']).items():
            _exact(projected.get(key), value, 'projection report/' + key)
        rollout_pair = next(r['report'] for r in binding['rollouts'] if r['pool'] == pool)
        rolled = _json(rollout_pair)
        for key, value in dict(**common, status='PASS_COMPLETE_S2_T0_CORRECTED_PROBE_ROLLOUT',
                projection=projection_pair, axes=ROLLOUT_AXES, models=recipient_report['models'],
                native_population=300, history_tokens=3, prefix_steps=10,
                source_sha256=sources['producers']['rollout']['sha256']).items():
            _exact(rolled.get(key), value, 'rollout report/' + key)
        values = validate_pool(recipient, donor, _arrays(projected['arrays']), projected['families'],
                               _arrays(rolled['arrays']), head_A, assignment, split=split)
        pools.append(dict(pool=pool, group=2*pool, recipient_cache=cache_pairs['recipient'],
            donor_cache=cache_pairs['donor'], projection=projection_pair, rollout=rollout_pair,
            truth_array_source=recipient_report['arrays'], head_A=head_pair, **values))
    _pair(binding_pair)
    result = dict(status=STATUS, split=split, count=n, pools=pools, complete_pools=3,
        complete_objective_strata=6, complete_families=3*n, complete_directions=12*n,
        axes=ROLLOUT_AXES, protocol_sha256=binding['protocol']['sha256'],
        sources_sha256=binding['sources']['sha256'], population_binding=binding_pair,
        raw_population_acceptance=binding['raw_acceptance'], donor_assignment=binding['donor_assignment'],
        parent_ids=parents[split+'_recipient'], donor_ids=parents[split+'_donor'],
        source_sha256=sha(__file__), q_g_weights_deserialized=False, q_g_outputs_computed=False,
        response_model_outputs_opened=False,
        outcome_statistics_computed=False, rows_removed=0, q_g_scoring_gate_passed=True,
        scope='Whole fixed split saved-data/source/raw-truth/latent-insertion linkage plus independent four-ray '
              'FP64 nonlinear g_A/region/norm/first-backoff acceptance. No independent QP optimality, '
              'image-encoder/dynamics recomputation or simulator rerun; no q_g/T1/error-based admission.')
    _write(output, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binding', required=True)
    parser.add_argument('--binding-sha256', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    accept_population(args.binding, args.binding_sha256, args.output)


if __name__ == '__main__':
    main()
