"""Draft complete S3 decision costs/statistics with a prediction-seal barrier.

No actual inputs, model or simulator are accessed at import. Future genuine
protocol/source/prediction/outcome receipts are required by the file entrypoints.
The upstream prediction gate, not this module, proves g_A and native/QP parity.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PHASE = HERE.parent
ROOT = PHASE.parents[1]
METRIC_SOURCE = ROOT / 'scripts/visual/feedback_ranking_metrics.py'
_spec = importlib.util.spec_from_file_location('_s3_existing_exact_tie_metrics', METRIC_SOURCE)
metrics = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(metrics)

CONDITIONS = ['T0', 'T1']
OBJECTIVES = ['decoded_teacher', 'physical_labels']
BRANCHES = ['free', 'actual', 'donor', 'reset']
AXES = dict(order=['recipient', 'pool', 'condition', 'objective', 'branch', 'candidate'],
    recipient_count=512, pools=[0, 1, 2], groups=[0, 2, 4], conditions=CONDITIONS,
    objectives=OBJECTIVES, branches=BRANCHES, candidate_count=32, horizon=25)
COST_SHAPE = (512, 3, 2, 2, 4, 32)
PRIMARY = ['C_T1_actual - C_T1_free', 'C_T1_actual - C_T1_donor']
PREDICTION_STATUS = 'S3_COMPLETE_PREDICTION_POPULATION_ACCEPTED_BEFORE_OUTCOME_JOIN'
SEAL_STATUS = 'S3_COMPLETE_PREDICTION_COSTS_SEALED_BEFORE_OUTCOMES'
OUTCOME_STATUS = 'S3_COMPLETE_PHYSICAL_CANDIDATE_OUTCOMES'


def require(value, message):
    if not value: raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b''): h.update(chunk)
    return h.hexdigest()


def _hex(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def pair(path):
    path = Path(path)
    require(path.is_absolute() and path.is_file(), 'Existing absolute file required')
    return dict(path=str(path), sha256=sha(path))


def checked(ref):
    require(set(ref) == {'path', 'sha256'} and _hex(ref['sha256']) and pair(ref['path']) == ref,
            'Changed or malformed actual file pair')
    return Path(ref['path'])


def document(ref): return json.loads(checked(ref).read_text())


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False); stream.write('\n')
    return pair(path)


def array(value, shape, dtype, name):
    x = np.asarray(value)
    require(x.shape == shape and x.dtype == np.dtype(dtype) and np.isfinite(x).all(),
            'Wrong complete shape/dtype/finite values: ' + name)
    return x


def validate_ids(parent_ids, donor_ids):
    for values, name in ((parent_ids, 'recipient'), (donor_ids, 'assigned donor')):
        x = array(values, (512,), 'int64', name)
        require((x >= 0).all() and len(np.unique(x)) == 512, 'Missing/duplicate parent: ' + name)
    require(not set(parent_ids) & set(donor_ids), 'Recipient/donor populations overlap')


def validate_models(records):
    order = [(p, c, o) for p in range(3) for c in CONDITIONS for o in OBJECTIVES]
    require(len(records) == 12 and [(r['pool'], r['condition'], r['objective']) for r in records] == order,
            'All12 fixed model records required in pool/condition/objective order')
    for record in records:
        require(set(record) == {'pool', 'group', 'condition', 'objective', 'checkpoint_sha256',
                'action_normalization_sha256', 'head_A_sha256'} and
                record['group'] == 2 * record['pool'] and
                all(_hex(record[k]) for k in ('checkpoint_sha256', 'action_normalization_sha256', 'head_A_sha256')),
                'Incomplete model/head/normalizer identity')
    for pool in range(3):
        require(len({r['head_A_sha256'] for r in records if r['pool'] == pool}) == 1,
                'One frozen g_A per pool is mandatory')


def validate_prediction_arrays(data, expected):
    require(set(data) == {'predicted_costs', 'recipient_ids', 'donor_ids', 'candidate_source_indices'},
            'Only complete prediction costs and fixed identities are allowed')
    costs = array(data['predicted_costs'], COST_SHAPE, 'float64', 'predicted costs')
    require((costs >= 0).all(), 'Squared pose costs must be nonnegative')
    validate_ids(data['recipient_ids'], data['donor_ids'])
    for role in ('recipient', 'donor'):
        require(data[role + '_ids'].tolist() == expected['expected_' + role + '_ids'],
                'Changed complete ordered ' + role + ' population')
    indices = array(data['candidate_source_indices'], (512, 3, 32), 'int64', 'candidate source indices')
    require((indices[:, :, 1] == -1).all() and (indices[:, :, 0] >= 0).all() and
            (indices[:, :, 0] < 300).all() and ((indices[:, :, 2:] >= 0) & (indices[:, :, 2:] < 300)).all(),
            'Selected/zero/sampled32 source-candidate roster changed')
    ordered = np.sort(np.concatenate([indices[:, :, :1], indices[:, :, 2:]], axis=2), axis=2)
    require((np.diff(ordered, axis=2) > 0).all(), 'Repeated source candidate index')
    return costs


def physical_candidate_costs(states, goals):
    states = array(states, (512, 3, 32, 7), 'float64', 'physical terminal states')
    goals = array(goals, (512, 7), 'float64', 'physical goal states')
    result = np.empty((512, 3, 32), np.float64)
    for i in range(512): result[i] = metrics.physical_goal_cost(states[i], goals[i])
    # Independent explicit reconstruction of the existing physical cost formula.
    difference = states[..., 4] - goals[:, None, None, 4]
    angle = np.arctan2(np.sin(difference), np.cos(difference))
    independent = np.sum((states[..., 2:4] - goals[:, None, None, 2:4]) ** 2, axis=-1) + 900 * angle ** 2
    np.testing.assert_array_equal(result, independent)
    return result


def selection_values(predicted_costs, physical_costs):
    p = array(predicted_costs, COST_SHAPE, 'float64', 'predicted costs')
    q = array(physical_costs, (512, 3, 32), 'float64', 'physical candidate costs')
    require((p >= 0).all() and (q >= 0).all(), 'Nonnegative physical/predicted costs required')
    ties = p == p.min(axis=-1, keepdims=True)
    costs = q[:, :, None, None, None, :]
    selected = np.sum(ties * costs, axis=-1) / ties.sum(axis=-1)
    lowest = np.take_along_axis(np.broadcast_to(costs, p.shape), p.argmin(axis=-1)[..., None], axis=-1)[..., 0]
    changed = np.any(ties[:, :, 1, :, 1] != ties[:, :, 1, :, 0], axis=-1)
    return dict(selected=selected, lowest=lowest, tie_counts=ties.sum(axis=-1),
                t1_actual_free_argmin_sets_differ=changed)


def contrasts(selected):
    s = array(selected, (512, 3, 2, 2, 4), 'float64', 'selected physical costs')
    return np.column_stack(((s[:, :, 1, :, 1] - s[:, :, 1, :, 0]).mean(axis=(1, 2)),
                            (s[:, :, 1, :, 1] - s[:, :, 1, :, 2]).mean(axis=(1, 2))))


def independent_selections(predicted, physical):
    """Original exact-tie metric per case; no producing vectorized selection."""
    selected = np.empty((512, 3, 2, 2, 4), np.float64)
    lowest = np.empty_like(selected)
    changed = np.empty((512, 3, 2), bool)
    for i, pool, condition, objective in np.ndindex(512, 3, 2, 2):
        sets = []
        for branch in range(4):
            r = metrics.ranking_metrics(predicted[i, pool, condition, objective, branch], physical[i, pool])
            selected[i, pool, condition, objective, branch] = r['selected_cost']
            lowest[i, pool, condition, objective, branch] = r['selected_cost_lowest']
            sets.append(r['argmin_indices'])
        if condition == 1: changed[i, pool, objective] = sets[1] != sets[0]
    vectors = np.empty((512, 2), np.float64)
    for i in range(512):
        for j, reference in enumerate((0, 2)):
            terms = [selected[i, pool, 1, objective, 1] - selected[i, pool, 1, objective, reference]
                     for pool in range(3) for objective in range(2)]
            vectors[i, j] = np.sum(terms, dtype=np.float64) / 6.
    return selected, lowest, changed, vectors


def bootstrap(vectors, *, seed, bit_generator, independent=False):
    vectors = array(vectors, (512, 2), 'float64', 'paired recipient contrasts')
    require(type(seed) is int and 0 <= seed < 2 ** 32 and bit_generator == 'PCG64',
            'An explicit frozen uint32 seed and PCG64 are required; no default')
    rng = np.random.Generator(np.random.PCG64(seed)); means = np.empty((20000, 2), np.float64)
    digest = hashlib.sha256()
    for lo in range(0, 20000, 200):
        ids = rng.integers(0, 512, size=(min(200, 20000 - lo), 512), dtype=np.int64)
        digest.update(ids.tobytes(order='C'))
        if independent:
            counts = np.stack([np.bincount(row, minlength=512) for row in ids]).astype(np.float64)
            means[lo:lo + len(ids)] = counts @ vectors / 512.
        else:
            means[lo:lo + len(ids)] = vectors[ids].mean(axis=1)
    return means, digest.hexdigest()


def summarize(values, vectors, bootstrap_means):
    selected = values['selected']; intervals = np.quantile(bootstrap_means, [.0125, .9875], axis=0, method='linear')
    primary = []
    for j, name in enumerate(PRIMARY):
        lo, hi = map(float, intervals[:, j])
        primary.append(dict(contrast=name, mean=float(vectors[:, j].mean()), lower=lo, upper=hi,
            interval_level=.975, classification='improvement' if hi < 0 else ('higher_cost' if lo > 0 else 'unresolved')))
    secondary = dict(free_training_change=float((selected[:, :, 1, :, 0] - selected[:, :, 0, :, 0]).mean()),
        benefit_training_change=float(((selected[:, :, 1, :, 0] - selected[:, :, 1, :, 1]) -
                                       (selected[:, :, 0, :, 0] - selected[:, :, 0, :, 1])).mean()),
        selection_change=float(values['t1_actual_free_argmin_sets_differ'].mean()),
        lowest_index_selected_cost_means=values['lowest'].mean(axis=0).tolist(),
        inference='Point descriptions only; no secondary intervals or significance family')
    return dict(primary=primary, joint_actual_specific_decision_improvement=all(r['upper'] < 0 for r in primary),
        secondary=secondary, descriptive_selected_cost_axes=['pool', 'condition', 'objective', 'branch'],
        descriptive_selected_cost_means=selected.mean(axis=0).tolist(),
        tie_count_means=values['tie_counts'].mean(axis=0).tolist(),
        units='Existing physical goal cost: block position squared plus900 times wrapped angle squared',
        uncertainty_scope='Paired recipients conditional on fixed six model pairs and realized donor bank/assignment; no training-population uncertainty.')


def complete_statistics(predicted, physical, *, seed, bit_generator, atol, rtol):
    require(type(atol) in (int, float) and type(rtol) in (int, float) and
            np.isfinite([atol, rtol]).all() and atol >= 0 and rtol >= 0,
            'Explicit frozen numerical comparison tolerances required')
    values = selection_values(predicted, physical); vectors = contrasts(values['selected'])
    selected, lowest, changed, reference_vectors = independent_selections(predicted, physical)
    for a, b in ((values['selected'], selected), (values['lowest'], lowest), (vectors, reference_vectors)):
        np.testing.assert_allclose(a, b, atol=atol, rtol=rtol)
    np.testing.assert_array_equal(values['t1_actual_free_argmin_sets_differ'], changed)
    draws, draw_sha = bootstrap(vectors, seed=seed, bit_generator=bit_generator)
    reference_draws, reference_sha = bootstrap(reference_vectors, seed=seed, bit_generator=bit_generator, independent=True)
    require(draw_sha == reference_sha, 'Different complete shared recipient draws')
    np.testing.assert_allclose(draws, reference_draws, atol=atol, rtol=rtol)
    result = summarize(values, vectors, draws)
    reference = summarize(dict(values, selected=selected, lowest=lowest,
        t1_actual_free_argmin_sets_differ=changed), reference_vectors, reference_draws)
    for a, b in zip(result['primary'], reference['primary'], strict=True):
        np.testing.assert_allclose([a[k] for k in ('mean', 'lower', 'upper')],
                                   [b[k] for k in ('mean', 'lower', 'upper')], atol=atol, rtol=rtol)
        require(a['classification'] == b['classification'], 'Independent interval classification differs')
    require(result['joint_actual_specific_decision_improvement'] == reference['joint_actual_specific_decision_improvement'],
            'Independent joint interpretation differs')
    result.update(bootstrap_seed=seed, bit_generator=bit_generator, bootstrap_iterations=20000,
        percentile_method='linear', shared_recipient_indices_sha256=draw_sha,
        independent_selection_and_statistics_verified=True, atol=atol, rtol=rtol,
        independent_scope='Existing independent exact-tie metrics and count-weighted NumPy resampling; no native model/simulator or QP replay.')
    return result, dict(goal_contrasts=vectors, independent_goal_contrasts=reference_vectors,
        bootstrap_means=draws, independent_bootstrap_means=reference_draws,
        selected_costs=values['selected'], lowest_index_selected_costs=values['lowest'],
        argmin_set_change=changed, exact_tie_counts=values['tie_counts'])


def load_seal_context(binding_pair):
    """Authenticate genuine complete prediction metadata, never physical outcomes.

    These are explicit future target receipts, not artifacts that exist now.
    Upstream acceptance must establish raw/model/head/action/region/rollout
    correctness; this adapter cannot certify those from costs alone.
    """
    b = document(binding_pair)
    require(set(b) == {'status', 'protocol', 'sources', 'prediction_acceptance',
            'expected_recipient_ids', 'expected_donor_ids', 'expected_model_records',
            'expected_candidate_actions_sha256', 'expected_donor_assignment_sha256',
            'expected_outcome_source_sha256'} and b['status'] == 'S3_DECISION_SEAL_INPUT_BINDING_FROZEN',
            'Exact genuine S3 seal-input binding required')
    protocol = document(b['protocol'])
    require(protocol.get('status') == 'S3_SCIENTIFIC_PROTOCOL_FROZEN' and
            protocol.get('scientific_protocol_frozen') is True and
            protocol.get('stage') == 'S3_matched_post_prefix_decision', 'Draft S3 protocol cannot execute')
    spec = protocol['primary_statistics']
    require(spec['contrasts'] == PRIMARY and spec['family_size'] == 2 and spec['iterations'] == 20000 and
            spec['interval_level'] == .975 and spec['percentile_method'] == 'linear' and
            spec['resampling_unit'] == 'recipient' and spec['bootstrap_rng'] == 'PCG64' and
            type(spec['bootstrap_seed']) is int and 0 <= spec['bootstrap_seed'] < 2 ** 32,
            'Frozen primary family, unit, seed or arithmetic changed')
    require(spec.get('verification_tolerances') == dict(atol=1e-9, rtol=1e-12),
            'Prespecified numerical verification tolerances required; no outcome tuning')
    population = protocol['population']
    require(all(population.get(k) == v for k, v in dict(recipient_count=512, donor_count=512,
        pools=[0, 1, 2], groups=[0, 2, 4], conditions=CONDITIONS, objectives=OBJECTIVES,
        models=12, model_pairs=6, branches=BRANCHES).items()), 'Changed fixed decision population')
    decision = protocol['decision_cost']
    require(decision['space'] == 'pose_only' and decision['orientation_weight'] == 900 and
            decision['tie_tolerance'] == 0 and decision['physical_outcomes'] == 49152,
            'Fixed physical cost or exact-tie semantics changed')
    sources = document(b['sources'])
    require(sources.get('status') == 'S3_DECISION_STATISTICS_SOURCES_FROZEN' and
            sources.get('protocol_sha256') == b['protocol']['sha256'], 'Wrong statistics source/protocol')
    source_paths = set(); retained = [binding_pair, b['protocol'], b['sources'], b['prediction_acceptance']]
    for name, digest in sources['files'].items():
        path = Path(name); path = path if path.is_absolute() else ROOT / path
        ref = dict(path=str(path), sha256=digest)
        checked(ref); retained.append(ref); source_paths.add(path.resolve())
    require({Path(__file__).resolve(), METRIC_SOURCE.resolve()} <= source_paths, 'Missing exact executing statistics sources')
    for key in ('expected_candidate_actions_sha256', 'expected_donor_assignment_sha256', 'expected_outcome_source_sha256'):
        require(_hex(b[key]), 'Missing actual action/donor/outcome-source identity')
    validate_models(b['expected_model_records'])
    for key in ('expected_recipient_ids', 'expected_donor_ids'):
        require(len(b[key]) == 512 and all(type(x) is int and x >= 0 for x in b[key]), 'Actual full parent roster required')
    validate_ids(np.asarray(b['expected_recipient_ids'], np.int64), np.asarray(b['expected_donor_ids'], np.int64))
    report = document(b['prediction_acceptance'])
    expected = dict(status=PREDICTION_STATUS, protocol_sha256=b['protocol']['sha256'],
        statistics_sources_sha256=b['sources']['sha256'], axes=AXES,
        model_records=b['expected_model_records'], candidate_actions_sha256=b['expected_candidate_actions_sha256'],
        donor_assignment_sha256=b['expected_donor_assignment_sha256'],
        goal_count=512, pool_count=3, model_count=12, branch_count=4, candidate_count=32,
        accepted_family_count=1536, all_family_checks_passed=True,
        all_native_and_identity_checks_passed=True, independent_gA_cost_checks_passed=True,
        physical_outcomes_opened=False, complete_population_accepted=True, rows_removed=0)
    require(all(report.get(k) == v for k, v in expected.items()), 'Incomplete, wrong-role or unrelated prediction population')
    payloads = report.get('retained_prediction_payloads', [])
    require(payloads and len({x['path'] for x in payloads}) == len(payloads), 'Complete prediction payload provenance required')
    for item in payloads: checked(item)  # Byte identities only; no outcomes or model weights.
    retained.extend(payloads); retained.append(report['arrays'])
    with np.load(checked(report['arrays']), allow_pickle=False) as archive:
        require(len(archive.files) == 4, 'Unexpected prediction archive members')
        data = {key: archive[key] for key in archive.files}
    validate_prediction_arrays(data, b)
    return dict(binding=b, binding_pair=binding_pair, protocol=protocol, spec=spec, sources=sources,
                prediction_report=report, predictions=data, retained=retained)


def _seal_value(ctx):
    return dict(status=SEAL_STATUS, input_binding=ctx['binding_pair'],
        protocol=ctx['binding']['protocol'], sources=ctx['binding']['sources'],
        prediction_acceptance=ctx['binding']['prediction_acceptance'],
        prediction_arrays=ctx['prediction_report']['arrays'], axes=AXES,
        primary_statistics=ctx['spec'], source_sha256=sha(__file__),
        physical_outcomes_opened_by_sealing=False,
        scope='Complete accepted prediction costs/identities sealed before outcome join; upstream parity evidence is authenticated, not independently rerun here.')


def seal_prediction_population(binding_pair, output):
    out = Path(output)
    require(out.is_absolute() and not out.exists() and not out.with_name(out.name + '.intent.json').exists(),
            'Fresh seal output and no prior intent required')
    ctx = load_seal_context(binding_pair)
    write(out.with_name(out.name + '.intent.json'), dict(status='S3_PREDICTION_SEAL_STARTED', input_binding=binding_pair))
    out.mkdir(exist_ok=False)
    for ref in ctx['retained']: checked(ref)
    return write(out / 'PREDICTIONS.lock.json', _seal_value(ctx))


def verify_prediction_seal(seal_pair, expected_binding_pair):
    # External caller identity is verified before any outcome loader exists.
    seal = document(seal_pair)
    require(seal.get('input_binding') == expected_binding_pair, 'Unrelated prediction seal')
    ctx = load_seal_context(expected_binding_pair)
    require(seal == _seal_value(ctx), 'Changed or incomplete prediction seal')
    return ctx


def evaluate_after_seal(seal_pair, expected_binding_pair, outcome_loader):
    """The callback is invoked only after complete seal/source/input validation."""
    ctx = verify_prediction_seal(seal_pair, expected_binding_pair)
    outcome_pair = outcome_loader()  # First physical-outcome receipt access.
    outcome = document(outcome_pair); b = ctx['binding']; data = ctx['predictions']
    expected = dict(status=OUTCOME_STATUS, protocol_sha256=b['protocol']['sha256'],
        sources_sha256=b['expected_outcome_source_sha256'], count=512, pools=[0, 1, 2], candidates=32,
        candidate_actions_sha256=b['expected_candidate_actions_sha256'], complete_population_accepted=True,
        rows_removed=0, all_fixed_candidates_retained=True)
    require(all(outcome.get(k) == v for k, v in expected.items()), 'Incomplete or changed physical outcome population')
    with np.load(checked(outcome['arrays']), allow_pickle=False) as archive:
        require(len(archive.files) == 4 and set(archive.files) == {'terminal_states', 'goal_states',
                'recipient_ids', 'candidate_source_indices'}, 'Unexpected/missing physical outcome fields')
        arrays = {key: archive[key] for key in archive.files}
    ids = array(arrays['recipient_ids'], (512,), 'int64', 'outcome recipient IDs')
    indices = array(arrays['candidate_source_indices'], (512, 3, 32), 'int64', 'outcome candidate IDs')
    require(np.array_equal(ids, data['recipient_ids']) and np.array_equal(indices, data['candidate_source_indices']),
            'Outcome recipient/candidate ordering differs from sealed predictions')
    physical = physical_candidate_costs(arrays['terminal_states'], arrays['goal_states'])
    spec = ctx['spec']
    summary, values = complete_statistics(data['predicted_costs'], physical,
        seed=spec['bootstrap_seed'], bit_generator=spec['bootstrap_rng'], **spec['verification_tolerances'])
    # Retain provenance and recheck after arithmetic, never allow a changed input.
    checked(seal_pair); checked(expected_binding_pair); checked(outcome_pair); checked(outcome['arrays'])
    for ref in ctx['retained']: checked(ref)
    summary.update(status='PASS_COMPLETE_S3_DECISION_STATISTICS_AND_INDEPENDENT_VERIFICATION',
        prediction_seal=seal_pair, input_binding=expected_binding_pair, outcome_acceptance=outcome_pair,
        physical_outcomes_opened_only_after_seal=True, source_sha256=sha(__file__), all512_recipient_scores_retained=True,
        historical_nonexposure_proven=False)
    values.update(physical_candidate_costs=physical, recipient_ids=data['recipient_ids'], donor_ids=data['donor_ids'])
    return summary, values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['seal', 'evaluate'])
    parser.add_argument('--binding', required=True); parser.add_argument('--binding-sha256', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--seal'); parser.add_argument('--seal-sha256')
    parser.add_argument('--outcomes'); parser.add_argument('--outcomes-sha256')
    args = parser.parse_args(); binding = dict(path=args.binding, sha256=args.binding_sha256)
    if args.stage == 'seal':
        require(all(getattr(args, k) is None for k in ('seal', 'seal_sha256', 'outcomes', 'outcomes_sha256')),
                'Seal stage cannot receive an outcome or prior seal argument')
        result = seal_prediction_population(binding, args.output)
    else:
        require(all(getattr(args, k) is not None for k in ('seal', 'seal_sha256', 'outcomes', 'outcomes_sha256')),
                'Actual seal and physical-outcome pairs required')
        out = Path(args.output)
        require(out.is_absolute() and not out.exists() and not out.with_name(out.name + '.intent.json').exists(),
                'Fresh evaluation output required; no overwrite or retry')
        seal = dict(path=args.seal, sha256=args.seal_sha256)
        write(out.with_name(out.name + '.intent.json'), dict(status='S3_DECISION_EVALUATION_STARTED', prediction_seal=seal, input_binding=binding))
        summary, arrays = evaluate_after_seal(seal, binding,
            lambda:dict(path=args.outcomes, sha256=args.outcomes_sha256))
        out.mkdir(exist_ok=False)
        with (out / 'complete_statistics.npz').open('xb') as stream: np.savez_compressed(stream, **arrays)
        summary['complete_arrays'] = pair(out / 'complete_statistics.npz')
        result = write(out / 'report.json', summary)
        (out / 'DONE').write_text('Complete fixed S3 decision statistics, including null and negative outcomes\n')
    print(json.dumps(result, sort_keys=True))


if __name__ == '__main__': main()
