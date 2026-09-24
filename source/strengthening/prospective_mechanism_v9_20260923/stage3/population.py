"""Complete fixed S3 population: latent pools, sealed costs, then outcomes.

Uses the existing native model, eight-direction projection and statistics
adapters. No extra training or historical experiment reproduction is performed.
"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import numpy as np

import model_runtime as runtime
import decision_statistics as stats
import projection_eight as projection

PHASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PHASE / 'scripts'))
from s1_readout import relu_forward
require, pair, document, write = stats.require, stats.pair, stats.document, stats.write


def arrays(ref):
    with np.load(stats.checked(ref), allow_pickle=False) as z:
        return {key: z[key] for key in z.files}


def save(path, values):
    with Path(path).open('xb') as stream:
        np.savez_compressed(stream, **values)
    return pair(path)


def context(binding_pair):
    b = document(binding_pair)
    require(b['status'] == 'S3_COMPLETE_POPULATION_BINDING_FROZEN', 'Wrong population binding')
    p = document(b['protocol']); runtime.validate_protocol(p, b['s2_completion'])
    source = document(b['statistics_sources'])
    require(source['status'] == 'S3_DECISION_STATISTICS_SOURCES_FROZEN' and
        source['protocol_sha256'] == b['protocol']['sha256'], 'Wrong source protocol')
    resolved = set()
    for name, digest in source['files'].items():
        path = Path(name); path = path if path.is_absolute() else runtime.ROOT / path
        stats.checked(dict(path=str(path), sha256=digest)); resolved.add(path.resolve())
    require({Path(__file__).resolve(), Path(runtime.__file__).resolve(), Path(stats.__file__).resolve(),
        Path(projection.__file__).resolve(), (PHASE / 'scripts/s1_readout.py').resolve()} <= resolved,
        'Missing actual executing source')
    admitted = {}; manifests = {}
    for role in ('recipient', 'donor'):
        c, a, models = runtime.runtime_admission(b['runtimes'][role], role)
        require(c['protocol'] == b['protocol'] and c['admissions'] == b['admissions'], 'Runtime input mismatch')
        admitted[role] = a
        manifests[role] = document(a['input_manifest'])
    require(len(b['heads_A']) == 3, 'Three fixed construction heads required')
    for pool, row in enumerate(b['heads_A']):
        require(row['pool'] == pool and row['group'] == 2 * pool, 'Wrong head pool mapping')
        stats.checked(row['file'])
        require(all(m['head_A_sha256'] == row['file']['sha256'] for m in models if m['pool'] == pool),
            'Readout differs from fixed training readout')
    assignment = np.random.Generator(np.random.PCG64(p['feedback_family']['donor_assignment_seed'])).permutation(512).astype(np.int64)
    return b, p, admitted, manifests, assignment


def predicted_map(roots, i):
    return {c: {o: roots[ci, oi, i] for oi, o in enumerate(stats.OBJECTIVES)}
            for ci, c in enumerate(stats.CONDITIONS)}


def latent_pool(binding_pair, pool):
    b, p, admitted, manifests, assignment = context(binding_pair)
    require(type(pool) is int and pool in range(3), 'Wrong fixed pool')
    out = Path(b['output_root']) / 'latent' / f'pool_{pool}'
    out.mkdir(parents=True, exist_ok=False)
    write(out / 'STARTED.json', dict(input_binding=binding_pair, pool=pool, job=os.environ['SLURM_JOB_ID']))
    started = time.monotonic(); timings = {}; heads = arrays(b['heads_A'][pool]['file'])
    roots = np.empty((2, 2, 512, 192), np.float32)
    observed = np.empty((512, 192), np.float32); goals = np.empty_like(observed)
    histories = np.empty((512, 3, 192), np.float32)
    donor_tokens = np.empty_like(observed); records = {}; models = []
    rp = b['runtimes']['recipient']; dp = b['runtimes']['donor']
    import torch
    torch.cuda.reset_peak_memory_stats()
    for ci, condition in enumerate(stats.CONDITIONS):
        for oi, objective in enumerate(stats.OBJECTIVES):
            h = runtime.load_recipient_model(pool, objective, condition,
                runtime_contract=rp['path'], runtime_contract_sha256=rp['sha256'])
            models.append(dict(pool=pool, group=2*pool, condition=condition, objective=objective,
                checkpoint_sha256=h.spec['checkpoint']['sha256'], action_normalization_sha256=h.normalization_sha256,
                head_A_sha256=h.spec['head_A_sha256']))
            records[ci, oi] = []
            for i in range(512):
                ref = admitted['recipient']['cases'][pool, i]['scorer_input']
                r = runtime.native_recipient_root(h, case_index=i, input_pair=ref)
                a = r['arrays']; roots[ci, oi, i] = a['predicted']
                if ci == oi == 0:
                    observed[i], histories[i], goals[i] = a['observed'], a['initial_history'], a['goal_token']
                else:
                    for left, right in ((observed[i], a['observed']), (histories[i], a['initial_history']), (goals[i], a['goal_token'])):
                        np.testing.assert_array_equal(left, right)
                records[ci, oi].append(r)
            del h; gc.collect(); torch.cuda.empty_cache()
            print(json.dumps(dict(stage='roots', pool=pool, condition=condition, objective=objective, count=512)), flush=True)
    require(len({x['action_normalization_sha256'] for x in models}) == 1, 'Pool action normalizers differ')
    h = runtime.load_donor_encoder(pool, runtime_contract=dp['path'], runtime_contract_sha256=dp['sha256'])
    require(h.normalization_sha256 == models[0]['action_normalization_sha256'], 'Donor normalization lineage differs')
    for i in range(512):
        r = runtime.encode_donor_current(h, case_index=i, input_pair=admitted['donor']['cases'][pool, i]['scorer_input'])
        donor_tokens[i] = r['arrays']['observed']
    donor_evidence = dict(spec=h.spec, model_state_sha256=h.model_sha256, normalization_sha256=h.normalization_sha256)
    del h; gc.collect(); torch.cuda.empty_cache()
    timings['native_roots_and_donor_seconds'] = time.monotonic() - started
    root_file = save(out / 'roots.npz', dict(predicted=roots, observed=observed, goals=goals,
        histories=histories, donor_tokens=donor_tokens, donor_assignment=assignment,
        recipient_ids=np.asarray(admitted['recipient']['parent_ids'], np.int64),
        donor_ids=np.asarray(admitted['donor']['parent_ids'], np.int64)))
    root_evidence = []
    for ci in range(2):
        for oi in range(2):
            root_evidence.append(dict(condition=stats.CONDITIONS[ci], objective=stats.OBJECTIVES[oi],
                cases=[{k: v for k, v in r.items() if k != 'arrays'} for r in records[ci, oi]]))
    write(out / 'root_evidence.json', dict(models=models, roots=root_evidence, donor=donor_evidence))
    replacement = np.empty((512, 2, 2, 2, 192), np.float32)
    directions = np.empty_like(replacement, dtype=np.float64); family_reports = []; failures = []
    begun = time.monotonic()
    for i in range(512):
        anchors = predicted_map(roots, i); guides = dict(actual=observed[i], donor=donor_tokens[assignment[i]])
        try:
            repl, delta, report = projection.project_matched_eight_family(anchors, guides, heads,
                pool=pool, goal_index=i, donor_index=int(assignment[i]))
            accepted = projection.accept_matched_eight_family(anchors, guides, heads, repl, delta, report,
                pool=pool, goal_index=i, donor_assignment=assignment)
            replacement[i], directions[i] = repl, delta
            family_reports.append(dict(index=i, projection=report, acceptance=accepted))
        except Exception as exc:
            failures.append(dict(index=i, error=repr(exc), detail=getattr(exc, 'detail', None)))
        if i % 64 == 63:
            print(json.dumps(dict(stage='projection', pool=pool, attempted=i+1, failures=len(failures))), flush=True)
    write(out / 'family_reports.json', dict(count=512, families=family_reports, failures=failures))
    require(not failures and len(family_reports) == 512, 'Complete eight-direction population blocked; no exclusions')
    projection_file = save(out / 'projection.npz', dict(replacements=replacement, directions=directions))
    timings['projection_seconds'] = time.monotonic() - begun
    terminal = np.empty((512, 2, 2, 4, 32, 192), np.float32)
    suffix_evidence = []; actions = np.empty((512, 32, 20, 2), np.float32)
    candidate_indices = np.empty((512, 32), np.int64); begun = time.monotonic()
    for ci, condition in enumerate(stats.CONDITIONS):
        for oi, objective in enumerate(stats.OBJECTIVES):
            h = runtime.load_recipient_model(pool, objective, condition,
                runtime_contract=rp['path'], runtime_contract_sha256=rp['sha256'])
            evidence = []
            for i in range(512):
                ref = admitted['recipient']['cases'][pool, i]['scorer_input']
                r = runtime.suffix_rollouts(h, case_index=i, input_pair=ref, root_record=records[ci, oi][i],
                    replacements=dict(actual=replacement[i, ci, oi, 0], donor=replacement[i, ci, oi, 1]))
                terminal[i, ci, oi] = np.stack([r['arrays'][branch][-1] for branch in stats.BRANCHES])
                evidence.append({k: v for k, v in r.items() if k != 'arrays'})
                if ci == oi == 0:
                    case = arrays(ref); actions[i] = case['suffix_actions']; candidate_indices[i] = case['source_candidate_indices']
            suffix_evidence.append(dict(condition=condition, objective=objective, cases=evidence))
            del h; gc.collect(); torch.cuda.empty_cache()
            print(json.dumps(dict(stage='suffixes', pool=pool, condition=condition, objective=objective, count=512)), flush=True)
    payload = save(out / 'terminal_tokens.npz', dict(terminal=terminal, goals=goals,
        candidate_actions=actions, candidate_source_indices=candidate_indices))
    write(out / 'suffix_evidence.json', suffix_evidence)
    timings['suffix_seconds'] = time.monotonic() - begun
    report = dict(status='PASS_S3_COMPLETE_LATENT_POOL', binding=binding_pair, pool=pool, count=512,
        models=models, roots=root_file, projection=projection_file, terminal=payload,
        root_evidence=pair(out / 'root_evidence.json'), suffix_evidence=pair(out / 'suffix_evidence.json'),
        family_reports=pair(out / 'family_reports.json'), accepted_families=512, scientific_rows_removed=0,
        physical_outcomes_opened=False, all_native_and_identity_checks_passed=True,
        timings=timings, elapsed_seconds=time.monotonic()-started,
        cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(), gpu=torch.cuda.get_device_name(),
        slurm_job_id=os.environ['SLURM_JOB_ID'], source_sha256=stats.sha(__file__))
    write(out / 'report.json', report); (out / 'DONE').write_text('Complete new S3 latent population\n')


def decode_checked(tokens, head):
    """Existing FP64 ReLU decoder plus separate transposed dense arithmetic."""
    tokens = np.asarray(tokens); flat = tokens.reshape(-1, 192); result = np.empty((len(flat), 6), np.float64)
    for start in range(0, len(flat), 512):
        z = flat[start:start+512]
        y = relu_forward(head, z, physical=True)
        x = ((np.asarray(z, np.float64) - head['mean']) / head['scale']).T
        a = np.maximum(head['0.weight'] @ x + head['0.bias'][:, None], 0)
        c = np.maximum(head['2.weight'] @ a + head['2.bias'][:, None], 0)
        ref = (head['4.weight'] @ c + head['4.bias'][:, None]).T * head['target_scale'] + head['target_mean']
        np.testing.assert_allclose(y, ref, atol=1e-9, rtol=1e-12)
        result[start:start+len(z)] = y
    return result.reshape(tokens.shape[:-1] + (6,))


def predicted_costs(tokens, goal_tokens, head):
    pose = decode_checked(tokens, head); goal = decode_checked(goal_tokens, head)[:, None, None, None, None]
    angle = np.arctan2(pose[..., 4], pose[..., 5]) - np.arctan2(goal[..., 4], goal[..., 5])
    cost = np.square(pose[..., 2:4]-goal[..., 2:4]).sum(-1) + 900*np.angle(np.exp(1j*angle))**2
    ref = ((pose[..., 2]-goal[..., 2])**2 + (pose[..., 3]-goal[..., 3])**2
        + 900*np.arctan2(np.sin(angle), np.cos(angle))**2)
    np.testing.assert_allclose(cost, ref, atol=1e-9, rtol=1e-12)
    require(np.isfinite(cost).all(), 'Nonfinite fixed cost')
    return cost


def seal(binding_pair):
    b, p, admitted, manifests, assignment = context(binding_pair)
    root = Path(b['output_root']); out = root / 'predictions'; out.mkdir(parents=True, exist_ok=False)
    reports = []; payloads = []; model_records = []; cost = np.empty(stats.COST_SHAPE, np.float64)
    candidates = np.empty((512, 3, 32), np.int64); action_hash = hashlib.sha256()
    # Authenticate all three complete populations/families before decoding even one cost.
    for pool in range(3):
        folder = root / 'latent' / f'pool_{pool}'; ref = pair(folder / 'report.json'); r = document(ref)
        require(r['status'] == 'PASS_S3_COMPLETE_LATENT_POOL' and r['binding'] == binding_pair
            and r['pool'] == pool and r['count'] == r['accepted_families'] == 512
            and r['scientific_rows_removed'] == 0 and r['physical_outcomes_opened'] is False
            and r['all_native_and_identity_checks_passed'] and (folder / 'DONE').is_file(), 'Incomplete latent population')
        v, q = arrays(r['roots']), arrays(r['projection']); families = document(r['family_reports'])
        require(not families['failures'] and len(families['families']) == 512, 'Unaccepted family')
        np.testing.assert_array_equal(v['donor_assignment'], assignment)
        for role in ('recipient', 'donor'):
            np.testing.assert_array_equal(v[role+'_ids'], np.asarray(admitted[role]['parent_ids'], np.int64))
        head = arrays(b['heads_A'][pool]['file'])
        for i, f in enumerate(families['families']):
            require(f['index'] == i, 'Reordered family')
            accepted = projection.accept_matched_eight_family(predicted_map(v['predicted'], i),
                dict(actual=v['observed'][i], donor=v['donor_tokens'][assignment[i]]), head,
                q['replacements'][i], q['directions'][i], f['projection'], pool=pool,
                goal_index=i, donor_assignment=assignment)
            require(accepted == f['acceptance'], 'Changed complete numerical family receipt')
        reports.append(r); model_records.extend(r['models']); payloads.append(ref)
        for key in ('roots', 'projection', 'terminal', 'root_evidence', 'suffix_evidence', 'family_reports'):
            stats.checked(r[key]); payloads.append(r[key])
    stats.validate_models(model_records)
    for pool, report in enumerate(reports):
        data = arrays(report['terminal']); head = arrays(b['heads_A'][pool]['file'])
        require(data['terminal'].shape == (512, 2, 2, 4, 32, 192) and data['terminal'].dtype == np.float32,
            'Incomplete terminal predictions')
        cost[:, pool] = predicted_costs(data['terminal'], data['goals'], head)
        candidates[:, pool] = data['candidate_source_indices']
        action_hash.update(np.ascontiguousarray(data['candidate_actions']).tobytes())
    prediction_arrays = save(out / 'costs.npz', dict(predicted_costs=cost, candidate_source_indices=candidates,
        recipient_ids=np.asarray(admitted['recipient']['parent_ids'], np.int64),
        donor_ids=np.asarray(admitted['donor']['parent_ids'], np.int64)[assignment]))
    report = dict(status=stats.PREDICTION_STATUS, protocol_sha256=b['protocol']['sha256'],
        statistics_sources_sha256=b['statistics_sources']['sha256'], axes=stats.AXES, model_records=model_records,
        candidate_actions_sha256=action_hash.hexdigest(), donor_assignment_sha256=runtime.array_sha(assignment),
        goal_count=512, pool_count=3, model_count=12, branch_count=4, candidate_count=32,
        accepted_family_count=1536, all_family_checks_passed=True, all_native_and_identity_checks_passed=True,
        independent_gA_cost_checks_passed=True, physical_outcomes_opened=False, complete_population_accepted=True,
        rows_removed=0, retained_prediction_payloads=payloads, arrays=prediction_arrays, population_binding=binding_pair)
    prediction_ref = write(out / 'acceptance.json', report)
    seal_binding = dict(status='S3_DECISION_SEAL_INPUT_BINDING_FROZEN', protocol=b['protocol'], sources=b['statistics_sources'],
        prediction_acceptance=prediction_ref, expected_recipient_ids=list(admitted['recipient']['parent_ids']),
        expected_donor_ids=np.asarray(admitted['donor']['parent_ids'], np.int64)[assignment].tolist(),
        expected_model_records=model_records, expected_candidate_actions_sha256=action_hash.hexdigest(),
        expected_donor_assignment_sha256=runtime.array_sha(assignment), expected_outcome_source_sha256=b['statistics_sources']['sha256'])
    bp = write(out / 'seal_binding.json', seal_binding)
    sealed = stats.seal_prediction_population(bp, str(root / 'prediction_seal'))
    write(out / 'COMPLETE.json', dict(status='COMPLETE_S3_PREDICTIONS_AND_SEAL', seal=sealed, binding=bp))
    (out / 'DONE').write_text('All candidate predictions sealed before outcomes\n')


def outcomes(binding_pair):
    b, p, admitted, manifests, assignment = context(binding_pair)
    root = Path(b['output_root']); completed = document(pair(root / 'predictions/COMPLETE.json'))
    ctx = stats.verify_prediction_seal(completed['seal'], completed['binding'])
    out = root / 'outcomes'; out.mkdir(parents=True, exist_ok=False)
    # Outcome arrays are first loaded here, after the complete externally saved seal.
    terminal = np.empty((512, 3, 32, 7), np.float64); goals = np.empty((512, 7), np.float64)
    candidates = np.empty((512, 3, 32), np.int64); action_hash = hashlib.sha256(); receipts = []
    raw_root = Path(p['raw_inputs']['output_root']) / 'common_prefix/recipient'
    for pool in range(3):
        report_ref = pair(raw_root / f'pool_{pool}/report.json'); report = document(report_ref); receipts.append(report_ref)
        require(report['status'] == 'PASS_S3_COMPLETE_COMMON_PREFIX_POOL' and report['count'] == 512 and
            report['parent_ids'] == list(admitted['recipient']['parent_ids']), 'Incomplete fixed physical outcomes')
        candidate_actions = []
        for i, row in enumerate(report['cases']):
            require(row['index'] == i and row['seed'] == admitted['recipient']['parent_ids'][i] and
                row['scorer_input'] == admitted['recipient']['cases'][pool, i]['scorer_input'], 'Outcome/input mismatch')
            physical, inputs = arrays(row['outcomes']), arrays(row['scorer_input'])
            terminal[i, pool] = physical['terminal_states']; candidates[i, pool] = inputs['source_candidate_indices']
            candidate_actions.append(inputs['suffix_actions'])
            if pool == 0: goals[i] = physical['goal_state']
            else: np.testing.assert_array_equal(goals[i], physical['goal_state'])
        action_hash.update(np.ascontiguousarray(np.stack(candidate_actions)).tobytes())
    require(action_hash.hexdigest() == ctx['binding']['expected_candidate_actions_sha256'], 'Different physical candidate actions')
    ref = save(out / 'complete_physical_states.npz', dict(terminal_states=terminal, goal_states=goals,
        recipient_ids=np.asarray(admitted['recipient']['parent_ids'], np.int64), candidate_source_indices=candidates))
    outcome_ref = write(out / 'acceptance.json', dict(status=stats.OUTCOME_STATUS, protocol_sha256=b['protocol']['sha256'],
        sources_sha256=b['statistics_sources']['sha256'], count=512, pools=[0, 1, 2], candidates=32,
        candidate_actions_sha256=action_hash.hexdigest(), complete_population_accepted=True, rows_removed=0,
        all_fixed_candidates_retained=True, arrays=ref, generator_reports=receipts,
        input_acceptance=b['admissions']['recipient'], prediction_seal=completed['seal']))
    summary, values = stats.evaluate_after_seal(completed['seal'], completed['binding'], lambda: outcome_ref)
    final = root / 'final_statistics'; final.mkdir(exist_ok=False)
    summary['complete_arrays'] = save(final / 'statistics.npz', values)
    write(final / 'report.json', summary); (final / 'DONE').write_text('Complete S3 result including negative outcomes\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('stage', choices=('latent-pool', 'seal', 'outcomes'))
    p.add_argument('--binding', required=True); p.add_argument('--binding-sha256', required=True)
    p.add_argument('--pool', type=int)
    a = p.parse_args(); require(os.environ.get('SLURM_JOB_ID', '').isdigit(), 'Actual allocation required')
    ref = dict(path=a.binding, sha256=a.binding_sha256)
    if a.stage == 'latent-pool': latent_pool(ref, a.pool)
    elif a.stage == 'seal': seal(ref)
    else: outcomes(ref)
