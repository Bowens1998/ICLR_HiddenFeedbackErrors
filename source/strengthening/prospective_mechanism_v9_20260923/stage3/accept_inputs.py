"""Accept the complete new S3 inputs without rerunning models or physics.

Checks retained pixels, every saved common-prefix input, and the generator's
physical output alignment. This is experiment validation, not raw reproduction.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import raw_inputs as raw
import common_prefix as common
import model_runtime as runtime


def load(ref):
    with np.load(runtime.bound_path(ref), allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def pixels(values, keys):
    result = set()
    for key in keys:
        if key not in values:
            continue
        a = values[key]
        raw.require(a.dtype == np.uint8 and a.shape[-3:] == (224, 224, 3), 'Invalid pixel field')
        for frame in a.reshape(-1, 224, 224, 3):
            result.add(hashlib.sha256(frame.tobytes(order='C')).hexdigest())
    return result


def validate_saved(case, inputs, context, actions, physics, action_row, role, pool, outcomes=None):
    values = common.validate_compact(context, actions, physics, action_row, 32 * pool)
    runtime.validate_case(inputs, pool=pool, index=case['index'], seed=case['seed'], role=role)
    expected = common._inputs(context, values, action_row, 32 * pool, physics['pixels'][3])
    if role == 'recipient':
        suffixes, indices = common.legacy().select_suffix_candidates(
            actions['population_actions'], int(actions['selected_index']), case['seed'], 32 * pool)
        expected.update(suffix_actions=suffixes, source_candidate_indices=indices)
    raw.require(set(inputs) == set(expected), 'Extra/missing scorer inputs')
    for key in expected:
        np.testing.assert_array_equal(inputs[key], expected[key])
    if role == 'donor':
        raw.require(outcomes is None and 'outcomes' not in case and
                    case['checks']['donor_suffixes_generated'] is False, 'Donor contains suffix outcomes')
        return
    raw.require(outcomes is not None, 'Complete physical suffix population missing')
    for key, value in [('seed', case['seed']), ('index', case['index']), ('reference_route', 32 * pool)]:
        raw.require(int(outcomes[key]) == value, 'Wrong outcome parent/route')
    for key, value in [('goal_state', context['goal_state']), ('history_states', context['history_states']),
                       ('execution_states', physics['states'][10:16]), ('current_state', physics['states'][15])]:
        np.testing.assert_array_equal(outcomes[key], value)
    states = outcomes['suffix_states']
    raw.require(states.shape == (32, 21, 7) and states.dtype.kind == 'f' and np.isfinite(states).all(),
                'Incomplete suffix-state population')
    np.testing.assert_array_equal(states[:, 0], np.broadcast_to(physics['states'][15], (32, 7)))
    np.testing.assert_allclose(states[0], physics['states'][15:], rtol=0, atol=1e-7)
    np.testing.assert_array_equal(outcomes['terminal_states'], states[:, -1])
    checks = case['checks']
    raw.require(checks['exact_history_replays'] == 32 and checks['exact_common_current_observation'] is True
        and checks['exact_common_execution_states'] is True and checks['selected_terminal_pixels_exact'] is True
        and checks['zero_suffix_fresh_replay_exact'] is True
        and checks['selected_suffix_anchor_max_abs'] <= 1e-7, 'Creation-time common-prefix checks failed')
    for key in ('out_of_view', 'contact_points', 'terminated', 'truncated'):
        raw.require(outcomes[key].shape[0] == 32 and np.isfinite(outcomes[key]).all(), 'Missing candidate flags')
    raw.require(checks['candidates_with_out_of_view'] == int(outcomes['out_of_view'].any(axis=1).sum()) and
        checks['candidates_with_contact'] == int((outcomes['contact_points'] > 0).any(axis=1).sum()),
        'Changed unfiltered physical flags')


def admit(protocol, protocol_sha, sources, sources_sha, output):
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    contexts = {role: raw.context(protocol, protocol_sha, sources, sources_sha, role) for role in raw.ROLES}
    ctx = contexts['recipient']
    files = ctx['sources']['files']
    for path in (Path(__file__), Path(runtime.__file__)):
        raw.require(files.get(str(path.relative_to(raw.ROOT))) == raw.sha(path), 'Unbound input acceptor source')
    prior_acceptance = raw.document(ctx, raw.document(ctx, ctx['sources']['seed_registry'])['raw_acceptance'])
    content_ref = prior_acceptance['content_lineage']
    old_content = runtime.document(content_ref)
    raw.require(old_content['status'] == 'PASS_S2_RETAINED_PIXEL_ISOLATION', 'Wrong prior content receipt')
    historical = set()
    for name, checksum in old_content['input_files_sha256'].items():
        raw.require(raw.sha(name) == checksum, 'Changed historical content source')
        if Path(name).suffix == '.npz':
            historical.update(pixels(load(dict(path=name, sha256=checksum)),
                ('history_pixels', 'goal_pixels', 'terminal_pixels', 'pixels')))
    head_pixels = set()
    for row in old_content['head_caches']:
        values = load({k: row[k] for k in ('path', 'sha256')})['pixel_sha256'].astype(str)
        raw.require(values.ndim == 1, 'Invalid head pixel identities')
        head_pixels.update(values.tolist())
    role_reports = {}; role_pixels = {}; parents = set(); checked_cases = 0
    for role, local in contexts.items():
        bank, role_doc, manifest = raw._bank(local)
        ids = [r['seed'] for r in manifest['cases']]
        raw.require(not parents.intersection(ids), 'Role parents overlap'); parents.update(ids)
        input_ref = raw.pair(local['artifacts'] / 'common_prefix' / role / 'input_manifest.json')
        im = runtime.document(input_ref)
        dummy = dict(status=runtime.ADMISSION_STATUS, protocol_sha256=protocol_sha, role=role, count=512,
            parent_ids=ids, input_manifest=input_ref)
        runtime.validate_admission(dummy, im, role=role, protocol_sha=protocol_sha)
        retained = set(); pool_reports = []
        for pool in range(3):
            folder = local['artifacts'] / 'common_prefix' / role / f'pool_{pool}'
            ref = raw.pair(folder / 'report.json'); report = runtime.document(ref)
            raw.require(report['status'] == 'PASS_S3_COMPLETE_COMMON_PREFIX_POOL' and (folder / 'DONE').is_file()
                and report['parent_ids'] == ids and report['count'] == 512 and report['pool'] == pool
                and all(report.get(k) == v for k, v in local['binding'].items()), 'Incomplete common-prefix route')
            ar = runtime.document(report['action_report']); pr = runtime.document(report['physics_report'])
            raw.require(len(report['cases']) == len(ar['cases']) == len(pr['cases']) == 512, 'Incomplete source population')
            for row, case, a, p, inp in zip(manifest['cases'], report['cases'], ar['cases'], pr['cases'],
                                           im['routes'][pool]['cases'], strict=True):
                index = row['index']
                raw.require(all(x['index'] == index and x['seed'] == row['seed'] for x in (case, a, p, inp))
                    and inp['scorer_input'] == case['scorer_input']
                    and case['bank'] == dict(path=str(bank / f'case_{index:03d}.npz'), sha256=row['sha256'])
                    and case['actions'] == dict(path=str(Path(report['action_report']['path']).parent / a['file']), sha256=a['file_sha256'])
                    and case['physics'] == dict(path=str(Path(report['physics_report']['path']).parent / p['file']), sha256=p['file_sha256'])
                    and p['all_two_replays_exact'] and p['action_file_sha256'] == a['file_sha256'], 'Wrong complete source chain')
                context, actions, physics, inputs = [load(case[k]) for k in ('bank', 'actions', 'physics', 'scorer_input')]
                outcome = load(case['outcomes']) if role == 'recipient' else None
                validate_saved(case, inputs, context, actions, physics, a, role, pool, outcome)
                retained.update(pixels(context, ('history_pixels', 'goal_pixels', 'terminal_pixels')))
                retained.update(pixels(physics, ('pixels',)))
                checked_cases += 1
            pool_reports.append(ref)
        role_pixels[role] = retained
        role_reports[role] = dict(**dummy, s2_completion=ctx['sources']['s2_completion'],
            sources_sha256=sources_sha, pool_reports=pool_reports,
            complete_saved_inputs_checked=True, creator_physics_checks_authenticated=True,
            raw_model_or_physics_regeneration_performed=False)
    overlap = dict(historical={r: len(p & historical) for r, p in role_pixels.items()},
        head={r: len(p & head_pixels) for r, p in role_pixels.items()},
        recipient_donor=len(role_pixels['recipient'] & role_pixels['donor']))
    raw.require(not any(overlap['historical'].values()) and not any(overlap['head'].values())
                and overlap['recipient_donor'] == 0, 'Retained-pixel overlap; entire population blocked')
    content = dict(status='PASS_S3_RETAINED_PIXEL_AND_COMPLETE_SAVED_INPUTS', protocol_sha256=protocol_sha,
        sources_sha256=sources_sha, prior_content=content_ref, overlap=overlap, checked_cases=checked_cases,
        role_unique_pixels={r: len(p) for r, p in role_pixels.items()}, source_sha256=raw.sha(__file__),
        scope='Retained new bank and selected-physics pixels versus prior S1/S2 retained pixels and bound head caches; '
              'not encoder pretraining or unretained historical pixels. No new model or simulator rerun.')
    raw.write_exclusive(output / 'content_and_saved_inputs.json', content)
    for role, report in role_reports.items():
        report['content_and_saved_inputs'] = raw.pair(output / 'content_and_saved_inputs.json')
        raw.write_exclusive(output / (role + '.json'), report)
    raw.write_exclusive(output / 'report.json', dict(status='PASS_S3_BOTH_COMPLETE_INPUT_ROLES',
        protocol_sha256=protocol_sha, sources_sha256=sources_sha,
        admissions={r: raw.pair(output / (r + '.json')) for r in raw.ROLES}, checked_cases=checked_cases))
    (output / 'DONE').write_text('Both complete S3 roles accepted for new model inference\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for name in ('protocol', 'protocol-sha256', 'sources', 'sources-sha256', 'output'):
        p.add_argument('--' + name, required=True)
    a = p.parse_args()
    admit(a.protocol, a.protocol_sha256, a.sources, a.sources_sha256, a.output)
