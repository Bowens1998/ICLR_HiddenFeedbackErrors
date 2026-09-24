"""Complete frozen eight-member projection shards; no measurement D or outcomes."""
import argparse
from pathlib import Path
import time

import numpy as np

from s1_common import atomic_json, atomic_npz, checked_json, load_protocol, namespace_seed, sha
from s1_projection import ProjectionFailure, project_eight_family


def validate_qualification(selected, qualification, protocol_sha, selected_sha, a_sha):
    roster = {(g,h) for g in [0,1] for h in ['C','D']}
    heads, rows = selected.get('heads', []), qualification.get('rows', [])
    if (selected.get('status') != 'ALL_FOUR_S1_HEADS_FROZEN_BEFORE_QUALIFICATION' or
            selected.get('protocol_sha256') != protocol_sha or len(heads) != 4 or
            {(r['group'],r['head_role']) for r in heads} != roster):
        raise ValueError('Incomplete frozen head roster')
    if (qualification.get('status') != 'COMPLETE_S1_QUALIFICATION_PASS' or
            qualification.get('all_gates_passed') is not True or qualification.get('all_four_completed') is not True or
            qualification.get('selected_lock_sha256') != selected_sha or
            qualification.get('protocol_sha256') != protocol_sha or
            qualification.get('a_bindings_sha256') != a_sha or len(rows) != 4 or
            {(r['group'],r['head_role']) for r in rows} != roster):
        raise ValueError('Every C/D qualification gate must pass before projection')
    hashes={(r['group'],r['head_role']):r['checkpoint_sha256'] for r in heads}
    for row in rows:
        if row['checkpoint_sha256'] != hashes[(row['group'],row['head_role'])] or row['gate']['passed'] is not True:
            raise ValueError('Qualification row does not match selected checkpoint')
        checks=row['gate']['checks']
        if set(checks) != {'expert','planner'}:
            raise ValueError('Missing qualification domain')
        for domain in checks.values():
            if set(domain) != {'six_normalized_mse','block_position_mse','agent_position_mse','wrapped_angle_mse'}:
                raise ValueError('Missing qualification metric')
            if any(v['passed'] is not True or v['ratio'] is None or not np.isfinite(v['ratio']) or not 0<=v['ratio']<=1.1 for v in domain.values()):
                raise ValueError('Failed qualification metric')


def read_cache(path, checksum, protocol_sha, kind, role, group, stream, count):
    r = checked_json(path, checksum)
    expected = dict(status='PASS_S1_OBSERVED_AND_FREE_CACHE', protocol_sha256=protocol_sha,
                    kind=kind, role=role, group=group, stream=stream, count=count)
    if any(r.get(k) != v for k, v in expected.items()):
        raise ValueError('Wrong, incomplete or unaccepted cache')
    objectives=['decoded_teacher','physical_labels'] if role=='recipient' else ['decoded_teacher']
    if r.get('objectives') != objectives or r.get('frozen_tensors_unchanged') is not True:
        raise ValueError('Changed objective order or model state')
    if role=='recipient' and r.get('native_endpoint_and_identity_replacement_exact') is not True:
        raise ValueError('Native arithmetic and identity parity must pass')
    a = r['arrays']
    if sha(a['path']) != a['sha256']:
        raise ValueError('Changed cache array')
    with np.load(a['path'], allow_pickle=False) as f:
        # Do not open future truth, pose predictions or any evaluator output.
        full_observed=f['observed']
        if full_observed.shape!=(count,5,192) or full_observed.dtype!=np.float32:
            raise ValueError('Changed complete observed-token interface')
        observed = full_observed[:, 0].copy()
        free=None
        if role=='recipient':
            full_free=f['free']
            if full_free.shape!=(2,count,5,192) or full_free.dtype!=np.float32:
                raise ValueError('Changed complete free-token interface')
            free=full_free[:,:,0].copy()
        elif 'free' in f.files:
            raise ValueError('Donor cache must not generate model predictions')
        seeds = f['seeds'].copy()
    if observed.shape != (count, 192) or not np.isfinite(observed).all():
        raise ValueError('Invalid guidance tokens')
    if role == 'recipient' and (free.shape != (2, count, 192) or not np.isfinite(free).all()):
        raise ValueError('Invalid free insertion tokens')
    if seeds.shape != (count,) or seeds.dtype!=np.int64 or len(np.unique(seeds)) != count:
        raise ValueError('Missing or repeated parents')
    return observed, free, seeds


def main():
    p = argparse.ArgumentParser()
    for key in ('protocol', 'selected-lock', 'qualification', 'a-bindings', 'recipient-cache', 'donor-cache'):
        p.add_argument('--' + key, required=True)
        p.add_argument('--' + key + '-sha256', required=True)
    p.add_argument('--kind', choices=['development', 'confirmation'], required=True)
    p.add_argument('--group', choices=[0, 1], type=int, required=True)
    p.add_argument('--stream', choices=range(4), type=int, required=True)
    p.add_argument('--start', type=int, required=True)
    p.add_argument('--stop', type=int, required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    cfg = load_protocol(a.protocol, a.protocol_sha256)
    selected = checked_json(a.selected_lock, a.selected_lock_sha256)
    qualification = checked_json(a.qualification, a.qualification_sha256)
    validate_qualification(selected, qualification, a.protocol_sha256, a.selected_lock_sha256, a.a_bindings_sha256)
    references = checked_json(a.a_bindings, a.a_bindings_sha256)
    if qualification.get('a_bindings_sha256') != a.a_bindings_sha256:
        raise ValueError('Reference head binding mismatch')
    ref = next(r for r in references['groups'] if r['group'] == a.group)['head_A']
    new = [r for r in selected['heads'] if r['group'] == a.group and r['head_role'] == 'C']
    if len(new) != 1 or sha(ref['path']) != ref['sha256'] or sha(new[0]['checkpoint']) != new[0]['checkpoint_sha256']:
        raise ValueError('Changed or ambiguous construction head')
    with np.load(ref['path'], allow_pickle=False) as z:
        head_A = dict(z)
    with np.load(new[0]['checkpoint'], allow_pickle=False) as z:
        head_C = dict(z)
    count = cfg['scores']['goal_count'] if a.kind == 'confirmation' else cfg['scores']['development_goal_count']
    if not 0 <= a.start < a.stop <= count:
        raise ValueError('Invalid fixed shard range')
    observed, free, seeds = read_cache(a.recipient_cache, a.recipient_cache_sha256, a.protocol_sha256,
                                     a.kind, 'recipient', a.group, a.stream, count)
    donors, _, donor_seeds = read_cache(a.donor_cache, a.donor_cache_sha256, a.protocol_sha256,
                                      a.kind, 'donor', a.group, a.stream, count)
    if set(seeds.tolist()) & set(donor_seeds.tolist()):
        raise ValueError('Recipient/donor parent overlap')
    ns = cfg['donor_assignment']['seed_namespace'].format(kind=a.kind, stream=a.stream)
    permutation = np.random.default_rng(namespace_seed(cfg['root_seed'], ns)).permutation(count)
    n = a.stop - a.start
    replacements = np.empty((2, 2, n, 2, 192), np.float32)
    directions = np.empty_like(replacements, dtype=np.float64)
    common_norm = np.empty(n, np.float64)
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=False)
    identity = dict(protocol_sha256=a.protocol_sha256, kind=a.kind, group=a.group, stream=a.stream,
                    count=count, start=a.start, stop=a.stop, selected_lock_sha256=a.selected_lock_sha256,
                    qualification_sha256=a.qualification_sha256, a_bindings_sha256=a.a_bindings_sha256,
                    recipient_cache_sha256=a.recipient_cache_sha256, donor_cache_sha256=a.donor_cache_sha256,
                    donor_seed_namespace=ns, head_A_sha256=ref['sha256'], head_C_sha256=new[0]['checkpoint_sha256'])
    reports = []
    started = time.monotonic()
    for local, goal in enumerate(range(a.start, a.stop)):
        pred = {o: free[j, goal] for j, o in enumerate(cfg['objectives'])}
        guides = dict(actual=observed[goal], donor=donors[permutation[goal]])
        try:
            tokens, raw, report = project_eight_family(pred, guides, head_A, head_C)
        except ProjectionFailure as exc:
            atomic_json(out / 'FAILURE.json', dict(status='BLOCKED_COMPLETE_FAMILY', **identity,
                        goal=goal, error=str(exc), detail=exc.detail, accepted_previous_families=len(reports)))
            raise
        for j, member in enumerate(report['members']):
            oi = cfg['objectives'].index(member['objective'])
            ci = cfg['scores']['constraints'].index(member['condition'])
            si = cfg['projection']['sources'].index(member['source'])
            replacements[oi, ci, local, si] = tokens[j]
            directions[oi, ci, local, si] = raw[j]
        common_norm[local] = report['effective_norm']
        reports.append(dict(goal=goal, recipient_seed=int(seeds[goal]), donor_index=int(permutation[goal]),
                            donor_seed=int(donor_seeds[permutation[goal]]), **report))
        atomic_json(out / 'progress.json', dict(**identity, accepted_families=len(reports)))
    arrays = out / 'projections.npz'
    atomic_npz(arrays, replacements=replacements, directions=directions, common_norm=common_norm,
               goal_indices=np.arange(a.start, a.stop, dtype=np.int64),
               donor_indices=permutation[a.start:a.stop].astype(np.int64))
    atomic_json(out / 'report.json', dict(status='ACCEPTED_COMPLETE_SHARD', **identity,
                arrays=dict(path=str(arrays.resolve()), sha256=sha(arrays)),
                matching_reports=reports, elapsed_seconds=time.monotonic()-started,
                source_sha256=sha(__file__), scope='No reserved D weights, predictions or future truth opened'))
    (out / 'DONE').write_text('complete_eight_member_families\n')


if __name__ == '__main__':
    main()
