"""Independent saved-array development checks; never load a readout or score effects.

Numerical projection feasibility is inherited from the separately executed QP
acceptor. This checker independently checks complete actual saved-array linkage,
branch identities, donor pairing and byte-exact insertion. It does not rerun the
image encoder, simulator, QP solver or world model.
"""
import argparse
import datetime
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def read(path, expected=None):
    if expected is not None and sha(path) != expected:
        raise ValueError('Changed input: ' + str(path))
    return json.loads(Path(path).read_text())


def require(ok, message):
    if not ok:
        raise ValueError(message)


def array(path, checksum):
    require(sha(path) == checksum, 'Changed array: ' + str(path))
    with np.load(path, allow_pickle=False) as z:
        return dict(z)


def check_route(g, s, root, binding, psha, ssha, selected_sha, qual_sha, kernels, a_sha):
    n = 64
    caches = {}
    ids = {}
    identities = {}
    for role in ('recipient', 'donor'):
        entry = next(x for x in binding['reports']
                     if (x['role'], x['group'], x['stream']) == (role, g, s))
        report = read(entry['report'], entry['report_sha256'])
        expected = dict(status='PASS_S1_OBSERVED_AND_FREE_CACHE', kind='development',
                        role=role, group=g, stream=s, count=n,
                        protocol_sha256=psha, sources_sha256=ssha,
                        cases=list(range(n)), donor_model_predictions_generated=False,
                        objectives=['decoded_teacher', 'physical_labels'] if role == 'recipient' else ['decoded_teacher'],
                        source_sha256=kernels['cache_inputs.py'])
        require(all(report.get(k) == v for k, v in expected.items()), 'Cache identity')
        require(report['frozen_tensors_unchanged'] is True, 'Model mutated')
        if role == 'recipient':
            require(report['native_endpoint_and_identity_replacement_exact'] is True,
                    'Native parity receipt missing')
        data = array(report['arrays']['path'], report['arrays']['sha256'])
        fields = {'initial', 'observed', 'truth', 'known_actions', 'seeds'}
        require(set(data) == (fields | {'free'} if role == 'recipient' else fields),
                'Unexpected cache fields or donor predictions')
        for name, shape, dtype in (('initial', (n, 3, 192), np.float32),
                                   ('truth', (n, 5, 6), np.float64),
                                   ('known_actions', (n, 10), np.float32)):
            require(data[name].shape == shape and data[name].dtype == dtype and
                    np.isfinite(data[name]).all(), 'Cache schema: ' + name)
        require(data['seeds'].shape == (n,) and data['seeds'].dtype == np.int64,
                'Bad seeds')
        require(len(set(data['seeds'].tolist())) == n, 'Repeated parent')
        require(data['observed'].shape == (n, 5, 192) and
                data['observed'].dtype == np.float32 and
                np.isfinite(data['observed']).all(), 'Observed tokens')
        caches[role] = (report, data, entry)
        ids[role] = data['seeds']
        identities[entry['report']] = entry['report_sha256']
        identities[report['arrays']['path']] = report['arrays']['sha256']
    require(not set(ids['recipient']) & set(ids['donor']), 'Cross-role parent overlap')
    cr, cache, entry = caches['recipient']
    require(cache['free'].shape == (2, n, 5, 192) and
            cache['free'].dtype == np.float32 and np.isfinite(cache['free']).all(),
            'Free cache shape/precision')
    qpath = root / 'projection_acceptance' / f'group_{g}_stream_{s}' / 'QP_LOCK.json'
    qp = read(qpath)
    expected = dict(status='ALL_S1_QP_SHARDS_ACCEPTED', kind='development',
                    group=g, stream=s, count=n, protocol_sha256=psha,
                    selected_lock_sha256=selected_sha, qualification_sha256=qual_sha,
                    recipient_cache_sha256=entry['report_sha256'],
                    donor_cache_sha256=caches['donor'][2]['report_sha256'],
                    all_goals_covered=True, source_sha256=kernels['freeze_qp_shards.py'])
    require(all(qp.get(k) == v for k, v in expected.items()), 'QP lock identity')
    require(len(qp['shards']) == 1 and np.isfinite(qp['maximum_rechecked_output_deviation'])
            and 0 <= qp['maximum_rechecked_output_deviation'] <= 1e-6,
            'Incomplete QP or infeasible output receipt')
    shard = qp['shards'][0]
    require((shard['start'], shard['stop']) == (0, n), 'Incomplete development shard')
    qr = read(shard['report'], shard['report_sha256'])
    required_qr = {k: v for k, v in expected.items()
                   if k not in ('all_goals_covered', 'status', 'source_sha256')}
    required_qr.update(status='ACCEPTED_COMPLETE_SHARD', start=0, stop=n,
                       a_bindings_sha256=a_sha, source_sha256=kernels['project_s1.py'])
    require(all(qr.get(k) == v for k, v in required_qr.items()) and
            len(qr['matching_reports']) == n, 'Missing or misbound projection families')
    require(qr['arrays'] == dict(path=shard['arrays'], sha256=shard['arrays_sha256']),
            'QP report/payload ancestry mismatch')
    q = array(shard['arrays'], shard['arrays_sha256'])
    require(q['replacements'].shape == (2, 2, n, 2, 192) and
            q['replacements'].dtype == np.float32 and
            np.isfinite(q['replacements']).all(), 'Bad replacements')
    require(q['goal_indices'].dtype == np.int64 and q['donor_indices'].dtype == np.int64,
            'QP index precision')
    np.testing.assert_array_equal(q['goal_indices'], np.arange(n, dtype=np.int64))
    seed = int.from_bytes(hashlib.sha256(
        f'20260923:v9_s1_donor/development/stream_{s}'.encode()).digest()[:4], 'big')
    perm = np.random.Generator(np.random.PCG64(seed)).permutation(n)
    np.testing.assert_array_equal(q['donor_indices'], perm)
    require(q['common_norm'].shape == (n,) and np.isfinite(q['common_norm']).all()
            and (q['common_norm'] >= 0).all(), 'Invalid dose')
    for i, family in enumerate(qr['matching_reports']):
        require(family['goal'] == i and family['family_size'] == 8 and
                family['status'] == 'ACCEPTED_S1_EIGHT_MEMBER_FAMILY' and
                family['recipient_seed'] == int(ids['recipient'][i]) and
                family['donor_seed'] == int(ids['donor'][perm[i]]) and
                family['effective_norm'] == q['common_norm'][i], 'Family identity')
    rrpath = root / 'rollout' / 'recipient' / f'group_{g}_stream_{s}' / 'report.json'
    rr = read(rrpath)
    expected = dict(status='PASS_S1_COMPLETE_ACCEPTED_ROLLOUT', kind='development',
                    role='recipient', group=g, stream=s, count=n,
                    protocol_sha256=psha, sources_sha256=ssha,
                    selected_lock_sha256=selected_sha, qualification_sha256=qual_sha,
                    objectives=['decoded_teacher', 'physical_labels'],
                    constraints=['A', 'AC'], branches=['free', 'actual', 'donor', 'reset'],
                    horizons=[5, 10, 15, 20, 25], head_weights_opened=False,
                    free_and_reset_bitwise_equal_across_constraints=True,
                    source_sha256=kernels['run_rollout.py'])
    require(all(rr.get(k) == v for k, v in expected.items()), 'Rollout identity')
    require(rr['qp_lock'] == dict(path=str(qpath), sha256=sha(qpath)) and
            rr['recipient_cache'] == dict(path=entry['report'], sha256=entry['report_sha256']),
            'Rollout ancestry')
    require(rr['models'] == cr['models'] and
            rr['input_reports_sha256'] == cr['input_reports_sha256'], 'Source/model drift')
    checks = rr['insertion_checks']
    require(len(checks) == 2*n and {(c['objective'], c['case']) for c in checks} ==
            {(o, i) for o in expected['objectives'] for i in range(n)}, 'Missing insertion checks')
    require(all(c[k] is True for c in checks for k in
                ('free_exact', 'identity_exact', 'inserted_fp32_exact', 'inputs_unchanged')),
            'Failed runtime insertion receipt')
    data = array(rr['arrays']['path'], rr['arrays']['sha256'])
    require(set(data) == {'tokens', 'truth', 'seeds', 'goal_indices'}, 'Rollout fields')
    require(data['seeds'].dtype == np.int64 and data['goal_indices'].dtype == np.int64 and
            data['truth'].dtype == np.float64 and data['truth'].shape == (n, 5, 6) and
            np.isfinite(data['truth']).all(), 'Rollout truth/index schema')
    tokens = data['tokens']
    require(tokens.shape == (2, 2, n, 4, 5, 192) and tokens.dtype == np.float32
            and np.isfinite(tokens).all(), 'Rollout shape/precision/nonfinite')
    np.testing.assert_array_equal(data['goal_indices'], np.arange(n, dtype=np.int64))
    np.testing.assert_array_equal(data['seeds'], cache['seeds'])
    np.testing.assert_array_equal(data['truth'], cache['truth'])
    for ci in range(2):
        np.testing.assert_array_equal(tokens[:, ci, :, 0], cache['free'])
        np.testing.assert_array_equal(tokens[:, ci, :, 1:3, 0], q['replacements'][:, ci])
        for oi in range(2):
            np.testing.assert_array_equal(tokens[oi, ci, :, 3, 0], cache['observed'][:, 0])
    np.testing.assert_array_equal(tokens[:, 0, :, 3], tokens[:, 1, :, 3])
    for path, checksum in ((qpath, sha(qpath)), (shard['report'], shard['report_sha256']),
                           (shard['arrays'], shard['arrays_sha256']),
                           (rrpath, sha(rrpath)), (rr['arrays']['path'], rr['arrays']['sha256'])):
        identities[str(path)] = checksum
    return dict(group=g, stream=s, count=n, matching_families=n, directions=8*n,
                zero_dose_families=int((q['common_norm'] == 0).sum()),
                maximum_rechecked_output_deviation=qp['maximum_rechecked_output_deviation'],
                qp_seconds=qr['elapsed_seconds'], rollout_seconds=rr['elapsed_seconds'],
                gpu=rr['gpu'], peak_allocated_bytes=rr['peak_allocated_bytes'],
                input_files_sha256=identities)


def main():
    p = argparse.ArgumentParser()
    names = ('protocol', 'sources', 'cache-bindings', 'selected-lock', 'qualification',
             'input-metadata-acceptance')
    for name in names:
        p.add_argument('--' + name, required=True)
        p.add_argument('--' + name + '-sha256', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    out = Path(a.output)
    require(not out.exists(), 'Never overwrite development acceptance')
    documents = {k: read(getattr(a, k.replace('-', '_')), getattr(a, k.replace('-', '_') + '_sha256'))
                 for k in names}
    cfg, sources, binding = (documents[k] for k in ('protocol', 'sources', 'cache-bindings'))
    require(cfg['study_id'] == 'prospective_mechanism_v9_s1_20260923' and
            cfg['scores']['development_goal_count'] == 64, 'Changed protocol')
    require(sources['status'] == 'S1_INPUT_SOURCES_FROZEN' and
            sources['qualification_gate_passed'] is True and
            sources['authorized_kinds'] == ['development'] and
            sources['protocol_sha256'] == a.protocol_sha256, 'Wrong source gate')
    require(sources['selected_heads']['sha256'] == a.selected_lock_sha256 and
            sources['qualification']['sha256'] == a.qualification_sha256,
            'Qualification not bound by source gate')
    prior = documents['input-metadata-acceptance']
    require(prior['status'] == 'PASS_COMPLETE_S1_DEVELOPMENT_INPUT_METADATA_ACCEPTANCE' and
            prior['protocol_sha256'] == a.protocol_sha256 and
            prior['sources_sha256'] == a.sources_sha256 and
            prior['cache_bindings_sha256'] == a.cache_bindings_sha256,
            'Independent exact cache/model/source metadata acceptance missing')
    require(binding['status'] == 'COMPLETE_S1_DEVELOPMENT_CACHE_BINDINGS' and
            binding['protocol_sha256'] == a.protocol_sha256 and
            binding['sources_sha256'] == a.sources_sha256, 'Wrong cache binding')
    require(len(binding['reports']) == 16 and
            {(r['role'], r['group'], r['stream']) for r in binding['reports']} ==
            {(role, g, s) for role in ('recipient', 'donor') for g in (0, 1) for s in range(4)},
            'Incomplete cache population')
    selected, qualification = documents['selected-lock'], documents['qualification']
    require(selected['status'] == 'ALL_FOUR_S1_HEADS_FROZEN_BEFORE_QUALIFICATION' and
            selected['protocol_sha256'] == a.protocol_sha256 and len(selected['heads']) == 4 and
            {(r['group'], r['head_role']) for r in selected['heads']} ==
            {(g, h) for g in (0, 1) for h in ('C', 'D')}, 'Selected head roster')
    require(qualification['status'] == 'COMPLETE_S1_QUALIFICATION_PASS' and
            qualification['protocol_sha256'] == a.protocol_sha256 and
            qualification['selected_lock_sha256'] == a.selected_lock_sha256,
            'Wrong qualification')
    independent = sources['independent_qualification']
    iq = read(independent['remote_path'], independent['sha256'])
    require(iq['status'] == 'PASS_INDEPENDENT_COMPLETE_QUALIFICATION_REPRODUCTION' and
            iq['qualification_sha256'] == a.qualification_sha256 and
            iq['selected_lock_sha256'] == a.selected_lock_sha256 and
            iq['metric_gates_passed'] == 32 and iq['metric_gates_verified'] == 32,
            'Independent full qualification gate missing')
    prefix = 'strengthening/prospective_mechanism_v9_20260923/'
    kernels = {name: sources['files'][prefix + 'scripts/' + name] for name in
               ('cache_inputs.py', 'project_s1.py', 'freeze_qp_shards.py', 'run_rollout.py')}
    a_sha = sources['files'][prefix + 'manifests/v8_encoder_bindings.json']
    require(qualification['a_bindings_sha256'] == a_sha, 'Reference head binding')
    root = Path(cfg['remote_output']) / 'artifacts/development'
    rows = [check_route(g, s, root, binding, a.protocol_sha256, a.sources_sha256,
                        a.selected_lock_sha256, a.qualification_sha256, kernels, a_sha)
            for g in (0, 1) for s in range(4)]
    report = dict(status='PASS_COMPLETE_S1_DEVELOPMENT_SAVED_ARRAY_GATE',
                  created_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  protocol_sha256=a.protocol_sha256, sources_sha256=a.sources_sha256,
                  cache_bindings_sha256=a.cache_bindings_sha256,
                  selected_lock_sha256=a.selected_lock_sha256,
                  qualification_sha256=a.qualification_sha256,
                  input_metadata_acceptance_sha256=a.input_metadata_acceptance_sha256,
                  count=64, route_count=8, complete_matching_families=512,
                  complete_projection_directions=4096, rows=rows,
                  reserved_D_weights_opened=False, intervention_effects_scored=False,
                  source_sha256=sha(__file__),
                  scope=__doc__.strip() + ' Exact model/source ancestry is inherited from the separately hash-bound independent input metadata acceptance.')
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('x') as f:
        json.dump(report, f, indent=2, allow_nan=False)
        f.write('\n')


if __name__ == '__main__':
    main()
