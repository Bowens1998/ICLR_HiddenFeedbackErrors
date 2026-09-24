"""Draft complete S2 saved-input admission; no model, solver or simulator run.

Authenticate the existing full generator/physics/content receipts and check
every saved bank/action/physics triple. Six role-specific runtime admissions
are emitted only after all four banks and eighteen routes pass. This does not
independently regenerate physics or recompute historical pixel isolation.
"""
import argparse
import datetime
import json
from pathlib import Path
import numpy as np
import raw_inputs as raw

STATUS = 'PASS_S2_COMPLETE_RAW_INPUT_POPULATION'
MANIFEST_STATUS = 'S2_COMPLETE_ROLE_STREAM_INPUT_MANIFEST'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def pair(path):
    return dict(path=str(Path(path).resolve()), sha256=raw.sha(path))


def numeric(a, shape, name, dtype=None):
    a = np.asarray(a)
    require(a.shape == shape and (a.dtype == dtype if dtype is not None else a.dtype.kind in 'fiu')
            and np.isfinite(a).all(), 'Wrong complete finite array: ' + name)
    return a


def integer(value, name, maximum=None):
    a = np.asarray(value)
    require(a.shape == () and a.dtype.kind in 'iu' and int(a) >= 0
            and (maximum is None or int(a) < maximum), 'Invalid integer: ' + name)
    return int(a)


def validate_bank_arrays(z, case):
    require(integer(z['seed'], 'bank seed') == case['seed'], 'Wrong bank parent')
    for key, shape in [('history_pixels', (3,224,224,3)), ('goal_pixels', (224,224,3)),
                       ('terminal_pixels', (32,224,224,3))]:
        numeric(z[key], shape, key, np.uint8)
    history = numeric(z['history_states'], (3,7), 'history states')
    goal = numeric(z['goal_state'], (7,), 'goal state')
    terminal = numeric(z['terminal_states'], (32,7), 'terminal states')
    require(all(v.dtype.kind == 'f' for v in (history,goal,terminal)), 'Physical states must remain floating point')
    numeric(z['prefix'], (10,2), 'prefix', np.float32)
    actions = numeric(z['actions'], (32,25,2), 'reference controls', np.float32)
    goal_actions = numeric(z['goal_actions'], (25,2), 'goal controls', np.float32)
    require(not actions[0].any() and np.max(np.abs(actions)) <= np.float32(.35)
            and np.max(np.abs(goal_actions)) <= np.float32(.35), 'Reference/goal input rule changed')
    for key in ('contacts','terminations'):
        values = numeric(z[key], (32,), key)
        require(values.dtype.kind in 'iu' and (values >= 0).all() and (values <= 25).all(), 'Invalid branch counters')
    displacement = float(np.linalg.norm(goal[2:4]-history[-1,2:4]))
    require(displacement >= 40 and case['all_shapes_visible_every_step'] is True, 'Changed input-only goal admission')
    np.testing.assert_allclose(displacement, case['initial_goal_block_distance'], rtol=1e-12, atol=0.)


def validate_case_arrays(bank, actions, physics, *, seed, action_row, physics_row):
    require(all(integer(z['seed'], name+' seed') == seed for name,z in
                [('bank',bank),('actions',actions),('physics',physics)]), 'Parent mismatch')
    prefix = numeric(bank['prefix'], (10,2), 'prefix', np.float32)
    pop = numeric(actions['population_actions'], (300,25,2), 'native population controls', np.float32)
    selected = numeric(actions['selected_actions'], (25,2), 'selected controls', np.float32)
    index = integer(actions['selected_index'], 'selected index', 300)
    iteration = integer(actions['selected_iteration'], 'selected iteration', 30)
    require(action_row['selected_index'] == index and action_row['selected_iteration'] == iteration,
            'Selected index/iteration receipt drift')
    np.testing.assert_array_equal(pop[index], selected)
    # All six fixed original policies use NonlinearPoseCost, whose frozen
    # readout/cost arithmetic is FP64 although controls and latent tokens are FP32.
    costs = numeric(actions['population_costs'], (300,), 'selected-population costs', np.float64)
    trace = numeric(actions['cost_trace'], (30,300), 'complete search scalar trace', np.float64)
    require(tuple(np.unravel_index(np.argmin(trace), trace.shape)) == (iteration,index), 'Search selection rule changed')
    np.testing.assert_array_equal(costs, trace[iteration])
    numeric(actions['selected_token'], (192,), 'retained original-policy selected token', np.float32)
    numeric(physics['actions'], (35,2), 'replayed controls', np.float32)
    np.testing.assert_array_equal(physics['actions'], np.concatenate([prefix,selected]))
    states = numeric(physics['states'], (36,7), 'all physical states')
    require(states.dtype.kind == 'f', 'Physical trajectory must remain floating point')
    pixels = numeric(physics['pixels'], (8,224,224,3), 'all rendered frames', np.uint8)
    np.testing.assert_array_equal(states[[0,5,10]], bank['history_states'])
    np.testing.assert_array_equal(pixels[:3], bank['history_pixels'])
    for name in ('contacts','terminations','boundary'):
        require(np.asarray(physics[name]).shape == (35,) and np.asarray(physics[name]).dtype == np.bool_,
                'Wrong complete physical flag array: '+name)
    require(physics_row['boundary_steps'] == int(physics['boundary'][10:].sum()), 'Boundary count changed')
    require(action_row['population_replay_exact'] is True and action_row['scored_candidates'] == 9000
            and physics_row['all_two_replays_exact'] is True, 'Missing complete producer parity/replay')
    expected_seed = int(np.random.SeedSequence([seed,955001]).generate_state(1)[0])
    require(action_row['torch_seed'] == expected_seed, 'Original reference search seed changed')
    return dict(selected_index=index, selected_iteration=iteration)


def validate_content(doc, protocol_sha256, sources_sha256):
    require(doc.get('status') == 'PASS_S2_RETAINED_PIXEL_ISOLATION' and
            doc.get('protocol_sha256') == protocol_sha256 and doc.get('sources_sha256') == sources_sha256,
            'Missing actual complete content gate')
    require(doc['seed_lineage']['status'] == 'PASS_S2_ALL_RECORDED_ATTEMPTS_DISJOINT' and
            doc['seed_lineage']['protocol_sha256'] == protocol_sha256 and
            doc['seed_lineage']['sources_sha256'] == sources_sha256, 'Wrong attempt-lineage source')
    historical = {k+'/'+r for k in ('development','confirmation') for r in ('recipient','donor')}
    require(set(doc['head_overlap_hashes']) == set(raw.ROLES) and
            all(v == [] for v in doc['head_overlap_hashes'].values()), 'Incomplete/head alias check')
    require(set(doc['s1_overlap_hashes']) == set(raw.ROLES) and
            all(set(v) == historical and all(x == [] for x in v.values()) for v in doc['s1_overlap_hashes'].values()),
            'Incomplete/S1 content alias check')
    expected = {(a,b) for i,a in enumerate(raw.ROLES) for b in raw.ROLES[i+1:]}
    rows = doc['new_role_pairs']
    require(len(rows) == 6 and {(r['left'],r['right']) for r in rows} == expected and
            all(r['overlap'] == [] for r in rows), 'Incomplete/new-role alias check')
    caches = doc['head_caches']
    require(len(caches) == 24 and len({r['path'] for r in caches}) == 24 and
            all(sum(r['group'] == g for r in caches) == 6 for g in (0,1,2,4)), 'Missing bound historical head exposure')
    require(set(doc['populations']) == {'S1/'+s for s in historical} | {'S2/'+s for s in raw.ROLES},
            'Wrong content population roster')
    for role in raw.ROLES:
        n = raw.COUNTS[role]; summary = doc['populations']['S2/'+role]
        require(summary['total_frames'] == n*(36+8*len(raw.routes(role))) and
                0 < summary['unique_pixels'] <= summary['total_frames'], 'Incomplete retained raw pixel count')
    require(bool(doc['input_files_sha256']), 'Empty content file binding')


def runtime_role(bank_role, stream_role):
    split,kind = bank_role.split('_')
    if kind == 'donor':
        require(stream_role == 'donor_observed', 'Donor requires its distinct encoder-only admission')
        return 'probe_donor_'+split
    require(stream_role in ('probe','response'), 'Unknown recipient stream role')
    return ('probe_'+split) if stream_role == 'probe' else (split+'_response')


def admit(protocol, protocol_sha256, sources, sources_sha256, content_path, content_sha256, output):
    output = Path(output).resolve()
    # Durable intention/output guard before population work; partial results stay.
    output.mkdir(parents=True, exist_ok=False)
    raw.write_exclusive(output/'started.json', dict(at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        protocol=pair(protocol), sources=pair(sources), content_lineage=pair(content_path), source_sha256=raw.sha(__file__)))
    contexts = {r:raw.context(protocol,protocol_sha256,sources,sources_sha256,r) for r in raw.ROLES}
    ctx = contexts[raw.ROLES[0]]
    bound = {str(raw.resolve(ctx,k).resolve()):v for k,v in ctx['sources']['files'].items()}
    require(bound.get(str(Path(__file__).resolve())) == raw.sha(__file__), 'Raw acceptor absent from frozen source closure')
    require(Path(content_path).resolve() == (ctx['artifacts']/'content_lineage.json').resolve(), 'Wrong content output namespace')
    content = raw.checked_json(content_path,content_sha256); validate_content(content,protocol_sha256,sources_sha256)
    observed = raw._read_pair(ctx,ctx['sources']['observed_cache_lock'])
    expected_head_caches = {(g['group'],str(Path(g['cache'])/name),checksum)
        for g in observed['groups'] if g['group'] in (0,1,2,4) for name,checksum in g['arrays'].items()}
    require(len(expected_head_caches) == 24 and
            {(r['group'],r['path'],r['sha256']) for r in content['head_caches']} == expected_head_caches,
            'Content head-cache roster differs from actual bound observed-cache lock')
    for name,checksum in content['input_files_sha256'].items():
        require(Path(name).is_absolute() and raw.sha(name) == checksum, 'Changed content source: '+name)
    for row in content['head_caches']:
        require(raw.sha(row['path']) == row['sha256'], 'Changed bound historical head cache')
    banks = []; route_outputs = {}; seen_parents = set(); checks = 0
    for role,c in contexts.items():
        root,role_meta,manifest = raw._bank(c); n = raw.COUNTS[role]; ids = [r['seed'] for r in manifest['cases']]
        require(not (set(ids)&seen_parents), 'Cross-role actual parent alias'); seen_parents.update(ids)
        acceptance = raw.checked_json(root/'acceptance.json',role_meta['acceptance_sha256'])
        require(acceptance['source_sha256'] == raw.sha(raw.ROOT/'scripts/visual/accept_adaptation_contexts.py'), 'Bank acceptor changed')
        require([r['index'] for r in acceptance['rows']] == list(range(n)), 'Incomplete bank acceptance rows')
        for row,verified in zip(manifest['cases'],acceptance['rows'],strict=True):
            bank_file = root/f"case_{row['index']:03d}.npz"
            require(raw.sha(bank_file) == row['sha256'] == verified['sha256'] and verified['seed'] == row['seed']
                    and verified['independently_replayed_branches'] == 33, 'Bank replay parent/hash coverage changed')
            with np.load(bank_file,allow_pickle=False) as z: validate_bank_arrays(z,row)
        banks.append(dict(bank_role=role,count=n,parent_ids=ids,manifest=pair(root/'manifest.json'),
                          role_metadata=pair(root/'role.json')))
        for index in raw.routes(role):
            aroot = raw._path(c,'actions')/f'route_{index}'; proot = raw._path(c,'physics')/f'route_{index}'
            reports = []
            for folder,stage in ((aroot,'PLAN'),(proot,'REPLAY')):
                a = json.loads((folder/'s2_admission.json').read_text())
                require(a['status'] == 'PASS_S2_RAW_'+stage and a['policy_index'] == index and a['count'] == n
                        and all(a.get(k) == v for k,v in c['binding'].items()), 'Wrong whole-route S2 admission')
                report = raw.checked_json(folder/'report.json',a['report_sha256'])
                binding = raw.checked_json(folder/'binding.json',raw.sha(folder/'binding.json'))
                require((folder/'DONE').is_file() and report['binding'] == binding and
                        all(binding.get(k) == v for k,v in c['binding'].items()) and
                        binding['source_sha256'] == raw.sha(raw.KERNEL) and
                        [r['index'] for r in report['cases']] == list(range(n)), 'Wrong complete route provenance')
                reports.append(report)
            ar,pr = reports
            require(ar['status'] == 'PASS_COMPACT_FIXED_REFERENCE_SEARCH' and ar['frozen_tensors_unchanged'] is True and
                    ar['binding']['row'] == raw._reference(c,index) and
                    ar['binding_sha256'] == raw.sha(aroot/'binding.json') and
                    ar['binding']['bank_manifest_sha256'] == raw.sha(root/'manifest.json') and
                    ar['binding']['bank_role_sha256'] == raw.sha(root/'role.json'), 'Wrong selected-action model/bank source')
            require(pr['status'] == 'PASS_COMPLETE_SELECTED_PHYSICS' and pr['independent_replays'] == 2*n and
                    pr['binding']['action_report_sha256'] == raw.sha(aroot/'report.json') and
                    pr['binding']['parent_manifest_sha256'] == raw.sha(root/'manifest.json') and
                    pr['binding']['bank_role_sha256'] == raw.sha(root/'role.json'), 'Wrong complete physics source')
            case_records = []
            for bank_row,action_row,physics_row in zip(manifest['cases'],ar['cases'],pr['cases'],strict=True):
                i = bank_row['index']; name = f'case_{i:03d}.npz'; seed = bank_row['seed']
                require(action_row['file'] == physics_row['file'] == name and
                        action_row['seed'] == physics_row['seed'] == seed, 'Wrong file or parent order')
                paths = dict(bank=root/name,actions=aroot/name,physics=proot/name)
                pairs = {k:pair(path) for k,path in paths.items()}
                require(pairs['bank']['sha256'] == bank_row['sha256'] == action_row['input_sha256'] == physics_row['input_sha256']
                        and pairs['actions']['sha256'] == action_row['file_sha256'] == physics_row['action_file_sha256']
                        and pairs['physics']['sha256'] == physics_row['file_sha256'] and
                        action_row['binding_sha256'] == raw.sha(aroot/'binding.json') and
                        physics_row['binding_sha256'] == raw.sha(proot/'binding.json'), 'Changed raw triple provenance')
                require(content['input_files_sha256'].get(str(paths['bank'])) == pairs['bank']['sha256'] and
                        content['input_files_sha256'].get(str(paths['physics'])) == pairs['physics']['sha256'], 'Raw triple absent from content acceptance')
                with np.load(paths['bank'],allow_pickle=False) as b, np.load(paths['actions'],allow_pickle=False) as a, np.load(paths['physics'],allow_pickle=False) as ph:
                    selected = validate_case_arrays(b,a,ph,seed=seed,action_row=action_row,physics_row=physics_row)
                case_records.append(dict(index=i,seed=seed,**pairs,**selected)); checks += 1
            route_outputs[(role,index)] = dict(pool=index//8,group=2*(index//8),policy_index=index,
                actions_report=pair(aroot/'report.json'),physics_report=pair(proot/'report.json'),cases=case_records)
    admissions = []
    for b in banks:
        role = b['bank_role']
        for stream in (('probe',) if role.endswith('_donor') else ('probe','response')):
            indices = [8*p+(stream=='response') for p in range(3)]
            folder = output/role/stream; manifest_path = folder/'input_manifest.json'
            # The physical policy is still the probe stream; runtime authority
            # is a distinct donor-observed role that cannot perform rollouts.
            admission_stream = 'donor_observed' if role.endswith('_donor') else stream
            common = dict(protocol_sha256=protocol_sha256,sources_sha256=sources_sha256,bank_role=role,
                          stream_role=admission_stream,count=b['count'],parent_ids=b['parent_ids'])
            raw.write_exclusive(manifest_path,dict(status=MANIFEST_STATUS,**common,
                routes=[route_outputs[(role,index)] for index in indices]))
            admission_path = folder/'admission.json'
            raw.write_exclusive(admission_path,dict(status='S2_FIXED_CONTEXTS_ACCEPTED',**common,
                bank_manifest=b['manifest'],input_manifest=pair(manifest_path)))
            admissions.append(dict(role=runtime_role(role,admission_stream),**pair(admission_path)))
    report = dict(status=STATUS,protocol_sha256=protocol_sha256,sources_sha256=sources_sha256,
        raw_sources_sha256=sources_sha256,protocol=pair(protocol),raw_sources=pair(sources),
        content_lineage=pair(content_path),banks=banks,admissions=admissions,bank_count=4,route_count=18,
        saved_triples_verified=checks,source_sha256=raw.sha(__file__),models_or_readouts_loaded=False,
        simulator_executed=False,pixel_isolation_independently_recomputed=False,
        scope='Complete saved bank/action/physics pairing and native selected-control/index/history integrity. '
              'Full bank33-branch and selected-future double physics replay plus retained-pixel isolation are inherited '
              'from hash-authenticated producer receipts, not independently regenerated here. No model outcomes opened.')
    raw.write_exclusive(output/'report.json',report)
    with (output/'DONE').open('x') as f:f.write('complete raw saved-input admission\n')
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('protocol','sources','content'):
        p.add_argument('--'+key,required=True); p.add_argument('--'+key+'-sha256',required=True)
    p.add_argument('--output',required=True); a=p.parse_args()
    report=admit(a.protocol,a.protocol_sha256,a.sources,a.sources_sha256,a.content,a.content_sha256,a.output)
    print(json.dumps({k:report[k] for k in ('status','bank_count','route_count','saved_triples_verified')}))


if __name__=='__main__':main()
