"""Locked fresh banks and compact unchanged reference-policy inputs for S1.

This module never fits or selects a model. Its planner and independent physics
replay retain the accepted legacy arithmetic; only namespace/role guards differ.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np

S1_PHASE = Path(__file__).resolve().parents[1]
S1_ROOT = S1_PHASE.parents[1]


def s1_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def s1_json(path, expected=None):
    if expected is not None and s1_sha(path) != expected:
        raise ValueError(f'Changed immutable file: {path}')
    return json.loads(Path(path).read_text())


def s1_write_json(path, value):
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + f'.{os.getpid()}.partial')
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    os.replace(tmp, p)


def s1_npz(path, **arrays):
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + f'.{os.getpid()}.partial')
    with tmp.open('wb') as f:
        np.savez_compressed(f, **arrays)
    os.replace(tmp, p)


def s1_arguments(p, role=True):
    for key in ['protocol', 'protocol-sha256', 'sources', 'sources-sha256']:
        p.add_argument('--' + key, required=True)
    p.add_argument('--kind', choices=['development', 'confirmation'], required=True)
    if role:
        p.add_argument('--role', choices=['recipient', 'donor'], required=True)


def s1_resolve(ctx, path):
    p = Path(path)
    return p if p.is_absolute() else ctx['root'] / p


def s1_context(a):
    cfg = s1_json(a.protocol, a.protocol_sha256)
    if (cfg['study_id'] != 'prospective_mechanism_v9_s1_20260923' or
            cfg['root_seed'] != 20260923 or cfg['groups'] != [0, 1] or
            cfg['objectives'] != ['decoded_teacher', 'physical_labels'] or
            cfg['scores']['native_population'] != 300):
        raise ValueError('Unexpected S1 scientific design')
    sources = s1_json(a.sources, a.sources_sha256)
    if (sources['status'] != 'S1_INPUT_SOURCES_FROZEN' or
            sources['protocol_sha256'] != a.protocol_sha256 or
            a.kind not in sources['authorized_kinds']):
        raise ValueError('Inputs/source gate does not authorize this stage')
    root = Path(sources['source_root']).resolve()
    if root != S1_ROOT.resolve():
        raise ValueError('Runtime tree differs from the frozen source root')
    ctx = dict(cfg=cfg, sources=sources, root=root, kind=a.kind,
               role=getattr(a, 'role', 'recipient'),
               protocol_sha256=a.protocol_sha256, sources_sha256=a.sources_sha256)
    if not sources['files']:
        raise ValueError('Empty source closure')
    for name, expected in sources['files'].items():
        if s1_sha(s1_resolve(ctx, name)) != expected:
            raise ValueError(f'Changed source closure: {name}')
    # The entrypoint itself must be frozen, not merely optional helper files.
    for name in ['fresh_inputs.py', Path(sys.argv[0]).name]:
        source = Path(__file__).with_name(name)
        if source.exists() and not any(s1_resolve(ctx, p).resolve() == source.resolve()
                                     for p in sources['files']):
            raise ValueError(f'Unbound entrypoint: {name}')
    ctx['spec'] = next(b for b in cfg['banks']
                       if b['kind'] == a.kind and b['role'] == ctx['role'])
    spec = ctx['spec']
    expected = int.from_bytes(hashlib.sha256(
        f"20260923:v9_s1_bank/{a.kind}/{ctx['role']}".encode()).digest()[:4], 'big')
    if spec['seed_start'] != expected or spec['max_seeds'] != 100000:
        raise ValueError('Changed bank seed namespace or attempt budget')
    if spec['count'] != (64 if a.kind == 'development' else 256):
        raise ValueError('Changed fixed bank size')
    ctx['base'] = Path(cfg['remote_base'])
    ctx['artifacts'] = Path(cfg['remote_output']) / 'artifacts' / a.kind
    ctx['binding'] = dict(protocol_sha256=a.protocol_sha256,
                          sources_sha256=a.sources_sha256, kind=a.kind,
                          role=ctx['role'])
    sys.path[:0] = [str(root / 'strengthening/adapters'), str(root / 'scripts/visual')]
    return ctx


def s1_path(ctx, stage, role=None):
    return ctx['artifacts'] / stage / (role or ctx['role'])


def s1_lineage(ctx):
    """Reject any overlap in the entire fixed attempt range, never skip seeds."""
    rows = []; old = set()
    manifests = ctx['sources']['legacy_bank_manifests']
    if not manifests:
        raise ValueError('Old bank registry is mandatory')
    for entry in manifests:
        d = s1_json(entry['path'], entry['sha256'])
        values = {int(x['seed']) for x in d['cases']}
        # Previously rejected seeds are excluded too, conservatively.
        values.update(int(x['seed']) for x in d.get('rejected_seeds', []) if 'seed' in x)
        old.update(values)
        rows.append(dict(**entry, unique_seed_count=len(values)))
    head_rows = ctx['sources']['head_planner_parent_metadata']
    if not head_rows:
        raise ValueError('All observed head planner roles must be bound')
    heads = set()
    for entry in head_rows:
        d = s1_json(entry['path'], entry['sha256'])
        if sorted(d['parent_ids'][entry['parent_key']]) != sorted(entry['parent_ids']):
            raise ValueError('Changed observed head parent roster')
        heads.update(map(int, entry['parent_ids']))
    ranges = []; overlaps = []
    for bank in ctx['cfg']['banks']:
        lo, hi = bank['seed_start'], bank['seed_start'] + bank['max_seeds']
        hit = sorted(x for x in old | heads if lo <= x < hi)
        overlaps += hit
        ranges.append(dict(kind=bank['kind'], role=bank['role'], start=lo, stop=hi,
                           old_or_head_overlap=hit))
    for i, left in enumerate(ranges):
        for right in ranges[i + 1:]:
            if max(left['start'], right['start']) < min(left['stop'], right['stop']):
                raise ValueError('Fresh fixed bank attempt ranges overlap')
    if overlaps:
        raise ValueError('Frozen seed range overlaps historical inputs; block, never resample')
    return dict(status='PASS_ALL_FIXED_SEED_RANGES_DISJOINT', **ctx['binding'],
                old_bank_manifests=rows, old_unique_seeds=len(old),
                head_planner_parent_metadata=head_rows, head_unique_parents=len(heads),
                attempt_ranges=ranges, policy='Any overlap blocks the unchanged design; no skipped seeds',
                scope='Complete frozen owned-project manifest registry and observed head planner parents. '
                      'Seed isolation does not claim isolation from encoder pretraining or expert pixels.')


def s1_prepare(ctx):
    out = s1_path(ctx, 'banks'); spec = ctx['spec']
    lineage = s1_lineage(ctx)  # Mandatory before any simulation/generation.
    if out.exists():
        raise ValueError('Output already exists; partial recovery requires engineering adjudication')
    sim = ctx['base'] / 'releases/visual-v1/stable-worldmodel'; start = time.monotonic()
    subprocess.run([sys.executable, str(ctx['root'] / 'scripts/visual/prepare_prospective_pusht.py'),
        '--source', str(sim), '--output', str(out), '--cases', str(spec['count']),
        '--seed-start', str(spec['seed_start']), '--max-seeds', str(spec['max_seeds'])], check=True)
    subprocess.run([sys.executable, str(ctx['root'] / 'scripts/visual/accept_adaptation_contexts.py'),
        '--simulator', str(sim), '--bank', str(out), '--count', str(spec['count']),
        '--seed-start', str(spec['seed_start'])], check=True)
    manifest = s1_json(out / 'manifest.json'); acceptance = s1_json(out / 'acceptance.json')
    if (len(manifest['cases']) != spec['count'] or
            acceptance['manifest_sha256'] != s1_sha(out / 'manifest.json')):
        raise ValueError('Incomplete or stale replay acceptance')
    s1_write_json(out / 'lineage.json', lineage)
    s1_write_json(out / 'role.json', dict(**ctx['binding'], count=spec['count'],
        seed_start=spec['seed_start'], max_seeds=spec['max_seeds'],
        parent_manifest_sha256=s1_sha(out / 'manifest.json'),
        acceptance_sha256=s1_sha(out / 'acceptance.json'), lineage_sha256=s1_sha(out / 'lineage.json'),
        source_sha256=s1_sha(__file__), elapsed_seconds=time.monotonic() - start,
        head_fit_role=False, purpose='Fresh mechanism recipient/donor input only'))
    (out / 'DONE').write_text('S1 fresh bank accepted\n')


def s1_bank(ctx):
    bank = s1_path(ctx, 'banks'); role = s1_json(bank / 'role.json')
    if any(role.get(k) != v for k, v in ctx['binding'].items()):
        raise ValueError('Bank belongs to another protocol, source gate or role')
    if (not (bank / 'DONE').exists() or role['count'] != ctx['spec']['count'] or
            role['parent_manifest_sha256'] != s1_sha(bank / 'manifest.json') or
            role['acceptance_sha256'] != s1_sha(bank / 'acceptance.json') or
            role['lineage_sha256'] != s1_sha(bank / 'lineage.json')):
        raise ValueError('Unaccepted bank')
    manifest = s1_json(bank / 'manifest.json')
    if [c['index'] for c in manifest['cases']] != list(range(role['count'])):
        raise ValueError('Fixed case roster changed')
    return bank, role, manifest


def s1_reference(ctx, index):
    if index not in range(8):
        raise ValueError('Only four streams for each of groups 0 and 1')
    entry = ctx['sources']['reference_policies']
    policies = s1_json(s1_resolve(ctx, entry['path']), entry['sha256'])
    row = policies['rows'][index]
    if row['index'] != index or row['entry']['adaptation_condition'] != 'original':
        raise ValueError('Changed fixed original reference stream')
    return row


def s1_content_audit(ctx):
    """Hash retained head pixels and every new history/goal/selected future frame.

    Only content identities are compared. No labels or model effects determine
    admission; any cross-role collision blocks the complete frozen population.
    """
    head_pixels = set(); head_bindings = []
    entries = ctx['sources']['observed_pixel_caches']
    observed_key = 'strengthening/prospective_mechanism_v9_20260923/manifests/v8_OBSERVED_CACHES.lock.json'
    observed_lock = s1_json(s1_resolve(ctx, observed_key), ctx['cfg']['immutable_source_bindings'][observed_key])
    expected_entries = {(g['group'], str(Path(g['cache']) / name), checksum)
                        for g in observed_lock['groups'] if g['group'] in [0, 1]
                        for name, checksum in g['arrays'].items()}
    actual_entries = {(r['group'], r['path'], r['sha256']) for r in entries}
    if len(entries) != 12 or len(actual_entries) != 12 or actual_entries != expected_entries:
        raise ValueError('All twelve accepted observed caches for C/D are required')
    for entry in entries:
        if s1_sha(entry['path']) != entry['sha256']:
            raise ValueError('Changed observed head pixel cache')
        with np.load(entry['path'], allow_pickle=False) as z:
            hashes = z['pixel_sha256'].astype(str)
        head_pixels.update(hashes.tolist()); head_bindings.append(dict(**entry, rows=len(hashes)))
    populations = {}; input_hashes = {}; row_counts = {}
    kinds = ['development'] if ctx['kind'] == 'development' else ['development', 'confirmation']
    sources_allowed = {ctx['sources_sha256'], *ctx['sources'].get('prior_input_source_sha256', [])}
    def digest(pixel):
        return hashlib.sha256(np.ascontiguousarray(pixel).tobytes()).hexdigest()
    for kind in kinds:
        for role in ['recipient', 'donor']:
            key = kind + '/' + role; population = set(); pixels_count = 0
            root = Path(ctx['cfg']['remote_output']) / 'artifacts' / kind
            bank = root / 'banks' / role; br = s1_json(bank / 'role.json')
            if (br.get('protocol_sha256') != ctx['protocol_sha256'] or
                    br.get('kind') != kind or br.get('role') != role or
                    br.get('sources_sha256') not in sources_allowed or not (bank / 'DONE').exists()):
                raise ValueError('Unbound fresh bank for cross-role content audit')
            bm = s1_json(bank / 'manifest.json', br['parent_manifest_sha256'])
            count = 64 if kind == 'development' else 256
            if len(bm['cases']) != count or br['count'] != count:
                raise ValueError('Incomplete pixel population')
            input_hashes[str(bank / 'role.json')] = s1_sha(bank / 'role.json')
            input_hashes[str(bank / 'manifest.json')] = br['parent_manifest_sha256']
            for case in bm['cases']:
                path = bank / f"case_{case['index']:03d}.npz"
                if s1_sha(path) != case['sha256']:
                    raise ValueError('Changed fresh context pixels')
                with np.load(path, allow_pickle=False) as z:
                    frames = [*z['history_pixels'], z['goal_pixels']]
                population.update(map(digest, frames)); pixels_count += len(frames)
                input_hashes[str(path)] = case['sha256']
            for route in range(8):
                physics = root / 'physics' / role / f'route_{route}'
                pr = s1_json(physics / 'report.json')
                expected = dict(protocol_sha256=ctx['protocol_sha256'], kind=kind, role=role)
                if (pr['status'] != 'PASS_COMPLETE_SELECTED_PHYSICS' or
                        any(pr['binding'].get(k) != v for k, v in expected.items()) or
                        pr['binding']['sources_sha256'] not in sources_allowed or
                        pr['binding']['parent_manifest_sha256'] != br['parent_manifest_sha256'] or
                        len(pr['cases']) != count or not (physics / 'DONE').exists()):
                    raise ValueError('Unaccepted complete physics for pixel audit')
                input_hashes[str(physics / 'report.json')] = s1_sha(physics / 'report.json')
                for case in pr['cases']:
                    path = physics / case['file']
                    if s1_sha(path) != case['file_sha256']:
                        raise ValueError('Changed rendered future')
                    with np.load(path, allow_pickle=False) as z:
                        frames = z['pixels']
                    if frames.shape != (8, 224, 224, 3):
                        raise ValueError('Missing rendered observed horizon')
                    population.update(map(digest, frames)); pixels_count += len(frames)
                    input_hashes[str(path)] = case['file_sha256']
            populations[key] = population; row_counts[key] = dict(total_frames=pixels_count, unique_pixels=len(population))
    head_overlaps = {k: sorted(v & head_pixels) for k, v in populations.items()}
    pairwise = []
    keys = list(populations)
    for i, left in enumerate(keys):
        for right in keys[i + 1:]:
            overlap = sorted(populations[left] & populations[right])
            pairwise.append(dict(left=left, right=right, overlap_count=len(overlap), pixel_sha256=overlap))
    passed = not any(head_overlaps.values()) and not any(r['overlap_count'] for r in pairwise)
    out = ctx['artifacts'] / 'content_lineage.json'
    if out.exists():
        raise ValueError('Never overwrite accepted or failed content audit')
    s1_write_json(out, dict(status='PASS_S1_CONTENT_ISOLATION' if passed else 'BLOCKED_S1_PIXEL_ALIAS',
        protocol_sha256=ctx['protocol_sha256'], sources_sha256=ctx['sources_sha256'], kind=ctx['kind'],
        head_caches=head_bindings, head_unique_pixels=len(head_pixels), populations=row_counts,
        head_overlap_hashes=head_overlaps, fresh_population_pairs=pairwise,
        input_files_sha256=input_hashes, source_sha256=s1_sha(__file__),
        scope='Exact uint8 rendered pixel bytes for all fresh history/goal/selected observed future frames '
              'against both architectures and every accepted expert/planner C/D source role; new roles mutually disjoint. '
              'Same-parent repetitions within a role retained. Historical old-bank raw pixel comparison and encoder pretraining isolation not asserted.',
        failure_policy='Block the whole unchanged population; no deletion, replacement or appended seeds'))
    if not passed:
        raise ValueError('Fresh pixel alias detected; complete result blocked')


def main():
    p = argparse.ArgumentParser(); s1_arguments(p)
    p.add_argument('--stage', choices=['audit', 'prepare', 'plan', 'replay', 'content-audit'], required=True)
    p.add_argument('--index', type=int)
    a = p.parse_args(); ctx = s1_context(a)
    if a.stage == 'audit':
        print(json.dumps(s1_lineage(ctx), indent=2)); return
    if a.stage == 'prepare':
        s1_prepare(ctx); return
    if a.stage == 'content-audit':
        s1_content_audit(ctx); return
    index = int(os.environ['SLURM_ARRAY_TASK_ID']) if a.index is None else a.index
    (s1_plan if a.stage == 'plan' else s1_replay)(ctx, index)


# Planner/replay functions below preserve the existing implementation explicitly.

def s1_plan(ctx, index):
    import torch
    from artifact_io import array_sha
    from controlled_search import search, expand_actions
    from factorial_model import make_model
    from image_planner_cost import ImagePlannerCost
    from nonlinear_pose_cost import NonlinearPoseCost
    from evaluation_precision import configure_evaluation_precision
    from score_feedback_ranking import state_hash
    sha, atomic_json, atomic_npz = s1_sha, s1_write_json, s1_npz
    ROOT = ctx['root']; base = ctx['base']
    bank, role, manifest = s1_bank(ctx); count = ctx['spec']['count']
    row = s1_reference(ctx, index); e = row['entry']; route = row['route']
    td = Path(e['training_path']); tr = json.loads((td/'summary.json').read_text())
    config = base/'assets/pusht-v1/models/config.json'
    assert sha(td/'summary.json') == e['training_summary_sha256']
    assert sha(td/'last_weights.pt') == e['weights_sha256'] and sha(config) == tr['config_sha256']
    precision = configure_evaluation_precision(); torch.set_num_threads(4)
    model = make_model(base/'releases/visual-v1/official', config, e['arm'], tr['seed'])
    model.load_state_dict(torch.load(td/'last_weights.pt', map_location='cpu', weights_only=True), strict=True)
    model = model.cuda().eval(); before = state_hash(model)
    heads = {}
    if e['score'].startswith('pose_'):
        for key in ['endpoint_head', 'goal_head']:
            assert sha(e[key]['path']) == e[key]['sha256']
            heads[key] = dict(np.load(e[key]['path']))
    out = s1_path(ctx, 'actions')/f'route_{index}'; out.mkdir(parents=True, exist_ok=True)
    binding = dict(**ctx['binding'], bank_manifest_sha256=sha(bank/'manifest.json'),
                   bank_role_sha256=sha(bank/'role.json'), reference_policies_sha256=ctx['sources']['reference_policies']['sha256'],
                   source_sha256=sha(__file__), row=row,
                   implementation_sha256={name:sha(ROOT/'scripts/visual'/name) for name in
                       ['controlled_search.py','image_planner_cost.py','nonlinear_pose_cost.py','factorial_model.py','evaluation_precision.py']})
    if (out/'binding.json').exists():
        assert json.loads((out/'binding.json').read_text()) == binding
    else:
        atomic_json(out/'binding.json', binding)
    started = time.monotonic(); rows = []
    with torch.inference_mode():
        for item in manifest['cases']:
            i = item['index']; path = bank/f'case_{i:03d}.npz'
            assert sha(path) == item['sha256']
            result = out/f'case_{i:03d}.json'; target = out/f'case_{i:03d}.npz'
            if result.exists():
                saved = json.loads(result.read_text())
                assert saved['binding_sha256'] == sha(out/'binding.json') and saved['input_sha256'] == sha(path)
                assert saved['file_sha256'] == sha(target)
                rows.append(saved); continue
            z = dict(np.load(path)); seed = int(z['seed'])
            cost = ImagePlannerCost(model, z['history_pixels'], z['goal_pixels'], z['prefix'],
                                    tr['normalization'], e['target_normalization'],
                                    'state' if e['score']=='state' else 'latent')
            if heads:
                cost = NonlinearPoseCost(cost, heads['endpoint_head'], heads['goal_head'])
            if i == 0:
                probe = torch.zeros(2,25,2,device='cuda'); probe[1] = .1
                cost.verify_native(probe)
            torch_seed = int(np.random.SeedSequence([seed,955001]).generate_state(1)[0])
            torch.cuda.synchronize(); start = time.monotonic()
            best, trace, mean = search(cost, algorithm=route['algorithm'], parameterization=route['parameterization'],
                                       device='cuda', seed=torch_seed)
            torch.cuda.synchronize(); elapsed = time.monotonic()-start
            pop = trace[best['iteration']]
            actions = expand_actions(torch.as_tensor(pop['parameters'], device='cuda'))
            pcost, tokens = cost(actions)
            np.testing.assert_array_equal(tokens.cpu().numpy(), pop['tokens'])
            np.testing.assert_array_equal(pcost.cpu().numpy(), pop['costs'])
            j = best['candidate']; selected = actions[j].cpu().numpy()
            np.testing.assert_array_equal(selected, expand_actions(best['parameters'][None])[0].cpu().numpy())
            # Keep every scalar cost and deterministic search seed, but no 9000-token prediction archive.
            cost_trace = np.stack([x['costs'] for x in trace])
            assert np.unravel_index(np.argmin(cost_trace), cost_trace.shape) == (best['iteration'], j)
            atomic_npz(target, population_actions=actions.cpu().numpy(), selected_actions=selected,
                selected_token=tokens[j].cpu().numpy(), population_costs=pcost.cpu().numpy(), cost_trace=cost_trace,
                distribution_mean=np.stack([x['mean'] for x in trace]), distribution_std=np.stack([x['std'] for x in trace]),
                final_distribution_mean=mean.cpu().numpy(), selected_index=np.asarray(j),
                selected_iteration=np.asarray(best['iteration']), seed=np.asarray(seed))
            saved = dict(index=i, seed=seed, torch_seed=torch_seed, selected_iteration=best['iteration'],
                selected_index=j, population_replay_exact=True, scored_candidates=9000,
                file=target.name, file_sha256=sha(target), input_sha256=sha(path),
                binding_sha256=sha(out/'binding.json'), planning_seconds=elapsed,
                full_trace_hashes=[{key:array_sha(x[key]) for key in ['parameters','costs','tokens','mean','std']} for x in trace])
            atomic_json(result,saved); rows.append(saved)
            print(json.dumps(dict(index=i,route=index,seconds=elapsed,accepted=True)),flush=True)
    assert len(rows) == count and state_hash(model) == before
    atomic_json(out/'report.json',dict(status='PASS_COMPACT_FIXED_REFERENCE_SEARCH',binding=binding,
        binding_sha256=sha(out/'binding.json'),cases=rows,precision=precision,gpu=torch.cuda.get_device_name(),
        elapsed_seconds=time.monotonic()-started,frozen_tensors_unchanged=True,
        scope='Input construction using original reference policies. Exact original search and selected-population arithmetic; physical replay is a separate acceptance stage. No new study model evaluated.'))
    (out/'DONE').write_text('source_actions_verified\n')


def s1_replay(ctx, index):
    s1_reference(ctx, index)
    sim = ctx['base']/'releases/visual-v1/stable-worldmodel'
    sys.path.insert(0, str(sim))
    from stable_worldmodel.envs.pusht.env import PushT
    sha, atomic_json, atomic_npz = s1_sha, s1_write_json, s1_npz
    bank, role, manifest = s1_bank(ctx); count = ctx['spec']['count']
    source = s1_path(ctx, 'actions')/f'route_{index}'
    report = s1_json(source/'report.json')
    if report['status'] != 'PASS_COMPACT_FIXED_REFERENCE_SEARCH' or not (source/'DONE').exists():
        raise ValueError('Unaccepted original reference action archive')
    if any(report['binding'].get(k) != v for k,v in ctx['binding'].items()):
        raise ValueError('Changed action role binding')
    if report['binding']['bank_manifest_sha256'] != sha(bank/'manifest.json') or report['binding']['source_sha256'] != sha(__file__):
        raise ValueError('Changed action source or bank')
    out = s1_path(ctx, 'physics')/f'route_{index}';out.mkdir(parents=True,exist_ok=True)
    binding=dict(**ctx['binding'],bank_role_sha256=sha(bank/'role.json'),parent_manifest_sha256=sha(bank/'manifest.json'),
        action_report_sha256=sha(source/'report.json'),source_sha256=sha(__file__),
        simulator_sha256=sha(sim/'stable_worldmodel/envs/pusht/env.py'))
    if (out/'binding.json').exists():
        if s1_json(out/'binding.json') != binding:raise ValueError('Changed physics binding')
    else:atomic_json(out/'binding.json',binding)
    def replay(seed,actions):
        env=PushT(resolution=224);obs,_=env.reset(seed=seed)
        states=[obs['state'].copy()];pixels=[env.render().copy()];contacts=[];terminated=[];boundary=[]
        for t,u in enumerate(actions):
            obs,_,term,_,_=env.step(u);states.append(obs['state'].copy())
            if (t+1)%5==0:pixels.append(env.render().copy())
            outside=False
            for body in [env.agent,env.block]:
                for shape in body.shapes:
                    bb=shape.cache_bb();outside|=min(bb.left,bb.bottom)<8 or max(bb.right,bb.top)>504
            boundary.append(outside);contacts.append(env.n_contact_points>0);terminated.append(term)
        env.close()
        return dict(pixels=np.stack(pixels),states=np.stack(states),actions=actions.copy(),
            contacts=np.asarray(contacts),terminations=np.asarray(terminated),boundary=np.asarray(boundary))
    rows=[];start=time.monotonic()
    for c,s in zip(manifest['cases'],report['cases'],strict=True):
        i=c['index'];assert s['index']==i and s['seed']==c['seed']
        context=bank/f'case_{i:03d}.npz';ap=source/s['file']
        assert sha(context)==c['sha256'] and sha(ap)==s['file_sha256']
        target=out/f'case_{i:03d}.npz';result=out/f'case_{i:03d}.json'
        if result.exists():
            row=json.loads(result.read_text());assert row['file_sha256']==sha(target) and row['binding_sha256']==sha(out/'binding.json')
            rows.append(row);continue
        z=dict(np.load(context));az=dict(np.load(ap));j=int(az['selected_index'])
        np.testing.assert_array_equal(az['population_actions'][j],az['selected_actions'])
        assert az['population_actions'].shape==(300,25,2)
        assert np.unravel_index(np.argmin(az['cost_trace']),az['cost_trace'].shape)==(int(az['selected_iteration']),j)
        full=np.concatenate([z['prefix'],az['selected_actions']]);left=replay(c['seed'],full);right=replay(c['seed'],full)
        for key in left:np.testing.assert_array_equal(left[key],right[key])
        assert left['pixels'].shape==(8,224,224,3) and left['states'].shape==(36,7)
        np.testing.assert_array_equal(left['pixels'][:3],z['history_pixels'])
        np.testing.assert_array_equal(left['states'][[0,5,10]],z['history_states'])
        atomic_npz(target,**left,seed=np.asarray(c['seed']))
        row=dict(index=i,seed=c['seed'],file=target.name,file_sha256=sha(target),input_sha256=sha(context),
            action_file_sha256=sha(ap),binding_sha256=sha(out/'binding.json'),all_two_replays_exact=True,
            boundary_steps=int(left['boundary'][10:].sum()),scope='All selected futures retained, including contacts and exits.')
        atomic_json(result,row);rows.append(row)
    assert len(rows)==count
    atomic_json(out/'report.json',dict(status='PASS_COMPLETE_SELECTED_PHYSICS',binding=binding,cases=rows,
        elapsed_seconds=time.monotonic()-start,independent_replays=2*len(rows)))
    (out/'DONE').write_text('accepted\n')


if __name__ == '__main__':
    main()
