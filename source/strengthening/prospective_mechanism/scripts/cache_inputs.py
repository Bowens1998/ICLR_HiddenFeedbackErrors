"""Observed-token and exact native 300-population free-rollout cache for S1."""
import argparse
from pathlib import Path
import sys
import time
import numpy as np
from fresh_inputs import (s1_arguments, s1_context, s1_bank, s1_path, s1_json,
                          s1_sha, s1_resolve, s1_npz, s1_write_json)


def s1_stream(ctx, group, stream):
    if group not in [0, 1] or stream not in range(4):
        raise ValueError('Unknown frozen group or stream')
    bank, _, manifest = s1_bank(ctx); route = 4 * group + stream
    actions = s1_path(ctx, 'actions') / f'route_{route}'
    physics = s1_path(ctx, 'physics') / f'route_{route}'
    ar, pr = s1_json(actions / 'report.json'), s1_json(physics / 'report.json')
    count = ctx['spec']['count']
    if (ar['status'] != 'PASS_COMPACT_FIXED_REFERENCE_SEARCH' or
            pr['status'] != 'PASS_COMPLETE_SELECTED_PHYSICS' or
            not all((p / 'DONE').exists() for p in [bank, actions, physics]) or
            len(ar['cases']) != count or len(pr['cases']) != count):
        raise ValueError('Unaccepted or incomplete fixed reference inputs')
    for report in [ar, pr]:
        if any(report['binding'].get(k) != v for k, v in ctx['binding'].items()):
            raise ValueError('Changed input binding')
    if (pr['binding']['action_report_sha256'] != s1_sha(actions / 'report.json') or
            ar['binding']['bank_manifest_sha256'] != s1_sha(bank / 'manifest.json')):
        raise ValueError('Broken source ancestry')
    return dict(bank=bank, actions=actions, physics=physics, bm=manifest, ar=ar, pr=pr,
                hashes={str(p): s1_sha(p) for p in [bank / 'manifest.json', bank / 'role.json',
                        actions / 'report.json', physics / 'report.json']})


def s1_case(ctx, refs, i):
    if i not in range(ctx['spec']['count']):
        raise ValueError('Case outside fixed population')
    c, ar, pr = refs['bm']['cases'][i], refs['ar']['cases'][i], refs['pr']['cases'][i]
    if c['index'] != ar['index'] or c['index'] != pr['index'] or c['index'] != i:
        raise ValueError('Changed case roster')
    if c['seed'] != ar['seed'] or c['seed'] != pr['seed']:
        raise ValueError('Changed physical parent')
    paths = [refs['bank'] / f'case_{i:03d}.npz', refs['actions'] / ar['file'], refs['physics'] / pr['file']]
    for p, expected in zip(paths, [c['sha256'], ar['file_sha256'], pr['file_sha256']], strict=True):
        if s1_sha(p) != expected:
            raise ValueError(f'Changed raw input: {p}')
    z, az, pz = [dict(np.load(p, allow_pickle=False)) for p in paths]
    if az['population_actions'].shape != (300, 25, 2) or pz['states'].shape != (36, 7):
        raise ValueError('Changed native population or physical horizon')
    np.testing.assert_array_equal(az['selected_actions'], pz['actions'][10:])
    np.testing.assert_array_equal(az['population_actions'][int(az['selected_index'])], az['selected_actions'])
    np.testing.assert_array_equal(z['history_pixels'], pz['pixels'][:3])
    np.testing.assert_array_equal(z['history_states'], pz['states'][[0, 5, 10]])
    return z, az, pz


def s1_load_model(ctx, group, objective):
    import torch
    from factorial_model import make_model
    from adaptation_freeze import configure_dynamics_only, verify_frozen
    from score_feedback_ranking import state_hash
    entry = ctx['sources']['model_bindings']
    manifest = s1_json(s1_resolve(ctx, entry['path']), entry['sha256'])
    gr = next(r for r in manifest['groups'] if r['group'] == group)
    if objective not in ctx['cfg']['objectives']:
        raise ValueError('Model objective outside fixed roster')
    selected = next(r for r in gr['models'] if r['objective'] == objective)
    original = gr['original']
    for path, expected in [(original['checkpoint'], original['checkpoint_sha256']),
                           (selected['checkpoint'], selected['checkpoint_sha256']),
                           (Path(selected['checkpoint']).parent / 'report.json', selected['report_sha256'])]:
        if s1_sha(path) != expected:
            raise ValueError(f'Changed model asset: {path}')
    tr = s1_json(original['summary_path'], original['summary_sha256'])
    config = ctx['base'] / 'assets/pusht-v1/models/config.json'
    if s1_sha(config) != tr['config_sha256']:
        raise ValueError('Changed original model configuration/normalization binding')
    model = make_model(ctx['base'] / 'releases/visual-v1/official', config, gr['architecture'], tr['seed'])
    model.load_state_dict(torch.load(original['checkpoint'], map_location='cpu', weights_only=True), strict=True)
    model = model.cuda(); boundary = configure_dynamics_only(model)
    model.load_state_dict(torch.load(selected['checkpoint'], map_location='cpu', weights_only=True), strict=True)
    model.eval(); verify_frozen(model, boundary)
    binding = dict(group=group, architecture=gr['architecture'], original=original,
                   selected=selected, model_bindings_sha256=entry['sha256'])
    return model, tr, boundary, state_hash(model), binding


def s1_cache(ctx, group, stream):
    import torch
    from ac_rollout import rollout_population
    from adaptation_freeze import verify_frozen
    from score_feedback_ranking import state_hash
    from evaluation_precision import configure_evaluation_precision
    from image_planner_cost import ImagePlannerCost
    refs = s1_stream(ctx, group, stream); n = ctx['spec']['count']
    content_path = ctx['artifacts'] / 'content_lineage.json'
    content = s1_json(content_path)
    if (content.get('status') != 'PASS_S1_CONTENT_ISOLATION' or
            content.get('protocol_sha256') != ctx['protocol_sha256'] or
            content.get('sources_sha256') != ctx['sources_sha256'] or
            content.get('kind') != ctx['kind']):
        raise ValueError('Complete raw content isolation must pass before model evaluation')
    for path, checksum in refs['hashes'].items():
        # Action reports contain no pixels; their ancestry is checked by s1_stream.
        if '/actions/' not in path and content['input_files_sha256'].get(path) != checksum:
            raise ValueError('Cache source is outside accepted pixel audit')
    out = s1_path(ctx, 'cache') / f'group_{group}_stream_{stream}'
    out.mkdir(parents=True, exist_ok=False)
    precision = configure_evaluation_precision(); torch.set_num_threads(4)
    initial = np.empty((n, 3, 192), np.float32); observed = np.empty((n, 5, 192), np.float32)
    truth = np.empty((n, 5, 6), np.float64); actions = np.empty((n, 10), np.float32)
    seeds = np.empty(n, np.int64)
    objectives = ctx['cfg']['objectives'] if ctx['role'] == 'recipient' else ctx['cfg']['objectives'][:1]
    free = np.empty((2, n, 5, 192), np.float32) if ctx['role'] == 'recipient' else None
    im = torch.tensor([.485, .456, .406], device='cuda')[None, None, :, None, None]
    sd = torch.tensor([.229, .224, .225], device='cuda')[None, None, :, None, None]
    start = time.monotonic(); models = []
    for mi, objective in enumerate(objectives):
        model, tr, boundary, before, binding = s1_load_model(ctx, group, objective)
        with torch.inference_mode():
            for i in range(n):
                z, az, pz = s1_case(ctx, refs, i)
                cost = ImagePlannerCost(model, z['history_pixels'], z['goal_pixels'], z['prefix'], tr['normalization'], None, 'latent')
                if mi == 0:
                    initial[i] = cost.initial.cpu().numpy(); seq = []
                    for pixel in pz['pixels'][3:]:
                        tensor = torch.as_tensor(pixel, device='cuda').permute(2, 0, 1)[None, None].float() / 255.
                        seq.append(model.encode({'pixels': (tensor - im) / sd})['emb'][0, 0])
                    observed[i] = torch.stack(seq).cpu().numpy()
                    ss = pz['states'][[15, 20, 25, 30, 35]]
                    truth[i] = np.c_[ss[:, :4], np.sin(ss[:, 4]), np.cos(ss[:, 4])]
                    actions[i] = az['selected_actions'][:5].reshape(10); seeds[i] = int(z['seed'])
                else:
                    np.testing.assert_array_equal(initial[i], cost.initial.cpu().numpy())
                if ctx['role'] == 'recipient':
                    population = torch.as_tensor(az['population_actions'], device='cuda')
                    encoded = model.action_encoder(cost.normalized_actions(population)); selected = int(az['selected_index'])
                    pred = rollout_population(model, cost.initial, encoded, selected)
                    _, native = cost(population)
                    torch.testing.assert_close(pred[-1], native[selected], rtol=0, atol=0)
                    identity = rollout_population(model, cost.initial, encoded, selected, pred[0].clone())
                    torch.testing.assert_close(pred, identity, rtol=0, atol=0)
                    if not torch.isfinite(pred).all():
                        raise ValueError('Nonfinite prediction; no recipient dropped')
                    free[mi, i] = pred.cpu().numpy()
        verify_frozen(model, boundary)
        if state_hash(model) != before:
            raise ValueError('Model mutated during cache construction')
        models.append(binding); del model, boundary, cost; torch.cuda.empty_cache()
    arrays = dict(initial=initial, observed=observed, truth=truth, known_actions=actions, seeds=seeds)
    if free is not None:
        arrays['free'] = free
    if not all(np.isfinite(a).all() for a in arrays.values()):
        raise ValueError('Nonfinite cache input')
    s1_npz(out / 'data.npz', **arrays)
    report = dict(status='PASS_S1_OBSERVED_AND_FREE_CACHE', **ctx['binding'],
        group=group, stream=stream, count=n, cases=list(range(n)), objectives=objectives,
        arrays=dict(path=str(out / 'data.npz'), sha256=s1_sha(out / 'data.npz')),
        input_reports_sha256=refs['hashes'], models=models,
        content_lineage=dict(path=str(content_path), sha256=s1_sha(content_path)),
        source_sha256=s1_sha(__file__), native_endpoint_and_identity_replacement_exact=ctx['role'] == 'recipient',
        frozen_tensors_unchanged=True, donor_model_predictions_generated=False,
        elapsed_seconds=time.monotonic() - start, precision=precision,
        gpu=torch.cuda.get_device_name(), peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        scope='Observed inputs and native free rollouts only; labels excluded from intervention construction. No fitted/evaluation head opened.')
    s1_write_json(out / 'report.json', report); (out / 'DONE').write_text('S1 cache accepted\n')


def main():
    p = argparse.ArgumentParser(); s1_arguments(p)
    p.add_argument('--group', type=int, required=True); p.add_argument('--stream', type=int, required=True)
    a = p.parse_args(); s1_cache(s1_context(a), a.group, a.stream)


if __name__ == '__main__':
    main()
