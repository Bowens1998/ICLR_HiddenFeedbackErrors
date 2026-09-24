"""Frozen S1 future propagation after independently accepted FP32 insertions.

No evaluator/constraint head is opened here. Free and reset conditions are
recomputed exactly once and copied bitwise across the two constraint labels.
"""
import argparse
from pathlib import Path
import time
import numpy as np
from fresh_inputs import (s1_arguments, s1_context, s1_path, s1_json, s1_sha,
                          s1_npz, s1_write_json)
from cache_inputs import s1_stream, s1_case, s1_load_model


def s1_read_qp(path, expected, *, ctx, group, stream, cache_sha):
    lock = s1_json(path, expected); n = ctx['spec']['count']
    identity = dict(protocol_sha256=ctx['protocol_sha256'], kind=ctx['kind'],
                    group=group, stream=stream, count=n)
    if (lock['status'] != 'ALL_S1_QP_SHARDS_ACCEPTED' or
            any(lock.get(k) != v for k, v in identity.items()) or
            lock['recipient_cache_sha256'] != cache_sha or
            not lock['selected_lock_sha256'] or not lock['qualification_sha256']):
        raise ValueError('QP lock not accepted or bound to this cache')
    replacements = np.empty((2, 2, n, 2, 192), np.float32)
    coverage = np.zeros(n, np.int64); rows = []
    for shard in lock['shards']:
        report = s1_json(shard['report'], shard['report_sha256'])
        if (report['status'] != 'ACCEPTED_COMPLETE_SHARD' or
                any(report.get(k) != v for k, v in identity.items()) or
                report['recipient_cache_sha256'] != cache_sha or
                report['donor_cache_sha256'] != lock['donor_cache_sha256'] or
                report['selected_lock_sha256'] != lock['selected_lock_sha256'] or
                report['qualification_sha256'] != lock['qualification_sha256']):
            raise ValueError('Changed shard acceptance identity')
        lo, hi = shard['start'], shard['stop']
        if (not 0 <= lo < hi <= n or report['start'] != lo or report['stop'] != hi or
                report['arrays'] != dict(path=shard['arrays'], sha256=shard['arrays_sha256']) or
                s1_sha(shard['arrays']) != shard['arrays_sha256']):
            raise ValueError('Invalid or mutated QP shard')
        with np.load(shard['arrays'], allow_pickle=False) as z:
            goals = z['goal_indices']; values = z['replacements']; norms = z['common_norm']
            if (goals.dtype != np.int64 or not np.array_equal(goals, np.arange(lo, hi)) or
                    values.dtype != np.float32 or values.shape != (2, 2, hi - lo, 2, 192) or
                    norms.shape != (hi - lo,) or not np.isfinite(norms).all() or
                    np.any(norms < 0) or not np.isfinite(values).all()):
                raise ValueError('Changed FP32 replacement shape or population')
            replacements[:, :, lo:hi] = values
        coverage[lo:hi] += 1; rows.append(shard)
    if not np.array_equal(coverage, np.ones(n, np.int64)):
        raise ValueError('QP shards must cover every goal exactly once')
    return replacements, lock


def s1_run(ctx, a):
    import torch
    from ac_rollout import rollout_population
    from adaptation_freeze import verify_frozen
    from score_feedback_ranking import state_hash
    from evaluation_precision import configure_evaluation_precision
    from image_planner_cost import ImagePlannerCost
    g, s = a.group, a.stream; n = ctx['spec']['count']
    cr = s1_json(a.recipient_cache, a.recipient_cache_sha256)
    expected = dict(status='PASS_S1_OBSERVED_AND_FREE_CACHE', **ctx['binding'], group=g, stream=s, count=n)
    if any(cr.get(k) != v for k, v in expected.items()):
        raise ValueError('Wrong cache, role or fixed population')
    if s1_sha(cr['arrays']['path']) != cr['arrays']['sha256']:
        raise ValueError('Changed cache arrays')
    obs = dict(np.load(cr['arrays']['path'], allow_pickle=False))
    if obs['free'].shape != (2, n, 5, 192) or obs['observed'].shape != (n, 5, 192):
        raise ValueError('Changed cache dimensions')
    replacements, qplock = s1_read_qp(a.qp_lock, a.qp_lock_sha256, ctx=ctx,
        group=g, stream=s, cache_sha=a.recipient_cache_sha256)
    refs = s1_stream(ctx, g, s)
    if refs['hashes'] != cr['input_reports_sha256']:
        raise ValueError('Raw source ancestry changed after caching')
    out = s1_path(ctx, 'rollout') / f'group_{g}_stream_{s}'; out.mkdir(parents=True, exist_ok=False)
    precision = configure_evaluation_precision(); torch.set_num_threads(4)
    tokens = np.empty((2, 2, n, 4, 5, 192), np.float32)
    checks = []; models = []; start = time.monotonic()
    for oi, objective in enumerate(ctx['cfg']['objectives']):
        model, tr, boundary, before, binding = s1_load_model(ctx, g, objective)
        if binding != cr['models'][oi]:
            raise ValueError('Model binding differs from accepted free cache')
        with torch.inference_mode():
            for i in range(n):
                z, az, _ = s1_case(ctx, refs, i)
                cost = ImagePlannerCost(model, z['history_pixels'], z['goal_pixels'], z['prefix'], tr['normalization'], None, 'latent')
                population = torch.as_tensor(az['population_actions'], device='cuda'); j = int(az['selected_index'])
                normalized = cost.normalized_actions(population); encoded = model.action_encoder(normalized)
                initial_before = cost.initial.clone(); normalized_before = normalized.clone()
                free = rollout_population(model, cost.initial, encoded, j)
                np.testing.assert_array_equal(free.cpu().numpy(), obs['free'][oi, i])
                np.testing.assert_array_equal(cost.initial.cpu().numpy(), obs['initial'][i])
                identity = rollout_population(model, cost.initial, encoded, j, free[0].clone())
                torch.testing.assert_close(identity, free, rtol=0, atol=0)
                reset = rollout_population(model, cost.initial, encoded, j, torch.as_tensor(obs['observed'][i, 0], device='cuda'))
                np.testing.assert_array_equal(reset[0].cpu().numpy(), obs['observed'][i, 0])
                for ci in range(2):
                    tokens[oi, ci, i, 0] = free.cpu().numpy()
                    tokens[oi, ci, i, 3] = reset.cpu().numpy()
                    for source in range(2):
                        replacement = replacements[oi, ci, i, source]
                        corrected = rollout_population(model, cost.initial, encoded, j,
                            torch.as_tensor(replacement, device='cuda'))
                        np.testing.assert_array_equal(corrected[0].cpu().numpy(), replacement)
                        if not torch.isfinite(corrected).all():
                            raise ValueError('Nonfinite corrected future; whole result blocked')
                        tokens[oi, ci, i, source + 1] = corrected.cpu().numpy()
                torch.testing.assert_close(cost.initial, initial_before, rtol=0, atol=0)
                torch.testing.assert_close(cost.normalized_actions(population), normalized_before, rtol=0, atol=0)
                checks.append(dict(objective=objective, case=i, free_exact=True,
                    identity_exact=True, inserted_fp32_exact=True, inputs_unchanged=True))
        verify_frozen(model, boundary)
        if state_hash(model) != before:
            raise ValueError('Model mutated during corrected propagation')
        models.append(binding); del model, boundary, cost; torch.cuda.empty_cache()
    np.testing.assert_array_equal(tokens[:, 0, :, 0], tokens[:, 1, :, 0])
    np.testing.assert_array_equal(tokens[:, 0, :, 3], tokens[:, 1, :, 3])
    if not np.isfinite(tokens).all():
        raise ValueError('Nonfinite complete rollout')
    s1_npz(out / 'data.npz', tokens=tokens, truth=obs['truth'], seeds=obs['seeds'],
           goal_indices=np.arange(n, dtype=np.int64))
    report = dict(status='PASS_S1_COMPLETE_ACCEPTED_ROLLOUT', **ctx['binding'],
        group=g, stream=s, count=n, objectives=ctx['cfg']['objectives'], constraints=['A', 'AC'],
        branches=['free', 'actual', 'donor', 'reset'], horizons=[5, 10, 15, 20, 25],
        arrays=dict(path=str(out / 'data.npz'), sha256=s1_sha(out / 'data.npz')),
        recipient_cache=dict(path=str(a.recipient_cache), sha256=a.recipient_cache_sha256),
        qp_lock=dict(path=str(a.qp_lock), sha256=a.qp_lock_sha256),
        selected_lock_sha256=qplock['selected_lock_sha256'], qualification_sha256=qplock['qualification_sha256'],
        input_reports_sha256=refs['hashes'], models=models, insertion_checks=checks,
        free_and_reset_bitwise_equal_across_constraints=True, head_weights_opened=False,
        source_sha256=s1_sha(__file__), elapsed_seconds=time.monotonic() - start, precision=precision,
        gpu=torch.cuda.get_device_name(), peak_allocated_bytes=torch.cuda.max_memory_allocated())
    s1_write_json(out / 'report.json', report); (out / 'DONE').write_text('S1 complete rollout accepted\n')


def main():
    p = argparse.ArgumentParser(); s1_arguments(p, role=False)
    for key in ['recipient-cache', 'recipient-cache-sha256', 'qp-lock', 'qp-lock-sha256']:
        p.add_argument('--' + key, required=True)
    p.add_argument('--group', type=int, required=True); p.add_argument('--stream', type=int, required=True)
    a = p.parse_args(); s1_run(s1_context(a), a)


if __name__ == '__main__':
    main()
