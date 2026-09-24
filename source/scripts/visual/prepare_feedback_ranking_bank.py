"""Replay a shared executed prefix before constructing a suffix-ranking bank.

The scorer archive contains only observations available at decision time and
action proposals. Future simulator states and the true goal state are written to
a separate outcomes directory. This is an engineering bank, not a study result.
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np


CANDIDATES = 32
SUBSAMPLE_SEED = 1378001


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def select_suffix_candidates(population, selected_index, seed, route):
    """Sample source indices without replacement; never inspect outcomes."""
    population = np.asarray(population)
    assert population.shape == (300, 25, 2)
    assert 0 <= selected_index < 300 and np.isfinite(population).all()
    rng = np.random.default_rng(np.random.SeedSequence([seed, route, SUBSAMPLE_SEED]))
    remaining = np.delete(np.arange(300), selected_index)
    sampled = rng.choice(remaining, size=CANDIDATES - 2, replace=False)
    indices = np.concatenate([[selected_index, -1], sampled]).astype(np.int64)
    suffixes = np.concatenate([
        population[selected_index:selected_index + 1, 5:],
        np.zeros((1, 20, 2), dtype=population.dtype),
        population[sampled, 5:],
    ])
    # Distinct source indices, not outcome-dependent action deduplication.
    assert len(np.unique(indices[indices >= 0])) == CANDIDATES - 1
    return suffixes, indices


def outside_view(env):
    return any(
        min(box.left, box.bottom) < 8 or max(box.right, box.top) > 504
        for body in (env.agent, env.block)
        for shape in body.shapes
        for box in (shape.cache_bb(),)
    )


def replay_branch(env_factory, seed, prefix, executed, suffix):
    """Restore the entire physical history by reset+actions, never pose-only reset."""
    assert prefix.shape == (10, 2) and executed.shape == (5, 2)
    assert suffix.shape == (20, 2)
    env = env_factory()
    try:
        obs, _ = env.reset(seed=int(seed))
        history_pixels = [env.render().copy()]
        history_states = [obs['state'].copy()]
        for step, action in enumerate(prefix, start=1):
            obs, *_ = env.step(action)
            if step % 5 == 0:
                history_pixels.append(env.render().copy())
                history_states.append(obs['state'].copy())
        execution_states = [obs['state'].copy()]
        for action in executed:
            obs, *_ = env.step(action)
            execution_states.append(obs['state'].copy())
        current_pixels = env.render().copy()
        suffix_states = [obs['state'].copy()]
        boundary, contacts, terminated, truncated = [], [], [], []
        for action in suffix:
            obs, _, term, trunc, _ = env.step(action)
            suffix_states.append(obs['state'].copy())
            boundary.append(outside_view(env))
            contacts.append(int(env.n_contact_points))
            terminated.append(bool(term))
            truncated.append(bool(trunc))
        return dict(
            history_pixels=np.asarray(history_pixels),
            history_states=np.asarray(history_states),
            execution_states=np.asarray(execution_states),
            current_pixels=current_pixels,
            suffix_states=np.asarray(suffix_states),
            terminal_pixels=env.render().copy(),
            out_of_view=np.asarray(boundary, dtype=bool),
            contact_points=np.asarray(contacts, dtype=np.int64),
            terminated=np.asarray(terminated, dtype=bool),
            truncated=np.asarray(truncated, dtype=bool),
        )
    finally:
        env.close()


def build_case(env_factory, context, archive, row, route):
    """Create separated inputs/outcomes and verify shared-prefix simulator anchors."""
    seed = int(context['seed'])
    assert seed == row['seed']
    iteration, selected = row['selected_iteration'], row['selected_candidate']
    population = archive['parameters'][iteration].reshape(300, 25, 2).copy()
    selected_actions = archive['selected_actions']
    assert selected_actions.shape == (25, 2)
    np.testing.assert_array_equal(population[selected], selected_actions)
    executed = selected_actions[:5].copy()
    suffixes, indices = select_suffix_candidates(population, selected, seed, route)
    results = []
    for suffix in suffixes:
        result = replay_branch(env_factory, seed, context['prefix'], executed, suffix)
        np.testing.assert_array_equal(result['history_pixels'], context['history_pixels'])
        np.testing.assert_array_equal(result['history_states'], context['history_states'])
        np.testing.assert_allclose(result['execution_states'], archive['selected_states'][:6], rtol=0, atol=1e-7)
        if results:
            for key in ('current_pixels', 'execution_states'):
                np.testing.assert_array_equal(result[key], results[0][key])
        assert np.isfinite(result['suffix_states']).all()
        results.append(result)
    np.testing.assert_allclose(results[0]['suffix_states'], archive['selected_states'][5:], rtol=0, atol=1e-7)
    np.testing.assert_array_equal(results[0]['terminal_pixels'], archive['terminal_pixels'])
    repeat = replay_branch(env_factory, seed, context['prefix'], executed, suffixes[1])
    for key, value in results[1].items():
        np.testing.assert_array_equal(repeat[key], value)
    scorer_input = dict(
        index=np.asarray(row['index']), seed=np.asarray(seed), reference_route=np.asarray(route),
        history_pixels=context['history_pixels'].copy(), prefix=context['prefix'].copy(),
        executed_actions=executed, current_pixels=results[0]['current_pixels'],
        goal_pixels=context['goal_pixels'].copy(), suffix_actions=suffixes,
        source_candidate_indices=indices, source_selected_index=np.asarray(selected),
        source_selected_iteration=np.asarray(iteration),
        source_first5_population=population[:, :5].copy(),
        # Retain full proposal shape for exact action-encoder arithmetic replay.
        # These are proposals, not observations of their future outcomes.
        source_population_actions=population,
    )
    outcomes = dict(
        index=np.asarray(row['index']), seed=np.asarray(seed), reference_route=np.asarray(route),
        goal_state=context['goal_state'].copy(),
        history_states=results[0]['history_states'], execution_states=results[0]['execution_states'],
        current_state=results[0]['suffix_states'][0],
        terminal_states=np.asarray([result['suffix_states'][-1] for result in results]),
        suffix_states=np.asarray([result['suffix_states'] for result in results]),
        **{key: np.asarray([result[key] for result in results])
           for key in ('out_of_view', 'contact_points', 'terminated', 'truncated')},
    )
    checks = dict(
        exact_history_replays=CANDIDATES,
        exact_common_current_observation=True,
        exact_common_execution_states=True,
        selected_suffix_anchor_max_abs=float(np.max(np.abs(results[0]['suffix_states'] - archive['selected_states'][5:]))),
        selected_terminal_pixels_exact=True,
        zero_suffix_fresh_replay_exact=True,
        candidates_with_out_of_view=int(np.sum(outcomes['out_of_view'].any(axis=1))),
        candidates_with_contact=int(np.sum((outcomes['contact_points'] > 0).any(axis=1))),
    )
    return scorer_input, outcomes, checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('plan', 'contexts', 'runs', 'simulator', 'output'):
        parser.add_argument('--' + key, required=True)
    parser.add_argument('--group', type=int, choices=range(6), required=True)
    parser.add_argument('--cases', type=int, default=8)
    args = parser.parse_args()
    started = time.monotonic()
    plan_path, contexts = Path(args.plan), Path(args.contexts)
    plan = json.loads(plan_path.read_text())
    route = 16 * args.group
    entry = plan['routes'][route]
    model = plan['models'][entry['model_index']]
    assert entry['algorithm'] == 'random' and entry['parameterization'] == 'full'
    assert entry['adaptation_condition'] == model['adaptation_condition'] == 'original'
    assert model['arm'].endswith('_jepa')
    run = Path(args.runs) / f'job_{route}'
    summary = json.loads((run / 'summary.json').read_text())
    acceptance = json.loads((run / 'acceptance.json').read_text())
    artifacts = json.loads((run / 'artifact_manifest.json').read_text())
    bank = json.loads((contexts / 'manifest.json').read_text())
    assert summary['route_index'] == route and summary['model_index'] == entry['model_index']
    assert summary['algorithm'] == 'random' and summary['parameterization'] == 'full'
    assert sha(run / 'summary.json') == artifacts['summary.json']
    assert summary['hashes']['model_manifest'] == sha(plan_path)
    assert summary['hashes']['weights'] == model['weights_sha256']
    assert summary['hashes']['bank_manifest'] == plan['bank_manifest_sha256'] == sha(contexts / 'manifest.json')
    assert acceptance['population_replay_exact_all_cases'] and acceptance['model_free_simulator_replay']
    assert acceptance['cases'] == len(summary['cases']) == len(bank['cases'])
    verifier = Path(__file__).with_name('accept_task_coordinate_population_planner.py')
    assert acceptance['verifier_sha256'] == sha(verifier)
    assert 1 <= args.cases <= len(bank['cases'])
    source_manifest = json.loads((run / 'source' / 'manifest.json').read_text())
    for name, expected in source_manifest.items():
        assert sha(run / 'source' / name) == expected, f'Changed saved source: {name}'
    simulator_file = Path(args.simulator) / 'stable_worldmodel/envs/pusht/env.py'
    assert sha(simulator_file) == source_manifest['env.py']
    sys.path.insert(0, args.simulator)
    from stable_worldmodel.envs.pusht.env import PushT
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    (out / 'scorer_inputs').mkdir()
    (out / 'outcomes').mkdir()
    rows = []
    for index in range(args.cases):
        item, row = bank['cases'][index], summary['cases'][index]
        assert item['index'] == row['index'] == index and item['seed'] == row['seed']
        context_path = contexts / f'case_{index:03d}.npz'
        archive_path = run / f'case_{index:03d}_predictions.npz'
        assert sha(context_path) == item['sha256']
        assert sha(archive_path) == artifacts[archive_path.name]
        with np.load(context_path, allow_pickle=False) as z:
            context = {key: z[key].copy() for key in ('seed', 'history_pixels', 'history_states', 'prefix', 'goal_pixels', 'goal_state')}
        with np.load(archive_path, allow_pickle=False) as z:
            archive = {key: z[key].copy() for key in ('parameters', 'selected_actions', 'selected_states', 'terminal_pixels')}
        inputs, outcomes, checks = build_case(lambda: PushT(resolution=224), context, archive, row, route)
        input_name, outcome_name = f'scorer_inputs/case_{index:03d}.npz', f'outcomes/case_{index:03d}.npz'
        np.savez_compressed(out / input_name, **inputs)
        np.savez_compressed(out / outcome_name, **outcomes)
        record = dict(index=index, seed=item['seed'], input_file=input_name, input_sha256=sha(out / input_name),
                      outcome_file=outcome_name, outcome_sha256=sha(out / outcome_name),
                      source_context_sha256=sha(context_path), source_archive_sha256=sha(archive_path),
                      source_selected_iteration=row['selected_iteration'], source_selected_index=row['selected_candidate'],
                      **checks)
        rows.append(record)
        print(json.dumps(record), flush=True)
    manifest = dict(
        status='ENGINEERING_SHARED_PREFIX_RANKING_BANK', group=args.group, reference_route=route,
        candidates=CANDIDATES, executed_actions=5, future_actions=20, subsample_seed_namespace=SUBSAMPLE_SEED,
        candidate_rule='Selected original suffix, zero suffix, then 30 source indices sampled uniformly without replacement from the other 299 candidates; no action deduplication or outcome filtering.',
        scope='Engineering diagnostic only. Shared real observation after an already executed prefix. No study efficacy result or independent confirmation claim.',
        cases=rows, elapsed_seconds=time.monotonic() - started,
        bindings=dict(plan_path=str(plan_path.resolve()), plan_sha256=sha(plan_path),
                      contexts_manifest_sha256=sha(contexts / 'manifest.json'),
                      source_run=str(run.resolve()), source_summary_sha256=sha(run / 'summary.json'),
                      source_acceptance_sha256=sha(run / 'acceptance.json'),
                      source_artifact_manifest_sha256=sha(run / 'artifact_manifest.json'),
                      source_code_manifest_sha256=sha(run / 'source' / 'manifest.json'),
                      generator_sha256=sha(__file__), simulator_sha256=sha(simulator_file),
                      source_acceptance_verifier_sha256=sha(verifier)),
        scorer_input_keys=sorted(inputs), outcome_keys=sorted(outcomes),
    )
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    result = dict(status='PASS_SHARED_PREFIX_SIMULATOR_BANK', cases=len(rows), candidates_per_case=CANDIDATES,
                  manifest_sha256=sha(out / 'manifest.json'), exact_history_replays=sum(r['exact_history_replays'] for r in rows),
                  common_current_observation_exact_all_cases=True, selected_suffix_matches_original_all_cases=True,
                  zero_fresh_replay_exact_all_cases=True, outcome_filtering=False,
                  scope='Creation-time simulator anchors and fresh zero replay; no model scoring or efficacy assertion.')
    (out / 'acceptance.json').write_text(json.dumps(result, indent=2) + '\n')
    (out / 'COMPLETE').write_text('Engineering bank; scorer inputs and simulator outcomes stored separately.\n')


if __name__ == '__main__':
    main()
