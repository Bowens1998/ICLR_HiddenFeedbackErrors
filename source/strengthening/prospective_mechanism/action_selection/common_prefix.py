"""Draft compact-archive adapter for the unchanged shared-prefix simulator rule.

Recipient suffix outcomes are separate from decision-time inputs. Donors expose
only their independently replayed selected prefix; they have no suffix bank.
This is data-flow separation, not operating-system access control.
"""
import importlib.util
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
LEGACY = ROOT / 'scripts/visual/prepare_feedback_ranking_bank.py'


def legacy():
    spec = importlib.util.spec_from_file_location('_s3_original_suffix_rule', LEGACY)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def require(value, message):
    if not value: raise ValueError(message)


def array(record, key, shape, dtype=None):
    value = record[key]
    require(isinstance(value, np.ndarray) and value.shape == shape, 'Wrong shape: ' + key)
    if dtype is not None: require(value.dtype == dtype, 'Wrong dtype: ' + key)
    require(np.isfinite(value).all(), 'Nonfinite array: ' + key)
    return value


def validate_compact(context, actions, physics, row, route):
    """Match the actual selected-iteration archive and full35-action replay."""
    require(type(route) is int and route in (0, 32, 64), 'Wrong legacy candidate namespace')
    seed = int(array(context, 'seed', (), np.int64))
    require(seed == row['seed'] and type(row['index']) is int and row['index'] >= 0, 'Wrong parent')
    prefix = array(context, 'prefix', (10, 2), np.float32)
    array(context, 'history_pixels', (3, 224, 224, 3), np.uint8)
    array(context, 'goal_pixels', (224, 224, 3), np.uint8)
    require(array(context, 'history_states', (3, 7)).dtype.kind == 'f', 'Nonfloating physical history')
    require(array(context, 'goal_state', (7,)).dtype.kind == 'f', 'Nonfloating physical goal')
    population = array(actions, 'population_actions', (300, 25, 2), np.float32)
    selected_actions = array(actions, 'selected_actions', (25, 2), np.float32)
    selected = int(array(actions, 'selected_index', (), np.int64))
    iteration = int(array(actions, 'selected_iteration', (), np.int64))
    require(0 <= selected < 300 and 0 <= iteration < 30, 'Wrong original search index')
    require((selected, iteration) == (row['selected_index'], row['selected_iteration']), 'Changed selection provenance')
    require(int(array(actions, 'seed', (), np.int64)) == seed, 'Wrong action parent')
    trace = array(actions, 'cost_trace', (30, 300), np.float64)
    costs = array(actions, 'population_costs', (300,), np.float64)
    require(np.unravel_index(np.argmin(trace), trace.shape) == (iteration, selected), 'Changed fixed original argmin')
    np.testing.assert_array_equal(costs, trace[iteration])
    np.testing.assert_array_equal(population[selected], selected_actions)
    states = array(physics, 'states', (36, 7))
    require(states.dtype.kind == 'f', 'Nonfloating physical replay')
    pixels = array(physics, 'pixels', (8, 224, 224, 3), np.uint8)
    np.testing.assert_array_equal(array(physics, 'actions', (35, 2), np.float32), np.concatenate([prefix, selected_actions]))
    require(int(array(physics, 'seed', (), np.int64)) == seed, 'Wrong physics parent')
    np.testing.assert_array_equal(pixels[:3], context['history_pixels'])
    np.testing.assert_array_equal(states[[0, 5, 10]], context['history_states'])
    return seed, selected, iteration, population, selected_actions


def _inputs(context, values, row, route, current):
    seed, selected, iteration, population, selected_actions = values
    return dict(index=np.asarray(row['index'], dtype=np.int64), seed=np.asarray(seed, dtype=np.int64),
        reference_route=np.asarray(route, dtype=np.int64), history_pixels=context['history_pixels'].copy(),
        prefix=context['prefix'].copy(), executed_actions=selected_actions[:5].copy(),
        current_pixels=current.copy(), goal_pixels=context['goal_pixels'].copy(),
        source_selected_index=np.asarray(selected, dtype=np.int64),
        source_selected_iteration=np.asarray(iteration, dtype=np.int64),
        source_first5_population=population[:, :5].copy(), source_population_actions=population.copy())


def build_donor_case(context, actions, physics, row, route):
    """No simulator construction, suffix selection, suffix replay, or outcomes."""
    values = validate_compact(context, actions, physics, row, route)
    return _inputs(context, values, row, route, physics['pixels'][3]), dict(
        history_and_action_alignment_exact=True, selected_physics_current_pixel_index=3,
        selected_physics_current_state_index=15, donor_suffixes_generated=False,
        scope='Selected prefix/current observation inherited from accepted full twice-replayed physics; no new replay.')


def build_compact_case(env_factory, context, actions, physics, row, route):
    """Reuse actual chooser/replay functions, never fabricate old parameters[]."""
    values = validate_compact(context, actions, physics, row, route)
    seed, selected, iteration, population, selected_actions = values
    old = legacy()
    suffixes, indices = old.select_suffix_candidates(population, selected, seed, route)
    results = []
    for suffix in suffixes:
        result = old.replay_branch(env_factory, seed, context['prefix'], selected_actions[:5], suffix)
        np.testing.assert_array_equal(result['history_pixels'], context['history_pixels'])
        np.testing.assert_array_equal(result['history_states'], context['history_states'])
        np.testing.assert_allclose(result['execution_states'], physics['states'][10:16], rtol=0, atol=1e-7)
        np.testing.assert_array_equal(result['current_pixels'], physics['pixels'][3])
        if results:
            for key in ('current_pixels', 'execution_states'):
                np.testing.assert_array_equal(result[key], results[0][key])
        require(np.isfinite(result['suffix_states']).all(), 'Nonfinite suffix state; no filtering')
        results.append(result)
    np.testing.assert_allclose(results[0]['suffix_states'], physics['states'][15:], rtol=0, atol=1e-7)
    np.testing.assert_array_equal(results[0]['terminal_pixels'], physics['pixels'][-1])
    repeat = old.replay_branch(env_factory, seed, context['prefix'], selected_actions[:5], suffixes[1])
    for key, value in results[1].items(): np.testing.assert_array_equal(repeat[key], value)
    inputs = _inputs(context, values, row, route, results[0]['current_pixels'])
    inputs.update(suffix_actions=suffixes, source_candidate_indices=indices)
    outcomes = dict(index=np.asarray(row['index'], dtype=np.int64), seed=np.asarray(seed, dtype=np.int64),
        reference_route=np.asarray(route, dtype=np.int64), goal_state=context['goal_state'].copy(),
        history_states=results[0]['history_states'], execution_states=results[0]['execution_states'],
        current_state=results[0]['suffix_states'][0], terminal_states=np.asarray([r['suffix_states'][-1] for r in results]),
        suffix_states=np.asarray([r['suffix_states'] for r in results]),
        **{key: np.asarray([r[key] for r in results]) for key in ('out_of_view', 'contact_points', 'terminated', 'truncated')})
    checks = dict(exact_history_replays=32, exact_common_current_observation=True,
        exact_common_execution_states=True,
        selected_suffix_anchor_max_abs=float(np.max(np.abs(results[0]['suffix_states'] - physics['states'][15:]))),
        selected_terminal_pixels_exact=True, zero_suffix_fresh_replay_exact=True,
        candidates_with_out_of_view=int(np.sum(outcomes['out_of_view'].any(axis=1))),
        candidates_with_contact=int(np.sum((outcomes['contact_points'] > 0).any(axis=1))))
    return inputs, outcomes, checks
