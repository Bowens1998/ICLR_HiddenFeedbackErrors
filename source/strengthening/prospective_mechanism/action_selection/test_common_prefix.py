"""Synthetic stateful-simulator parity, real legacy chooser/replay, no real data."""
import copy
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import common_prefix as adapter


class StatefulSimulator:
    constructions = 0
    def __init__(self):
        type(self).constructions += 1
        shape = SimpleNamespace(cache_bb=lambda: SimpleNamespace(left=20, bottom=20, right=40, top=40))
        self.agent = self.block = SimpleNamespace(shapes=[shape]); self.n_contact_points = 0

    def reset(self, seed):
        self.state = np.array([100. + seed % 13, 120., 150., 160., 0., 0., 0.]); self.steps = 0
        return {'state': self.state.copy()}, {}

    def step(self, action):
        velocity = self.state[5:7].copy(); self.state[:2] += action
        self.state[2:4] += .3 * action + .2 * velocity
        self.state[4] += float(action[0] - action[1]) / 100; self.state[5:7] = action
        self.steps += 1; self.n_contact_points = int(np.linalg.norm(action) > .1)
        return {'state': self.state.copy()}, 0., False, False, {}

    def render(self):
        return np.full((224, 224, 3), int(self.state[:4].sum() + self.steps) % 256, np.uint8)

    def close(self): pass


def fixture():
    rng = np.random.default_rng(812); seed, selected, iteration = 73, 7, 17
    prefix = rng.uniform(-.35, .35, (10, 2)).astype(np.float32)
    population = rng.uniform(-.35, .35, (300, 25, 2)).astype(np.float32)
    selected_actions = population[selected].copy(); full = np.concatenate([prefix, selected_actions])
    env = StatefulSimulator(); obs, _ = env.reset(seed)
    states = [obs['state']]; pixels = [env.render()]
    for step, action in enumerate(full, 1):
        obs, *_ = env.step(action); states.append(obs['state'])
        if step % 5 == 0: pixels.append(env.render())
    states, pixels = np.stack(states), np.stack(pixels)
    context = dict(seed=np.asarray(seed, np.int64), prefix=prefix, history_pixels=pixels[:3],
        history_states=states[[0, 5, 10]], goal_pixels=pixels[-1], goal_state=states[-1])
    trace = np.ones((30, 300), np.float64); trace[iteration, selected] = -1
    actions = dict(population_actions=population, selected_actions=selected_actions,
        selected_index=np.asarray(selected, np.int64), selected_iteration=np.asarray(iteration, np.int64),
        seed=np.asarray(seed, np.int64), cost_trace=trace, population_costs=trace[iteration].copy())
    physics = dict(states=states, pixels=pixels, actions=full, seed=np.asarray(seed, np.int64))
    row = dict(index=0, seed=seed, selected_index=selected, selected_iteration=iteration)
    return context, actions, physics, row


class CompactTests(unittest.TestCase):
    def test_exact_old_build_case_parity_with_real_legacy_functions(self):
        context, actions, physics, row = fixture(); old = adapter.legacy()
        # Legacy parameters[] exists only in this explicit synthetic parity fixture.
        parameters = np.zeros((30, 300, 5, 10), np.float32)
        parameters[row['selected_iteration']] = actions['population_actions'].reshape(300, 5, 10)
        archive = dict(parameters=parameters, selected_actions=actions['selected_actions'],
            selected_states=physics['states'][10:], terminal_pixels=physics['pixels'][-1])
        old_row = dict(row, selected_candidate=row['selected_index'])
        expected = old.build_case(StatefulSimulator, context, archive, old_row, 64)
        actual = adapter.build_compact_case(StatefulSimulator, context, actions, physics, row, 64)
        for left, right in zip(expected[:2], actual[:2], strict=True):
            self.assertEqual(set(left), set(right))
            for key in left: np.testing.assert_array_equal(left[key], right[key])
        self.assertEqual(expected[2], actual[2])

    def test_all32_branches_share_exact_prefix_including_velocity(self):
        context, actions, physics, row = fixture()
        inputs, outcomes, checks = adapter.build_compact_case(StatefulSimulator, context, actions, physics, row, 32)
        self.assertTrue(checks['exact_common_execution_states'])
        np.testing.assert_array_equal(outcomes['suffix_states'][:, 0], np.repeat(physics['states'][15:16], 32, axis=0))
        env = StatefulSimulator(); env.reset(row['seed'])
        for u in np.concatenate([context['prefix'], actions['selected_actions'][:5], inputs['suffix_actions'][2]]): obs, *_ = env.step(u)
        np.testing.assert_array_equal(outcomes['terminal_states'][2], obs['state'])
        self.assertTrue(np.any(outcomes['current_state'][5:] != 0))

    def test_hidden_physical_goal_does_not_enter_inputs(self):
        context, actions, physics, row = fixture()
        before = adapter.build_compact_case(StatefulSimulator, context, actions, physics, row, 0)
        altered = dict(context, goal_state=np.full(7, 1e9))
        after = adapter.build_compact_case(StatefulSimulator, altered, actions, physics, row, 0)
        for key in before[0]: np.testing.assert_array_equal(before[0][key], after[0][key])
        self.assertFalse(np.array_equal(before[1]['goal_state'], after[1]['goal_state']))
        self.assertFalse(any('state' in key or key in ('terminal_pixels', 'out_of_view') for key in before[0]))

    def test_donor_has_no_suffix_generation_or_new_physics(self):
        context, actions, physics, row = fixture(); constructed = StatefulSimulator.constructions
        with patch.object(adapter, 'legacy', side_effect=AssertionError('No suffix chooser or replay for donors')):
            inputs, checks = adapter.build_donor_case(context, actions, physics, row, 64)
        self.assertEqual(StatefulSimulator.constructions, constructed)
        self.assertFalse(checks['donor_suffixes_generated'])
        self.assertNotIn('suffix_actions', inputs); self.assertNotIn('source_candidate_indices', inputs)
        np.testing.assert_array_equal(inputs['current_pixels'], physics['pixels'][3])

    def test_shape_dtype_selection_route_corruption_blocks(self):
        for kind in ['shape', 'dtype', 'selection', 'action', 'history', 'route', 'seed']:
            with self.subTest(kind=kind):
                context, actions, physics, row = fixture(); route = 0
                if kind == 'shape': actions['population_actions'] = actions['population_actions'][:299]
                if kind == 'dtype': actions['cost_trace'] = actions['cost_trace'].astype(np.float32)
                if kind == 'selection': actions['cost_trace'][1, 2] = -2
                if kind == 'action': physics['actions'][10, 0] += 1
                if kind == 'history': physics['states'][0, 0] += 1
                if kind == 'route': route = 8
                if kind == 'seed': actions['seed'] = np.asarray(74, np.int64)
                with self.assertRaises((ValueError, AssertionError)):
                    adapter.build_donor_case(context, actions, physics, row, route)

    def test_selected_suffix_physics_corruption_is_not_filtered(self):
        context, actions, physics, row = fixture(); physics['states'][-1, 2] += .1
        with self.assertRaises(AssertionError):
            adapter.build_compact_case(StatefulSimulator, context, actions, physics, row, 0)

    def test_candidate_namespace_and_duplicate_actions_retained(self):
        context, actions, physics, row = fixture(); old = adapter.legacy()
        _, first = old.select_suffix_candidates(actions['population_actions'], row['selected_index'], row['seed'], 0)
        _, last = old.select_suffix_candidates(actions['population_actions'], row['selected_index'], row['seed'], 64)
        self.assertFalse(np.array_equal(first, last))
        population = np.zeros((300, 25, 2), np.float32)
        suffixes, indices = old.select_suffix_candidates(population, 7, 73, 64)
        self.assertEqual(len(indices), 32); self.assertEqual(len(set(indices[2:])), 30)
        self.assertFalse(suffixes.any())


if __name__ == '__main__': unittest.main()
