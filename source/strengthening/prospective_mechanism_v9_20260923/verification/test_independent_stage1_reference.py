"""Analytical tests only; no real experiment or v9 effects are loaded."""
import unittest
import copy
import math
import numpy as np
from independent_stage1_reference import (goal_vectors, count_weighted_bootstrap,
                                          independent_gelu_forward, validate_accepted_population,
                                          validate_reserved_head_binding)


def fixture(n=4, streams=3):
    p = np.zeros((2, 2, 2, n, streams, 4, 5, 6), dtype=np.float64)
    truth = np.zeros((2, n, streams, 5, 6), dtype=np.float64)
    for objective, scale in enumerate((1, 2)):
        for constraint, values in enumerate(((10, 7, 11, 0), (10, 8, 12, 0))):
            for branch, x in enumerate(values):
                p[:, objective, constraint, :, :, branch, :, 2] = x * scale
    return p, truth


class TestIndependentReference(unittest.TestCase):
    def test_hand_computed_effects_and_no_division_by_two(self):
        p, truth = fixture()
        v, d = goal_vectors(p, truth, goal_count=4, stream_count=3)
        np.testing.assert_array_equal(v, np.tile([36, 80, 144, 320, 15, 5, 60, 20], (4, 1)))
        np.testing.assert_array_equal(d['decoded_teacher']['U_A_goal'], np.full(4, 51))

    def test_group_and_stream_average_before_goal_resampling(self):
        p, truth = fixture()
        # Unequal group/stream effects; a transparent nested scalar calculation is oracle.
        for g in range(2):
            for i in range(4):
                for s in range(3):
                    p[g, 0, 1, i, s, 1, -1, 2] += g + i + s
        v, _ = goal_vectors(p, truth, goal_count=4, stream_count=3)
        manual = [sum(100 - (8 + g + i + s) ** 2 for g in range(2) for s in range(3)) / 6 for i in range(4)]
        np.testing.assert_allclose(v[:, 0], manual, rtol=0, atol=1e-13)

    def test_multiplicity_bootstrap_has_hand_calculated_quantiles(self):
        v = np.repeat(np.arange(1, 5, dtype=float)[:, None], 8, axis=1)
        ix = np.array([[0, 0, 0, 0], [1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3]], dtype=np.int64)
        rows, boot = count_weighted_bootstrap(v, ix, expected_draws=4)
        np.testing.assert_array_equal(boot, v)
        self.assertAlmostEqual(rows[0]['lower'], 1 + 3 * .00625)
        self.assertAlmostEqual(rows[0]['upper'], 1 + 3 * .99375)
        self.assertEqual(rows[0]['mean'], 2.5)
        self.assertEqual([r['family'] for r in rows], ['primary'] * 4 + ['secondary'] * 4)

    def test_null_and_negative_results_are_retained(self):
        v = np.zeros((4, 8)); v[:, 1] = -2
        rows, _ = count_weighted_bootstrap(v, np.tile(np.arange(4, dtype=np.int64), (4, 1)), expected_draws=4)
        self.assertEqual(len(rows), 8)
        self.assertEqual(rows[0]['classification'], 'unresolved')
        self.assertEqual(rows[1]['classification'], 'negative')

    def test_missing_goals_and_changed_free_copy_rejected(self):
        p, truth = fixture()
        with self.assertRaises(ValueError): goal_vectors(p[:, :, :, :3], truth, 4, 3)
        p[0, 0, 1, 0, 0, 0, 0, 2] += 1
        with self.assertRaises(ValueError): goal_vectors(p, truth, 4, 3)

    def test_nonfinite_and_bad_bootstrap_rejected(self):
        p, truth = fixture(); p[0, 0, 0, 0, 0, 1, 0, 2] = np.nan
        with self.assertRaises(ValueError): goal_vectors(p, truth, 4, 3)
        v = np.zeros((4, 8)); ix = np.zeros((4, 4), dtype=np.int64);ix[0, 0] = 4
        with self.assertRaises(ValueError): count_weighted_bootstrap(v, ix, 4)
        with self.assertRaises(ValueError): count_weighted_bootstrap(v, ix.astype(np.int32), 4)

    def test_independent_forward_constant_plus_linear_skip(self):
        width, dim, output = 4, 3, 2
        h = {'mean': np.array([1., 2., 3.]), 'scale': np.array([2., 3., 4.]),
             'target_mean': np.array([20., 30.]), 'target_scale': np.array([5., 7.])}
        dims = {'input': (width, dim), 'output': (output, width), 'skip': (output, dim)}
        for block in range(2):
            for layer in ['fc1', 'fc2']: dims[f'blocks.{block}.{layer}'] = (width, width)
        for name, shape in dims.items():
            h[name + '.weight'] = np.zeros(shape)
            if name != 'skip': h[name + '.bias'] = np.zeros(shape[0])
        h['output.bias'][:] = [2, -1]
        h['skip.weight'][:] = [[1, 2, 3], [-1, 0, 2]]
        x = np.array([[1., 2., 3.], [3., 5., 7.]])
        expected = np.array([[30., 23.], [60., 30.]])
        np.testing.assert_array_equal(independent_gelu_forward(x, h, width=width), expected)
        h['scale'][0] = 0
        with self.assertRaises(ValueError): independent_gelu_forward(x, h, width=width)

    def test_nonlinear_two_residual_blocks_against_scalar_erf(self):
        h = {'mean': np.array([2.]), 'scale': np.array([3.]),
             'target_mean': np.array([5.]), 'target_scale': np.array([7.])}
        for name in ['input', 'output', 'blocks.0.fc1', 'blocks.0.fc2', 'blocks.1.fc1', 'blocks.1.fc2']:
            h[name + '.weight'] = np.ones((1, 1))
            h[name + '.bias'] = np.zeros(1)
        h['skip.weight'] = np.array([[.25]])
        z = np.array([[-1.], [2.], [5.]])
        expected = []
        gelu = lambda x: x * (1 + math.erf(x / math.sqrt(2))) / 2
        for row in z:
            x = (float(row[0]) - 2) / 3
            hidden = gelu(x)
            hidden += .5 * gelu(hidden)
            hidden += .5 * gelu(hidden)
            expected.append([(hidden + .25 * x) * 7 + 5])
        np.testing.assert_allclose(independent_gelu_forward(z, h, width=1), expected,
                                   rtol=1e-14, atol=1e-14)


    def test_independent_population_receipt_checks_all_eight_streams(self):
        binding = dict(study_id='synthetic', design=dict(sha256='protocol'), axis_labels={'synthetic': True})
        accepted = dict(status='FULL_S1_CONFIRMATION_ACCEPTED_BEFORE_D_SCORING', study_id='synthetic',
            protocol_sha256='protocol', axis_labels={'synthetic': True}, complete_population_accepted=True,
            population_count=8, goal_count=256, D_weights_deserialized=False, D_predictions_opened=False, effects_computed=False,
            populations=[dict(group=g, stream=s, count=256, complete_finite_shapes=True,
                native_identity_checks=True, insertion_and_full_function_checks=True) for g in [0, 1] for s in range(4)])
        validate_accepted_population(accepted, binding)
        changes = [lambda a: a['populations'].pop(),
            lambda a: a['populations'].__setitem__(7, copy.deepcopy(a['populations'][6])),
            lambda a: a['populations'][0].update(count=255),
            lambda a: a['populations'][0].update(insertion_and_full_function_checks=False),
            lambda a: a.update(D_predictions_opened=True), lambda a: a.update(protocol_sha256='wrong')]
        for i, change in enumerate(changes):
            with self.subTest(corruption=i):
                bad=copy.deepcopy(accepted); change(bad)
                with self.assertRaises(ValueError): validate_accepted_population(bad, binding)

    def test_reserved_head_must_match_accepted_D_hash(self):
        accepted = dict(heads=[dict(group=g, head_role=role, sha256=str(g)*64)
            for g in [0, 1] for role in ['A', 'C', 'D']])
        supplied = [dict(group=g, path=f'unopened_head_{g}', sha256=str(g)*64) for g in [0, 1]]
        validate_reserved_head_binding(accepted, supplied)
        bad=copy.deepcopy(supplied); bad[1]['sha256']='f'*64
        with self.assertRaises(ValueError): validate_reserved_head_binding(accepted, bad)
        with self.assertRaises(ValueError): validate_reserved_head_binding(accepted, supplied[::-1])
        bad=copy.deepcopy(accepted); bad['heads']=[h for h in bad['heads'] if not (h['group']==1 and h['head_role']=='D')]
        with self.assertRaises(ValueError): validate_reserved_head_binding(bad, supplied)


if __name__ == '__main__':
    unittest.main(verbosity=2)
