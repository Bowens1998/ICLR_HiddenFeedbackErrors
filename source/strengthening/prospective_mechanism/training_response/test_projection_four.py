"""Focused synthetic-only four-family coupling/acceptance tests; no real arrays."""
import copy
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import projection_four as p


def fixture():
    d = p.LATENT_DIM
    h = dict(mean=np.zeros(d), scale=np.ones(d), target_mean=np.zeros(6), target_scale=np.arange(1., 7.),
             **{'0.weight': np.zeros((2, d)), '0.bias': np.ones(2),
                '2.weight': np.eye(2), '2.bias': np.ones(2),
                '4.weight': np.zeros((6, 2)), '4.bias': np.zeros(6)})
    h['0.weight'][0, 0] = 1.
    h['4.weight'][0, 0] = 1.
    predicted = {o: np.zeros(d, np.float32) for o in p.OBJECTIVES}
    predicted['physical_labels'][2] = 2.
    guides = dict(actual=np.ones(d, np.float32), donor=-np.ones(d, np.float32))
    directions = []
    for magnitude in (4., 2., 3., 1.):
        delta = np.zeros(d, np.float64); delta[1] = magnitude; directions.append(delta)
    return h, predicted, guides, directions


def solver_for(directions, fail_at=None):
    state = {'calls': 0}
    def solve(heads, token, observed, reference_head=None):
        j = state['calls']; state['calls'] += 1
        if len(heads) != 1 or heads[0] is not reference_head:
            raise AssertionError('Only g_A in its metric is allowed')
        if j == fail_at:
            raise RuntimeError('Synthetic solver failure')
        return SimpleNamespace(delta=directions[j].copy(), solver=dict(status='solved', head_count=1, iterations=1))
    return solve, state


def construct(head=None, predicted=None, guides=None, directions=None, goal=1, donor=0):
    h, z, obs, ds = fixture()
    head = h if head is None else head; predicted = z if predicted is None else predicted
    guides = obs if guides is None else guides; directions = ds if directions is None else directions
    solver, state = solver_for(directions)
    result = p.project_t0_four_family(predicted, guides, head, goal_index=goal,
                                     donor_index=donor, projector=solver)
    return head, predicted, guides, result, state


def accept(head, predicted, guides, result, assignment=None, goal=1):
    replacements, directions, report = result
    return p.accept_t0_four_family(predicted, guides, head, replacements, directions, report,
        goal_index=goal, donor_assignment=np.array([1, 0, 2], np.int64) if assignment is None else assignment)


class FourFamilyTests(unittest.TestCase):
    def test_fourth_member_controls_both_objectives_and_sources(self):
        h, z, obs, result, state = construct()
        replacement, directions, report = result
        self.assertEqual(state['calls'], 4)
        self.assertEqual(report['native_norms'], [4., 2., 3., 1.])
        self.assertEqual(report['effective_norm'], 1.)
        self.assertEqual(report['members'], [list(x) for x in p.MEMBERS])
        for oi, objective in enumerate(p.OBJECTIVES):
            for si in range(2):
                np.testing.assert_array_equal(replacement[oi, si] - z[objective], np.eye(1, 192, 1, dtype=np.float32)[0])
        self.assertEqual(accept(h, z, obs, result)['family_size'], 4)

    def test_g_a_metric_normalization_is_shared(self):
        h, z, obs, ds = fixture(); h['scale'][1] = 7.5; h['mean'][1] = -12.
        h, z, obs, result, _ = construct(head=h, predicted=z, guides=obs, directions=ds)
        np.testing.assert_array_equal(result[0][:, :, 1], np.full((2, 2), 7.5, np.float32))
        self.assertEqual(accept(h, z, obs, result)['effective_norm_squared'], 1.)

    def test_legitimate_zero_couples_all_four_without_skipping_qp(self):
        h, z, obs, ds = fixture(); ds[2][:] = 0.
        h, z, obs, result, state = construct(head=h, predicted=z, guides=obs, directions=ds)
        self.assertEqual(state['calls'], 4)
        self.assertIs(result[2]['legitimate_zero_norm'], True)
        for oi, objective in enumerate(p.OBJECTIVES):
            for si in range(2): np.testing.assert_array_equal(result[0][oi, si], z[objective])
        self.assertEqual(accept(h, z, obs, result)['effective_norm'], 0.)

    def test_a_single_numerical_problem_shrinks_the_whole_family(self):
        h, z, obs, ds = fixture(); ds[0][0] = 8e-6
        h, z, obs, result, _ = construct(head=h, predicted=z, guides=obs, directions=ds)
        report = result[2]
        self.assertLess(report['shrink_factor'], 1.)
        self.assertEqual(len(report['attempts']), p.SHRINK_FACTORS.index(report['shrink_factor']) + 1)
        self.assertTrue(all(not all(c['accepted'] for c in a['checks']) for a in report['attempts'][:-1]))
        self.assertEqual(accept(h, z, obs, result)['effective_norm_squared'], report['effective_norm'] ** 2)
        for oi in range(2):
            for si in range(2):
                norm = np.linalg.norm((result[0][oi, si].astype(float) - z[p.OBJECTIVES[oi]]) / h['scale'])
                self.assertAlmostEqual(norm, report['effective_norm'], places=6)

    def test_no_accepted_backoff_blocks_complete_family(self):
        h, z, obs, ds = fixture(); ds[0][0] = 1.
        solve, state = solver_for(ds)
        with self.assertRaises(p.FourFamilyFailure) as caught:
            p.project_t0_four_family(z, obs, h, goal_index=1, donor_index=0, projector=solve)
        self.assertEqual(state['calls'], 4)
        self.assertEqual(caught.exception.detail['stage'], 'verification')
        self.assertEqual(len(caught.exception.detail['attempts']), 9)

    def test_solver_failure_is_not_a_zero_or_partial_success(self):
        h, z, obs, ds = fixture(); solve, state = solver_for(ds, fail_at=2)
        with self.assertRaises(p.FourFamilyFailure) as caught:
            p.project_t0_four_family(z, obs, h, goal_index=1, donor_index=0, projector=solve)
        self.assertEqual(state['calls'], 3)
        self.assertEqual(caught.exception.detail['completed_directions'], 2)

    def test_nonfinite_or_unsolved_direction_rejected(self):
        for mode in ('nan', 'unsolved', 'dual_head'):
            with self.subTest(mode=mode):
                h, z, obs, ds = fixture()
                if mode == 'nan': ds[0][3] = np.nan
                solve, _ = solver_for(ds)
                if mode != 'nan':
                    original = solve
                    def solve(*args, _original=original, **kwargs):
                        r = _original(*args, **kwargs)
                        if mode == 'unsolved': r.solver['status'] = 'maximum iterations reached'
                        else: r.solver['head_count'] = 2
                        return r
                with self.assertRaises(p.FourFamilyFailure):
                    p.project_t0_four_family(z, obs, h, goal_index=1, donor_index=0, projector=solve)

    def test_wrong_objective_or_guidance_roster_rejected(self):
        h, z, obs, ds = fixture()
        cases = [(dict(z, T1=np.zeros(192, np.float32)), obs), ({'decoded_teacher': z['decoded_teacher']}, obs),
                 (z, dict(obs, evaluator=np.zeros(192, np.float32))), (z, {'actual': obs['actual']})]
        for pred, guides in cases:
            with self.subTest(keys=(tuple(pred), tuple(guides))):
                with self.assertRaises(ValueError):
                    p.project_t0_four_family(pred, guides, h, goal_index=1, donor_index=0)
        with self.assertRaises(TypeError):
            p.project_t0_four_family(z, obs, h, goal_index=1, donor_index=0, q_g={})

    def test_shape_dtype_normalizer_and_nonfinite_boundary(self):
        for mode in ('float64', 'dim', 'nan', 'scale', 'goal'):
            h, z, obs, ds = fixture()
            goal = 1
            if mode == 'float64': z['decoded_teacher'] = z['decoded_teacher'].astype(float)
            if mode == 'dim': obs['donor'] = np.zeros(191, np.float32)
            if mode == 'nan': obs['actual'][0] = np.nan
            if mode == 'scale': h['scale'][1] = 0.
            if mode == 'goal': goal = True
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                p.project_t0_four_family(z, obs, h, goal_index=goal, donor_index=0)

    def test_global_donor_permutation_not_local_shard_index(self):
        h, z, obs, result, _ = construct(goal=7, donor=1)
        assignment = np.array([0, 7, 2, 3, 4, 5, 6, 1], np.int64)
        self.assertEqual(accept(h, z, obs, result, assignment=assignment, goal=7)['donor_index'], 1)
        for assignment in (np.arange(8, dtype=np.int64), np.zeros(8, dtype=np.int64), np.arange(8, dtype=np.int32)):
            with self.assertRaises(ValueError): accept(h, z, obs, result, assignment=assignment, goal=7)

    def test_independent_acceptance_rejects_payload_corruption(self):
        h, z, obs, original, _ = construct()
        for mode in ('replacement', 'direction', 'dtype', 'nan'):
            replacements, directions, report = copy.deepcopy(original)
            if mode == 'replacement': replacements[0, 0, 1] += 1e-3
            if mode == 'direction': directions[0, 0, 2] += .5
            if mode == 'dtype': replacements = replacements.astype(np.float64)
            if mode == 'nan': directions[0, 0, 1] = np.nan
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                accept(h, z, obs, (replacements, directions, report))

    def test_independent_acceptance_rejects_receipt_corruption(self):
        h, z, obs, original, _ = construct()
        mutations = [lambda r: r.update(family_size=8), lambda r: r.update(model_role='T1'),
            lambda r: r.update(donor_index=2), lambda r: r.update(effective_norm=2.),
            lambda r: r['native_norms'].__setitem__(0, 99.),
            lambda r: r['members'].reverse(), lambda r: r['attempts'][0]['checks'][0].update(accepted=False),
            lambda r: r['solvers'][0].update(head_count=2), lambda r: r.update(legitimate_zero_norm=True),
            lambda r: r['defaults'].update(tolerance=1.), lambda r: r.update(attempts=[])]
        for i, mutate in enumerate(mutations):
            result = copy.deepcopy(original); mutate(result[2])
            with self.subTest(mutation=i), self.assertRaises(ValueError): accept(h, z, obs, result)

    def test_later_accepted_backoff_is_rejected_even_if_feasible(self):
        h, z, obs, result, _ = construct()
        corrupted = copy.deepcopy(result)
        corrupted[2]['attempts'].append(copy.deepcopy(corrupted[2]['attempts'][0]))
        corrupted[2]['attempts'][1]['shrink'] = .5
        with self.assertRaisesRegex(ValueError, 'first common accepted'):
            accept(h, z, obs, corrupted)

    def test_acceptance_uses_independent_full_function_and_region_math(self):
        h, z, obs, ds = fixture()
        h['0.bias'][0] = 0.; h['4.weight'][:] = 0.
        original = np.zeros(192, np.float32); candidate = original.copy(); candidate[0] = -2e-6
        # Constant readout output does not license crossing the original ReLU region.
        check = p._independent_token_check(original, candidate, p._head(h), 2e-6)
        self.assertEqual(check['heads'][0]['normalized_output_deviation'], 0.)
        self.assertGreater(check['heads'][0]['region_violation'], p.TOLERANCE)
        self.assertFalse(check['accepted'])

    def test_dense_independent_roundoff_physical_receipt_is_scale_aware(self):
        h, z, obs, _ = fixture()
        rng = np.random.default_rng(404)
        h['scale'] = np.exp(rng.uniform(-1., 1., 192))
        h['target_scale'] = np.full(6, 10000.)
        h['0.weight'] = rng.normal(0., .1, (256, 192)); h['0.bias'] = np.full(256, 10.)
        h['2.weight'] = rng.normal(0., .1, (256, 256)); h['2.bias'] = np.full(256, 50.)
        h['4.weight'] = rng.normal(0., .1, (6, 256))
        ds = []
        from s1_projection import relu_geometry
        for objective, _ in p.MEMBERS:
            jac = relu_geometry(h, z[objective])[-1]
            _, _, vh = np.linalg.svd(jac, full_matrices=False)
            raw = rng.normal(size=192); raw -= vh.T @ (vh @ raw)
            ds.append(.01 * raw / np.linalg.norm(raw))
        h, z, obs, result, _ = construct(head=h, predicted=z, guides=obs, directions=ds)
        accepted = accept(h, z, obs, result)
        self.assertTrue(all(c['accepted'] for c in accepted['checks']))
        damaged = copy.deepcopy(result)
        damaged[2]['attempts'][-1]['checks'][0]['heads'][0]['physical_output_max_deviation'] += .001
        with self.assertRaises(ValueError): accept(h, z, obs, damaged)

    def test_input_value_changes_invalidate_receipt(self):
        h, z, obs, result, _ = construct()
        for target in ('head', 'anchor', 'guide'):
            hh, zz, oo = copy.deepcopy((h, z, obs))
            if target == 'head': hh['target_scale'][0] += .1
            if target == 'anchor': zz['physical_labels'][5] += .1
            if target == 'guide': oo['donor'][1] += .1
            with self.subTest(target=target), self.assertRaises(ValueError): accept(hh, zz, oo, result)


if __name__ == '__main__':
    unittest.main()
