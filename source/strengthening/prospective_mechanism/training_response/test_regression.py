"""Synthetic-only S2 regression tests; no scientific files or outcomes loaded."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import regression as r

STRATA = tuple(f'synthetic_pool{p}/{o}' for p in range(3) for o in ['teacher', 'physical'])
TOLS = dict(verification_atol=1e-10, verification_rtol=1e-10)
BOOT = dict(bootstrap_seed=123456, bit_generator='PCG64', quantile_method='linear')
BINDINGS = {name: hashlib.sha256(('SYNTHETIC_ONLY/' + name).encode()).hexdigest() for name in r.REQUIRED_BINDINGS}


def sample(n, seed):
    rng = np.random.Generator(np.random.PCG64(seed))
    x = np.exp(rng.normal(0., .4, size=(n, 6, 4)))
    g = rng.normal(size=(n, 6))
    y = np.arange(6)[None, :] * 10 + 3 * np.square(x[..., 0] - 1) + .3 * x[..., 1] - 1.5 * g
    return x, g, y


def fit_sample():
    x, g, y = sample(256, 101)
    return r.fit_calibration(x, g, y, recipient_ids=np.arange(256), donor_ids=np.arange(1000, 1256),
                            stratum_ids=STRATA, **TOLS)


def prediction_sample(fit):
    x, g, _ = sample(512, 202)
    return r.predict_test(fit, x, g, recipient_ids=np.arange(10000, 10512),
                          donor_ids=np.arange(20000, 20512), stratum_ids=STRATA)


class ScalingAndFit(unittest.TestCase):
    def test_quadratic_column_order(self):
        z = np.tile(np.array([1., 2., 3., 4.]), (2, 6, 1))
        expected = np.array([1., 2., 3., 4., 1., 4., 9., 16., 2., 3., 4., 6., 8., 12.])
        np.testing.assert_array_equal(r.quadratic_basis(z), np.tile(expected, (2, 6, 1)))

    def test_zero_variance_decimal_constants_have_scale_one(self):
        x = np.full((256, 6, 4), .1); g = np.full((256, 6), -.3)
        m, b, a = r.calibration_preprocessing(x, g)
        for key in ['raw_scale', 'basis_scale', 'g_scale']:
            np.testing.assert_array_equal(m[key], np.ones_like(m[key]))
        np.testing.assert_array_equal(b, np.zeros((256, 6, 14)))
        np.testing.assert_array_equal(a, np.zeros((256, 6, 15)))

    def test_two_stage_population_scaling_and_shared_baseline(self):
        x, g, _ = sample(256, 3); m, b, a = r.calibration_preprocessing(x, g)
        np.testing.assert_allclose(m['raw_mean'], x.mean(axis=0), atol=0, rtol=0)
        np.testing.assert_allclose(m['raw_scale'], x.std(axis=0, ddof=0), atol=0, rtol=0)
        np.testing.assert_allclose(b.mean(axis=0), 0, atol=1e-14)
        np.testing.assert_allclose(b.std(axis=0, ddof=0), 1, atol=1e-14)
        np.testing.assert_array_equal(a[..., :14], b)
        np.testing.assert_array_equal(a[..., 14], (g - m['g_mean']) / m['g_scale'])

    def test_test_scaling_never_refits_moments(self):
        x, g, _ = sample(256, 4); m, _, _ = r.calibration_preprocessing(x, g)
        before = {k: a.copy() for k, a in m.items()}
        test = np.full((512, 6, 4), 20.)
        b, _ = r.apply_preprocessing(test, np.full((512, 6), 30.), m)
        self.assertGreater(abs(b.mean()), 1)
        for key in m:
            np.testing.assert_array_equal(m[key], before[key])

    def test_exact_objective_and_unscaled_ridge_factor(self):
        fit = fit_sample(); z = fit.arrays; u = z['calibration_response'] / z['target_scale']
        for name, k in [('baseline', 14), ('augmented', 15)]:
            design = np.c_[np.tile(np.eye(6), (256, 1)), z['calibration_basis_' + name].reshape(1536, k)]
            penalty = np.diag(np.r_[np.zeros(6), np.full(k, 1536 * .01)])
            independent = np.linalg.solve(design.T @ design + penalty, design.T @ u.ravel())
            got = np.r_[z[name + '_intercepts'], z[name + '_slopes']]
            np.testing.assert_allclose(got, independent, atol=1e-10, rtol=1e-10)
            wrong = np.linalg.solve(design.T @ design + np.diag(np.r_[np.zeros(6), np.full(k, .01)]), design.T @ u.ravel())
            self.assertGreater(np.max(abs(wrong - got)), 1e-5)
            residual = u - z[name + '_calibration_prediction'] / z['target_scale']
            objective = np.mean(residual**2) + .01 * np.sum(z[name + '_slopes']**2)
            self.assertAlmostEqual(objective, fit.metadata['diagnostics'][name]['objective'], places=14)

    def test_six_unpenalized_intercepts_and_zero_target_scale_rule(self):
        x, g, _ = sample(256, 5); values = np.array([.1, 10., 20., 30., 40., 50.])
        y = np.tile(values, (256, 1))
        fit = r.fit_calibration(x, g, y, recipient_ids=np.arange(256), donor_ids=np.arange(1000, 1256),
                                stratum_ids=STRATA, **TOLS)
        self.assertEqual(float(fit.arrays['target_scale']), 1.)
        for name in ['baseline', 'augmented']:
            np.testing.assert_array_equal(fit.arrays[name + '_intercepts'], values)
            np.testing.assert_array_equal(fit.arrays[name + '_slopes'], np.zeros(14 if name == 'baseline' else 15))
            np.testing.assert_array_equal(fit.arrays[name + '_calibration_prediction'], y)

    def test_common_target_scale_is_calibration_rms(self):
        fit = fit_sample(); y = fit.arrays['calibration_response']
        self.assertEqual(float(fit.arrays['target_scale']), float(np.sqrt(np.mean((y - y.mean(0))**2))))
        self.assertEqual(fit.arrays['baseline_slopes'].shape, (14,))
        self.assertEqual(fit.arrays['augmented_slopes'].shape, (15,))

    def test_g_is_signed_and_has_no_interactions(self):
        x, g, y = sample(256, 6)
        kw = dict(recipient_ids=np.arange(256), donor_ids=np.arange(1000, 1256), stratum_ids=STRATA, **TOLS)
        a = r.fit_calibration(x, g, y, **kw); b = r.fit_calibration(x, -g, y, **kw)
        np.testing.assert_array_equal(a.arrays['calibration_basis_baseline'], b.arrays['calibration_basis_baseline'])
        np.testing.assert_array_equal(a.arrays['calibration_basis_augmented'][..., -1], -b.arrays['calibration_basis_augmented'][..., -1])
        np.testing.assert_allclose(a.arrays['augmented_slopes'][-1], -b.arrays['augmented_slopes'][-1], atol=1e-13)

    def test_no_incomplete_or_nonfinite_calibration(self):
        x, g, y = sample(256, 7)
        kw = dict(recipient_ids=np.arange(256), donor_ids=np.arange(1000, 1256), stratum_ids=STRATA, **TOLS)
        with self.assertRaises(ValueError):
            r.fit_calibration(x[:255], g[:255], y[:255], **kw)
        x[0, 0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, 'nonfinite'):
            r.fit_calibration(x, g, y, **kw)

    def test_parent_overlap_and_stratum_reordering_rejected(self):
        fit = fit_sample(); x, g, _ = sample(512, 8)
        with self.assertRaisesRegex(ValueError, 'parent overlap'):
            r.predict_test(fit, x, g, recipient_ids=np.arange(512), donor_ids=np.arange(20000, 20512), stratum_ids=STRATA)
        with self.assertRaisesRegex(ValueError, 'order changed'):
            r.predict_test(fit, x, g, recipient_ids=np.arange(10000, 10512), donor_ids=np.arange(20000, 20512), stratum_ids=STRATA[::-1])

    def test_changed_calibration_record_rejected(self):
        fit = fit_sample(); fit.metadata['ridge_lambda'] = 0.1
        with self.assertRaisesRegex(ValueError, 'changed'):
            prediction_sample(fit)


class PredictionBarrierAndStatistics(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.out = Path(self.tmp.name) / 'synthetic_seal'
        self.fit = fit_sample(); self.pred = prediction_sample(self.fit)

    def tearDown(self):
        self.tmp.cleanup()

    def seal(self):
        return r.seal_test_predictions(self.out, self.fit, self.pred, bindings=BINDINGS, **BOOT)

    def response(self):
        return dict(response=sample(512, 202)[2], recipient_ids=self.pred.arrays['test_recipient_ids'],
            donor_ids=self.pred.arrays['test_donor_ids'], stratum_ids=STRATA,
            source_sha256=hashlib.sha256(b'SYNTHETIC_RESPONSE').hexdigest())

    def response_receipt(self, seal, data=None, *, contract=None):
        data = self.response() if data is None else data
        arrays = self.out / 'synthetic_response.npz'
        np.savez_compressed(arrays, response=data['response'], recipient_ids=data['recipient_ids'], donor_ids=data['donor_ids'])
        receipt = dict(status='ACCEPTED_S2_HELDOUT_RESPONSE', count=512,
            protocol_sha256=BINDINGS['protocol_sha256'], prediction_lock_sha256=seal['sha256'],
            response_contract_sha256=contract or BINDINGS['response_contract_sha256'],
            stratum_ids=list(data['stratum_ids']), arrays=dict(path=str(arrays), sha256=r._file_sha(arrays)),
            scope='SYNTHETIC UNIT TEST ONLY; no scientific source, results or frozen experiment')
        path = self.out / 'synthetic_response_receipt.json'; path.write_text(json.dumps(receipt))
        return dict(receipt_path=str(path), receipt_sha256=r._file_sha(path))

    def test_seal_reconstructs_and_preserves_all_calibration_evidence(self):
        seal = self.seal(); lock, arrays, meta = r.verify_prediction_seal(seal['path'], seal['sha256'], expected_bindings=BINDINGS)
        np.testing.assert_array_equal(arrays['baseline_prediction'], self.pred.arrays['baseline_prediction'])
        self.assertEqual(lock['statistical_settings']['bootstrap_seed'], BOOT['bootstrap_seed'])
        with np.load(self.out / 'calibration.npz', allow_pickle=False) as z:
            for key in ['calibration_ordinary', 'calibration_signed_g', 'calibration_response',
                        'raw_mean', 'raw_scale', 'basis_mean', 'basis_scale',
                        'baseline_calibration_residual', 'augmented_calibration_residual']:
                np.testing.assert_array_equal(z[key], self.fit.arrays[key])

    def test_missing_numerical_choices_block_seal(self):
        with self.assertRaisesRegex(ValueError, 'bootstrap seed'):
            r.seal_test_predictions(self.out, self.fit, self.pred, bindings=BINDINGS,
                                     bootstrap_seed=None, bit_generator='PCG64', quantile_method='linear')
        self.assertFalse(self.out.exists())

    def test_exclusive_seal_never_overwrites_predictions(self):
        self.seal()
        with self.assertRaises(FileExistsError):
            self.seal()

    def test_tamper_blocks_response_loader_before_any_access(self):
        seal = self.seal(); calls = []
        with (self.out / 'predictions.npz').open('ab') as f:
            f.write(b'TAMPER')
        def loader():
            calls.append(True); return self.response_receipt(seal)
        with self.assertRaisesRegex(ValueError, 'payload changed'):
            r.evaluate_response_after_seal(seal['path'], seal['sha256'], expected_bindings=BINDINGS, response_loader=loader)
        self.assertEqual(calls, [])

    def test_wrong_protocol_blocks_response_loader(self):
        seal = self.seal(); calls = []
        def loader():
            calls.append(True); return self.response_receipt(seal)
        bindings = dict(BINDINGS, protocol_sha256='0' * 64)
        with self.assertRaisesRegex(ValueError, 'identity mismatch'):
            r.evaluate_response_after_seal(seal['path'], seal['sha256'], expected_bindings=bindings, response_loader=loader)
        self.assertEqual(calls, [])

    def test_response_parent_order_is_not_silently_repaired(self):
        seal = self.seal()
        def loader():
            x = self.response(); x['recipient_ids'] = x['recipient_ids'][::-1]
            return self.response_receipt(seal, x)
        with self.assertRaisesRegex(ValueError, 'binding mismatch'):
            r.evaluate_response_after_seal(seal['path'], seal['sha256'], expected_bindings=BINDINGS, response_loader=loader)

    def test_complete_synthetic_seal_to_response_path(self):
        seal = self.seal(); calls = []
        def loader():
            calls.append(True)
            self.assertTrue(Path(seal['path']).exists())
            return self.response_receipt(seal)
        result = r.evaluate_response_after_seal(seal['path'], seal['sha256'], expected_bindings=BINDINGS, response_loader=loader)
        self.assertEqual(calls, [True]); self.assertEqual(result['bootstrap_draws'], 20000)
        self.assertEqual(result['paired_recipient_scores'].shape, (512,))
        self.assertEqual(result['protocol_sha256'], BINDINGS['protocol_sha256'])

    def test_response_source_contract_must_match_predeclared_seal(self):
        seal = self.seal()
        with self.assertRaisesRegex(ValueError, 'Response contract'):
            r.evaluate_response_after_seal(seal['path'], seal['sha256'], expected_bindings=BINDINGS,
                response_loader=lambda: self.response_receipt(seal, contract='0' * 64))

    def test_response_receipt_cannot_be_mutated_after_hash(self):
        seal = self.seal()
        def loader():
            row = self.response_receipt(seal)
            with Path(row['receipt_path']).open('a') as f:
                f.write(' ')
            return row
        with self.assertRaisesRegex(ValueError, 'receipt hash'):
            r.evaluate_response_after_seal(seal['path'], seal['sha256'], expected_bindings=BINDINGS, response_loader=loader)

    def test_whole_recipient_bootstrap_matches_independent_stream(self):
        rng = np.random.Generator(np.random.PCG64(99))
        y = rng.normal(size=(512, 6)); b = rng.normal(size=(512, 6)); a = rng.normal(size=(512, 6))
        result = r.paired_test_statistics(y, b, a, **BOOT)
        expected = np.mean((y - b)**2 - (y - a)**2, axis=1)
        np.testing.assert_array_equal(result['paired_recipient_scores'], expected)
        draws = np.empty(20000); ref = np.random.Generator(np.random.PCG64(BOOT['bootstrap_seed']))
        for start in range(0, 20000, 500):
            ix = ref.integers(0, 512, size=(500, 512), dtype=np.int64)
            draws[start:start + 500] = expected[ix].mean(axis=1)
        np.testing.assert_array_equal(result['bootstrap_means'], draws)
        np.testing.assert_array_equal(result['ci_95'], np.quantile(draws, [.025, .975], method='linear'))
        self.assertAlmostEqual(result['paired_improvement'], expected.mean())

    def test_negative_and_zero_results_retained_without_relative_division(self):
        zeros = np.zeros((512, 6)); ones = np.ones((512, 6))
        negative = r.paired_test_statistics(zeros, zeros, ones, **BOOT)
        self.assertEqual(negative['paired_improvement'], -1.); self.assertFalse(negative['positive_support'])
        np.testing.assert_array_equal(negative['ci_95'], [-1., -1.]); self.assertIsNone(negative['relative_mse_reduction'])
        zero = r.paired_test_statistics(zeros, zeros, zeros, **BOOT)
        self.assertEqual(zero['paired_improvement'], 0.); self.assertFalse(zero['positive_support'])
        np.testing.assert_array_equal(zero['ci_95'], [0., 0.])

    def test_heldout_population_and_resampling_choices_cannot_shrink(self):
        zeros = np.zeros((512, 6))
        with self.assertRaises(ValueError):
            r.paired_test_statistics(zeros[:511], zeros[:511], zeros[:511], **BOOT)
        with self.assertRaisesRegex(ValueError, 'PCG64'):
            r.paired_test_statistics(zeros, zeros, zeros, bootstrap_seed=12, bit_generator='MT19937', quantile_method='linear')
        with self.assertRaisesRegex(ValueError, 'linear quantiles'):
            r.paired_test_statistics(zeros, zeros, zeros, bootstrap_seed=12, bit_generator='PCG64', quantile_method='nearest')


if __name__ == '__main__':
    unittest.main()
