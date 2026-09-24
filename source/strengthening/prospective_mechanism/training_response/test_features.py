"""Synthetic arithmetic checks; no real S2 data, readouts or protocol."""
import copy
import unittest
import numpy as np
import features as f


class Features(unittest.TestCase):
    def setUp(self):
        self.n = 256
        self.p = np.zeros((256, 3, 2, 4, 5, 6), np.float64)
        self.t = np.zeros((256, 3, 5, 6), np.float64)
        # Different strata and times reveal transpose/order/endpoint errors.
        for pool in range(3):
            for objective in range(2):
                for branch in range(4):
                    for h in range(5):
                        self.p[:, pool, objective, branch, h, 2] = 1 + 10*pool + 3*objective + branch + h
        self.p[..., :2] = 1000.  # Agent errors must not enter the block metric.
        self.o = self.p[:, :, :, 0].copy()
        self.o[..., -1, 2] += 7
        self.m = np.broadcast_to(np.array([0., 2., 3.]), (256, 3)).copy()
        self.axes = dict(groups=list(f.GROUPS), objectives=list(f.OBJECTIVES), branches=list(f.BRANCHES),
                         horizons=list(f.HORIZONS), pose=list(f.POSE))

    def run_probe(self, **kwargs):
        return f.probe_features(kwargs.get('p', self.p), kwargs.get('o', self.o), self.t,
                                kwargs.get('m', self.m), split='calibration', axes=kwargs.get('axes', self.axes))

    def test_exact_signed_feature_endpoint_dose_and_strata(self):
        r = self.run_probe()
        for pool in range(3):
            for objective in range(2):
                k = 2*pool + objective; x = 1 + 10*pool + 3*objective
                np.testing.assert_array_equal(r['ordinary'][:, k],
                    np.tile([x*x, (x+4)**2, (x+11)**2, self.m[0, pool]**2], (256, 1)))
                np.testing.assert_array_equal(r['signed_g'][:, k], np.full(256, (x+6)**2-(x+5)**2))
        self.assertEqual(r['stratum_ids'], f.STRATA)
        # The pure extractor does not manufacture G=0 from the supplied norm.
        self.assertNotEqual(r['signed_g'][0, 0], 0)

    def test_actual_minus_donor_is_not_the_definition(self):
        x = self.run_probe()['signed_g'].copy(); swapped = self.p.copy()
        swapped[:, :, :, [1, 2]] = swapped[:, :, :, [2, 1]]
        np.testing.assert_array_equal(self.run_probe(p=swapped)['signed_g'], -x)

    def test_truth_is_physical_and_pool_specific(self):
        self.t[:, 1, :, 2] = 2.
        r = self.run_probe()
        self.assertEqual(r['ordinary'][0, 2, 0], (11-2)**2)
        self.assertEqual(r['ordinary'][0, 3, 0], (14-2)**2)

    def test_first_observed_history_must_equal_free(self):
        self.o[0, 0, 0, 0, 2] += 1
        with self.assertRaises(AssertionError): self.run_probe()

    def test_complete_axes_dtype_and_finite_guards(self):
        for key in self.axes:
            axes = copy.deepcopy(self.axes); axes[key].reverse()
            with self.subTest(key=key), self.assertRaises(ValueError):self.run_probe(axes=axes)
        for p in (self.p[:-1], self.p.astype(np.float32)):
            with self.assertRaises(ValueError):self.run_probe(p=p)
        bad = self.p.copy();bad[0, 0, 0, 1, 3, 4] = np.nan
        with self.assertRaises(ValueError):self.run_probe(p=bad)
        badm = self.m.copy();badm[0, 0] = -1
        with self.assertRaises(ValueError):self.run_probe(m=badm)

    def test_response_signed_T0_minus_T1_and_units(self):
        p = self.p[:, :, :, :2].copy()
        axes = {k:v for k,v in self.axes.items() if k != 'branches'};axes['conditions'] = list(f.CONDITIONS)
        r = f.response_values(p, self.t, split='calibration', axes=axes)
        np.testing.assert_array_equal(r['response'][0], [(1+10*j+3*k+4)**2-(1+10*j+3*k+5)**2 for j in range(3) for k in range(2)])
        self.assertEqual(r['response_forecast_mse_units'], 'position_to_the_fourth')
        axes['conditions'].reverse()
        with self.assertRaises(ValueError):f.response_values(p, self.t, split='calibration', axes=axes)

    def test_test_count_is_512_without_subsampling(self):
        r = f.probe_features(np.tile(self.p, (2, 1, 1, 1, 1, 1)), np.tile(self.o, (2, 1, 1, 1, 1)),
                             np.tile(self.t, (2, 1, 1, 1)), np.tile(self.m, (2, 1)), split='test', axes=self.axes)
        self.assertEqual(r['ordinary'].shape, (512, 6, 4))
        with self.assertRaises(ValueError):f.probe_features(self.p, self.o, self.t, self.m, split='test', axes=self.axes)


if __name__ == '__main__':
    unittest.main()
