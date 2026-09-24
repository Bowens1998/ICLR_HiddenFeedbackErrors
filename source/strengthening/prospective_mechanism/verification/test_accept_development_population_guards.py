"""Independent synthetic corruption checks for the development saved-array gate.

Fixtures are deliberately artificial, not QP-feasibility or scientific evidence.
Each corruption is re-hashed through all downstream receipts so a rejection
tests a semantic guard, rather than merely detecting stale file checksums.
"""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np


SOURCE = Path(__file__).with_name('accept_development_population.py')
SPEC = importlib.util.spec_from_file_location('development_gate_under_test', SOURCE)
GATE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GATE)


class SyntheticRoute:
    def __init__(self, root):
        self.root = Path(root)
        self.psha, self.ssha, self.selected, self.qual, self.asha = [c * 64 for c in '12345']
        self.kernels = {name: c * 64 for name, c in zip(
            ['cache_inputs.py', 'project_s1.py', 'freeze_qp_shards.py', 'run_rollout.py'], '6789')}
        self.objectives = ['decoded_teacher', 'physical_labels']
        n = 64
        self.caches = {}
        for role, first in [('recipient', 100), ('donor', 200)]:
            self.caches[role] = dict(
                initial=np.zeros((n, 3, 192), np.float32),
                observed=np.arange(n * 5 * 192, dtype=np.float32).reshape(n, 5, 192) / 1024,
                truth=np.arange(n * 5 * 6, dtype=np.float64).reshape(n, 5, 6) / 100,
                known_actions=np.zeros((n, 10), np.float32),
                seeds=np.arange(first, first + n, dtype=np.int64))
        free = np.arange(2 * n * 5 * 192, dtype=np.float32).reshape(2, n, 5, 192) / 512
        self.caches['recipient']['free'] = free
        seed = int.from_bytes(hashlib.sha256(b'20260923:v9_s1_donor/development/stream_0').digest()[:4], 'big')
        perm = np.random.Generator(np.random.PCG64(seed)).permutation(n)
        replacements = np.arange(2 * 2 * n * 2 * 192, dtype=np.float32).reshape(2, 2, n, 2, 192) / 256
        self.q = dict(replacements=replacements, goal_indices=np.arange(n, dtype=np.int64),
                      donor_indices=perm, common_norm=np.ones(n, np.float64))
        tokens = np.arange(2 * 2 * n * 4 * 5 * 192, dtype=np.float32).reshape(2, 2, n, 4, 5, 192) / 2048
        for ci in range(2):
            tokens[:, ci, :, 0] = free
            tokens[:, ci, :, 1:3, 0] = replacements[:, ci]
            for oi in range(2):
                tokens[oi, ci, :, 3, 0] = self.caches['recipient']['observed'][:, 0]
        tokens[:, 1, :, 3] = tokens[:, 0, :, 3]
        self.rollout = dict(tokens=tokens, truth=self.caches['recipient']['truth'].copy(),
                            seeds=self.caches['recipient']['seeds'].copy(), goal_indices=np.arange(n, dtype=np.int64))
        self.cache_overrides = {'recipient': {}, 'donor': {}}
        self.qp_overrides, self.qr_overrides, self.rr_overrides = {}, {}, {}
        self.family_mutation = None

    @staticmethod
    def json_file(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, sort_keys=True, indent=2) + '\n')
        return GATE.sha(path)

    @staticmethod
    def npz_file(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **value)
        return dict(path=str(path), sha256=GATE.sha(path))

    def emit(self):
        n = 64
        self.binding = {'reports': []}
        cache_reports = {}
        for role in ['recipient', 'donor']:
            folder = self.root / 'cache' / role / 'group_0_stream_0'
            arr = self.npz_file(folder / 'data.npz', self.caches[role])
            report = dict(status='PASS_S1_OBSERVED_AND_FREE_CACHE', kind='development', role=role,
                group=0, stream=0, count=n, protocol_sha256=self.psha, sources_sha256=self.ssha,
                cases=list(range(n)), donor_model_predictions_generated=False,
                objectives=self.objectives if role == 'recipient' else self.objectives[:1],
                source_sha256=self.kernels['cache_inputs.py'], frozen_tensors_unchanged=True,
                native_endpoint_and_identity_replacement_exact=role == 'recipient', arrays=arr,
                models=[{'synthetic_fixed_model': o} for o in (self.objectives if role == 'recipient' else self.objectives[:1])],
                input_reports_sha256={'synthetic_input_receipt': 'a' * 64})
            report.update(self.cache_overrides[role])
            path = folder / 'report.json'
            digest = self.json_file(path, report)
            self.binding['reports'].append(dict(role=role, group=0, stream=0, report=str(path), report_sha256=digest))
            cache_reports[role] = report
        rc, dc = self.binding['reports']
        arr = self.npz_file(self.root / 'projection' / 'data.npz', self.q)
        common = dict(kind='development', group=0, stream=0, count=n, protocol_sha256=self.psha,
            selected_lock_sha256=self.selected, qualification_sha256=self.qual,
            recipient_cache_sha256=rc['report_sha256'], donor_cache_sha256=dc['report_sha256'])
        families = [dict(goal=i, family_size=8, status='ACCEPTED_S1_EIGHT_MEMBER_FAMILY',
            recipient_seed=int(self.caches['recipient']['seeds'][i]),
            donor_seed=int(self.caches['donor']['seeds'][int(self.q['donor_indices'][i])]),
            effective_norm=float(self.q['common_norm'][i])) for i in range(n)]
        if self.family_mutation:
            self.family_mutation(families)
        qr = dict(common, status='ACCEPTED_COMPLETE_SHARD', start=0, stop=n,
            a_bindings_sha256=self.asha, source_sha256=self.kernels['project_s1.py'],
            arrays=arr, matching_reports=families, elapsed_seconds=1.)
        qr.update(self.qr_overrides)
        qrpath = self.root / 'projection' / 'report.json'
        qrsha = self.json_file(qrpath, qr)
        qp = dict(common, status='ALL_S1_QP_SHARDS_ACCEPTED', all_goals_covered=True,
            source_sha256=self.kernels['freeze_qp_shards.py'], maximum_rechecked_output_deviation=0.,
            shards=[dict(start=0, stop=n, report=str(qrpath), report_sha256=qrsha,
                         arrays=arr['path'], arrays_sha256=arr['sha256'])])
        qp.update(self.qp_overrides)
        qpath = self.root / 'projection_acceptance/group_0_stream_0/QP_LOCK.json'
        qsha = self.json_file(qpath, qp)
        arr = self.npz_file(self.root / 'rollout/recipient/group_0_stream_0/data.npz', self.rollout)
        rr = dict(status='PASS_S1_COMPLETE_ACCEPTED_ROLLOUT', kind='development', role='recipient',
            group=0, stream=0, count=n, protocol_sha256=self.psha, sources_sha256=self.ssha,
            selected_lock_sha256=self.selected, qualification_sha256=self.qual,
            objectives=self.objectives, constraints=['A', 'AC'], branches=['free', 'actual', 'donor', 'reset'],
            horizons=[5, 10, 15, 20, 25], head_weights_opened=False,
            free_and_reset_bitwise_equal_across_constraints=True, source_sha256=self.kernels['run_rollout.py'],
            qp_lock=dict(path=str(qpath), sha256=qsha), recipient_cache=dict(path=rc['report'], sha256=rc['report_sha256']),
            models=copy.deepcopy(cache_reports['recipient']['models']),
            input_reports_sha256=copy.deepcopy(cache_reports['recipient']['input_reports_sha256']),
            insertion_checks=[dict(objective=o, case=i, free_exact=True, identity_exact=True,
                inserted_fp32_exact=True, inputs_unchanged=True) for o in self.objectives for i in range(n)],
            arrays=arr, elapsed_seconds=1., gpu='synthetic', peak_allocated_bytes=1)
        rr.update(self.rr_overrides)
        self.json_file(self.root / 'rollout/recipient/group_0_stream_0/report.json', rr)

    def check(self):
        return GATE.check_route(0, 0, self.root, self.binding, self.psha, self.ssha,
                               self.selected, self.qual, self.kernels, self.asha)


class DevelopmentSavedArrayGuards(unittest.TestCase):
    def run_case(self, mutation=None, message=None):
        with tempfile.TemporaryDirectory() as d:
            fixture = SyntheticRoute(d)
            if mutation:
                mutation(fixture)
            fixture.emit()
            if mutation:
                with self.assertRaises((ValueError, AssertionError), msg=message):
                    fixture.check()
            else:
                result = fixture.check()
                self.assertEqual((result['count'], result['matching_families'], result['directions']), (64, 64, 512))

    def test_complete_axis_distinct_fixture_passes(self):
        self.run_case()

    def test_cache_truth_and_input_schema_corruptions(self):
        changes = {
            'truth_nan': lambda f: f.caches['recipient']['truth'].__setitem__((0, 0, 0), np.nan),
            'truth_shape': lambda f: f.caches['recipient'].__setitem__('truth', np.zeros((64, 5, 5), np.float64)),
            'truth_dtype': lambda f: f.caches['recipient'].__setitem__('truth', f.caches['recipient']['truth'].astype(np.float32)),
            'initial_nan': lambda f: f.caches['donor']['initial'].__setitem__((0, 0, 0), np.nan),
            'action_dtype': lambda f: f.caches['recipient'].__setitem__('known_actions', np.zeros((64, 10), np.float64)),
            'donor_free': lambda f: f.caches['donor'].__setitem__('free', f.caches['recipient']['free'].copy()),
        }
        for name, mutation in changes.items():
            with self.subTest(name=name):
                self.run_case(mutation, name)

    def test_seed_and_index_dtypes_are_not_value_only(self):
        changes = {
            'cache_seed': lambda f: f.caches['recipient'].__setitem__('seeds', f.caches['recipient']['seeds'].astype(np.float64)),
            'qp_goals': lambda f: f.q.__setitem__('goal_indices', f.q['goal_indices'].astype(np.float64)),
            'qp_donors': lambda f: f.q.__setitem__('donor_indices', f.q['donor_indices'].astype(np.float64)),
            'rollout_seed': lambda f: f.rollout.__setitem__('seeds', f.rollout['seeds'].astype(np.float64)),
            'rollout_goals': lambda f: f.rollout.__setitem__('goal_indices', f.rollout['goal_indices'].astype(np.float64)),
        }
        for name, mutation in changes.items():
            with self.subTest(name=name):
                self.run_case(mutation, name)

    def test_rollout_truth_nan_shape_dtype(self):
        for name, value in [('nan', np.full((64, 5, 6), np.nan)),
                            ('shape', np.zeros((64, 5, 5), np.float64)),
                            ('dtype', np.zeros((64, 5, 6), np.float32))]:
            with self.subTest(name=name):
                self.run_case(lambda f, value=value: f.rollout.__setitem__('truth', value), name)

    def test_family_ancestry_not_just_status(self):
        changes = {
            'wrong_qualification': lambda f: f.qr_overrides.update(qualification_sha256='a' * 64),
            'wrong_selected': lambda f: f.qr_overrides.update(selected_lock_sha256='a' * 64),
            'wrong_cache': lambda f: f.qr_overrides.update(recipient_cache_sha256='a' * 64),
            'wrong_arrays': lambda f: f.qr_overrides.update(arrays=dict(path='wrong.npz', sha256='a' * 64)),
            'wrong_family_parent': lambda f: setattr(f, 'family_mutation', lambda families: families[0].update(donor_seed=-1)),
        }
        for name, mutation in changes.items():
            with self.subTest(name=name):
                self.run_case(mutation, name)

    def test_branch_insertion_and_duplicate_reset_identity(self):
        for name, index in [('actual', (0, 0, 0, 1, 0, 0)), ('donor', (1, 1, 1, 2, 0, 1)),
                            ('free', (1, 1, 1, 0, 3, 1)), ('reset', (0, 1, 0, 3, 1, 0))]:
            with self.subTest(name=name):
                self.run_case(lambda f, index=index: f.rollout['tokens'].__setitem__(index, -99.), name)

    def test_bound_source_and_model_mismatches(self):
        changes = {
            'cache_source': lambda f: f.cache_overrides['recipient'].update(source_sha256='a' * 64),
            'qp_source': lambda f: f.qp_overrides.update(source_sha256='a' * 64),
            'project_source': lambda f: f.qr_overrides.update(source_sha256='a' * 64),
            'rollout_source': lambda f: f.rr_overrides.update(source_sha256='a' * 64),
            'model_drift': lambda f: f.rr_overrides.update(models=[{'wrong_model': True}]),
            'input_drift': lambda f: f.rr_overrides.update(input_reports_sha256={'wrong': 'a' * 64}),
            'objective_order': lambda f: f.cache_overrides['recipient'].update(objectives=list(reversed(f.objectives))),
        }
        for name, mutation in changes.items():
            with self.subTest(name=name):
                self.run_case(mutation, name)

    def test_qp_residual_receipt_must_be_finite_nonnegative(self):
        for value in [-np.inf, np.nan, -1., 1.1e-6]:
            with self.subTest(value=value):
                self.run_case(lambda f, value=value: f.qp_overrides.update(maximum_rechecked_output_deviation=value))


if __name__ == '__main__':
    unittest.main()
