"""Metadata/access/native-interface checks only; no Torch or real weight loads."""
import copy
import inspect
from pathlib import Path
from types import SimpleNamespace
import subprocess
import sys
import unittest
from unittest import mock

import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import model_runtime as m


def synthetic_admission(role):
    bank_role, count, stream = m.ROLE_INFO[role]
    return dict(status=m.INPUT_STATUS, protocol_sha256=m.value_sha('synthetic protocol'),
        bank_role=bank_role, count=count, stream_role=stream, parent_ids=list(range(count)),
        bank_manifest=dict(path='/synthetic/bank.json', sha256=m.value_sha('synthetic bank')),
        input_manifest=dict(path='/synthetic/inputs.json', sha256=m.value_sha('synthetic inputs')))


def inputs():
    return dict(history_pixels=np.zeros((3, 224, 224, 3), np.uint8),
                goal_pixels=np.zeros((224, 224, 3), np.uint8), prefix_actions=np.zeros((10, 2), np.float32),
                population_actions=np.zeros((300, 25, 2), np.float32), selected_index=299,
                observed_pixels=np.zeros((5, 224, 224, 3), np.uint8))


def handle(role='probe_test', condition='T0'):
    admission = m.validate_input_admission(synthetic_admission(role), role=role,
                                          protocol_sha256=m.value_sha('synthetic protocol'))
    norm = dict(mean=[.1, -.1], std=[2., 3.])
    return m.RuntimeModel(None, {}, 'synthetic unexecuted model', norm, m.value_sha(norm),
                          dict(condition=condition), admission, dict(role=role))


class RuntimeInterfaceTests(unittest.TestCase):
    def test_import_does_not_import_torch_or_load_checkpoint(self):
        code = "import sys;sys.path.insert(0," + repr(str(Path(m.__file__).parent)) + ");import model_runtime;assert 'torch' not in sys.modules;print('PASS')"
        result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_all_twelve_descriptions_match_existing_audited_metadata(self):
        # Actual local metadata only, no remote stat, array, head or weight reads.
        seen = set()
        for pool in range(3):
            for objective in m.OBJECTIVES:
                for condition in ('T0', 'T1'):
                    spec = m.describe_fixed_model(pool, objective, condition)
                    self.assertEqual(spec['group'], 2 * pool)
                    self.assertEqual(spec['condition'], condition)
                    self.assertEqual(spec['architecture'], 'transformer_jepa')
                    self.assertTrue(spec['checkpoint']['path'].endswith(f'pool{pool}_{objective}_{condition}/last_weights.pt'))
                    self.assertEqual(set(spec['metadata_sha256']), set(m.METADATA))
                    self.assertNotIn('q_g', spec)
                    seen.add(spec['checkpoint']['sha256'])
        self.assertEqual(len(seen), 12)

    def test_no_t2_or_unknown_objective_or_pool(self):
        for values in [(0, 'decoded_teacher', 'T2'), (3, 'decoded_teacher', 'T0'),
                       (True, 'decoded_teacher', 'T0'), (0, 'latent', 'T0')]:
            with self.subTest(values=values), self.assertRaises(ValueError): m.describe_fixed_model(*values)

    def test_probe_signature_has_no_model_condition_or_readout(self):
        keys = set(inspect.signature(m.load_probe_model).parameters)
        self.assertFalse(keys & {'condition', 'T1', 'T2', 'q_g', 'head', 'response_contract'})
        with self.assertRaises(TypeError): m.load_probe_model(0, 'decoded_teacher', condition='T1')

    def test_probe_selects_only_t0_from_validated_role_catalog(self):
        models = [dict(pool=0, objective='decoded_teacher', condition='T0')]
        c = dict(input_admission={'synthetic': True})
        with mock.patch.object(m, '_runtime_admission', return_value=(c, {'count': 256}, models)) as admission, mock.patch.object(m, '_load_bound_model', return_value='synthetic') as load:
            self.assertEqual(m.load_probe_model(0, 'decoded_teacher', runtime_contract='unused', runtime_contract_sha256='unused', split='calibration'), 'synthetic')
            self.assertEqual(admission.call_args.args[2], 'probe_calibration')
            self.assertEqual(load.call_args.args[0]['condition'], 'T0')

    def test_calibration_and_test_input_roles_cannot_be_interchanged(self):
        for role in m.ROLE_INFO:
            r = synthetic_admission(role)
            validated = m.validate_input_admission(r, role=role, protocol_sha256=r['protocol_sha256'])
            self.assertEqual(validated['count'], 256 if 'calibration' in role else 512)
            for wrong in set(m.ROLE_INFO) - {role}:
                with self.subTest(role=role, wrong=wrong), self.assertRaises(ValueError):
                    m.validate_input_admission(r, role=wrong, protocol_sha256=r['protocol_sha256'])

    def test_input_admission_rejects_missing_duplicate_or_wrong_count(self):
        for mutate in (lambda r: r.update(count=512), lambda r: r['parent_ids'].pop(),
                       lambda r: r['parent_ids'].__setitem__(1, 0), lambda r: r.update(stream_role='probe'),
                       lambda r: r.update(bank_role='test_recipient')):
            r = synthetic_admission('calibration_response'); mutate(r)
            with self.assertRaises(ValueError):
                m.validate_input_admission(r, role='calibration_response', protocol_sha256=m.value_sha('synthetic protocol'))

    def test_unfrozen_runtime_cannot_reach_metadata_or_weights(self):
        with mock.patch.object(m, 'checked_json', return_value=dict(status='DRAFT')) as read, mock.patch.object(m, '_load_bound_model') as load:
            with self.assertRaises(ValueError): m.load_probe_model(0, 'decoded_teacher', runtime_contract='unused', runtime_contract_sha256='unused', split='test')
            self.assertEqual(read.call_count, 1); load.assert_not_called()

    def test_calibration_response_has_own256_gate_and_no_test_bypass(self):
        models = [dict(pool=1, objective='physical_labels', condition='T1')]
        c = dict(input_admission={'synthetic': True})
        with mock.patch.object(m, '_runtime_admission', return_value=(c, {'count': 256}, models)) as admission, mock.patch.object(m, '_load_bound_model', return_value='synthetic'):
            m.load_calibration_response_model(1, 'physical_labels', 'T1', runtime_contract='unused', runtime_contract_sha256='unused')
            self.assertEqual(admission.call_args.args[2], 'calibration_response')

    def test_invalid_test_prediction_seal_stops_before_contract_and_load(self):
        fake = SimpleNamespace(verify_prediction_seal=mock.Mock(side_effect=ValueError('synthetic invalid seal')))
        with mock.patch.dict(sys.modules, {'regression': fake}), mock.patch.object(m, 'checked_json') as read, mock.patch.object(m, '_load_bound_model') as load:
            with self.assertRaisesRegex(ValueError, 'invalid seal'):
                m.load_test_response_model(0, 'decoded_teacher', 'T1', prediction_lock='unused', prediction_lock_sha256='unused',
                    response_contract='unused', response_contract_sha256='unused', expected_bindings={})
            read.assert_not_called(); load.assert_not_called()

    def test_unbound_test_response_contract_stops_before_opening_it(self):
        fake = SimpleNamespace(verify_prediction_seal=mock.Mock(return_value=(dict(bindings={'response_contract_sha256': 'bound'}), {}, {})))
        with mock.patch.dict(sys.modules, {'regression': fake}), mock.patch.object(m, 'checked_json') as read, mock.patch.object(m, '_load_bound_model') as load:
            with self.assertRaises(ValueError):
                m.load_test_response_model(0, 'decoded_teacher', 'T0', prediction_lock='unused', prediction_lock_sha256='unused',
                    response_contract='unused', response_contract_sha256='different', expected_bindings={})
            read.assert_not_called(); load.assert_not_called()

    def test_test_response_valid_path_orders_seal_before_model_access(self):
        events = []
        def seal(*args, **kwargs):
            events.append('seal')
            return dict(bindings={'response_contract_sha256': 'bound', 'protocol_sha256': 'protocol'}), {'test_recipient_ids': np.arange(512)}, {}
        response = dict(status=m.RESPONSE_STATUS, protocol_sha256='protocol', role='test_response', count=512,
                        runtime_contract={'path': 'synthetic runtime', 'sha256': 'hash'}, input_admission={'synthetic': True})
        def read(*args): events.append('contract'); return response
        def admit(*args):
            events.append('admission'); self.assertEqual(args[2], 'test_response')
            return dict(protocol={'sha256': 'protocol'}, input_admission={'synthetic': True}), {'parent_ids': tuple(range(512))}, [dict(pool=0, objective='decoded_teacher', condition='T1')]
        def load(*args): events.append('load'); return 'synthetic handle'
        with mock.patch.dict(sys.modules, {'regression': SimpleNamespace(verify_prediction_seal=seal)}), mock.patch.object(m, 'checked_json', side_effect=read), mock.patch.object(m, '_runtime_admission', side_effect=admit), mock.patch.object(m, '_load_bound_model', side_effect=load):
            self.assertEqual(m.load_test_response_model(0, 'decoded_teacher', 'T1', prediction_lock='unused', prediction_lock_sha256='unused', response_contract='unused', response_contract_sha256='bound', expected_bindings={}), 'synthetic handle')
        self.assertEqual(events, ['seal', 'contract', 'admission', 'load'])

    def test_response_t2_rejected_before_prediction_seal(self):
        with mock.patch.dict(sys.modules, {'regression': SimpleNamespace(verify_prediction_seal=mock.Mock())}) as _:
            with self.assertRaises(ValueError):
                m.load_test_response_model(0, 'decoded_teacher', 'T2', prediction_lock='unused', prediction_lock_sha256='unused', response_contract='unused', response_contract_sha256='unused', expected_bindings={})

    def test_exact_native_population_and_pixel_interfaces(self):
        case = inputs(); m.validate_native_inputs(**case)
        for name, value in [('population_actions', np.zeros((1,25,2), np.float32)),
                            ('population_actions', np.zeros((300,25,2), np.float64)),
                            ('prefix_actions', np.zeros((9,2), np.float32)),
                            ('observed_pixels', np.zeros((4,224,224,3), np.uint8)),
                            ('selected_index',300), ('selected_index',True)]:
            changed = dict(case); changed[name] = value
            with self.subTest(name=name), self.assertRaises(ValueError): m.validate_native_inputs(**changed)

    def test_nonfinite_action_or_invalid_normalizer_rejected(self):
        case=inputs(); case['population_actions'][1,2,0]=np.nan
        with self.assertRaises(ValueError): m.validate_native_inputs(**case)
        for value in [dict(mean=[0.,0.],std=[0.,1.]),dict(mean=[0.],std=[1.,1.]),dict(mean=[0.,0.],std=[1.,np.nan])]:
            with self.assertRaises(ValueError): m._normalization(value)
        normalized=m._normalization(dict(mean=[0.,0.],std=[1.,2.],documented_metadata='retained'))
        self.assertEqual(normalized['documented_metadata'],'retained')

    def test_model_role_and_normalization_guard_before_torch(self):
        for mutation in ('normalizer','T1','parent'):
            h=handle(); parent=1
            if mutation=='normalizer':h.normalization['std'][0]=99.
            if mutation=='T1':h.spec['condition']='T1'
            if mutation=='parent':parent=10000
            with self.subTest(mutation=mutation), mock.patch.object(m,'_torch_dependencies') as backend:
                with self.assertRaises(ValueError):m.native_rollouts(h,parent_id=parent,**inputs())
                backend.assert_not_called()

    def test_response_has_no_observed_or_corrected_probe_branch(self):
        h=handle('calibration_response','T1')
        with mock.patch.object(m,'_torch_dependencies') as backend:
            with self.assertRaises(ValueError):m.native_rollouts(h,parent_id=1,**inputs())
            case=inputs();case['observed_pixels']=None
            with self.assertRaises(ValueError):m.native_rollouts(h,parent_id=1,replacements={'actual':np.zeros(192,np.float32)},**case)
            backend.assert_not_called()

    def test_corrected_probe_requires_both_finite_fp32_sources(self):
        h=handle()
        for replacement in ({'actual':np.zeros(192,np.float32)},
                            {'actual':np.zeros(192,np.float32),'donor':np.zeros(192,np.float64)}):
            with mock.patch.object(m,'_torch_dependencies') as backend:
                with self.assertRaises(ValueError):m.native_rollouts(h,parent_id=1,replacements=replacement,**inputs())
                backend.assert_not_called()

    def test_donor_loader_has_independent_split_role_and_fixed_t0(self):
        for split in ('calibration', 'test'):
            models = [dict(pool=1, objective='decoded_teacher', condition='T0')]
            with mock.patch.object(m, '_runtime_admission', return_value=({'input_admission': {}}, {}, models)) as admit, mock.patch.object(m, '_load_bound_model', return_value='synthetic') as load:
                m.load_donor_encoder(1, 'decoded_teacher', runtime_contract='unused', runtime_contract_sha256='unused', split=split)
                self.assertEqual(admit.call_args.args[2], 'probe_donor_' + split)
                self.assertEqual(load.call_args.args[0]['condition'], 'T0')
                self.assertEqual(load.call_args.args[2]['permitted_operation'], 'encode_donor_observations_only')
        self.assertNotIn('condition', inspect.signature(m.load_donor_encoder).parameters)
        with self.assertRaises(TypeError): m.load_donor_encoder(1, 'decoded_teacher', condition='T1')

    def test_donor_admission_never_authorizes_native_rollout(self):
        for role in ('probe_donor_calibration', 'probe_donor_test'):
            with mock.patch.object(m, '_torch_dependencies') as backend:
                with self.assertRaisesRegex(ValueError, 'never rollout'): m.native_rollouts(handle(role), parent_id=1, **inputs())
                backend.assert_not_called()

    def test_donor_encoding_rejects_recipient_response_t1_and_wrong_frames(self):
        for role in ('probe_test', 'calibration_response', 'probe_donor_test'):
            h = handle(role, 'T1' if role == 'probe_donor_test' else 'T0')
            h.access_receipt['permitted_operation'] = 'encode_donor_observations_only'
            with mock.patch.object(m, '_torch_dependencies') as backend:
                with self.assertRaises(ValueError): m.encode_donor_observations(h, parent_id=1, observed_pixels=inputs()['observed_pixels'])
                backend.assert_not_called()
        h = handle('probe_donor_test'); h.access_receipt['permitted_operation'] = 'encode_donor_observations_only'
        for pixels in (np.zeros((1,224,224,3),np.uint8), np.zeros((5,224,224,3),np.float32)):
            with mock.patch.object(m, '_torch_dependencies') as backend:
                with self.assertRaises(ValueError): m.encode_donor_observations(h, parent_id=1, observed_pixels=pixels)
                backend.assert_not_called()


if __name__=='__main__':unittest.main()
