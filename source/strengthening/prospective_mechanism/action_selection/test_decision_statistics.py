"""Synthetic full-schema S3 statistics/seal tests; no model or real outcome data."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('s3_decision_statistics_tested', HERE / 'decision_statistics.py')
d = importlib.util.module_from_spec(spec); spec.loader.exec_module(d)


def predictions():
    p = np.full(d.COST_SHAPE, 10., np.float64)
    p[..., 0, 3] = 0.
    p[..., 1, 1:3] = 0.
    p[..., 2, 4] = 0.
    p[..., 3, 0] = 0.
    return p


def models():
    return [dict(pool=p, group=2*p, condition=c, objective=o,
        checkpoint_sha256=f'{p*4+ci*2+oi+1:064x}', action_normalization_sha256='a'*64,
        head_A_sha256=f'{100+p:064x}')
        for p in range(3) for ci,c in enumerate(d.CONDITIONS) for oi,o in enumerate(d.OBJECTIVES)]


class Fixture:
    """Future target receipts here are synthetic; no upstream native gate is run."""
    def __init__(self, root):
        self.root = Path(root)
        self.protocol = json.loads((HERE.parent / 'protocol/S3_PROTOCOL.draft.json').read_text())
        self.protocol['status'] = 'S3_SCIENTIFIC_PROTOCOL_FROZEN'
        self.protocol['scientific_protocol_frozen'] = True
        self.protocol['primary_statistics']['verification_tolerances'] = dict(atol=1e-9, rtol=1e-12)
        protocol = self.doc('protocol.json', self.protocol)
        sources = self.doc('sources.json', dict(status='S3_DECISION_STATISTICS_SOURCES_FROZEN',
            protocol_sha256=protocol['sha256'], files={str(HERE/'decision_statistics.py'):d.sha(HERE/'decision_statistics.py'),
            str(d.METRIC_SOURCE):d.sha(d.METRIC_SOURCE)}))
        ids = np.arange(512, dtype=np.int64)
        idx = np.broadcast_to(np.array([0,-1,*range(1,31)], np.int64), (512,3,32)).copy()
        self.data = dict(predicted_costs=predictions(), recipient_ids=ids, donor_ids=ids+10000,
            candidate_source_indices=idx)
        arrays = self.archive('predictions.npz', self.data)
        self.payload = self.root/'retained_prediction_payload.bin'; self.payload.write_bytes(b'synthetic predicted-token payload')
        self.expected = dict(expected_recipient_ids=ids.tolist(), expected_donor_ids=(ids+10000).tolist(),
            expected_model_records=models(), expected_candidate_actions_sha256='b'*64,
            expected_donor_assignment_sha256='c'*64, expected_outcome_source_sha256='d'*64)
        self.report = dict(status=d.PREDICTION_STATUS, protocol_sha256=protocol['sha256'], statistics_sources_sha256=sources['sha256'],
            axes=d.AXES, model_records=models(), candidate_actions_sha256='b'*64, donor_assignment_sha256='c'*64,
            goal_count=512,pool_count=3,model_count=12,branch_count=4,candidate_count=32,
            accepted_family_count=1536,all_family_checks_passed=True,all_native_and_identity_checks_passed=True,
            independent_gA_cost_checks_passed=True,physical_outcomes_opened=False,complete_population_accepted=True,
            rows_removed=0,retained_prediction_payloads=[d.pair(self.payload)],arrays=arrays)
        acceptance = self.doc('prediction_acceptance.json', self.report)
        self.binding = dict(status='S3_DECISION_SEAL_INPUT_BINDING_FROZEN', protocol=protocol,
            sources=sources,prediction_acceptance=acceptance,**self.expected)
        self.binding_pair = self.doc('binding.json',self.binding)
        self.outcome_arrays = dict(terminal_states=np.zeros((512,3,32,7), np.float64),
            goal_states=np.zeros((512,7),np.float64),recipient_ids=ids.copy(),candidate_source_indices=idx.copy())
        self.outcome_report = dict(status=d.OUTCOME_STATUS,protocol_sha256=protocol['sha256'],sources_sha256='d'*64,
            count=512,pools=[0,1,2],candidates=32,candidate_actions_sha256='b'*64,complete_population_accepted=True,
            rows_removed=0,all_fixed_candidates_retained=True,arrays=self.archive('outcomes.npz', self.outcome_arrays))
        self.outcome_pair = self.doc('outcomes.json',self.outcome_report)

    def doc(self,name,obj):
        path=self.root/name;path.write_text(json.dumps(obj,sort_keys=True,allow_nan=False));return d.pair(path)

    def archive(self,name,values):
        path=self.root/name
        with path.open('wb') as stream:np.savez_compressed(stream,**values)
        return d.pair(path)

    def rebound_report(self):
        self.binding['prediction_acceptance']=self.doc('prediction_acceptance.json',self.report)
        self.binding_pair=self.doc('binding.json',self.binding)
        return self.binding_pair


class DecisionStatisticsTests(unittest.TestCase):
    def test_exact_tie_uniform_mean_and_no_tolerance(self):
        p=predictions();q=np.broadcast_to(np.arange(32,dtype=np.float64),(512,3,32)).copy()
        values=d.selection_values(p,q)
        self.assertEqual(values['selected'][0,0,1,0,1],1.5)
        self.assertEqual(values['lowest'][0,0,1,0,1],1.)
        self.assertEqual(values['tie_counts'][0,0,1,0,1],2)
        p[0,0,1,0,1,2]=np.nextafter(0.,1.)
        values=d.selection_values(p,q)
        self.assertEqual(values['selected'][0,0,1,0,1],1.)
        self.assertEqual(values['tie_counts'][0,0,1,0,1],1)

    def test_sign_axes_pool_within_recipient_and_descriptions(self):
        selected=np.zeros((512,3,2,2,4),np.float64)
        selected[:,:,0,:,0]=10;selected[:,:,0,:,1]=8
        selected[:,:,1,:,0]=20;selected[:,:,1,:,1]=14;selected[:,:,1,:,2]=25
        selected[:,2,1,1,1]=2
        vectors=d.contrasts(selected)
        np.testing.assert_array_equal(vectors,np.broadcast_to([-8.,-13.],(512,2)))
        v=dict(selected=selected,lowest=selected.copy(),tie_counts=np.ones_like(selected,dtype=np.int64),
               t1_actual_free_argmin_sets_differ=np.ones((512,3,2),bool))
        report=d.summarize(v,vectors,np.broadcast_to([-8.,-13.],(20000,2)))
        self.assertEqual(report['secondary']['free_training_change'],10.)
        self.assertEqual(report['secondary']['benefit_training_change'],6.)
        self.assertEqual(report['secondary']['selection_change'],1.)

    def test_joint_interpretation_null_negative_and_no_secondary_ci(self):
        selected=np.zeros((512,3,2,2,4),np.float64)
        v=dict(selected=selected,lowest=selected,tie_counts=np.ones_like(selected,dtype=np.int64),
               t1_actual_free_argmin_sets_differ=np.zeros((512,3,2),bool))
        for row,labels,joint in [([-2.,-1.],['improvement']*2,True),([-2.,1.],['improvement','higher_cost'],False),
                               ([-2.,0.],['improvement','unresolved'],False)]:
            out=d.summarize(v,np.broadcast_to(row,(512,2)),np.broadcast_to(row,(20000,2)))
            self.assertEqual([x['classification'] for x in out['primary']],labels)
            self.assertEqual(out['joint_actual_specific_decision_improvement'],joint)
            self.assertFalse(any('interval' in k or 'lower' in k or 'upper' in k for k in out['secondary']))

    def test_original_physical_cost_block_axes_wrapped_angle(self):
        states=np.zeros((512,3,32,7),np.float64);goals=np.zeros((512,7),np.float64)
        states[...,0:2]=99999;states[...,2]=3;states[...,3]=4;states[...,4]=2*np.pi+.5
        result=d.physical_candidate_costs(states,goals)
        np.testing.assert_allclose(result,250.,rtol=0,atol=1e-9)
        bad=states.copy();bad[-1,-1,-1,-1]=np.nan
        with self.assertRaises(ValueError):d.physical_candidate_costs(bad,goals)

    def test_complete_statistics_original_helper_and_count_weighted_reconstruction(self):
        p=predictions();q=np.broadcast_to(np.arange(32,dtype=np.float64)*70,(512,3,32)).copy()
        q+=np.arange(512)[:,None,None]*1000.
        p[:100,...,1,:]=10.;p[:100,...,1,31]=0.
        result,values=d.complete_statistics(p,q,seed=1169911584,bit_generator='PCG64',atol=1e-9,rtol=1e-12)
        self.assertEqual(values['goal_contrasts'].shape,(512,2));self.assertEqual(values['bootstrap_means'].shape,(20000,2))
        self.assertTrue(result['independent_selection_and_statistics_verified'])
        self.assertEqual(result['bootstrap_iterations'],20000)
        expected=np.broadcast_to([-105.,-175.],(512,2)).copy();expected[:100]=[1960.,1890.]
        np.testing.assert_array_equal(values['goal_contrasts'],expected)
        np.testing.assert_allclose(values['bootstrap_means'],values['independent_bootstrap_means'],atol=1e-9,rtol=1e-12)

    def test_seed_and_full_array_guards(self):
        for seed,rng in [(None,'PCG64'),(True,'PCG64'),(-1,'PCG64'),(5,'MT19937')]:
            with self.assertRaises(ValueError):d.bootstrap(np.zeros((512,2),np.float64),seed=seed,bit_generator=rng)
        for p in [np.zeros((511,3,2,2,4,32)),np.zeros(d.COST_SHAPE,np.float32)]:
            with self.assertRaises(ValueError):d.selection_values(p,np.zeros((512,3,32),np.float64))

    def test_full_file_seal_and_valid_outcome_join_without_upstream_simulation(self):
        with tempfile.TemporaryDirectory() as tmp:
            f=Fixture(tmp);seal=d.seal_prediction_population(f.binding_pair,Path(tmp)/'sealed')
            d.verify_prediction_seal(seal,f.binding_pair)
            loader=Mock(return_value=f.outcome_pair)
            # The complete numerical algorithm is exercised separately above; this checks the genuine file barrier.
            with patch.object(d,'complete_statistics',return_value=({'synthetic_math_mock':True},{})):
                report,values=d.evaluate_after_seal(seal,f.binding_pair,loader)
            self.assertEqual(loader.call_count,1);self.assertTrue(report['physical_outcomes_opened_only_after_seal'])
            self.assertFalse(report['historical_nonexposure_proven']);self.assertEqual(values['physical_candidate_costs'].shape,(512,3,32))
            with self.assertRaises(ValueError):d.seal_prediction_population(f.binding_pair,Path(tmp)/'sealed')

    def test_wrong_seal_or_changed_payload_never_invokes_outcome_loader(self):
        with tempfile.TemporaryDirectory() as tmp:
            f=Fixture(tmp);seal=d.seal_prediction_population(f.binding_pair,Path(tmp)/'sealed')
            loader=Mock(return_value=f.outcome_pair)
            f.payload.write_bytes(b'changed synthetic payload')
            with self.assertRaises(ValueError):d.evaluate_after_seal(seal,f.binding_pair,loader)
            loader.assert_not_called()

    def test_prediction_receipt_role_axis_partial_and_exposure_guards(self):
        with tempfile.TemporaryDirectory() as tmp:
            f=Fixture(tmp);original=copy.deepcopy(f.report)
            edits=[('goal_count',511),('accepted_family_count',1535),('rows_removed',1),('physical_outcomes_opened',True),
                ('all_native_and_identity_checks_passed',False),('independent_gA_cost_checks_passed',False),
                ('model_records',models()[:-1]),('candidate_actions_sha256','e'*64),
                ('axes',dict(d.AXES,conditions=['T1','T0']))]
            for key,value in edits:
                f.report=copy.deepcopy(original);f.report[key]=value
                with self.subTest(key=key),self.assertRaises(ValueError):d.load_seal_context(f.rebound_report())

    def test_prediction_ids_source_indices_full_population_guards(self):
        with tempfile.TemporaryDirectory() as tmp:
            f=Fixture(tmp)
            bad=copy.deepcopy(f.data);bad['recipient_ids'][-1]=bad['recipient_ids'][0]
            with self.assertRaises(ValueError):d.validate_prediction_arrays(bad,f.binding)
            bad=copy.deepcopy(f.data);bad['donor_ids']=bad['donor_ids'][::-1]
            with self.assertRaises(ValueError):d.validate_prediction_arrays(bad,f.binding)
            bad=copy.deepcopy(f.data);bad['candidate_source_indices'][-1,-1,-1]=0
            with self.assertRaises(ValueError):d.validate_prediction_arrays(bad,f.binding)
            bad=copy.deepcopy(f.data);bad['predicted_costs'][-1,-1,-1,-1,-1,-1]=np.inf
            with self.assertRaises(ValueError):d.validate_prediction_arrays(bad,f.binding)
            bad=copy.deepcopy(f.data);bad['physical_costs']=np.zeros((512,3,32))
            with self.assertRaises(ValueError):d.validate_prediction_arrays(bad,f.binding)

    def test_exact_model_order_and_same_pool_head_guards(self):
        good=models();d.validate_models(good)
        for change in ('order','head','normalizer'):
            bad=copy.deepcopy(good)
            if change=='order':bad[0],bad[1]=bad[1],bad[0]
            elif change=='head':bad[1]['head_A_sha256']='e'*64
            else:bad[0]['action_normalization_sha256']=None
            with self.subTest(change=change),self.assertRaises(ValueError):d.validate_models(bad)

    def test_outcome_order_ancestry_or_partial_guards(self):
        with tempfile.TemporaryDirectory() as tmp:
            f=Fixture(tmp);seal=d.seal_prediction_population(f.binding_pair,Path(tmp)/'sealed')
            original=copy.deepcopy(f.outcome_report)
            for key,value in [('count',511),('candidate_actions_sha256','e'*64),('sources_sha256','f'*64),('rows_removed',1)]:
                bad=copy.deepcopy(original);bad[key]=value;ref=f.doc('outcomes.json',bad)
                with self.subTest(key=key),self.assertRaises(ValueError):d.evaluate_after_seal(seal,f.binding_pair,lambda:ref)
            f.outcome_arrays['recipient_ids']=f.outcome_arrays['recipient_ids'][::-1]
            original['arrays']=f.archive('outcomes.npz',f.outcome_arrays);ref=f.doc('outcomes.json',original)
            with self.assertRaises(ValueError):d.evaluate_after_seal(seal,f.binding_pair,lambda:ref)

    def test_changed_retained_source_during_outcome_join_rejected_before_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            f=Fixture(tmp);seal=d.seal_prediction_population(f.binding_pair,Path(tmp)/'sealed')
            def open_outcome():
                f.payload.write_bytes(b'mid-run changed source');return f.outcome_pair
            with patch.object(d,'complete_statistics',return_value=({},{})),self.assertRaises(ValueError):
                d.evaluate_after_seal(seal,f.binding_pair,open_outcome)

    def test_changed_draft_protocol_or_numerical_tolerances_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            f=Fixture(tmp)
            for key,value in [('status','DRAFT_NOT_EXECUTION_AUTHORIZATION'),('scientific_protocol_frozen',False),
                    ('stage','S2_wrong_stage'),('primary_statistics',
                    dict(f.protocol['primary_statistics'],verification_tolerances=dict(atol=1e-3,rtol=1e-3)))]:
                bad=copy.deepcopy(f.protocol);bad[key]=value
                f.binding['protocol']=f.doc('protocol.json',bad);f.binding_pair=f.doc('binding.json',f.binding)
                with self.subTest(key=key),self.assertRaises(ValueError):d.load_seal_context(f.binding_pair)


if __name__=='__main__':unittest.main()
