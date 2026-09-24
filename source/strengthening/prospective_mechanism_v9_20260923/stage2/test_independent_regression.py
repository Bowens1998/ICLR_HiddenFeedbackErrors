"""Synthetic-only parity/corruption tests; never accesses actual S2 artifacts."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import regression as production
import independent_regression as independent

STRATA = [f'synthetic_pool{p}/{o}' for p in range(3) for o in ['teacher','physical']]
TOL = {'atol':1e-9,'rtol':1e-10}


def dump(path,value):
    path.write_text(json.dumps(value,sort_keys=True,allow_nan=False,
                    default=lambda x:x.tolist() if isinstance(x,np.ndarray) else (_ for _ in ()).throw(TypeError())))


def desc(path):
    return {'path':str(path),'sha256':independent.file_sha(path)}


def fixture(root,constant=False):
    rng = np.random.default_rng(98213)
    x = np.exp(rng.normal(0,.4,(256,6,4)))
    tx = np.exp(rng.normal(.2,.7,(512,6,4)))
    g = rng.normal(size=(256,6))
    tg = rng.normal(size=(512,6))
    y = np.arange(6)[None,:]*10+3*(x[:,:,0]-1)**2+.3*x[:,:,1]-1.5*g
    ty = np.arange(6)[None,:]*10+3*(tx[:,:,0]-1)**2+.3*tx[:,:,1]-1.5*tg
    if constant:
        x[:]=.1;tx[:]=.1;g[:]=.1;tg[:]=.1;y[:]=0.;ty[:]=0.
    fit = production.fit_calibration(x,g,y,recipient_ids=np.arange(256),donor_ids=np.arange(1000,1256),
              stratum_ids=STRATA,verification_atol=TOL['atol'],verification_rtol=TOL['rtol'])
    pred = production.predict_test(fit,tx,tg,recipient_ids=np.arange(10000,10512),
              donor_ids=np.arange(20000,20512),stratum_ids=STRATA)
    bindings = {k:format(i+1,'064x') for i,k in enumerate(sorted(independent.BINDINGS))}
    lock = production.seal_test_predictions(root/'seal',fit,pred,bindings=bindings,
             bootstrap_seed=123456,bit_generator='PCG64',quantile_method='linear')
    response_path = root/'response.npz'
    np.savez_compressed(response_path,response=ty,recipient_ids=pred.arrays['test_recipient_ids'],
                        donor_ids=pred.arrays['test_donor_ids'])
    receipt_path = root/'response.json'
    dump(receipt_path,dict(status='ACCEPTED_S2_HELDOUT_RESPONSE',count=512,
         protocol_sha256=bindings['protocol_sha256'],prediction_lock_sha256=lock['sha256'],
         response_contract_sha256=bindings['response_contract_sha256'],stratum_ids=STRATA,arrays=desc(response_path)))
    stats = production.evaluate_response_after_seal(lock['path'],lock['sha256'],expected_bindings=bindings,
              response_loader=lambda:dict(receipt_path=str(receipt_path),receipt_sha256=independent.file_sha(receipt_path)))
    stats_path = root/'statistics.json';dump(stats_path,stats)
    return dict(schema='s2_independent_regression_binding_v1_draft',prediction_lock=lock,
             response_receipt=desc(receipt_path),production_statistics=desc(stats_path),expected_bindings=bindings,
             expected_stratum_ids=STRATA.copy(),expected_statistical_settings=independent.settings(123456),
             comparison_tolerances=TOL.copy(),production_verification_tolerances=TOL.copy(),
             expected_parent_ids={**{k:fit.arrays[k].tolist() for k in ['calibration_recipient_ids','calibration_donor_ids']},
                                  **{k:pred.arrays[k].tolist() for k in ['test_recipient_ids','test_donor_ids']} })


def rewrite_seal(c,edit_cal=None,edit_pred=None,edit_meta=None):
    """Produce self-consistent hashes so tests inspect semantics, not just hashes."""
    path=Path(c['prediction_lock']['path']);root=path.parent
    ca=independent.load_npz(root/'calibration.npz');pa=independent.load_npz(root/'predictions.npz')
    cm=independent.read_json(root/'calibration.json');pm=independent.read_json(root/'predictions.json')
    if edit_cal:edit_cal(ca)
    if edit_pred:edit_pred(pa)
    if edit_meta:edit_meta(cm)
    csha=independent.content_digest(ca,cm);pm['calibration_sha256']=csha
    psha=independent.content_digest(pa,pm)
    np.savez_compressed(root/'calibration.npz',**ca);np.savez_compressed(root/'predictions.npz',**pa)
    dump(root/'calibration.json',cm);dump(root/'predictions.json',pm)
    lock=independent.read_json(path);lock.update(calibration_sha256=csha,predictions_sha256=psha)
    lock['files']={k:independent.file_sha(root/k) for k in lock['files']};dump(path,lock)
    c['prediction_lock']=desc(path)


class IndependentRegressionTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.root=Path(self.tmp.name)
        self.contract=fixture(self.root)
    def tearDown(self):
        self.tmp.cleanup()
    def verify(self):
        return independent.verify_artifacts(self.contract)
    def no_response_open_reject(self,pattern):
        response_path=str(self.root/'response.npz');opened=[]
        original=independent.load_npz
        def tracking(path):
            opened.append(str(path));return original(path)
        with mock.patch.object(independent,'load_npz',side_effect=tracking):
            with self.assertRaisesRegex(ValueError,pattern):self.verify()
        self.assertNotIn(response_path,opened)
    def test_complete_calibration_prediction_scores_and_bootstrap_parity(self):
        report,arrays=self.verify()
        self.assertEqual(report['status'],'PASS_S2_INDEPENDENT_REGRESSION_DRAFT')
        self.assertEqual(arrays['statistics__paired_recipient_scores'].shape,(512,))
        self.assertEqual(arrays['statistics__bootstrap_means'].shape,(20000,))
        self.assertGreater(len(report['maximum_absolute_differences']),30)
        original=independent.read_json(self.contract['production_statistics']['path'])
        self.assertEqual(report['positive_support'],original['positive_support'])
    def test_full_design_penalty_and_unpenalized_intercepts(self):
        rng=np.random.default_rng(419)
        x=rng.normal(size=(256,6,14));y=2*x[:,:,0]+np.arange(6)[None,:]*7
        alpha,beta,gradient,_=independent.full_ridge(x,y)
        xc=x-x.mean(axis=0);yc=y-y.mean(axis=0)
        normal=xc.reshape(-1,14).T@xc.reshape(-1,14)
        rhs=xc.reshape(-1,14).T@yc.ravel()
        correct=np.linalg.solve(normal+15.36*np.eye(14),rhs)
        wrong=np.linalg.solve(normal+.01*np.eye(14),rhs)
        np.testing.assert_allclose(beta,correct,atol=1e-12,rtol=1e-12)
        self.assertGreater(np.max(np.abs(beta-wrong)),.01)
        np.testing.assert_allclose(alpha,y.mean(axis=0)-x.mean(axis=0)@beta,atol=1e-12)
        self.assertLess(np.max(np.abs(gradient)),1e-12)
    def test_constant_population_scaling_and_zero_outcome_retained(self):
        with tempfile.TemporaryDirectory() as d:
            c=fixture(Path(d),constant=True);report,arrays=independent.verify_artifacts(c)
        self.assertFalse(report['positive_support'])
        for name in ['raw_scale','basis_scale','g_scale','target_scale']:
            np.testing.assert_array_equal(arrays['calibration__'+name],np.ones_like(arrays['calibration__'+name]))
        np.testing.assert_array_equal(arrays['statistics__bootstrap_means'],np.zeros(20000))
    def test_semantic_coefficient_drift_with_valid_hashes(self):
        rewrite_seal(self.contract,edit_cal=lambda a:a['baseline_slopes'].__setitem__(0,a['baseline_slopes'][0]+.1))
        self.no_response_open_reject('baseline_slopes.*mismatch')
    def test_basis_order_drift_with_valid_hashes(self):
        rewrite_seal(self.contract,edit_cal=lambda a:a.__setitem__('calibration_basis_baseline',a['calibration_basis_baseline'][:,:,::-1].copy()))
        self.no_response_open_reject('calibration_basis_baseline.*mismatch')
    def test_scaling_drift_with_valid_hashes(self):
        rewrite_seal(self.contract,edit_cal=lambda a:a['raw_scale'].__imul__(1.1))
        self.no_response_open_reject('raw_scale.*mismatch')
    def test_raw_declared_order_drift(self):
        rewrite_seal(self.contract,edit_meta=lambda m:m.__setitem__('raw_columns',list(reversed(m['raw_columns']))))
        self.no_response_open_reject('algorithm/schema')
    def test_external_stratum_and_parent_order_are_required(self):
        self.contract['expected_stratum_ids'].reverse()
        self.no_response_open_reject('Stratum order')
        self.contract['expected_stratum_ids'].reverse()
        self.contract['expected_parent_ids']['test_recipient_ids'].reverse()
        self.no_response_open_reject('parent ordering')
    def test_response_contract_drift_rejected_before_response_array_open(self):
        p=Path(self.contract['response_receipt']['path']);r=independent.read_json(p)
        r['response_contract_sha256']='e'*64;dump(p,r);self.contract['response_receipt']=desc(p)
        self.no_response_open_reject('Response contract')
    def test_seal_drift_rejected_before_any_array_open(self):
        self.contract['prediction_lock']['sha256']='f'*64
        with mock.patch.object(independent,'load_npz',side_effect=AssertionError('Opened an array before seal check')):
            with self.assertRaisesRegex(ValueError,'file hash mismatch'):self.verify()
    def test_response_parent_order_drift(self):
        p=self.root/'response.npz';a=independent.load_npz(p);a['recipient_ids']=a['recipient_ids'][::-1]
        np.savez_compressed(p,**a);rp=Path(self.contract['response_receipt']['path']);r=independent.read_json(rp)
        r['arrays']=desc(p);dump(rp,r);self.contract['response_receipt']=desc(rp)
        with self.assertRaisesRegex(ValueError,'Response parent ordering'):self.verify()
    def test_statistics_single_bootstrap_draw_corruption(self):
        p=Path(self.contract['production_statistics']['path']);s=independent.read_json(p)
        s['bootstrap_means'][19876]+=.1;dump(p,s);self.contract['production_statistics']=desc(p)
        with self.assertRaisesRegex(ValueError,'bootstrap_means.*mismatch'):self.verify()
    def test_explicit_seed_tolerance_and_family_are_required(self):
        for key,value,pattern in [('expected_statistical_settings',{**independent.settings(123456),'quantile_method':'nearest'},'configuration'),
                                 ('comparison_tolerances',{'atol':1e-9},'atol/rtol'),
                                 ('expected_bindings',{**self.contract['expected_bindings'],'protocol_sha256':None},'provenance')]:
            old=self.contract[key];self.contract[key]=value
            self.no_response_open_reject(pattern);self.contract[key]=old
    def test_signed_g_and_test_distribution_do_not_refit_moments(self):
        ca=independent.load_npz(self.root/'seal/calibration.npz');pa=independent.load_npz(self.root/'seal/predictions.npz')
        changed=copy.deepcopy(pa);changed['test_ordinary']*=100;changed['test_signed_g']=-changed['test_signed_g']
        c1,p1,_=independent.recompute(ca,pa);c2,p2,_=independent.recompute(ca,changed)
        for name in c1:np.testing.assert_array_equal(c1[name],c2[name])
        self.assertGreater(np.max(np.abs(p1['augmented_prediction']-p2['augmented_prediction'])),100.)
        gm,gs=c1['g_mean'],c1['g_scale']
        np.testing.assert_allclose(p2['test_basis_augmented'][:,:,-1],(-pa['test_signed_g']-gm)/gs)
    def test_negative_and_zero_statistics_no_dropping(self):
        y=np.zeros((512,6));b=np.zeros_like(y);a=np.ones_like(y)
        stats=independent.independent_statistics(y,b,a,77)
        self.assertEqual(stats['paired_improvement'],-1.)
        self.assertIsNone(stats['relative_mse_reduction'])
        self.assertFalse(stats['positive_support'])
        np.testing.assert_array_equal(stats['paired_recipient_scores'],-np.ones(512))
        np.testing.assert_array_equal(stats['ci_95'],[-1.,-1.])
    def test_nonfinite_complete_population_rejected(self):
        rewrite_seal(self.contract,edit_pred=lambda a:a['test_ordinary'].__setitem__((511,5,3),np.nan))
        self.no_response_open_reject('nonfinite complete population')
    def test_wrong_dtype_complete_population_rejected(self):
        rewrite_seal(self.contract,edit_cal=lambda a:a.__setitem__('calibration_ordinary',a['calibration_ordinary'].astype(np.float32)))
        self.no_response_open_reject('FP64 shape')


if __name__=='__main__':
    unittest.main()
