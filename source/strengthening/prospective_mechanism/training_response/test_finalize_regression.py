"""Focused synthetic two-stage interface tests; no actual S2 inputs or jobs."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import finalize_regression as f


def dump(path,value):
    Path(path).write_text(json.dumps(value,sort_keys=True,allow_nan=False))


def pair(path):return f.descriptor(path)


def fixture(root):
    protocol=root/'protocol.json'
    spec=dict(calibration_count=256,test_count=512,stratum_ids=f.CANONICAL_STRATA,
         raw_columns=list(f.production.RAW_COLUMNS),ordinary_basis_columns=list(f.production.BASIS_COLUMNS),ridge_lambda=.01,
         production_verification_tolerances=dict(atol=1e-9,rtol=1e-10),comparison_tolerances=dict(atol=1e-9,rtol=1e-10),
         statistical_settings=f.independent.settings(3040612462))
    dump(protocol,dict(status='S2_SCIENTIFIC_PROTOCOL_FROZEN',synthetic_fixture_only=True,regression=spec))
    sources=root/'sources.json'
    dump(sources,dict(status='S2_REGRESSION_SOURCES_FROZEN',protocol_sha256=pair(protocol)['sha256'],
         files={name:pair(f.HERE/name) for name in ['finalize_regression.py','regression.py','independent_regression.py']}))
    rosters={name:list(range(start,start+n)) for name,start,n in [
       ('calibration_recipient_ids',0,256),('calibration_donor_ids',1000,256),
       ('test_recipient_ids',10000,512),('test_donor_ids',20000,512)]}
    isolation=root/'isolation.json'
    dump(isolation,dict(status='PASS_COMPLETE_S2_DATA_ISOLATION',protocol_sha256=pair(protocol)['sha256'],
         sources_sha256=pair(sources)['sha256'],expected_parent_ids=rosters))
    response_contract=root/'response_contract.json'
    dump(response_contract,dict(status='S2_TEST_RESPONSE_CONTRACT_FROZEN',protocol_sha256=pair(protocol)['sha256'],
         role='test_response',count=512,synthetic_fixture_only=True))
    b=dict(status='S2_REGRESSION_INPUT_BINDINGS_ACCEPTED',protocol=pair(protocol),sources=pair(sources),
           data_isolation=pair(isolation),response_contract=pair(response_contract),expected_parent_ids=rosters)
    rng=np.random.default_rng(83493);test_y=None
    for key,(role,split,n) in f.ROLES.items():
        x=np.exp(rng.normal(0,.3,(n,6,4)));g=rng.normal(size=(n,6))
        y=np.arange(6)[None,:]+.4*x[:,:,1]-2*g
        arrays=dict(ordinary=x,signed_g=g,recipient_ids=np.asarray(rosters[split+'_recipient_ids'],dtype=np.int64),
                    donor_ids=np.asarray(rosters[split+'_donor_ids'],dtype=np.int64))
        if split=='calibration':arrays['response']=y
        else:test_y=y
        ap=root/(key+'.npz');np.savez_compressed(ap,**arrays)
        upstream=root/(key+'_upstream.json')
        fields=dict(role=role,split=split,count=n,stratum_ids=f.CANONICAL_STRATA,
             protocol_sha256=pair(protocol)['sha256'],sources_sha256=pair(sources)['sha256'],complete_population_accepted=True)
        dump(upstream,dict(status='PASS_COMPLETE_S2_SCALAR_POPULATION',**fields,scalar_arrays=pair(ap),
             recipient_ids=rosters[split+'_recipient_ids'],donor_ids=rosters[split+'_donor_ids']))
        receipt=root/(key+'_receipt.json')
        dump(receipt,dict(status='PASS_COMPLETE_S2_SCALAR_INPUTS',**fields,upstream_acceptance=pair(upstream),arrays=pair(ap)))
        b[key]=pair(receipt)
    bp=root/'inputs.json';dump(bp,b)
    return pair(bp),test_y


class FinalizeRegressionTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.binding,self.test_y=fixture(self.root)
    def tearDown(self):self.temp.cleanup()
    def seal(self):return f.seal(self.binding,self.root/'sealed')
    def read_binding(self):return f.independent.read_json(self.binding['path'])
    def save_binding(self,b):dump(self.binding['path'],b);self.binding=pair(self.binding['path'])
    def modify_receipt(self,key,edit):
        b=self.read_binding();p=Path(b[key]['path']);d=f.independent.read_json(p);edit(d);dump(p,d);b[key]=pair(p);self.save_binding(b)
    def modify_arrays(self,key,edit):
        b=self.read_binding();rp=Path(b[key]['path']);row=f.independent.read_json(rp);ap=Path(row['arrays']['path'])
        a=f.independent.load_npz(ap);edit(a);np.savez_compressed(ap,**a);row['arrays']=pair(ap)
        up=Path(row['upstream_acceptance']['path']);u=f.independent.read_json(up);u['scalar_arrays']=pair(ap);dump(up,u)
        row['upstream_acceptance']=pair(up);dump(rp,row);b[key]=pair(rp);self.save_binding(b)
    def response(self,seal_pair):
        seal=f.independent.read_json(seal_pair['path']);lock=seal['prediction_lock'];b=self.read_binding()
        path=self.root/'opened_response.npz';np.savez_compressed(path,response=self.test_y,
            recipient_ids=np.asarray(b['expected_parent_ids']['test_recipient_ids'],dtype=np.int64),
            donor_ids=np.asarray(b['expected_parent_ids']['test_donor_ids'],dtype=np.int64))
        rp=self.root/'response_receipt.json'
        receipt=dict(status='ACCEPTED_S2_HELDOUT_RESPONSE',count=512,stratum_ids=f.CANONICAL_STRATA,
            protocol_sha256=b['protocol']['sha256'],response_contract_sha256=b['response_contract']['sha256'],
            prediction_lock_sha256=lock['sha256'],arrays=pair(path))
        dump(rp,receipt)
        access=self.root/'response_access.json'
        dump(access,dict(status='PASS_COMPLETE_S2_TEST_RESPONSE_ACCESS',protocol_sha256=b['protocol']['sha256'],
            response_contract_sha256=b['response_contract']['sha256'],prediction_lock_sha256=lock['sha256'],
            response_receipt_sha256=pair(rp)['sha256'],count=512,stratum_ids=f.CANONICAL_STRATA,
            recipient_ids=b['expected_parent_ids']['test_recipient_ids'],donor_ids=b['expected_parent_ids']['test_donor_ids'],
            prediction_seal_verified_before_response_inference=True))
        return pair(rp),pair(access)
    def test_seal_exactly_two_models_and_no_test_response_artifact(self):
        with mock.patch.object(f.production,'_ridge',wraps=f.production._ridge) as fit:
            seal=self.seal();self.assertEqual(fit.call_count,2)
        self.assertFalse((self.root/'opened_response.npz').exists())
        row=f.independent.read_json(seal['path']);self.assertIs(row['test_response_opened_by_this_stage'],False)
        pred=f.independent.load_npz(self.root/'sealed/predictions.npz')
        self.assertEqual(pred['baseline_prediction'].shape,(512,6))
        with self.assertRaisesRegex(ValueError,'already exists'):self.seal()
    def test_complete_two_stages_and_independent_cli(self):
        seal=self.seal();response,access=self.response(seal)
        with mock.patch.object(f.production,'fit_calibration',side_effect=AssertionError('Refit forbidden after seal')):
            accepted=f.evaluate(self.binding,seal,response,access,self.root/'evaluated')
        report=f.independent.read_json(accepted['path'])
        self.assertEqual(report['status'],'PASS_COMPLETE_S2_REGRESSION_AND_INDEPENDENT_STATISTICS')
        independent=f.independent.read_json(report['independent_report']['path'])
        self.assertEqual(independent['test_count'],512)
        self.assertEqual(independent['bootstrap_draws'],20000)
    def test_both_cli_stages(self):
        cmd=[sys.executable,str(f.HERE/'finalize_regression.py'),'seal','--inputs',self.binding['path'],
             '--inputs-sha256',self.binding['sha256'],'--output',str(self.root/'sealed')]
        result=subprocess.run(cmd,check=True,capture_output=True,text=True);sealed=json.loads(result.stdout)
        response,access=self.response(sealed)
        cmd=[sys.executable,str(f.HERE/'finalize_regression.py'),'evaluate','--inputs',self.binding['path'],
             '--inputs-sha256',self.binding['sha256'],'--seal',sealed['path'],'--seal-sha256',sealed['sha256'],
             '--response-receipt',response['path'],'--response-receipt-sha256',response['sha256'],
             '--response-access',access['path'],'--response-access-sha256',access['sha256'],
             '--output',str(self.root/'evaluated')]
        result=subprocess.run(cmd,check=True,capture_output=True,text=True)
        accepted=json.loads(result.stdout);self.assertEqual(accepted['sha256'],f.independent.file_sha(accepted['path']))
    def test_wrong_role_rejected_before_any_fit(self):
        self.modify_receipt('test_probe',lambda r:r.__setitem__('role','test_response'))
        with mock.patch.object(f.production,'fit_calibration',side_effect=AssertionError('Unexpected fit')):
            with self.assertRaisesRegex(ValueError,'wrong role/source'):self.seal()
    def test_wrong_source_binding_rejected(self):
        self.modify_receipt('calibration',lambda r:r.__setitem__('sources_sha256','0'*64))
        with self.assertRaisesRegex(ValueError,'wrong role/source'):self.seal()
    def test_partial_population_rejected_even_with_new_hashes(self):
        self.modify_arrays('test_probe',lambda a:a.__setitem__('ordinary',a['ordinary'][:-1]))
        with self.assertRaisesRegex(ValueError,'FP64 shape'):self.seal()
    def test_test_response_member_forbidden_at_seal(self):
        self.modify_arrays('test_probe',lambda a:a.__setitem__('response',self.test_y))
        original=np.lib.npyio.NpzFile.__getitem__;opened=[]
        def track(archive,key):
            if Path(archive.fid.name).name=='test_probe.npz':opened.append(key)
            return original(archive,key)
        with mock.patch.object(np.lib.npyio.NpzFile,'__getitem__',track):
            with self.assertRaisesRegex(ValueError,'unexpected fields'):self.seal()
        self.assertEqual(opened,[])
    def test_assigned_parent_order_cannot_be_repaired_implicitly(self):
        self.modify_arrays('calibration',lambda a:a.__setitem__('donor_ids',a['donor_ids'][::-1]))
        with self.assertRaisesRegex(ValueError,'parent order'):self.seal()
    def test_missing_population_acceptance_rejected(self):
        self.modify_receipt('test_probe',lambda r:r.__setitem__('complete_population_accepted',False))
        with self.assertRaisesRegex(ValueError,'incomplete scalar population'):self.seal()
    def test_changed_executing_source_rejected(self):
        b=self.read_binding();sp=Path(b['sources']['path']);s=f.independent.read_json(sp)
        s['files']['regression.py']['sha256']='f'*64;dump(sp,s);b['sources']=pair(sp);self.save_binding(b)
        with self.assertRaisesRegex(ValueError,'file hash mismatch'):self.seal()
    def test_prediction_seal_checked_before_response_receipt_or_access(self):
        sealed=self.seal();response,access=self.response(sealed)
        p=self.root/'sealed/PREDICTIONS.lock.json';p.write_text(p.read_text()+' ')
        original=f.document;opened=[]
        def track(pair,name):opened.append(name);return original(pair,name)
        with mock.patch.object(f,'document',side_effect=track):
            with self.assertRaisesRegex(ValueError,'file hash mismatch'):f.evaluate(self.binding,sealed,response,access,self.root/'fail')
        self.assertNotIn('Accepted test-response access',opened)
        self.assertNotIn('Actual held-out response receipt',opened)
    def test_unknown_response_contract_crossbinding_rejected(self):
        sealed=self.seal();response,access=self.response(sealed)
        p=Path(response['path']);r=f.independent.read_json(p);r['response_contract_sha256']='f'*64;dump(p,r);response=pair(p)
        ap=Path(access['path']);a=f.independent.read_json(ap);a['response_receipt_sha256']=response['sha256'];dump(ap,a);access=pair(ap)
        original=f.independent.load_npz;opened=[]
        def track(path):opened.append(str(path));return original(path)
        with mock.patch.object(f.independent,'load_npz',side_effect=track):
            with self.assertRaisesRegex(ValueError,'cross-binding'):f.evaluate(self.binding,sealed,response,access,self.root/'fail')
        self.assertNotIn(str(self.root/'opened_response.npz'),opened)
    def test_response_access_cannot_reference_a_different_seal(self):
        sealed=self.seal();response,access=self.response(sealed)
        p=Path(access['path']);a=f.independent.read_json(p);a['prediction_lock_sha256']='e'*64;dump(p,a);access=pair(p)
        with self.assertRaisesRegex(ValueError,'Response access'):f.evaluate(self.binding,sealed,response,access,self.root/'fail')
    def test_independent_failure_keeps_outputs_without_acceptance(self):
        sealed=self.seal();response,access=self.response(sealed)
        fake=subprocess.CompletedProcess([],1,stdout='',stderr='synthetic rejection')
        with mock.patch.object(f.subprocess,'run',return_value=fake):
            with self.assertRaisesRegex(ValueError,'Independent verifier rejected'):
                f.evaluate(self.binding,sealed,response,access,self.root/'evaluated')
        self.assertTrue((self.root/'evaluated/production_statistics.json').exists())
        self.assertFalse((self.root/'evaluated/ACCEPTANCE.json').exists())


if __name__=='__main__':unittest.main()
