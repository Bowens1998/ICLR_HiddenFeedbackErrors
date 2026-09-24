"""Synthetic joins/access-order tests; native/readout producers are mocked."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import assemble_scalars as a
import test_finalize_regression as old_fixture


def dump(path, value):
    Path(path).write_text(json.dumps(value,sort_keys=True));return a.pair(path)


def arbitrary_file(root,name):return dump(root/name,dict(synthetic_only=True))


class ScalarAssemblyTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        original,_=old_fixture.fixture(self.root)
        original=a.read(original,'synthetic input binding');self.protocol=original['protocol']
        self.rosters=original['expected_parent_ids'];self.probes={};self.responses={};self.log=[]
        measurement=arbitrary_file(self.root,'measurement.json');raw_sources=arbitrary_file(self.root,'raw_sources.json')
        admission=arbitrary_file(self.root,'response_admission.json')
        runtime=dump(self.root/'runtime.json',dict(status='S2_MODEL_RUNTIME_CONTRACT_FROZEN',role='test_response',
                      protocol=self.protocol,input_admission=admission))
        response_sources=dump(self.root/'response_sources.json',dict(runtime_contracts={'test_response':runtime}))
        source_doc=dict(status=a.SOURCE_STATUS,protocol=self.protocol,source_root=str(a.ROOT),files={},
            numerical_sources=original['sources'],measurement_sources=measurement,
            response_latent_sources=response_sources,raw_sources=raw_sources)
        self.source=dump(self.root/'assembly_sources.json',source_doc)
        self.ctx=dict(protocol=a.read(self.protocol,'protocol'),protocol_pair=self.protocol,sources=source_doc,source_pair=self.source)
        self.contract=dump(self.root/'response_contract.json',dict(status='S2_TEST_RESPONSE_CONTRACT_FROZEN',
            role='test_response',count=512,protocol_sha256=self.protocol['sha256'],scalar_assembly_sources=self.source,
            response_latent_sources=response_sources,measurement_sources=measurement,runtime_contract=runtime,input_admission=admission))
        content=arbitrary_file(self.root,'content.json')
        banks=[]
        for role in ('calibration_recipient','calibration_donor','test_recipient','test_donor'):
            ids=self.rosters[role+'_ids'];banks.append(dict(bank_role=role,count=len(ids),parent_ids=ids,
                manifest=arbitrary_file(self.root,role+'_manifest.json'),role_metadata=arbitrary_file(self.root,role+'_role.json')))
        rawdoc=dict(status='PASS_S2_COMPLETE_RAW_INPUT_POPULATION',protocol_sha256=self.protocol['sha256'],
            sources_sha256=raw_sources['sha256'],raw_sources_sha256=raw_sources['sha256'],banks=banks,
            admissions=[dict(role=r,**arbitrary_file(self.root,r+'.json')) for r in sorted(a.cache_population.ALL_ADMISSIONS)],
            content_lineage=content,source_sha256=a.gate.sha(a.HERE/'accept_raw_inputs.py'),bank_count=4,route_count=18,saved_triples_verified=6912)
        self.raw=dump(self.root/'raw.json',rawdoc)
        pairs={}
        for split,key in [('calibration','calibration'),('test','test_probe')]:
            receipt=a.read(original[key],key);z=a.finalizer.independent.load_npz(receipt['arrays']['path'])
            data={k:v for k,v in z.items() if k!='response'}
            # Deliberately non-identity assignment. A second permutation or direct
            # bank order would corrupt the final regression provenance.
            data['donor_ids']=data['donor_ids'][::-1].copy()
            report=dict(status='PASS_COMPLETE_S2_PROBE_SCALARS',split=split,count=len(data['recipient_ids']),
                recipient_ids=data['recipient_ids'].tolist(),donor_ids=data['donor_ids'].tolist(),raw_acceptance=self.raw,
                probe_acceptance=arbitrary_file(self.root,split+'_gate.json'))
            desc=dump(self.root/(split+'_probe.json'),report);pairs[split]=desc
            self.probes[desc['path']]=dict(report=report,arrays=data)
            self.responses[split]=self.response_fixture(split)
        response_pair=dump(self.root/'cal_response.json',self.responses['calibration']['report'])
        self.prepare_doc=dict(status=a.PREPARE_STATUS,protocol=self.protocol,sources=self.source,probe_scalars=pairs,
            calibration_response=response_pair,raw_acceptance=self.raw,response_contract=self.contract,output_root=str(self.root/'assembled'))
        self.binding=dump(self.root/'assembly_binding.json',self.prepare_doc)
        self.scorer=SimpleNamespace(verify_completed_probe=self.probe_verify,
            authenticate_measurement_sources=self.authenticate,authorize_measurement_context=self.authorize,
            decode_pool_tokens=self.decode)
        self.producer=SimpleNamespace(verify_completed_population=self.response_verify)
        self.patches=[mock.patch.object(a,'source_context',return_value=self.ctx),
            mock.patch.object(a,'modules',return_value=(self.scorer,self.producer)),
            mock.patch.object(a.accept_raw_inputs,'validate_content')]
        for patch in self.patches:patch.start()
    def tearDown(self):
        for patch in reversed(self.patches):patch.stop()
        self.tmp.cleanup()
    def response_fixture(self,split):
        n=a.gate._size(split);ids=self.rosters[split+'_recipient_ids'];pools=[]
        models=[]
        for pool in range(3):
            free=np.zeros((2,2,n,5,192),np.float32)
            for oi in range(2):
                values=(np.arange(n,dtype=np.float32)*.002+.4+.1*pool+.03*oi)
                free[oi,0,:,:,2]=values[:,None];free[oi,1,:,:,2]=.5*values[:,None]
                for condition in ('T0','T1'):models.append(dict(pool=pool,objective=a.features.OBJECTIVES[oi],condition=condition,synthetic_only=True))
            arrays=dict(free=free,truth=np.zeros((n,5,6),np.float64),seeds=np.asarray(ids,np.int64))
            pools.append(dict(pool=pool,arrays=arrays,report={'synthetic_only':True}))
        report=dict(status='PASS_COMPLETE_S2_RESPONSE_LATENT_POPULATION',protocol=self.protocol,
            producer_sources=self.ctx['sources']['response_latent_sources'],split=split,raw_acceptance=self.raw,
            parent_ids=ids,models=models,prediction_seal_verified_before_response_inference=split=='test')
        return dict(report=report,pools=pools,parent_ids=ids)
    def probe_verify(self,descriptor,**kwargs):
        self.log.append('probe:'+kwargs['expected_split']);row=copy.deepcopy(self.probes[descriptor['path']])
        self.assertEqual(kwargs['protocol_pair'],self.protocol);self.assertEqual(kwargs['measurement_sources_pair'],self.ctx['sources']['measurement_sources'])
        return row
    def response_verify(self,descriptor,**kwargs):
        split=kwargs['expected_split'];self.log.append('response:'+split)
        self.assertEqual(kwargs['protocol_pair'],self.protocol);self.assertEqual(kwargs['expected_raw_acceptance'],self.raw)
        return copy.deepcopy(self.responses[split])
    def authenticate(self,source,protocol):
        self.log.append('authenticate');self.assertEqual(source,self.ctx['sources']['measurement_sources']);self.assertEqual(protocol,self.protocol)
        return dict(authorized=False)
    def authorize(self,ctx,gate):self.log.append('authorize');return dict(authorized=True)
    def decode(self,ctx,pool,tokens):
        self.assertTrue(ctx['authorized']);self.log.append('decode:'+str(pool))
        return dict(poses=tokens[...,:6].astype(np.float64),verification={'synthetic_only':True,'pool':pool})
    def prepare(self):return a.prepare(self.binding)
    def input_pair(self,report):return a.read(report,'report')['regression_inputs']
    def make_test_binding(self,inputs,seal):
        seal_doc=a.read(seal,'seal');num=a.finalizer.load_context(inputs)
        self.responses['test']['report'].update(prediction_lock=seal_doc['prediction_lock'],orchestration_seal=seal,
            response_contract=self.contract,expected_bindings=num['expected_bindings'])
        rp=dump(self.root/'test_response_latent.json',self.responses['test']['report'])
        binding=dict(status=a.TEST_STATUS,regression_inputs=inputs,orchestration_seal=seal,sources=self.source,
            response_latent_population=rp,output_root=str(self.root/'test_assembled'))
        return dump(self.root/'test_binding.json',binding)

    def test_complete_prepare_materializes_existing_finalizer_schema_and_assigned_donors(self):
        report=self.prepare();inputs=self.input_pair(report);ctx=a.finalizer.load_context(inputs)
        for key,split in [('calibration','calibration'),('test_probe','test')]:
            scalar=a.finalizer.load_scalar(ctx,key)
            np.testing.assert_array_equal(scalar['donor_ids'],np.asarray(self.rosters[split+'_donor_ids'][::-1],np.int64))
            self.assertEqual(scalar['ordinary'].shape,(a.gate._size(split),6,4))
        self.assertNotIn('response',a.finalizer.load_scalar(ctx,'test_probe'))
        self.assertEqual(self.log[:3],['probe:calibration','probe:test','response:calibration'])
        self.assertLess(self.log.index('authorize'),self.log.index('decode:0'))
        self.assertNotIn('response:test',self.log)

    def test_complete_pipeline_reuses_frozen_regressors_and_independent_statistics(self):
        inputs=self.input_pair(self.prepare());seal=a.finalizer.seal(inputs,self.root/'sealed')
        bound=self.make_test_binding(inputs,seal)
        with mock.patch.object(a.finalizer.production,'fit_calibration',side_effect=AssertionError('Refit forbidden')):
            result=a.complete_test(bound);r=a.read(result,'heldout scalar report')
            final=a.finalizer.evaluate(inputs,seal,r['response_receipt'],r['response_access'],self.root/'evaluated')
        self.assertEqual(a.read(final,'final')['status'],'PASS_COMPLETE_S2_REGRESSION_AND_INDEPENDENT_STATISTICS')
        d=a.finalizer.independent.load_npz(a.read(r['response_receipt'],'response')['arrays']['path'])
        self.assertEqual(d['response'].shape,(512,6));self.assertEqual(set(d),{'response','recipient_ids','donor_ids'})

    def test_responds_to_separate_response_tokens_with_exact_sign_and_axes(self):
        inputs=self.input_pair(self.prepare());ctx=a.finalizer.load_context(inputs);z=a.finalizer.load_scalar(ctx,'calibration')
        expected=[]
        for pool in self.responses['calibration']['pools']:
            p=pool['arrays']['free'].astype(np.float64)
            expected.append((p[:,0,:,-1,2]**2-p[:,1,:,-1,2]**2).T)
        np.testing.assert_array_equal(z['response'],np.concatenate(expected,axis=1))
        self.assertGreater(z['response'][0,4],z['response'][0,0])

    def test_parent_reordering_blocks_before_decoder_or_output(self):
        self.responses['calibration']['parent_ids']=self.responses['calibration']['parent_ids'][::-1]
        with self.assertRaisesRegex(ValueError,'population mismatch'):self.prepare()
        self.assertNotIn('authenticate',self.log);self.assertFalse((self.root/'assembled').exists())

    def test_probe_unexpected_response_member_rejected(self):
        row=next(iter(self.probes.values()));row['arrays']['response']=np.zeros((256,6))
        with self.assertRaisesRegex(ValueError,'unexpected fields'):self.prepare()
        self.assertNotIn('response:calibration',self.log)

    def test_wrong_raw_ancestry_or_bank_donor_coverage_rejected(self):
        self.probes[self.prepare_doc['probe_scalars']['test']['path']]['report']['raw_acceptance']={'path':'/changed','sha256':'f'*64}
        with self.assertRaisesRegex(ValueError,'different raw ancestry'):self.prepare()

    def test_missing_response_pool_does_not_produce_accepted_outputs(self):
        self.responses['calibration']['pools'].pop()
        with self.assertRaisesRegex(ValueError,'All response pools'):self.prepare()
        self.assertTrue((self.root/'assembled.intent.json').exists())
        self.assertFalse((self.root/'assembled/report.json').exists())
        with self.assertRaisesRegex(ValueError,'unused absolute'):self.prepare()

    def test_wrong_predeclared_response_computation_rejected(self):
        c=a.read(self.contract,'contract');c['measurement_sources']={'path':'/wrong','sha256':'f'*64}
        self.prepare_doc['response_contract']=dump(self.root/'bad_contract.json',c)
        self.binding=dump(self.root/'assembly_binding.json',self.prepare_doc)
        with self.assertRaisesRegex(ValueError,'computation changed'):self.prepare()
        self.assertEqual(self.log,[])

    def test_seal_verified_before_response_producer_receipt_access(self):
        inputs=self.input_pair(self.prepare());seal=a.finalizer.seal(inputs,self.root/'sealed');bound=self.make_test_binding(inputs,seal)
        doc=a.read(seal,'seal');Path(doc['prediction_lock']['path']).write_text('changed')
        self.log=[]
        with self.assertRaises(ValueError):a.complete_test(bound)
        self.assertNotIn('response:test',self.log);self.assertNotIn('authenticate',self.log)

    def test_actual_response_must_bind_same_lock_and_access_order(self):
        inputs=self.input_pair(self.prepare());seal=a.finalizer.seal(inputs,self.root/'sealed');bound=self.make_test_binding(inputs,seal)
        self.responses['test']['report']['prediction_lock']={'path':'/wrong','sha256':'f'*64}
        with self.assertRaisesRegex(ValueError,'different prediction_lock'):a.complete_test(bound)
        self.responses['test']['report']['prediction_lock']=a.read(seal,'seal')['prediction_lock']
        self.responses['test']['report']['prediction_seal_verified_before_response_inference']=False
        with self.assertRaisesRegex(ValueError,'seal-first'):a.complete_test(bound)
        self.assertFalse((self.root/'test_assembled').exists())

    def test_incomplete_nan_wrong_dtype_response_blocks_scalar_acceptance(self):
        original=copy.deepcopy(self.responses['calibration'])
        for change in ('shape','nan','dtype'):
            self.responses['calibration']=copy.deepcopy(original)
            arr=self.responses['calibration']['pools'][0]['arrays']
            if change=='shape':arr['free']=arr['free'][:,:,:255]
            if change=='nan':arr['free'][0,0,0,0,0]=np.nan
            if change=='dtype':arr['free']=arr['free'].astype(np.float64)
            self.prepare_doc['output_root']=str(self.root/change)
            self.binding=dump(self.root/'assembly_binding.json',self.prepare_doc)
            with self.assertRaises(ValueError):self.prepare()
            self.assertFalse((self.root/change/'report.json').exists())

    def test_existing_output_no_overwrite_and_no_accidental_fitting(self):
        with mock.patch.object(a.finalizer.production,'fit_calibration',side_effect=AssertionError('Unexpected fit')):
            self.prepare()
            with self.assertRaisesRegex(ValueError,'unused absolute'):self.prepare()

    def test_unfrozen_source_guard_precedes_payloads(self):
        self.patches[0].stop()
        protocol=dump(self.root/'draft.json',dict(status='DRAFT_NOT_EXECUTION_AUTHORIZATION'))
        with self.assertRaisesRegex(ValueError,'Draft protocol'):a.source_context(self.source,protocol)
        self.patches[0].start()

    def test_real_assembly_source_authentication_and_missing_module(self):
        self.patches[0].stop()
        source=a.read(self.source,'synthetic assembly source')
        names=['assemble_scalars.py','features.py','finalize_regression.py','regression.py',
            'independent_regression.py','score_probe_population.py','response_population.py',
            'accept_probe_population.py','accept_raw_inputs.py','cache_population.py']
        source['files']={str(a.HERE/name):a.gate.sha(a.HERE/name) for name in names}
        complete=dump(self.root/'authenticated_sources.json',source)
        ctx=a.source_context(complete,self.protocol)
        self.assertEqual(ctx['source_pair'],complete)
        del source['files'][str(a.HERE/'response_population.py')]
        incomplete=dump(self.root/'missing_sources.json',source)
        with self.assertRaisesRegex(ValueError,'Missing direct'):a.source_context(incomplete,self.protocol)
        self.patches[0].start()

    def test_decoder_source_cannot_pollute_exact_numerical_source(self):
        self.patches[0].stop()
        source=a.read(self.source,'synthetic assembly source')
        names=['assemble_scalars.py','features.py','finalize_regression.py','regression.py',
            'independent_regression.py','score_probe_population.py','response_population.py',
            'accept_probe_population.py','accept_raw_inputs.py','cache_population.py']
        source['files']={str(a.HERE/name):a.gate.sha(a.HERE/name) for name in names}
        numerical=a.read(source['numerical_sources'],'numerical sources')
        numerical['files']['score_probe_population.py']=a.pair(a.HERE/'score_probe_population.py')
        source['numerical_sources']=dump(self.root/'polluted_numerical.json',numerical)
        polluted=dump(self.root/'polluted_sources.json',source)
        with self.assertRaisesRegex(ValueError,'unexpected fields'):a.source_context(polluted,self.protocol)
        self.patches[0].start()


if __name__=='__main__':unittest.main()
