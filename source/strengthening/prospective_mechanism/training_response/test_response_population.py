"""Focused synthetic/mock response producer tests; no actual model/data run."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
import response_population as r


def write(path,value):
    path.write_text(json.dumps(value,sort_keys=True));return r.pair(path)


def synthetic_context(root,split='calibration'):
    n=r.gate._size(split);ids=list(range(1000,1000+n));empty=write(root/'metadata.json',{})
    b=dict(status=r.INPUT_STATUS,split=split,protocol=empty,sources=empty,output_root=str(root/'output'))
    if split=='test':
        b.update(prediction_lock=empty,response_contract=empty,orchestration_seal=empty,
                 expected_bindings={k:empty['sha256'] for k in r.regression.REQUIRED_BINDINGS})
    bp=write(root/'input.json',b)
    pools=[]
    for p in range(3):
        cases=[dict(index=i,seed=seed,parent_id=seed,selected_index=i%300,selected_iteration=i%30,
                    pool=p,bank=empty,actions=empty,physics=empty) for i,seed in enumerate(ids)]
        models=[dict(pool=p,group=2*p,objective=o,condition=c,synthetic=True)
                for o in r.runtime.OBJECTIVES for c in r.CONDITIONS]
        pools.append(dict(pool=p,cases=cases,models=models,normalizers=['1'*64]*4))
    return dict(binding=bp,input=b,split=split,role=split+'_response',count=n,output=root/'output',sources={},
        protocol=empty,producer_sources=empty,runtime_contract=empty,raw_acceptance=empty,role_admission=empty,
        input_manifest=empty,content_lineage=empty,parent_ids=ids,pools=pools,admission_status=r.ADMITTED,
        prediction_seal_verified_before_response_inference=split=='test')


def mock_inputs(case):
    value=case['index']/8+case['pool']/4
    return dict(history_pixels=None,goal_pixels=None,prefix_actions=np.full((10,2),value,np.float32),
        population_actions=None,selected_index=case['selected_index'],observed_pixels='must never reach response runtime',
        selected_actions=np.full((25,2),value+.5,np.float32),states=np.full((36,7),value,np.float64))


def truth(states):
    # Independent analytical fixture mapping, not claimed as actual simulator pose.
    return np.full((5,6),float(states[0,0]),np.float64)


def handle(ctx,pool,objective,condition):
    spec=next(m for m in ctx['pools'][pool]['models'] if m['objective']==objective and m['condition']==condition)
    return SimpleNamespace(spec=spec,admission={'parent_ids':ctx['parent_ids'],'stream_role':'response'},
        access_receipt=r._access(ctx),normalization_sha256='1'*64,model_sha256='2'*64)


def result(model,parent_id):
    value=(parent_id-1000)/16+model.spec['pool']+.25*r.runtime.OBJECTIVES.index(model.spec['objective'])+.125*r.CONDITIONS.index(model.spec['condition'])
    array=np.full((5,192),value,np.float32)
    return dict(arrays={'free':array},checks=dict(native_endpoint_exact=True,identity_exact=True,
        model_unchanged=True,actions_and_normalization_unchanged=True,observed_history_first_exact=False),
        model_binding_sha256=r.runtime.value_sha(model.spec),access_receipt=model.access_receipt)


class ResponsePopulationTests(unittest.TestCase):
    def execute(self,ctx,*,fail_at=None):
        events=[];calls=[0]
        def seal(binding):events.append('seal');return ctx['parent_ids']
        def load_cal(pool,objective,condition,**kwargs):
            events.append(('calibration',pool,objective,condition));return handle(ctx,pool,objective,condition)
        def load_test(pool,objective,condition,**kwargs):
            self.assertTrue(events and events[0]=='seal')
            self.assertEqual(kwargs['prediction_lock'],ctx['input']['prediction_lock']['path'])
            self.assertEqual(kwargs['expected_bindings'],ctx['input']['expected_bindings'])
            events.append(('sealed-test',pool,objective,condition));return handle(ctx,pool,objective,condition)
        def native(model,**kwargs):
            self.assertNotIn('observed_pixels',kwargs);self.assertNotIn('replacements',kwargs)
            calls[0]+=1
            if calls[0]==fail_at:raise RuntimeError('synthetic native failure')
            return result(model,kwargs['parent_id'])
        with (mock.patch.object(r,'_seal',side_effect=seal),
              mock.patch.object(r.runtime,'load_calibration_response_model',side_effect=load_cal) as cal,
              mock.patch.object(r.runtime,'load_test_response_model',side_effect=load_test) as test,
              mock.patch.object(r.runtime,'load_probe_model') as probe,
              mock.patch.object(r.runtime,'native_rollouts',side_effect=native),
              mock.patch.object(r.cache,'case_inputs',side_effect=mock_inputs),mock.patch.object(r.cache,'physical_truth',side_effect=truth)):
            report=r.produce(ctx);probe.assert_not_called()
            if ctx['split']=='test':cal.assert_not_called();self.assertEqual(test.call_count,12)
            else:test.assert_not_called();self.assertEqual(cal.call_count,12)
        return report,events,calls[0]

    def verify(self,ctx):
        rp=r.pair(ctx['output']/'report.json')
        with (mock.patch.object(r,'load_context',return_value=ctx) as load,
              mock.patch.object(r.cache,'case_inputs',side_effect=mock_inputs),mock.patch.object(r.cache,'physical_truth',side_effect=truth),
              mock.patch.object(r.runtime,'load_calibration_response_model') as cal,
              mock.patch.object(r.runtime,'load_test_response_model') as test):
            accepted=r.verify_completed_population(rp,protocol_pair=ctx['protocol'],expected_split=ctx['split'],
                expected_raw_acceptance=ctx['raw_acceptance'],expected_seal=ctx['input'].get('orchestration_seal'))
            cal.assert_not_called();test.assert_not_called();self.assertFalse(load.call_args.kwargs['require_unused_output'])
        return accepted

    def test_import_has_no_torch(self):
        done=subprocess.run([sys.executable,'-c','import sys;sys.path.insert(0,'+repr(str(Path(r.__file__).parent))+');import response_population;assert "torch" not in sys.modules'],capture_output=True,text=True)
        self.assertEqual(done.returncode,0,done.stderr)

    def test_complete_calibration256_twelve_models_native_only_and_saved_verifier(self):
        with tempfile.TemporaryDirectory() as tmp:
            ctx=synthetic_context(Path(tmp));report,events,calls=self.execute(ctx)
            self.assertEqual(calls,3072);self.assertNotIn('seal',events);self.assertEqual(report['completed_cases'],3072)
            self.assertNotIn('prediction_lock',report);self.assertEqual(report['status'],r.STATUS)
            checked=self.verify(ctx);self.assertEqual(len(checked['pools']),3)
            for pool in checked['pools']:
                self.assertEqual(pool['arrays']['free'].shape,(2,2,256,5,192))
                self.assertEqual(pool['report']['policy_index'],8*pool['pool']+1)
                self.assertEqual(pool['report']['axes']['conditions'],['T0','T1'])
            self.assertTrue((ctx['output']/'DONE').exists())

    def test_complete_test512_seal_before_all_twelve_public_loaders(self):
        with tempfile.TemporaryDirectory() as tmp:
            ctx=synthetic_context(Path(tmp),'test');report,events,calls=self.execute(ctx)
            self.assertEqual(events[0],'seal');self.assertEqual(calls,6144);self.assertEqual(len(report['models']),12)
            self.assertTrue(report['prediction_seal_verified_before_response_inference'])
            self.assertEqual(report['prediction_lock'],ctx['input']['prediction_lock'])
            self.assertEqual(report['orchestration_seal'],ctx['input']['orchestration_seal'])
            self.assertEqual(self.verify(ctx)['parent_ids'],ctx['parent_ids'])

    def test_seal_failure_prevents_any_model_or_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            ctx=synthetic_context(Path(tmp),'test')
            with mock.patch.object(r,'_seal',side_effect=ValueError('bad actual seal')),mock.patch.object(r.runtime,'load_test_response_model') as load:
                with self.assertRaisesRegex(ValueError,'bad actual seal'):r.produce(ctx)
                load.assert_not_called()
            self.assertFalse(ctx['output'].exists())

    def test_outer_inner_and_actual_five_bindings_cross_check(self):
        with tempfile.TemporaryDirectory() as tmp:
            ctx=synthetic_context(Path(tmp),'test');b=ctx['input'];outer={'input_binding':ctx['binding']}
            finalctx=dict(binding=dict(protocol=b['protocol'],response_contract=b['response_contract'],
                expected_parent_ids={'test_recipient_ids':ctx['parent_ids']}),expected_bindings=b['expected_bindings'])
            with (mock.patch.object(r.gate,'_json',return_value=outer),mock.patch.object(r.finalizer,'load_context',return_value=finalctx),
                  mock.patch.object(r.finalizer,'verify_seal',return_value=b['prediction_lock']) as verify):
                self.assertEqual(r._seal(b),ctx['parent_ids']);verify.assert_called_once_with(finalctx,b['orchestration_seal'])
                changed=copy.deepcopy(b);changed['expected_bindings']['data_isolation_sha256']='1'*64
                with self.assertRaises(ValueError):r._seal(changed)
                changed=copy.deepcopy(b);changed['prediction_lock']=dict(b['prediction_lock'],sha256='1'*64)
                with self.assertRaises(ValueError):r._seal(changed)
                changed=copy.deepcopy(b);changed['expected_bindings']['extra']='0'*64
                with self.assertRaises(ValueError):r._seal(changed)

    def test_native_response_rejects_probe_arrays_false_parity_or_changed_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            ctx=synthetic_context(Path(tmp));h=handle(ctx,0,'decoded_teacher','T0')
            evidence=r._model_evidence(h,ctx,h.spec,'1'*64);value=result(h,1000)
            r._native(value,evidence)
            changes=(lambda v:v['arrays'].update(actual=v['arrays']['free']),lambda v:v['checks'].update(identity_exact=False),
                     lambda v:v['checks'].pop('native_endpoint_exact'),lambda v:v['checks'].update(observed_history_first_exact=True),
                     lambda v:v.update(model_binding_sha256='bad'),lambda v:v.update(access_receipt={'role':'probe_calibration'}),
                     lambda v:v['arrays']['free'].__setitem__((4,191),np.nan))
            for change in changes:
                altered=copy.deepcopy(value);change(altered)
                with self.assertRaises(ValueError):r._native(altered,evidence)

    def test_model_norm_parent_stream_and_access_mismatch_block(self):
        with tempfile.TemporaryDirectory() as tmp:
            ctx=synthetic_context(Path(tmp));spec=ctx['pools'][0]['models'][0]
            changes=(lambda h:h.spec.update(condition='T2'),lambda h:h.admission.update(parent_ids=[0]*256),
                     lambda h:h.admission.update(stream_role='probe'),lambda h:h.access_receipt.update(role='probe_calibration'),
                     lambda h:setattr(h,'normalization_sha256','0'*64),lambda h:setattr(h,'model_sha256',''))
            for change in changes:
                h=copy.deepcopy(handle(ctx,0,'decoded_teacher','T0'));change(h)
                with self.assertRaises(ValueError):r._model_evidence(h,ctx,spec,'1'*64)

    def test_response_manifest_requires_original_1_9_17_and_full_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);ctx=synthetic_context(root);p=ctx['protocol'];n=256
            manifest=dict(status='S2_COMPLETE_ROLE_STREAM_INPUT_MANIFEST',protocol_sha256=p['sha256'],sources_sha256='s',
                bank_role='calibration_recipient',stream_role='response',count=n,parent_ids=ctx['parent_ids'],
                routes=[dict(pool=d['pool'],group=2*d['pool'],policy_index=8*d['pool']+1,cases=d['cases'],
                             actions_report=p,physics_report=p) for d in ctx['pools']])
            kwargs=dict(split='calibration',protocol_sha256=p['sha256'],raw_sources_sha256='s')
            self.assertEqual([x['policy_index'] for x in r.validate_response_manifest(manifest,{'parent_ids':ctx['parent_ids']},**kwargs)],[1,9,17])
            changes=(lambda m:m['routes'][1].update(policy_index=8),lambda m:m['routes'][2]['cases'].pop(),
                     lambda m:m.update(stream_role='probe'),lambda m:m['parent_ids'].reverse(),lambda m:m.update(count=512))
            for change in changes:
                bad=copy.deepcopy(manifest);change(bad)
                with self.assertRaises(ValueError):r.validate_response_manifest(bad,{'parent_ids':ctx['parent_ids']},**kwargs)

    def test_existing_partial_and_failure_block_retry(self):
        with tempfile.TemporaryDirectory() as tmp:
            ctx=synthetic_context(Path(tmp))
            with self.assertRaisesRegex(RuntimeError,'synthetic native'):self.execute(ctx,fail_at=3)
            failure=json.loads((ctx['output']/'failure.json').read_text());self.assertEqual(failure['completed_cases'],2)
            self.assertTrue((ctx['output']/'pool_0/decoded_teacher_T0/case_001.npz').exists())
            self.assertFalse((ctx['output']/'DONE').exists());self.assertFalse((ctx['output']/'report.json').exists())
            with mock.patch.object(r.runtime,'load_calibration_response_model') as load,self.assertRaises(ValueError):r.produce(ctx)
            load.assert_not_called()

    def test_saved_verifier_rejects_externally_wrong_split_seal_and_top_models(self):
        with tempfile.TemporaryDirectory() as tmp:
            ctx=synthetic_context(Path(tmp));self.execute(ctx);path=ctx['output']/'report.json';report=json.loads(path.read_text())
            with self.assertRaises(ValueError):r.verify_completed_population(r.pair(path),protocol_pair=ctx['protocol'],expected_split='test',expected_raw_acceptance=ctx['raw_acceptance'])
            report['models'][0]['normalization_sha256']='3'*64;write(path,report)
            with self.assertRaises(ValueError):self.verify(ctx)

    def test_source_context_rejects_draft_and_extra_test_fields_before_runtime(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);ctx=synthetic_context(root);b=ctx['input']
            for change in (dict(b,status='DRAFT'),dict(b,prediction_lock=ctx['protocol']),dict(b,output_root='relative')):
                bp=write(root/'bad.json',change)
                with mock.patch.object(r.runtime,'_runtime_admission') as rt,self.assertRaises(ValueError):r.load_context(bp['path'],bp['sha256'])
                rt.assert_not_called()

    def test_full_source_preflight_uses_response_cases_and_both_runtime_allowlists(self):
        # Hash actual implementation/metadata source files, but mock raw admission
        # and fixed model declarations. Never read real head/model/input arrays.
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);ctx=synthetic_context(root);ids=ctx['parent_ids'];b=ctx['input'];p=write(root/'protocol.json',{'status':'S2_SCIENTIFIC_PROTOCOL_FROZEN'})
            rawp=write(root/'raw.json',{});runtimep=write(root/'runtime.json',{});accepted=write(root/'accepted.json',{})
            admission=dict(parent_ids=tuple(ids),stream_role='response',bank_manifest=accepted)
            routes=[dict(pool=d['pool'],group=2*d['pool'],policy_index=8*d['pool']+1,cases=d['cases'],actions_report=accepted,physics_report=accepted) for d in ctx['pools']]
            manifest=dict(status='S2_COMPLETE_ROLE_STREAM_INPUT_MANIFEST',protocol_sha256=p['sha256'],sources_sha256=rawp['sha256'],bank_role='calibration_recipient',stream_role='response',count=256,parent_ids=ids,routes=routes)
            admission['input_manifest']=write(root/'manifest.json',manifest)
            rawctx=dict(root=r.runtime.ROOT,sources={'files':{}})
            paths=[Path(r.__file__),Path(r.cache.__file__),Path(r.raw.__file__),Path(r.runtime.__file__),Path(r.gate.__file__),
                Path(r.finalizer.__file__),Path(r.regression.__file__),r.HERE/'independent_regression.py',r.HERE/'accept_raw_inputs.py',
                r.HERE/'projection_four.py',r.cache.POSE_SOURCE,*r.runtime.METADATA.values()]
            source=dict(status=r.SOURCE_STATUS,protocol_sha256=p['sha256'],source_root=str(r.runtime.ROOT),
                files={str(path):r.gate.sha(path) for path in paths},raw_sources=rawp,raw_acceptance=accepted,
                runtime_contracts={role:runtimep for role in ('calibration_response','test_response')})
            b.update(protocol=p,sources=write(root/'source.json',source));bp=write(root/'input.json',b)
            allspecs=[m for d in ctx['pools'] for m in d['models']]
            calls=[]
            def runtime_admission(path,sha,role):
                calls.append(role);return dict(protocol=p,input_admission=accepted,files={}),admission,allspecs
            def auth(rawcontext):return ctx['pools'][rawcontext['info']['pool']]['cases']
            with (mock.patch.object(r.raw,'context',return_value=rawctx),
                  mock.patch.object(r.gate,'_raw_admission',return_value=({'content_lineage':accepted},{'calibration_recipient':ids},
                      {role:(accepted,None,None) for role in ('calibration_response','test_response')})),
                  mock.patch.object(r.runtime,'_runtime_admission',side_effect=runtime_admission),
                  mock.patch.object(r.cache,'authenticate_reports',side_effect=auth),mock.patch.object(r,'_model_metadata',return_value='1'*64) as metadata,
                  mock.patch.object(r.runtime,'load_calibration_response_model') as load):
                actual=r.load_context(bp['path'],bp['sha256']);load.assert_not_called()
            self.assertEqual(calls,['calibration_response','test_response']);self.assertEqual(metadata.call_count,12)
            self.assertEqual(actual['admission_status'],r.ADMITTED);self.assertEqual(actual['parent_ids'],ids)


if __name__=='__main__':unittest.main()
