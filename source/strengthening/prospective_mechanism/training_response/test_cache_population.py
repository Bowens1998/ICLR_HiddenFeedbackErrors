"""Synthetic/source-interface tests only; no real raw data or model execution."""
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
import cache_population as c


def fake_pair(name):return dict(path='/synthetic/'+name,sha256='0'*64)


def population():
    banks=[]
    for i,role in enumerate(c.raw.ROLES):
        n=c.raw.COUNTS[role]
        banks.append(dict(bank_role=role,count=n,parent_ids=list(range(10000*(i+1),10000*(i+1)+n)),
                          manifest=fake_pair(role+'manifest'),role_metadata=fake_pair(role+'role')))
    return dict(status='PASS_S2_COMPLETE_RAW_INPUT_POPULATION',protocol_sha256='p',sources_sha256='s',raw_sources_sha256='s',
                banks=banks,admissions=[dict(role=role,**fake_pair(role)) for role in sorted(c.ALL_ADMISSIONS)],content_lineage=fake_pair('content'))


def input_manifest(role='calibration_recipient'):
    info=c.role_info(role,0);n=info['count'];ids=list(range(n));stream='donor_observed' if info['donor'] else 'probe'
    admission=dict(parent_ids=ids,stream_role=stream)
    routes=[]
    for pool in range(3):
        cases=[dict(index=i,seed=i,selected_index=0,selected_iteration=0,
                    **{k:fake_pair(k+str(pool)+'_'+str(i)) for k in ('bank','actions','physics')}) for i in range(n)]
        routes.append(dict(pool=pool,group=2*pool,policy_index=8*pool,cases=cases,
                           actions_report=fake_pair('ar'+str(pool)),physics_report=fake_pair('pr'+str(pool))))
    return dict(status='S2_COMPLETE_ROLE_STREAM_INPUT_MANIFEST',protocol_sha256='p',sources_sha256='s',
                bank_role=role,stream_role=stream,count=n,parent_ids=ids,routes=routes),admission,info


def raw_case():
    states=np.zeros((36,7),np.float64);states[:,4]=np.arange(36)*.1
    pixels=np.zeros((8,224,224,3),np.uint8);prefix=np.zeros((10,2),np.float32)
    bank=dict(seed=np.asarray(7),history_pixels=pixels[:3],goal_pixels=pixels[-1],prefix=prefix,history_states=states[[0,5,10]])
    action=dict(seed=np.asarray(7),selected_index=np.asarray(0),selected_iteration=np.asarray(0),
        population_actions=np.zeros((300,25,2),np.float32),selected_actions=np.zeros((25,2),np.float32),
        cost_trace=np.zeros((30,300),np.float64),population_costs=np.zeros(300,np.float64))
    physics=dict(seed=np.asarray(7),pixels=pixels,states=states,actions=np.zeros((35,2),np.float32))
    evidence=dict(seed=7,selected_index=0,selected_iteration=0,bank={},actions={},physics={})
    return bank,action,physics,evidence


def mocked_context(directory,role):
    info=c.role_info(role,0);ids=list(range(info['count']));out=Path(directory)/role/'pool_0'
    ctx=dict(info=info,admission={'parent_ids':ids},output=out,runtime_contract=fake_pair('runtime'),
             cache_binding=dict(protocol_sha256='p',cache_sources_sha256='s',raw_sources_sha256='r',source_sha256=c.raw.sha(c.__file__)))
    cases=[dict(index=i,seed=i,parent_id=i) for i in ids]
    return ctx,cases


def mock_handle(ctx,objective):
    info=ctx['info'];spec=dict(pool=info['pool'],group=info['group'],objective=objective,condition='T0',
        original={'synthetic':'fixed original'},summary={'synthetic':'summary'},config={'synthetic':'config'})
    return SimpleNamespace(spec=spec,access_receipt={'role':info['runtime_role']},normalization_sha256='synthetic norm',
                           model_sha256='synthetic model',admission=ctx['admission'])


def mock_inputs(_):
    return dict(history_pixels=None,goal_pixels=None,prefix_actions=np.zeros((10,2),np.float32),population_actions=None,
        selected_index=0,observed_pixels=None,selected_actions=np.zeros((25,2),np.float32),states=np.zeros((36,7),np.float64))


def mock_native(handle,**kwargs):
    free=np.full((5,192),3 if handle.spec['objective']=='decoded_teacher' else 4,np.float32)
    observed=np.full((5,192),2,np.float32);history=free.copy();reset=free.copy();reset[0]=observed[0]
    return dict(arrays=dict(free=free,observed=observed,observed_history=history,reset=reset),
        checks={key:True for key in ('native_endpoint_exact','identity_exact','observed_history_first_exact','model_unchanged','actions_and_normalization_unchanged')},
        model_binding_sha256=c.runtime.value_sha(handle.spec),access_receipt=handle.access_receipt)


class CachePopulationTests(unittest.TestCase):
    def test_import_is_lazy(self):
        script='import sys;sys.path.insert(0,'+repr(str(Path(c.__file__).parent))+');import cache_population;assert "torch" not in sys.modules'
        result=subprocess.run([sys.executable,'-c',script],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)

    def test_only_probe_roles_fixed_pools_and_original_indices(self):
        for role in c.raw.ROLES:
            for pool in range(3):
                info=c.role_info(role,pool);self.assertEqual(info['policy_index'],8*pool)
                self.assertEqual(info['count'],256 if role.startswith('calibration') else 512)
        for role,pool in [('test_response',0),('test_recipient',3),('test_donor',True)]:
            with self.assertRaises(ValueError):c.role_info(role,pool)

    def test_unfrozen_cache_source_rejected_before_raw_or_runtime(self):
        with mock.patch.object(c.raw,'checked_json',return_value={'status':'DRAFT'}),mock.patch.object(c.raw,'context') as rc,mock.patch.object(c.runtime,'_runtime_admission') as rt:
            with self.assertRaises(ValueError):c.context('unused','p','unused','s','test_recipient',0)
            rc.assert_not_called();rt.assert_not_called()

    def test_complete_six_admissions_four_banks_and_disjoint_parents(self):
        p=population();c.validate_population(p,protocol_sha256='p',raw_sources_sha256='s')
        mutations=(lambda p:p['admissions'].pop(),lambda p:p['banks'][0]['parent_ids'].pop(),
                   lambda p:p['banks'][1]['parent_ids'].__setitem__(0,p['banks'][0]['parent_ids'][0]),
                   lambda p:p.update(raw_sources_sha256='wrong'))
        for mutation in mutations:
            q=copy.deepcopy(p);mutation(q)
            with self.assertRaises(ValueError):c.validate_population(q,protocol_sha256='p',raw_sources_sha256='s')

    def test_full_role_manifest_and_donor_distinction(self):
        for role in c.raw.ROLES:
            m,a,i=input_manifest(role);route=c.validate_manifest(m,a,i,'p','s')
            self.assertEqual(route['policy_index'],0)
            if i['donor']:self.assertEqual(m['stream_role'],'donor_observed')

    def test_missing_case_reordered_parent_wrong_route_or_source_block(self):
        m,a,i=input_manifest()
        changes=(lambda m:m['routes'].pop(),lambda m:m['routes'][0]['cases'].pop(),
                 lambda m:m['routes'][0].update(policy_index=1),lambda m:m['parent_ids'].reverse(),
                 lambda m:m['routes'][0]['cases'][0].update(selected_index=300),lambda m:m.update(sources_sha256='wrong'))
        for change in changes:
            x=copy.deepcopy(m);change(x)
            with self.assertRaises(ValueError):c.validate_manifest(x,a,i,'p','s')

    def test_raw_array_file_hash_verified_before_loading(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'synthetic.npz';np.savez(path,x=np.arange(3))
            self.assertEqual(c.read_arrays(dict(path=str(path),sha256=c.raw.sha(path)))['x'].tolist(),[0,1,2])
            with mock.patch.object(c.np,'load') as load:
                with self.assertRaises(ValueError):c.read_arrays(dict(path=str(path),sha256='0'*64))
                load.assert_not_called()

    def test_native_raw_case_checks_complete_controls_states_and_history(self):
        b,a,p,e=raw_case()
        with mock.patch.object(c,'read_arrays',side_effect=[b,a,p]):inputs=c.case_inputs(e)
        self.assertEqual(inputs['population_actions'].shape,(300,25,2));self.assertEqual(inputs['observed_pixels'].shape,(5,224,224,3))
        mutations=(lambda b,a,p:a.update(selected_index=np.asarray(1)),lambda b,a,p:a.update(population_actions=np.zeros((1,25,2),np.float32)),
                   lambda b,a,p:p['actions'].__setitem__((12,0),.1),lambda b,a,p:p['states'].__setitem__((5,0),1),
                   lambda b,a,p:a['cost_trace'].__setitem__((1,1),-1),lambda b,a,p:p.update(seed=np.asarray(7.5)),
                   lambda b,a,p:a.update(cost_trace=a['cost_trace'].astype(np.float32)))
        for mutation in mutations:
            b,a,p,e=raw_case();mutation(b,a,p)
            with mock.patch.object(c,'read_arrays',side_effect=[b,a,p]),self.assertRaises((ValueError,AssertionError)):c.case_inputs(e)

    def test_physical_truth_uses_exact_bound_v8_mapping(self):
        b,a,p,e=raw_case();truth=c.physical_truth(p['states'])
        expected=np.c_[p['states'][[15,20,25,30,35],:4],np.sin(p['states'][[15,20,25,30,35],4]),np.cos(p['states'][[15,20,25,30,35],4])]
        self.assertEqual(truth.dtype,np.float64);np.testing.assert_array_equal(truth,expected)

    def test_runtime_parity_report_and_complete_tokens_required(self):
        with tempfile.TemporaryDirectory() as temp:
            ctx,_=mocked_context(temp,'calibration_recipient');handle=mock_handle(ctx,'decoded_teacher')
            result=mock_native(handle);c.validate_token_result(result,ctx['info'])
            for mutate in (lambda x:x['checks'].update(identity_exact=False),lambda x:x['arrays'].pop('reset'),
                           lambda x:x['arrays']['observed_history'].__setitem__((0,0),99),lambda x:x['arrays']['free'].__setitem__((1,0),np.nan)):
                altered=copy.deepcopy(result);mutate(altered)
                with self.assertRaises((ValueError,AssertionError)):c.validate_token_result(altered,ctx['info'])

    def test_model_role_and_parent_binding_rejects_t1(self):
        with tempfile.TemporaryDirectory() as temp:
            ctx,_=mocked_context(temp,'test_recipient');info=dict(ctx['info'],parent_ids=ctx['admission']['parent_ids'])
            h=mock_handle(ctx,'decoded_teacher');h.spec['condition']='T1'
            with self.assertRaises(ValueError):c.model_evidence(h,info,'decoded_teacher')
            h=mock_handle(ctx,'decoded_teacher');h.admission={'parent_ids':[999]*512}
            with self.assertRaises(ValueError):c.model_evidence(h,info,'decoded_teacher')

    def test_complete_recipient_population_written_only_after_both_models(self):
        with tempfile.TemporaryDirectory() as temp:
            ctx,cases=mocked_context(temp,'calibration_recipient')
            def load(pool,objective,**kwargs):return mock_handle(ctx,objective)
            with (mock.patch.object(c,'authenticate_reports',return_value=cases),mock.patch.object(c,'case_inputs',side_effect=mock_inputs),
                 mock.patch.object(c.runtime,'load_probe_model',side_effect=load) as loader,mock.patch.object(c.runtime,'native_rollouts',side_effect=mock_native) as native,
                 mock.patch.object(c.runtime,'load_donor_encoder') as donor):
                report=c.produce(ctx)
            self.assertEqual(loader.call_count,2);self.assertEqual(native.call_count,512);donor.assert_not_called()
            self.assertTrue(report['observed_shared_encoder_exact']);self.assertEqual(report['axes'],c.AXES)
            with np.load(report['arrays']['path']) as z:
                self.assertEqual(z['free'].shape,(2,256,5,192));self.assertEqual(z['truth'].shape,(256,5,6))
                self.assertEqual(z['prefix_actions'].shape,(256,10,2));self.assertEqual(z['selected_actions'].shape,(256,25,2))
                np.testing.assert_array_equal(z['seeds'],np.arange(256));self.assertNotIn('population_actions',z.files)
            self.assertTrue((ctx['output']/'DONE').exists());self.assertEqual(len(report['cases']),256)

    def test_donor_uses_only_own_encoder_and_saves_observed5(self):
        with tempfile.TemporaryDirectory() as temp:
            ctx,cases=mocked_context(temp,'test_donor')
            def load(*args,**kwargs):return mock_handle(ctx,'decoded_teacher')
            def encode(h,**kwargs):return dict(observed=np.ones((5,192),np.float32),model_binding_sha256=c.runtime.value_sha(h.spec),access_receipt=h.access_receipt)
            with (mock.patch.object(c,'authenticate_reports',return_value=cases),mock.patch.object(c,'case_inputs',side_effect=mock_inputs),
                 mock.patch.object(c.runtime,'load_donor_encoder',side_effect=load) as loader,mock.patch.object(c.runtime,'encode_donor_observations',side_effect=encode) as encoder,
                 mock.patch.object(c.runtime,'load_probe_model') as probe,mock.patch.object(c.runtime,'native_rollouts') as rollout,mock.patch.object(c,'physical_truth') as truth):
                report=c.produce(ctx)
            self.assertEqual(loader.call_count,1);self.assertEqual(encoder.call_count,512);probe.assert_not_called();rollout.assert_not_called();truth.assert_not_called()
            self.assertTrue(report['donor_encoder_only']);self.assertFalse(report['donor_model_predictions_generated'])
            with np.load(report['arrays']['path']) as z:self.assertEqual(set(z.files),{'observed5','seeds'});self.assertEqual(z['observed5'].shape,(512,192))

    def test_cross_objective_encoder_or_observed_token_drift_blocks(self):
        for failure in ('encoder','observed'):
            with tempfile.TemporaryDirectory() as temp:
                ctx,cases=mocked_context(temp,'calibration_recipient')
                def load(pool,objective,**kwargs):
                    h=mock_handle(ctx,objective)
                    if objective=='physical_labels' and failure=='encoder':h.spec['original']={'synthetic':'changed encoder'}
                    return h
                def native(h,**kwargs):
                    result=mock_native(h)
                    if h.spec['objective']=='physical_labels' and failure=='observed':result['arrays']['observed'][1,0]=99.
                    return result
                with (mock.patch.object(c,'authenticate_reports',return_value=cases),mock.patch.object(c,'case_inputs',side_effect=mock_inputs),
                      mock.patch.object(c.runtime,'load_probe_model',side_effect=load),mock.patch.object(c.runtime,'native_rollouts',side_effect=native)):
                    with self.assertRaises((ValueError,AssertionError)):c.produce(ctx)
                self.assertFalse((ctx['output']/'DONE').exists());self.assertFalse((ctx['output']/'report.json').exists())

    def test_failed_case_keeps_partial_and_blocks_second_attempt(self):
        with tempfile.TemporaryDirectory() as temp:
            ctx,cases=mocked_context(temp,'calibration_recipient')
            with (mock.patch.object(c,'authenticate_reports',return_value=cases),mock.patch.object(c,'case_inputs',side_effect=RuntimeError('synthetic case failed')),
                 mock.patch.object(c.runtime,'load_probe_model',return_value=mock_handle(ctx,'decoded_teacher')) as loader):
                with self.assertRaisesRegex(RuntimeError,'case failed'):c.produce(ctx)
                with self.assertRaises(ValueError):c.produce(ctx)
                self.assertEqual(loader.call_count,1)
            self.assertTrue(ctx['output'].is_dir());self.assertFalse((ctx['output']/'report.json').exists());self.assertFalse((ctx['output']/'DONE').exists())

    def test_preflight_failure_or_prior_intent_prevents_model_load(self):
        with tempfile.TemporaryDirectory() as temp:
            ctx,_=mocked_context(temp,'test_donor')
            with mock.patch.object(c,'authenticate_reports',side_effect=ValueError('synthetic raw mismatch')),mock.patch.object(c.runtime,'load_donor_encoder') as loader:
                with self.assertRaises(ValueError):c.produce(ctx)
                loader.assert_not_called();self.assertFalse(ctx['output'].exists())
            path=ctx['output'].parent/(ctx['output'].name+'.intent.json');c.raw.write_exclusive(path,{'synthetic':True})
            with mock.patch.object(c,'authenticate_reports') as preflight:
                with self.assertRaises(ValueError):c.produce(ctx)
                preflight.assert_not_called()


if __name__=='__main__':unittest.main()
