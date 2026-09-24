"""Synthetic/mock wrapper tests; no real heads, weights, banks or Torch run."""
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
sys.path.insert(0, str(Path(__file__).resolve().parent))
import rollout_population as r
from test_accept_probe_population import fixture, json_file


def save(path, arrays):
    np.savez_compressed(path, **arrays)
    return r.file_pair(path)


def model(pool, objective, ids, split):
    spec = dict(pool=pool,group=2*pool,objective=objective,condition='T0',
                original={'synthetic':'fixed original'},summary={'synthetic':'summary'},config={'synthetic':'config'})
    handle = SimpleNamespace(spec=spec,admission={'parent_ids':ids},
        access_receipt={'role':'probe_'+split},normalization_sha256='synthetic norm',model_sha256='synthetic state')
    info = dict(pool=pool,group=2*pool,parent_ids=ids,runtime_role='probe_'+split)
    return handle,r.cache.model_evidence(handle,info,objective)


def context(root, data, *, corrupt_final_family=False):
    """Real temporary hashes; upstream source/raw admission is mocked explicitly."""
    split = data['split']; n = r.gate._size(split); projection_out = root/'projection';projection_out.mkdir()
    empty = json_file(root/'synthetic_metadata.json',{})
    projected_arrays = save(root/'projection_arrays.npz',data['projection'])
    head = save(root/'synthetic_head.npz',data['head_A'])
    input_doc = dict(protocol=empty,sources=empty,donor_assignment=empty)
    input_pair = json_file(root/'projection_input.json',input_doc)
    ids = data['recipient']['seeds'].tolist(); pools=[]; rows=[]
    for pool in range(3):
        cases = [dict(index=i,seed=ids[i],parent_id=ids[i],selected_index=int(data['recipient']['selected_index'][i])) for i in range(n)]
        models = [model(pool,o,ids,split)[1] for o in r.four.OBJECTIVES]
        report = dict(cases=cases,models=models,runtime_contract=empty,input_admission=empty,input_manifest=empty)
        families = copy.deepcopy(data['families'])
        if corrupt_final_family and pool == 2:families[-1]['effective_norm'] = 999.
        folder=projection_out/f'pool_{pool}';folder.mkdir()
        projected = dict(status=r.project.POOL_STATUS,split=split,pool=pool,group=2*pool,count=n,model_role='T0',
            protocol_sha256=empty['sha256'],probe_sources_sha256=empty['sha256'],source_sha256=r.gate.sha(r.project.__file__),
            recipient_cache=empty,donor_cache=empty,donor_assignment=empty,head_A=head,axes=r.gate.PROJECTION_AXES,
            input_binding=input_pair,attempted_families=n,completed_families=n,failed_families=0,rows_removed=0,
            q_g_weights_deserialized=False,q_g_outputs_computed=False,response_models_deserialized=False,
            later_horizon_values_used_for_projection=False,arrays=projected_arrays,families=families,
            zero_families=len(range(0,n,17)))
        pair=json_file(folder/'report.json',projected);rows.append(dict(pool=pool,report=pair,status=r.project.POOL_STATUS))
        pools.append(dict(pool=pool,group=2*pool,recipient_cache=empty,donor_cache=empty,head_A_file=head,head_A=data['head_A'],
            predicted=data['recipient']['free'][:,:,0].copy(),actual=data['recipient']['observed'][:,0].copy(),donor=data['donor']['observed5'].copy(),
            recipient_report=report,donor_report={},recipient_arrays=data['recipient'],donor_arrays=data['donor']))
    population=dict(status=r.project.POPULATION_STATUS,input_binding=input_pair,protocol_sha256=empty['sha256'],
        probe_sources_sha256=empty['sha256'],source_sha256=r.gate.sha(r.project.__file__),split=split,count=n,pool_count=3,
        attempted_families=3*n,completed_families=3*n,failed_families=0,rows_removed=0,
        q_g_weights_deserialized=False,q_g_outputs_computed=False,response_models_deserialized=False,
        zero_families=3*len(range(0,n,17)),projections=rows)
    pp=json_file(projection_out/'report.json',population)
    binding=json_file(root/'rollout_input.json',dict(status=r.INPUT_STATUS,projection_population=pp,output_root=str(root/'rollouts')))
    ctx=dict(binding=input_pair,input=input_doc,split=split,count=n,sources={'producers':{'rollout':r.file_pair(r.__file__)}},
        output=projection_out,assignment=data['assignment'].copy(),recipient_ids=data['recipient']['seeds'].copy(),
        donor_ids=data['donor']['seeds'].copy(),pools=pools)
    return ctx,binding


def inputs(data, evidence):
    i=evidence['index']
    return dict(history_pixels=None,goal_pixels=None,prefix_actions=data['recipient']['prefix_actions'][i],
        population_actions=None,selected_index=int(data['recipient']['selected_index'][i]),observed_pixels=None,
        selected_actions=data['recipient']['selected_actions'][i],states=None)


def native_result(data, handle, parent_id, replacements):
    goal=int(parent_id)-int(data['recipient']['seeds'][0]);oi=r.four.OBJECTIVES.index(handle.spec['objective'])
    arrays={key:data['recipient'][key][oi,goal].copy() for key in ('free','observed_history','reset')}
    arrays['observed']=data['recipient']['observed'][goal].copy()
    arrays['actual']=data['rollout']['tokens'][oi,goal,1].copy();arrays['donor']=data['rollout']['tokens'][oi,goal,2].copy()
    for source in r.four.SOURCES:np.testing.assert_array_equal(arrays[source][0],replacements[source])
    return dict(arrays=arrays,checks={key:True for key in ('native_endpoint_exact','identity_exact','observed_history_first_exact',
        'model_unchanged','actions_and_normalization_unchanged')},model_binding_sha256=r.runtime.value_sha(handle.spec),
        access_receipt=handle.access_receipt)


class RolloutProducerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.data=fixture()

    def admitted(self,root,data=None,**kwargs):
        data=self.data if data is None else data
        ctx,pair=context(root,data,**kwargs)
        with mock.patch.object(r.project,'load_context',return_value=ctx) as upstream:
            result=r.load_context(pair['path'],pair['sha256'])
        self.assertEqual(upstream.call_args.kwargs,dict(require_unused_output=False,retain_caches=True))
        return result

    def execute(self,ctx,data,*,fail_at=None,wrong_model=False):
        calls=[0]
        def load(pool,objective,**kwargs):
            self.assertEqual(kwargs['split'],data['split'])
            handle,_=model(pool,objective,ctx['recipient_ids'].tolist(),data['split'])
            if wrong_model:handle.spec['condition']='T1'
            return handle
        def run(handle,**kwargs):
            calls[0]+=1
            if calls[0]==fail_at:raise RuntimeError('synthetic backend failure')
            return native_result(data,handle,kwargs['parent_id'],kwargs['replacements'])
        with (mock.patch.object(r.runtime,'load_probe_model',side_effect=load),
              mock.patch.object(r.runtime,'native_rollouts',side_effect=run),
              mock.patch.object(r.cache,'case_inputs',side_effect=lambda evidence:inputs(data,evidence)),
              mock.patch.object(r.runtime,'load_donor_encoder') as donor,
              mock.patch.object(r.runtime,'load_calibration_response_model') as cal,
              mock.patch.object(r.runtime,'load_test_response_model') as test):
            result=r.produce(ctx)
            donor.assert_not_called();cal.assert_not_called();test.assert_not_called()
        return result,calls[0]

    def test_lazy_import_does_not_import_torch(self):
        result=subprocess.run([sys.executable,'-c','import sys;sys.path.insert(0,'+repr(str(Path(r.__file__).parent))+');import rollout_population;assert "torch" not in sys.modules'],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)

    def test_complete_calibration_three_pools_zeros_and_existing_value_gate(self):
        with tempfile.TemporaryDirectory() as temp:
            ctx=self.admitted(Path(temp));result,calls=self.execute(ctx,self.data)
            self.assertEqual(calls,1536);self.assertEqual(result['completed_cases'],1536)
            self.assertEqual(result['status'],r.POPULATION_STATUS);self.assertTrue((ctx['output']/'DONE').exists())
            for row in result['rollouts']:
                report=r.gate._json(row['report']);arrays=r.gate._arrays(report['arrays'])
                self.assertTrue(report['identity_native_endpoint_saved_from_checked_free'])
                self.assertIn('not separately returned',report['identity_endpoint_provenance'])
                self.assertEqual(report['models'],ctx['pools'][row['pool']]['recipient_report']['models'])
                self.assertEqual(report['axes'],r.gate.ROLLOUT_AXES)
                accepted=r.gate.validate_pool(**dict(self.data,rollout=arrays))
                self.assertEqual(accepted['zero_families'],len(range(0,256,17)))
                records=[json.loads(line) for line in Path(report['journal']['path']).read_text().splitlines()]
                self.assertEqual(len(records),512)
                self.assertTrue(all(record['identity_native_endpoint_saved_from_checked_free'] for record in records))

    def test_fixed_test512_population_produces_all3072_cases(self):
        data=fixture('test')
        with tempfile.TemporaryDirectory() as temp:
            ctx=self.admitted(Path(temp),data);result,calls=self.execute(ctx,data)
            self.assertEqual(calls,3072);self.assertEqual(result['count'],512)
            self.assertEqual(len(result['rollouts']),3)

    def test_last_projection_corruption_rejected_before_any_model(self):
        with tempfile.TemporaryDirectory() as temp:
            ctx,pair=context(Path(temp),self.data,corrupt_final_family=True)
            with mock.patch.object(r.project,'load_context',return_value=ctx),mock.patch.object(r.runtime,'load_probe_model') as load:
                with self.assertRaises(ValueError):r.load_context(pair['path'],pair['sha256'])
                load.assert_not_called()
            self.assertFalse((Path(temp)/'rollouts').exists())

    def test_missing_pool_failed_population_and_wrong_self_source_block(self):
        for change in ('missing_pool','failed','wrong_source'):
            with tempfile.TemporaryDirectory() as temp:
                root=Path(temp);ctx,pair=context(root,self.data)
                binding=json.loads(Path(pair['path']).read_text());population=r.gate._json(binding['projection_population'])
                if change=='missing_pool':population['projections'].pop()
                elif change=='failed':population['status']='BLOCKED_COMPLETE_S2_PROJECTION_POPULATION'
                else:ctx['sources']['producers']['rollout']=ctx['binding']
                binding['projection_population']=json_file(root/'projection/report.json',population)
                pair=json_file(root/'rollout_input.json',binding)
                with mock.patch.object(r.project,'load_context',return_value=ctx),self.assertRaises(ValueError):
                    r.load_context(pair['path'],pair['sha256'])

    def test_input_status_extra_fields_or_changed_hash_block(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);_,pair=context(root,self.data)
            original=json.loads(Path(pair['path']).read_text())
            for binding in (dict(original,status='DRAFT'),dict(original,q_g={}),dict(original,output_root='relative')):
                changed=json_file(root/'bad.json',binding)
                with mock.patch.object(r.project,'load_context') as load,self.assertRaises(ValueError):
                    r.load_context(changed['path'],changed['sha256'])
                load.assert_not_called()
            Path(pair['path']).write_text('{}')
            with self.assertRaises(ValueError):r.load_context(pair['path'],pair['sha256'])

    def test_missing_or_false_native_checks_block_copy_materialization(self):
        handle,model_evidence=model(0,'decoded_teacher',self.data['recipient']['seeds'].tolist(),'calibration')
        replacement={s:self.data['projection']['replacements'][0,0,i] for i,s in enumerate(r.four.SOURCES)}
        result=native_result(self.data,handle,1000,replacement)
        for key in result['checks']:
            for absent in (True,False):
                changed=copy.deepcopy(result)
                if absent:changed['checks'].pop(key)
                else:changed['checks'][key]=False
                with self.subTest(key=key,absent=absent),self.assertRaises(ValueError):
                    r.validate_native_result(changed,model=model_evidence,cached=self.data['recipient'],objective_index=0,goal=0,replacements=replacement,zero=True)

    def test_corrupted_tokens_model_access_or_zero_trajectory_rejected(self):
        handle,evidence=model(0,'decoded_teacher',self.data['recipient']['seeds'].tolist(),'calibration')
        replacement={s:self.data['projection']['replacements'][0,0,i] for i,s in enumerate(r.four.SOURCES)}
        result=native_result(self.data,handle,1000,replacement)
        changes=(lambda x:x['arrays']['free'].__setitem__((4,191),99),lambda x:x['arrays']['donor'].__setitem__((4,191),99),
                 lambda x:x['arrays']['actual'].__setitem__((0,191),99),lambda x:x['arrays']['observed'].__setitem__((4,191),99),
                 lambda x:x.update(model_binding_sha256='wrong'),lambda x:x.update(access_receipt={'role':'test_response'}),
                 lambda x:x['arrays'].update(response=np.zeros((5,192),np.float32)))
        for change in changes:
            altered=copy.deepcopy(result);change(altered)
            with self.assertRaises(ValueError):r.validate_native_result(altered,model=evidence,cached=self.data['recipient'],
                objective_index=0,goal=0,replacements=replacement,zero=True)

    def test_failure_retains_partial_no_done_no_population_and_no_retry(self):
        with tempfile.TemporaryDirectory() as temp:
            ctx=self.admitted(Path(temp))
            with self.assertRaisesRegex(RuntimeError,'synthetic backend'):self.execute(ctx,self.data,fail_at=3)
            self.assertFalse((ctx['output']/'DONE').exists());self.assertFalse((ctx['output']/'report.json').exists())
            failure=json.loads((ctx['output']/'failure.json').read_text());self.assertEqual(failure['completed_cases'],2)
            self.assertEqual(failure['location']['goal_index'],2)
            self.assertTrue((ctx['output']/'pool_0/decoded_teacher/case_001.npz').exists())
            with mock.patch.object(r.runtime,'load_probe_model') as load,self.assertRaises(ValueError):r.produce(ctx)
            load.assert_not_called()

    def test_t1_substitution_fails_and_prior_intent_blocks_before_model(self):
        with tempfile.TemporaryDirectory() as temp:
            ctx=self.admitted(Path(temp))
            with self.assertRaises(ValueError):self.execute(ctx,self.data,wrong_model=True)
            self.assertFalse((ctx['output']/'pool_0/arrays.npz').exists())
        with tempfile.TemporaryDirectory() as temp:
            ctx=self.admitted(Path(temp));ctx['output'].with_name('rollouts.intent.json').write_text('keep')
            with mock.patch.object(r.runtime,'load_probe_model') as load,self.assertRaises(ValueError):r.produce(ctx)
            load.assert_not_called();self.assertFalse(ctx['output'].exists())

    def test_incomplete_preflight_and_population_count_cannot_execute(self):
        with tempfile.TemporaryDirectory() as temp:
            ctx=self.admitted(Path(temp))
            for change in ('status','count','pool'):
                invalid=dict(ctx)
                if change=='status':invalid['admission_status']='PARTIAL'
                elif change=='count':invalid['count']=255
                else:invalid['pools']=invalid['pools'][:-1]
                with mock.patch.object(r.runtime,'load_probe_model') as load,self.assertRaises(ValueError):r.produce(invalid)
                load.assert_not_called()


if __name__ == '__main__':unittest.main()
