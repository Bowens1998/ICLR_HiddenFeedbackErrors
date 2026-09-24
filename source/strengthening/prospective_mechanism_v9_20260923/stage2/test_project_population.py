"""Synthetic S2 producer checks; no actual data or default OSQP execution."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
import project_population as p
from test_accept_probe_population import fixture, json_file


class ProjectionProducerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data=fixture()

    def write_inputs(self, root, *, split='calibration'):
        data=self.data if split=='calibration' else fixture('test');n=p.gate._size(split)
        def save(name,arrays):
            path=root/name;np.savez_compressed(path,**arrays);return p.file_pair(path)
        head=save('synthetic_head.npz',data['head_A'])
        assignment=save('assignment.npz',dict(donor_assignment=data['assignment'],recipient_ids=data['recipient']['seeds'],donor_ids=data['donor']['seeds']))
        empty=json_file(root/'empty.json',{})
        binding=dict(status=p.INPUT_STATUS,split=split,protocol=empty,sources=empty,raw_acceptance=empty,
            caches=[dict(pool=pool,role=role,report=empty) for pool in range(3) for role in ('recipient','donor')],
            donor_assignment=assignment,output_root=str(root/'outputs'))
        bp=json_file(root/'binding.json',binding)
        pools=[dict(pool=pool,group=2*pool,recipient_cache=empty,donor_cache=empty,head_A_file=head,head_A=data['head_A'],
            predicted=data['recipient']['free'][:,:,0].copy(),actual=data['recipient']['observed'][:,0].copy(),
            donor=data['donor']['observed5'].copy()) for pool in range(3)]
        ctx=dict(binding=bp,input=binding,split=split,count=n,sources={},output=root/'outputs',assignment=data['assignment'].copy(),
            recipient_ids=data['recipient']['seeds'].copy(),donor_ids=data['donor']['seeds'].copy(),pools=pools,
            admission_status='ALL_SIX_COMPLETE_CACHES_THREE_HEADS_RAW_AND_ASSIGNMENT_AUTHENTICATED')
        return data,ctx,head

    def factory(self,data,events=None,fail_at=()):
        calls=[0]
        def run(predicted,guides,head,*,goal_index,donor_index):
            ordinal=calls[0];calls[0]+=1
            if events is not None:events.append(('project',ordinal))
            if ordinal in fail_at:
                raise p.four.FourFamilyFailure('synthetic unsolved direction',dict(stage='solver',member=['physical_labels','donor'],
                    completed_directions=3,residual=np.nan))
            self.assertEqual(donor_index,int(data['assignment'][goal_index]))
            for i,o in enumerate(p.four.OBJECTIVES):
                np.testing.assert_array_equal(predicted[o],data['recipient']['free'][i,goal_index,0])
            np.testing.assert_array_equal(guides['actual'],data['recipient']['observed'][goal_index,0])
            np.testing.assert_array_equal(guides['donor'],data['donor']['observed5'][donor_index])
            return (data['projection']['replacements'][goal_index].copy(),data['projection']['directions'][goal_index].copy(),
                    copy.deepcopy(data['families'][goal_index]))
        return run,calls

    def test_complete_synthetic_calibration_output_consumed_by_existing_gate(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);data,ctx,_=self.write_inputs(root);run,calls=self.factory(data)
            with mock.patch.object(p.four,'project_t0_four_family',side_effect=run):result=p.produce(ctx)
            self.assertEqual(calls[0],768);self.assertEqual(result['status'],p.POPULATION_STATUS)
            self.assertEqual(result['attempted_families'],768);self.assertEqual(result['failed_families'],0)
            self.assertEqual(result['zero_families'],3*len(range(0,256,17)))
            self.assertTrue((ctx['output']/'DONE').is_file())
            for row in result['projections']:
                report=p.gate._json(row['report']);arrays=p.gate._arrays(report['arrays'])
                self.assertEqual(report['status'],p.POOL_STATUS);self.assertEqual(report['axes'],p.gate.PROJECTION_AXES)
                self.assertEqual(report['source_sha256'],p.gate.sha(p.__file__))
                accepted=p.gate.validate_pool(**dict(data,projection=arrays,families=report['families']))
                self.assertEqual(accepted['directions'],1024)
                journal=[json.loads(x) for x in Path(report['journal']['path']).read_text().splitlines()]
                self.assertEqual([r['goal_index'] for r in journal],list(range(256)))
                np.testing.assert_array_equal(np.asarray(journal[1]['replacements'],np.float32),arrays['replacements'][1])

    def test_complete_synthetic_test_count_remains512(self):
        # Exercise full 1,536 fixed test families; the analytical producer is
        # injected, while actual dense independent feasibility checks still run.
        with tempfile.TemporaryDirectory() as temp:
            data,ctx,_=self.write_inputs(Path(temp),split='test');run,calls=self.factory(data)
            with mock.patch.object(p.four,'project_t0_four_family',side_effect=run):result=p.produce(ctx)
            self.assertEqual(calls[0],1536);self.assertEqual(result['count'],512)
            self.assertEqual(result['completed_families'],1536)

    def test_failure_retained_without_zero_substitution_or_population_reduction(self):
        with tempfile.TemporaryDirectory() as temp:
            data,ctx,_=self.write_inputs(Path(temp));run,calls=self.factory(data,fail_at=(1,514))
            with mock.patch.object(p.four,'project_t0_four_family',side_effect=run):
                with self.assertRaises(p.ProjectionPopulationBlocked):p.produce(ctx)
            result=json.loads((ctx['output']/'report.json').read_text())
            self.assertEqual(calls[0],768);self.assertEqual(result['failed_families'],2)
            self.assertEqual(result['completed_families'],766);self.assertEqual(result['attempted_families'],768)
            self.assertFalse((ctx['output']/'DONE').exists())
            for pool in (0,2):
                folder=ctx['output']/f'pool_{pool}';report=json.loads((folder/'report.json').read_text())
                self.assertEqual(report['status'],'BLOCKED_COMPLETE_S2_PROJECTION_POOL')
                self.assertNotIn('arrays',report);self.assertFalse((folder/'arrays.npz').exists());self.assertFalse((folder/'DONE').exists())
                rows=[json.loads(x) for x in (folder/'families.jsonl').read_text().splitlines()]
                self.assertEqual(len(rows),256)
                failed=[x for x in rows if x['status']=='FAILED_S2_PROJECTED_FAMILY']
                self.assertEqual(len(failed),1);self.assertNotIn('replacements',failed[0])
                self.assertEqual(failed[0]['detail']['residual'],{'nonfinite_diagnostic':'nan'})
            self.assertTrue((ctx['output']/'pool_1/DONE').is_file())

    def test_independent_family_rejection_becomes_recorded_technical_failure(self):
        with tempfile.TemporaryDirectory() as temp:
            data,ctx,_=self.write_inputs(Path(temp));factory,calls=self.factory(data)
            def corrupt(*args,**kwargs):
                replacement,direction,report=factory(*args,**kwargs)
                if calls[0]==1:report['effective_norm']=99.
                return replacement,direction,report
            with mock.patch.object(p.four,'project_t0_four_family',side_effect=corrupt):
                with self.assertRaises(p.ProjectionPopulationBlocked):p.produce(ctx)
            result=json.loads((ctx['output']/'report.json').read_text())
            self.assertEqual(result['failed_families'],1);self.assertEqual(calls[0],768)

    def test_no_output_or_retry_after_prior_intent(self):
        with tempfile.TemporaryDirectory() as temp:
            _,ctx,_=self.write_inputs(Path(temp));intent=ctx['output'].with_name('outputs.intent.json');intent.write_text('retained')
            with mock.patch.object(p.four,'project_t0_four_family') as compute:
                with self.assertRaises(ValueError):p.produce(ctx)
                compute.assert_not_called()
            self.assertEqual(intent.read_text(),'retained');self.assertFalse(ctx['output'].exists())

    def test_changed_binding_or_incomplete_admission_prevents_projection(self):
        with tempfile.TemporaryDirectory() as temp:
            _,ctx,_=self.write_inputs(Path(temp));Path(ctx['binding']['path']).write_text('{}')
            with mock.patch.object(p.four,'project_t0_four_family') as compute:
                with self.assertRaises(ValueError):p.produce(ctx)
                compute.assert_not_called()
            self.assertFalse(ctx['output'].exists())
            ctx['admission_status']='PARTIAL'
            with self.assertRaises(ValueError):p.produce(ctx)

    def test_complete_preflight_all_six_caches_before_compute_and_no_future_tokens_retained(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);data,ctx,head=self.write_inputs(root);events=[]
            sources=dict(heads_A=[dict(pool=i,group=2*i,file=head) for i in range(3)])
            parents=dict(calibration_recipient=ctx['recipient_ids'].tolist(),calibration_donor=ctx['donor_ids'].tolist())
            model=dict(spec={'head_A_sha256':head['sha256']})
            def cache(*args,**kw):
                events.append(('cache',kw['pool'],kw['donor']))
                return dict(models=[model] if kw['donor'] else [model,model]),data['donor' if kw['donor'] else 'recipient']
            with mock.patch.object(p,'configuration',return_value=('calibration',256,{'donor_assignment_seeds':{'calibration':123}},sources,ctx['output'])), \
                 mock.patch.object(p.gate,'_raw_admission',return_value=({},parents,{})),mock.patch.object(p.gate,'_cache',side_effect=cache), \
                 mock.patch.object(p.four,'project_t0_four_family') as compute:
                admitted=p.load_context(ctx['binding']['path'],ctx['binding']['sha256']);compute.assert_not_called()
            self.assertEqual(events,[('cache',pool,donor) for pool in range(3) for donor in (False,True)])
            for pool in admitted['pools']:
                self.assertEqual(pool['predicted'].shape,(2,256,192));self.assertEqual(pool['actual'].shape,(256,192))
                self.assertFalse(set(pool)&{'truth','free','reset','observed_history','response','q_g'})
            with mock.patch.object(p,'configuration',return_value=('calibration',256,{'donor_assignment_seeds':{'calibration':123}},sources,ctx['output'])) as config, \
                 mock.patch.object(p.gate,'_raw_admission',return_value=({},parents,{})),mock.patch.object(p.gate,'_cache',side_effect=cache):
                retained=p.load_context(ctx['binding']['path'],ctx['binding']['sha256'],require_unused_output=False,retain_caches=True)
                self.assertFalse(config.call_args.kwargs['require_unused_output'])
                self.assertEqual(len(events),12)
                self.assertIs(retained['pools'][0]['recipient_arrays'],data['recipient'])

    def test_late_last_cache_failure_blocks_all_projection(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);data,ctx,head=self.write_inputs(root)
            sources=dict(heads_A=[dict(pool=i,group=2*i,file=head) for i in range(3)])
            parents=dict(calibration_recipient=ctx['recipient_ids'].tolist(),calibration_donor=ctx['donor_ids'].tolist())
            model=dict(spec={'head_A_sha256':head['sha256']})
            def cache(*args,**kw):
                if kw['pool']==2 and kw['donor']:raise ValueError('synthetic changed final cache')
                return dict(models=[model] if kw['donor'] else [model,model]),data['donor' if kw['donor'] else 'recipient']
            with mock.patch.object(p,'configuration',return_value=('calibration',256,{'donor_assignment_seeds':{'calibration':123}},sources,ctx['output'])), \
                 mock.patch.object(p.gate,'_raw_admission',return_value=({},parents,{})),mock.patch.object(p.gate,'_cache',side_effect=cache), \
                 mock.patch.object(p.four,'project_t0_four_family') as compute:
                with self.assertRaisesRegex(ValueError,'final cache'):p.load_context(ctx['binding']['path'],ctx['binding']['sha256'])
                compute.assert_not_called()
            self.assertFalse(ctx['output'].exists())

    def test_complete_cache_shapes_roles_and_finite_values_before_qp(self):
        p.validate_cache_values(self.data['recipient'],self.data['donor'],split='calibration')
        bad=dict(self.data['recipient']);bad['free']=bad['free'][:,:-1]
        with self.assertRaises(ValueError):p.validate_cache_values(bad,self.data['donor'],split='calibration')
        bad=dict(self.data['recipient']);bad['truth']=bad['truth'].copy();bad['truth'][-1,-1,-1]=np.nan
        with self.assertRaises(ValueError):p.validate_cache_values(bad,self.data['donor'],split='calibration')
        bad=dict(self.data['donor'],predictions=self.data['recipient']['free'])
        with self.assertRaises(ValueError):p.validate_cache_values(self.data['recipient'],bad,split='calibration')
        bad=dict(self.data['recipient']);bad['observed_history']=bad['observed_history'].copy();bad['observed_history'][1,255,0,191]+=1
        with self.assertRaises(ValueError):p.validate_cache_values(bad,self.data['donor'],split='calibration')

    def test_protocol_freeze_output_namespace_and_top_keys_are_explicit(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);_,ctx,_=self.write_inputs(root);b=ctx['input']
            with self.assertRaises(ValueError):p.configuration(dict(b,q_g={}))
            with self.assertRaises(ValueError):p.configuration(dict(b,output_root='relative'))
            b['protocol']=json_file(root/'protocol.json',{'status':'DRAFT'})
            with self.assertRaisesRegex(ValueError,'freeze'):p.configuration(b)
            (root/'outputs').mkdir()
            with self.assertRaisesRegex(ValueError,'Existing'):p.configuration(b)

    def test_complete_source_contract_checks_and_readonly_reuse_flag(self):
        # Real local source bytes, fake future contracts in a temporary directory;
        # no checkpoint, observed cache or actual experiment input is accessed.
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);_,ctx,head=self.write_inputs(root);b=ctx['input']
            protocol=json_file(root/'protocol.json',dict(status='S2_SCIENTIFIC_PROTOCOL_FROZEN',probe_population=dict(
                axes=p.gate.ROLLOUT_AXES,native_population=300,history_tokens=3,prefix_steps=10,insertion_relative_step=5,
                shrink_factors=list(p.four.SHRINK_FACTORS),tolerance=p.four.TOLERANCE,receipt_atol=p.four.RECEIPT_ATOL,
                donor_assignment_seeds={'calibration':123,'test':456})))
            raw=json_file(root/'raw.json',dict(status='S2_RAW_INPUT_SOURCES_FROZEN',protocol_sha256=protocol['sha256'],files={}))
            cache=json_file(root/'cache.json',dict(status='S2_LATENT_CACHE_SOURCES_FROZEN',protocol_sha256=protocol['sha256'],
                files={},raw_input_sources=raw,raw_population_acceptance=b['raw_acceptance']))
            future=root/'synthetic_rollout.py';future.write_text('# synthetic unexecuted source fixture\n')
            paths=[Path(p.__file__),Path(p.gate.__file__),Path(p.four.__file__),p.HERE/'model_runtime.py',
                p.HERE/'cache_population.py',p.HERE/'raw_inputs.py',p.HERE/'accept_raw_inputs.py',*p.gate.model_runtime.METADATA.values(),
                p.PHASE/'scripts/s1_projection.py',p.PHASE/'scripts/s1_common.py',p.PHASE/'scripts/s1_readout.py',
                p.ROOT/'strengthening/adapters/verifier.py',future]
            source=dict(status='S2_PROBE_POPULATION_SOURCES_FROZEN',protocol_sha256=protocol['sha256'],
                producers={'projection':p.file_pair(p.__file__),'rollout':p.file_pair(future)},files={str(x):p.gate.sha(x) for x in paths},
                raw_sources=raw,cache_sources=cache,heads_A=[dict(pool=i,group=2*i,file=head) for i in range(3)])
            b['protocol']=protocol;b['sources']=json_file(root/'sources.json',source)
            self.assertEqual(p.configuration(b)[1],256)
            (root/'outputs').mkdir()
            with self.assertRaisesRegex(ValueError,'Existing'):p.configuration(b)
            self.assertEqual(p.configuration(b,require_unused_output=False)[1],256)
            altered=copy.deepcopy(source);del altered['files'][str(Path(p.gate.__file__))]
            b['sources']=json_file(root/'bad_sources.json',altered)
            with self.assertRaisesRegex(ValueError,'Missing'):p.configuration(b,require_unused_output=False)
            altered=copy.deepcopy(source);altered['producers']['projection']=p.file_pair(future)
            b['sources']=json_file(root/'bad_sources.json',altered)
            with self.assertRaisesRegex(ValueError,'identity'):p.configuration(b,require_unused_output=False)


if __name__=='__main__':unittest.main()
