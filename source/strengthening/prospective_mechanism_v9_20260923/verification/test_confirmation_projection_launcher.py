"""Synthetic-only launcher contract tests; never SSH, arrays or real manifests."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import shlex
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

OPERATIONS = Path(__file__).resolve().parents[1] / 'operations'
sys.path.insert(0, str(OPERATIONS))
import submit_confirmation_projection_rollout as launcher


class Fixture:
    def __init__(self, root):
        self.phase = Path(root); self.rows = []
        self.kernels = {n: chr(97 + i) * 64 for i, n in enumerate(
            ['cache_inputs.py', 'project_s1.py', 'freeze_qp_shards.py', 'run_rollout.py'])}
        self.source = self.phase / 'sources.json'
        self.write(self.source, {'files': {launcher.REL + '/scripts/' + k: v for k, v in self.kernels.items()}})
        self.args = SimpleNamespace(sources=self.source, sources_sha256=launcher.sha(self.source),
            cache_bindings=self.phase / 'cache-bindings.json',
            input_metadata_acceptance=self.phase / 'input-metadata.json')
        self.bindings = {'protocol': dict(path='/synthetic/protocol.json', sha256=launcher.PROTOCOL_SHA),
                         'sources': dict(path='/synthetic/sources.json', sha256=self.args.sources_sha256)}
        for role in ['recipient', 'donor']:
            for g in [0, 1]:
                for s in range(4):
                    relative = f'reports/cache/{role}/group_{g}_stream_{s}/report.json'
                    remote = launcher.REMOTE + f'/artifacts/confirmation/cache/{role}/group_{g}_stream_{s}/report.json'
                    row = dict(role=role, group=g, stream=s, report=remote, local_report=relative)
                    report = dict(status='PASS_S1_OBSERVED_AND_FREE_CACHE', protocol_sha256=launcher.PROTOCOL_SHA,
                        sources_sha256=self.args.sources_sha256, kind='confirmation', role=role, group=g, stream=s,
                        count=256, cases=list(range(256)), frozen_tensors_unchanged=True,
                        donor_model_predictions_generated=False, source_sha256=self.kernels['cache_inputs.py'],
                        objectives=['decoded_teacher', 'physical_labels'] if role == 'recipient' else ['decoded_teacher'],
                        native_endpoint_and_identity_replacement_exact=role == 'recipient',
                        arrays=dict(path=str(Path(remote).with_name('data.npz')), sha256='e' * 64),
                        content_lineage=dict(path=launcher.REMOTE + '/artifacts/confirmation/content_lineage.json', sha256='f' * 64))
                    self.write(self.phase / relative, report); row['report_sha256'] = launcher.sha(self.phase / relative)
                    self.rows.append(row)
        self.refresh()

    @staticmethod
    def write(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, sort_keys=True) + '\n')

    def refresh(self):
        manifest = dict(status='COMPLETE_S1_CONFIRMATION_CACHE_BINDINGS', protocol_sha256=launcher.PROTOCOL_SHA,
                        sources_sha256=self.args.sources_sha256, reports=self.rows)
        self.write(self.args.cache_bindings, manifest)
        self.args.cache_bindings_sha256 = launcher.sha(self.args.cache_bindings)
        audit = dict(status='PASS_COMPLETE_S1_CONFIRMATION_INPUT_METADATA_ACCEPTANCE',
            protocol_sha256=launcher.PROTOCOL_SHA, sources_sha256=self.args.sources_sha256,
            cache_bindings_sha256=self.args.cache_bindings_sha256,
            exact_cache_report_count=16, all_16_cache_reports_verified=True,
            exact_cache_coverage=[{k: row[k] for k in ('role', 'group', 'stream')} for row in self.rows])
        self.write(self.args.input_metadata_acceptance, audit)
        self.args.input_metadata_acceptance_sha256 = launcher.sha(self.args.input_metadata_acceptance)

    def base(self):
        return dict(bindings=copy.deepcopy(self.bindings), source_root=launcher.SOURCE, remote_root=launcher.REMOTE,
                    relative_phase=launcher.REL, gate_producer=launcher.input_helper.GATE_PRODUCER)

    def plan(self):
        # The separately tested original source/qualification gate is a boundary here.
        with mock.patch.object(launcher, 'PHASE', self.phase), \
             mock.patch.object(launcher.input_helper, 'load_spec', return_value=self.base()), \
             mock.patch.object(launcher.subprocess, 'run', side_effect=AssertionError('No process permitted during dry-run')):
            return launcher.build_plan(self.args)

    def mutate_report(self, key, value, index=0):
        row = self.rows[index]; path = self.phase / row['local_report']
        report = json.loads(path.read_text()); report[key] = value
        self.write(path, report); row['report_sha256'] = launcher.sha(path); self.refresh()


class ConfirmationProjectionLauncherTests(unittest.TestCase):
    def test_exact_plan_resources_routes_and_self_contained_workers(self):
        with tempfile.TemporaryDirectory() as d:
            f = Fixture(d); plan = f.plan()
            self.assertEqual(len(plan['spec']['routes']), 8)
            self.assertEqual([(r['group'], r['stream']) for r in plan['spec']['routes']], [(g,s) for g in [0,1] for s in range(4)])
            self.assertTrue(all(r['shard_dir'].endswith('/shard_0_256') for r in plan['spec']['routes']))
            stages = plan['stages']
            self.assertEqual([(r['array'],r['cpus'],r['memory'],r['time']) for r in stages],
                [('0-7%8',2,'8G','01:00:00'),('0-7%8',2,'8G','00:30:00'),('0-7%4',4,'16G','01:00:00')])
            self.assertEqual([s['dependency'] for s in stages], [None,'afterok:whole_projection_array','afterok:whole_acceptance_array'])
            for stage in stages:
                body = stage['body']; self.assertEqual(hashlib.sha256(body.encode()).hexdigest(),stage['body_sha256'])
                code = body.split("<<'S1_CONFIRMATION_PROJECTION_WORKER'\n",1)[1].rsplit('S1_CONFIRMATION_PROJECTION_WORKER\n',1)[0]
                prefix = code.rsplit('\nruntime_worker(SPEC, ',1)[0]
                namespace = {}; exec(compile(prefix,'<synthetic-worker>','exec'),namespace)
                self.assertTrue(callable(namespace['validate_confirmation']))
                self.assertTrue(callable(namespace['runtime_worker']))

    def test_complete_roster_and_metadata_gates(self):
        changes = {
            'missing_report': lambda f: f.rows.pop(),
            'duplicate_report': lambda f: f.rows.__setitem__(15,copy.deepcopy(f.rows[0])),
            'wrong_kind_path': lambda f: f.rows[0].__setitem__('report',f.rows[0]['report'].replace('/confirmation/','/development/')),
            'escaped_local_path': lambda f: f.rows[0].__setitem__('local_report','../../outside/report.json'),
        }
        for name, change in changes.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as d:
                f=Fixture(d); change(f); f.refresh()
                with self.assertRaises(ValueError): f.plan()
        for key,value in [('all_16_cache_reports_verified',False),('exact_cache_report_count',15),
                          ('cache_bindings_sha256','0'*64),('sources_sha256','0'*64),
                          ('status','PENDING')]:
            with self.subTest(key=key), tempfile.TemporaryDirectory() as d:
                f=Fixture(d); path=f.args.input_metadata_acceptance; audit=json.loads(path.read_text());audit[key]=value
                f.write(path,audit);f.args.input_metadata_acceptance_sha256=launcher.sha(path)
                with self.assertRaises(ValueError):f.plan()

    def test_cache_semantics_even_after_rehashing(self):
        changes=[('count',64),('cases',list(range(255))),('kind','development'),
            ('source_sha256','0'*64),('frozen_tensors_unchanged',False),('donor_model_predictions_generated',True),
            ('native_endpoint_and_identity_replacement_exact',False),('objectives',['physical_labels','decoded_teacher']),
            ('arrays',dict(path='/wrong/data.npz',sha256='a'*64)),
            ('content_lineage',dict(path=launcher.REMOTE+'/artifacts/development/content_lineage.json',sha256='a'*64))]
        for key,value in changes:
            with self.subTest(key=key),tempfile.TemporaryDirectory() as d:
                f=Fixture(d);f.mutate_report(key,value)
                with self.assertRaises(ValueError):f.plan()

    def test_missing_or_changed_actual_hash_blocks_plan(self):
        with tempfile.TemporaryDirectory() as d:
            f=Fixture(d); f.args.cache_bindings_sha256='0'*64
            with self.assertRaises(ValueError):f.plan()
        with tempfile.TemporaryDirectory() as d:
            f=Fixture(d);f.args.input_metadata_acceptance.unlink()
            with self.assertRaises(FileNotFoundError):f.plan()

    def test_allocated_worker_keeps_exact_frozen_cli_and_whole_shard(self):
        with tempfile.TemporaryDirectory() as d:
            f=Fixture(d);plan=f.plan();spec=copy.deepcopy(plan['spec']);row=spec['routes'][0]
            work=Path(d)/'allocated';spec['remote_root']=str(work);spec['source_root']=str(work/'source')
            row.update(shard_dir=str(work/'projection/group_0_stream_0/shard_0_256'),
                acceptance_dir=str(work/'projection_acceptance/group_0_stream_0'),
                rollout_dir=str(work/'rollout/recipient/group_0_stream_0'))
            bindings=spec['bindings']
            for key in ['selected-lock','qualification','a-bindings']:
                bindings[key]=dict(path='/synthetic/'+key+'.json',sha256='9'*64)
            documents={v['path']:{} for v in bindings.values() if isinstance(v,dict) and 'path' in v}
            documents[bindings['sources']['path']]={'files':{}}
            documents[bindings['cache-bindings']['path']]=json.loads(f.args.cache_bindings.read_text())
            documents[bindings['input-metadata-acceptance']['path']]=json.loads(f.args.input_metadata_acceptance.read_text())
            for r in f.rows:
                documents[r['report']]=json.loads((f.phase/r['local_report']).read_text())
            real_checked=launcher.checked_json
            def read(path,digest):
                return copy.deepcopy(documents[str(path)]) if str(path) in documents else real_checked(path,digest)
            identity=dict(protocol_sha256=launcher.PROTOCOL_SHA,kind='confirmation',group=0,stream=0,count=256,
                selected_lock_sha256=bindings['selected-lock']['sha256'],qualification_sha256=bindings['qualification']['sha256'],
                recipient_cache_sha256=row['recipient']['report_sha256'],donor_cache_sha256=row['donor']['report_sha256'])
            calls=[]
            def kernel(command,**kwargs):
                calls.append(command);name=Path(command[1]).name
                pairs=dict(zip(command[2::2],command[3::2]))
                self.assertNotIn('score',name);self.assertNotIn('readout',name)
                if name=='project_s1.py':
                    self.assertEqual((pairs['--kind'],pairs['--start'],pairs['--stop']),('confirmation','0','256'))
                    self.assertEqual(pairs['--recipient-cache-sha256'],row['recipient']['report_sha256'])
                    self.assertEqual(pairs['--donor-cache-sha256'],row['donor']['report_sha256'])
                    report=dict(identity,status='ACCEPTED_COMPLETE_SHARD',start=0,stop=256,
                        a_bindings_sha256=bindings['a-bindings']['sha256'],source_sha256=f.kernels[name],
                        arrays=dict(path=str(Path(row['shard_dir'])/'projections.npz'),sha256='a'*64))
                    f.write(Path(row['shard_dir'])/'report.json',report)
                elif name=='freeze_qp_shards.py':
                    b=json.loads(Path(pairs['--bindings']).read_text())
                    self.assertEqual([(r['start'],r['stop']) for r in b['shards']],[(0,256)])
                    f.write(Path(pairs['--output']),dict(identity,status='ALL_S1_QP_SHARDS_ACCEPTED',
                        all_goals_covered=True,source_sha256=f.kernels[name],bindings_sha256=launcher.sha(pairs['--bindings']),
                        shards=b['shards']))
                elif name=='run_rollout.py':
                    self.assertEqual(pairs['--kind'],'confirmation')
                    self.assertEqual(pairs['--sources-sha256'],f.args.sources_sha256)
                    self.assertEqual(pairs['--qp-lock-sha256'],launcher.sha(Path(row['acceptance_dir'])/'QP_LOCK.json'))
                    f.write(Path(row['rollout_dir'])/'report.json',dict(identity,status='PASS_S1_COMPLETE_ACCEPTED_ROLLOUT'))
                else:
                    self.fail('Unexpected command '+name)
                return SimpleNamespace(returncode=0)
            with mock.patch.object(launcher,'checked_json',side_effect=read), \
                 mock.patch.object(launcher,'validate_confirmation'), \
                 mock.patch.dict('os.environ',{'SLURM_ARRAY_TASK_ID':'0','SLURM_JOB_ID':'synthetic'}), \
                 mock.patch.object(launcher.subprocess,'run',side_effect=kernel):
                for stage in ['projection','acceptance','rollout']:
                    launcher.runtime_worker(spec,stage)
                with self.assertRaises(FileExistsError):launcher.runtime_worker(spec,'rollout')
            self.assertEqual([Path(c[1]).name for c in calls],['project_s1.py','freeze_qp_shards.py','run_rollout.py'])

    def test_submission_dependencies_and_durable_no_retry(self):
        with tempfile.TemporaryDirectory() as d:
            phase=Path(d);f=Fixture(d);plan=f.plan();calls=[]
            def scheduler(argv,**kw):
                self.assertTrue((phase/'manifests/intent_confirmation_projection_rollout_dag.json').is_file())
                name=plan['stages'][len(calls)]['name']
                self.assertTrue((phase/f'manifests/intent_confirmation_projection_rollout_{name}.json').is_file())
                calls.append((argv,kw));return SimpleNamespace(stdout=str(100+len(calls)).encode(),stderr=b'')
            with mock.patch.object(launcher,'PHASE',phase),mock.patch.object(launcher.subprocess,'run',side_effect=scheduler):
                result=launcher.submit(plan,'a'*64)
                with self.assertRaises(ValueError):launcher.submit(plan,'a'*64)
            self.assertEqual(result['status'],'SUBMITTED');self.assertEqual(len(calls),3)
            argv=[shlex.split(x[0][-1]) for x in calls]
            self.assertFalse(any(v.startswith('--dependency=') for v in argv[0]))
            self.assertIn('--dependency=afterok:101',argv[1]);self.assertIn('--dependency=afterok:102',argv[2])
            self.assertIn('--gres=gpu:rtx_pro_6000:1',argv[2])

    def test_ambiguous_reply_stops_after_intent(self):
        with tempfile.TemporaryDirectory() as d:
            f=Fixture(d);plan=f.plan()
            with mock.patch.object(launcher,'PHASE',Path(d)),mock.patch.object(launcher.subprocess,'run',return_value=SimpleNamespace(stdout=b'uncertain',stderr=b'')) as run:
                with self.assertRaises(ValueError):launcher.submit(plan,'a'*64)
                with self.assertRaises(ValueError):launcher.submit(plan,'a'*64)
                self.assertEqual(run.call_count,1)

    def test_reviewed_plan_hash_required_before_submission(self):
        with tempfile.TemporaryDirectory() as d:
            f=Fixture(d);plan=f.plan()
            argv=['launcher']
            for name in ['sources','development-gate','development-metadata','cache-bindings','input-metadata-acceptance']:
                argv += ['--'+name,'unused.json','--'+name+'-sha256','a'*64]
            argv += ['--sources-remote','/unused','--submit','--reviewed-plan-sha256','b'*64]
            with mock.patch.object(sys,'argv',argv),mock.patch.object(launcher,'build_plan',return_value=plan),mock.patch.object(launcher,'submit') as submit:
                with self.assertRaises(ValueError):launcher.main()
                submit.assert_not_called()


if __name__=='__main__':
    unittest.main()
