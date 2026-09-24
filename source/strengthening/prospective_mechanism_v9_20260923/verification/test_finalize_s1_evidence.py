"""Synthetic evidence-finalizer guards; no model, real arrays, SSH or scheduler."""
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

SOURCE = Path(__file__).resolve().parents[1] / 'operations/finalize_s1_evidence.py'
SPEC = importlib.util.spec_from_file_location('s1_evidence_finalizer_under_test', SOURCE)
finalizer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(finalizer)
PROTOCOL_SHA = '5ce812fe7259bb7489bef214d733d2f370d6eb7650db0f3b7a9d1860f3e41fe3'
NAMES = ['decoded_teacher/U_AC', 'decoded_teacher/S_AC', 'physical_labels/U_AC', 'physical_labels/S_AC',
         'decoded_teacher/U_A_minus_U_AC', 'decoded_teacher/insertion_displacement_suppression',
         'physical_labels/U_A_minus_U_AC', 'physical_labels/insertion_displacement_suppression']


def statistics():
    intervals = [(1., 3.), (-3., -1.), (-1., 1.), (0., 2.), (-2., 0.), (2., 4.), (-4., -2.), (-2., 2.)]
    rows, independent = [], []
    values = []
    for j, (name, ci) in enumerate(zip(NAMES, intervals)):
        mean = sum(ci) / 2
        rows.append(dict(id=name, estimate=mean, ci=list(ci), coverage=.9875, positive_support=ci[0] > 0, recipient_count=256))
        independent.append(dict(id=name, family='primary' if j < 4 else 'secondary', nominal_coverage=.9875,
            mean=mean, lower=ci[0], upper=ci[1], classification='positive' if ci[0] > 0 else 'negative' if ci[1] < 0 else 'unresolved'))
        values.append(mean + (np.arange(256, dtype=np.float64) - 127.5) * .001 * (j + 1))
    return dict(primary=rows[:4], secondary=rows[4:]), dict(contrasts=independent), np.column_stack(values)


class Fixture:
    """Temporary invented metadata with an explicit mocked protocol-hash boundary."""
    def __init__(self, root):
        self.root = Path(root); self.output = self.root / 'result'; self.records = {}; self.calls = []
        self.protocol_path = self.root / 'protocol.json'
        self.phase = self.root / 'source/strengthening/prospective_mechanism_v9_20260923'
        scripts = ['scripts/accept_s1_population.py', 'scripts/score_reserved_readouts.py',
                   'scripts/summarize_s1.py', 'verification/independent_stage1_reference.py']
        files = {}
        for script in scripts:
            f = self.phase / script; f.parent.mkdir(parents=True, exist_ok=True); f.write_text('# Synthetic inert source identity only\n')
            files[str(f.relative_to(self.root / 'source'))] = finalizer.sha(f)
        self.cfg = dict(study_id='synthetic-only', remote_output=str(self.root / 'remote'), groups=[0, 1],
            objectives=['decoded_teacher', 'physical_labels'], scores=dict(constraints=['A', 'AC'], streams=[0, 1, 2, 3],
            branches=['free', 'actual', 'donor', 'reset'], horizons_primitive=[5, 10, 15, 20, 25], pose_order=['a','b','x','y','c','d']),
            statistics=dict(primary=[dict(id=n) for n in NAMES[:4]], secondary=[dict(id=n) for n in NAMES[4:]]))
        self.save(self.protocol_path, self.cfg); self.records['protocol'] = dict(path=str(self.protocol_path), sha256=PROTOCOL_SHA)
        sources = dict(status='S1_INPUT_SOURCES_FROZEN', authorized_kinds=['confirmation'], protocol_sha256=PROTOCOL_SHA,
                       source_root=str(self.root / 'source'), files=files)
        self.put('sources', sources)
        for key in ['selected-lock','qualification','a-bindings','head-implementation']: self.put(key, {'synthetic':key})
        self.cache_rows = [dict(role=role,group=g,stream=s,report=f'/synthetic/cache/{role}/{g}/{s}/report.json',report_sha256='e'*64)
                           for role in ['recipient','donor'] for g in [0,1] for s in range(4)]
        self.cache = dict(status='COMPLETE_S1_CONFIRMATION_CACHE_BINDINGS', protocol_sha256=PROTOCOL_SHA,
                          sources_sha256=self.records['sources']['sha256'], reports=self.cache_rows)
        self.put('cache-bindings', self.cache)
        self.meta = dict(status='PASS_COMPLETE_S1_CONFIRMATION_INPUT_METADATA_ACCEPTANCE', protocol_sha256=PROTOCOL_SHA,
            sources_sha256=self.records['sources']['sha256'], cache_bindings_sha256=self.records['cache-bindings']['sha256'],
            exact_cache_report_count=16, all_16_cache_reports_verified=True,
            exact_cache_coverage=[{k:r[k] for k in ('role','group','stream')} for r in self.cache_rows])
        self.put('cache-metadata', self.meta)
        self.rollout_paths = []
        for g in [0,1]:
            for s in range(4):
                r = next(r for r in self.cache_rows if (r['role'],r['group'],r['stream']) == ('recipient',g,s))
                path = self.root / f'remote/artifacts/confirmation/rollout/recipient/group_{g}_stream_{s}/report.json'
                self.save(path,dict(status='PASS_S1_COMPLETE_ACCEPTED_ROLLOUT',group=g,stream=s,count=256,
                    recipient_cache=dict(path=r['report'],sha256=r['report_sha256'])))
                (path.parent/'DONE').write_text('synthetic\n');self.rollout_paths.append(path)
        self.acceptance_status = 'FULL_S1_CONFIRMATION_ACCEPTED_BEFORE_D_SCORING'
        self.forward_count = 163840; self.forward_status = 'PASS_COMPLETE_RESERVED_FORWARD'
        self.production_change = None; self.independent_change = None

    @staticmethod
    def save(path, obj):
        path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(obj)+'\n')

    def put(self, key, obj):
        path=self.root/(key+'.json');self.save(path,obj);self.records[key]=dict(path=str(path),sha256=finalizer.sha(path))

    def kernel(self, command, **kw):
        name=Path(command[1]).name;self.calls.append(name);args=dict(zip(command[2::2],command[3::2]));out=Path(args['--output'])
        self.assert_binding(args)
        if name=='accept_s1_population.py':
            bp=Path(args['--binding']);assert finalizer.sha(bp)==args['--binding-sha256']
            binding=json.loads(bp.read_text());assert len(binding['populations'])==8
            assert [(x['group'],x['stream']) for x in binding['populations']]==[(g,s) for g in [0,1] for s in range(4)]
            assert (self.output/'execution_inputs.json').exists()
            self.save(out,dict(status=self.acceptance_status))
        elif name=='score_reserved_readouts.py':
            assert self.calls == ['accept_s1_population.py','score_reserved_readouts.py']
            assert json.loads(Path(args['--acceptance']).read_text())['status']=='FULL_S1_CONFIRMATION_ACCEPTED_BEFORE_D_SCORING'
            assert (self.output/'population_binding.json').exists()
            self.save(out/'production_binding.json',{'synthetic':True});self.save(out/'independent_binding.json',{'synthetic':True})
        elif name=='summarize_s1.py':
            prod,_,v=statistics();prod['status']='COMPLETE_PRODUCTION_SUMMARY_PENDING_INDEPENDENT_VERIFICATION'
            prod.update(protocol_sha256=PROTOCOL_SHA,acceptance_sha256=args['--acceptance-sha256'],binding_sha256=args['--binding-sha256'])
            if self.production_change:self.production_change(prod)
            self.save(out/'summary.json',prod);np.savez(out/'goal_contrasts.npz',**dict(zip(NAMES,v.T)))
        elif name=='independent_stage1_reference.py':
            _,ind,v=statistics();out.mkdir(parents=True,exist_ok=False)
            np.savez(out/'independent_vectors_and_bootstrap.npz',goal_vectors=v,contrast_ids=np.asarray(NAMES))
            ind.update(status='PASS_INDEPENDENT_STAGE1_ARITHMETIC',forward=dict(status=self.forward_status,token_count=self.forward_count),
                       vectors_sha256=finalizer.sha(out/'independent_vectors_and_bootstrap.npz'),design_sha256=PROTOCOL_SHA,
                       acceptance_sha256=finalizer.sha(self.output/'acceptance.json'),binding_sha256=args['--binding-sha256'],
                       reference_source_sha256=finalizer.sha(self.phase/'verification/independent_stage1_reference.py'))
            if self.independent_change:self.independent_change(ind)
            self.save(out/'report.json',ind)
        else:raise AssertionError('Unexpected process '+name)

    @staticmethod
    def assert_binding(args):
        for flag,path in args.items():
            if flag.endswith('-sha256') or flag=='--output':continue
            if flag=='--protocol': assert args['--protocol-sha256']==PROTOCOL_SHA
            else: assert finalizer.sha(path)==args[flag+'-sha256']

    def run(self):
        argv=['finalizer']
        for key,r in self.records.items():argv+=['--'+key,r['path'],'--'+key+'-sha256',r['sha256']]
        argv+=['--output',str(self.output)]
        original_sha=finalizer.sha
        def synthetic_protocol_hash(path):
            return PROTOCOL_SHA if Path(path)==self.protocol_path else original_sha(path)
        with mock.patch.object(sys,'argv',argv),mock.patch.object(finalizer,'sha',side_effect=synthetic_protocol_hash), \
             mock.patch.object(finalizer.subprocess,'run',side_effect=self.kernel), \
             mock.patch.dict(os.environ,{'SLURM_JOB_ID':'synthetic-no-scheduler'}):finalizer.main()


class StatisticsComparisonTests(unittest.TestCase):
    def test_all_eight_signed_and_boundary_classifications_retained(self):
        p,i,v=statistics();r=finalizer.compare_statistics(p,i,v,v.copy(),NAMES)
        self.assertTrue(r['all_eight_classifications_equal']);self.assertEqual((r['atol'],r['rtol']),(1e-9,1e-12))
        self.assertEqual({r['classification'] for r in i['contrasts']},{'positive','negative','unresolved'})

    def test_changed_vectors_shape_finite_roster_or_dtype_rejected(self):
        for kind in ['shape','nan','difference','dtype','order','family','coverage','count','classification','positive','interval']:
            with self.subTest(kind=kind):
                p,i,v=statistics();w=v.copy();names=NAMES.copy()
                if kind=='shape':w=w[:-1]
                elif kind=='nan':w[0,0]=np.nan
                elif kind=='difference':w[0,0]+=.01
                elif kind=='dtype':v=v.astype(np.float32);w=v.copy()
                elif kind=='order':names=names[::-1]
                elif kind=='family':i['contrasts'][4]['family']='primary'
                elif kind=='coverage':p['primary'][0]['coverage']=.95
                elif kind=='count':p['primary'][0]['recipient_count']=255
                elif kind=='classification':i['contrasts'][1]['classification']='positive'
                elif kind=='positive':p['primary'][1]['positive_support']=True
                elif kind=='interval':p['primary'][0]['ci']=[3.,1.]
                with self.assertRaises((ValueError,AssertionError)):finalizer.compare_statistics(p,i,v,w,names)

    def test_tolerance_fixed_and_interval_disagreement_rejected(self):
        p,i,v=statistics();w=v+1e-10
        finalizer.compare_statistics(p,i,v,w,NAMES)
        i['contrasts'][0]['upper']+=.01
        with self.assertRaises(AssertionError):finalizer.compare_statistics(p,i,v,v,NAMES)


class PipelineTests(unittest.TestCase):
    def test_exact_order_complete_binding_no_overwrite_and_negative_results_retained(self):
        with tempfile.TemporaryDirectory() as d:
            f=Fixture(d);f.run()
            self.assertEqual(f.calls,['accept_s1_population.py','score_reserved_readouts.py','summarize_s1.py','independent_stage1_reference.py'])
            report=json.loads((f.output/'report.json').read_text())
            self.assertEqual(report['status'],'COMPLETE_S1_SAVED_TOKEN_AND_STATISTICS_VERIFIED')
            self.assertEqual(len(report['primary'])+len(report['secondary']),8)
            self.assertFalse(report['raw_image_world_model_regeneration_complete'])
            self.assertEqual(report['independent_forward']['token_count'],163840)
            self.assertTrue((f.output/'DONE').is_file())
            with self.assertRaises(FileExistsError):f.run()
            self.assertEqual(len(f.calls),4)

    def test_failed_admission_never_opens_D(self):
        with tempfile.TemporaryDirectory() as d:
            f=Fixture(d);f.acceptance_status='BLOCKED'
            with self.assertRaises(ValueError):f.run()
            self.assertEqual(f.calls,['accept_s1_population.py']);self.assertFalse((f.output/'DONE').exists())

    def test_rollout_and_cache_coverage_fail_before_any_kernel(self):
        for kind in ['missing_rollout','missing_DONE','count','recipient','cache_missing','cache_duplicate','meta_count','meta_verified','meta_roster']:
            with self.subTest(kind=kind),tempfile.TemporaryDirectory() as d:
                f=Fixture(d)
                if kind=='missing_rollout':f.rollout_paths[-1].unlink()
                elif kind=='missing_DONE':f.rollout_paths[-1].with_name('DONE').unlink()
                elif kind in ['count','recipient']:
                    path=f.rollout_paths[-1];doc=json.loads(path.read_text());doc['count' if kind=='count' else 'recipient_cache']=255 if kind=='count' else {};f.save(path,doc)
                elif kind.startswith('cache_'):
                    if kind=='cache_missing':f.cache['reports'].pop()
                    else:f.cache['reports'][-1]=copy.deepcopy(f.cache['reports'][0])
                    f.put('cache-bindings',f.cache);f.meta['cache_bindings_sha256']=f.records['cache-bindings']['sha256'];f.put('cache-metadata',f.meta)
                else:
                    f.meta[{'meta_count':'exact_cache_report_count','meta_verified':'all_16_cache_reports_verified','meta_roster':'exact_cache_coverage'}[kind]]=15 if kind=='meta_count' else False if kind=='meta_verified' else f.meta['exact_cache_coverage'][:-1]
                    f.put('cache-metadata',f.meta)
                with self.assertRaises((ValueError,FileNotFoundError)):f.run()
                self.assertEqual(f.calls,[])

    def test_incomplete_reserved_forward_or_omitted_contrast_cannot_complete(self):
        for kind in ['forward_count','forward_status','omit_negative']:
            with self.subTest(kind=kind),tempfile.TemporaryDirectory() as d:
                f=Fixture(d)
                if kind=='forward_count':f.forward_count=163839
                elif kind=='forward_status':f.forward_status='NOT_CHECKED'
                else:f.production_change=lambda p:p['primary'].pop(1)
                with self.assertRaises(ValueError):f.run()
                self.assertFalse((f.output/'DONE').exists())

    def test_bound_input_or_frozen_source_change_blocks_pipeline(self):
        for kind in ['bound','source']:
            with self.subTest(kind=kind),tempfile.TemporaryDirectory() as d:
                f=Fixture(d)
                path=Path(f.records['qualification']['path']) if kind=='bound' else f.phase/'scripts/score_reserved_readouts.py'
                path.write_text(path.read_text()+' ')
                with self.assertRaises(ValueError):f.run()
                self.assertEqual(f.calls,[])

    def test_statistical_output_identity_drift_cannot_complete(self):
        changes = [('production','protocol_sha256'),('production','acceptance_sha256'),
                   ('production','binding_sha256'),('independent','design_sha256'),
                   ('independent','acceptance_sha256'),('independent','binding_sha256'),
                   ('independent','reference_source_sha256'),('independent','vectors_sha256')]
        for side,key in changes:
            with self.subTest(side=side,key=key),tempfile.TemporaryDirectory() as d:
                f=Fixture(d)
                def mutate(doc,key=key):doc[key]='0'*64
                if side=='production':f.production_change=mutate
                else:f.independent_change=mutate
                with self.assertRaises(ValueError):f.run()
                self.assertFalse((f.output/'DONE').exists())


if __name__=='__main__':unittest.main()
