"""S3 metadata/interfaces and actual source bodies on a NumPy fake backend.

No scientific arrays, trained models, real Torch/GPU or simulator are executed.
"""
import copy
from contextlib import nullcontext
import importlib.util
import inspect
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

# Explicit unique module name prevents an S2/S3 basename collision in discovery.
_spec=importlib.util.spec_from_file_location('_test_s3_runtime',Path(__file__).with_name('model_runtime.py'))
m=importlib.util.module_from_spec(_spec);sys.modules[_spec.name]=m;_spec.loader.exec_module(m)


class Tensor:
    def __init__(self,value):self.a=np.asarray(value)
    @property
    def shape(self):return self.a.shape
    @property
    def ndim(self):return self.a.ndim
    def __len__(self):return len(self.a)
    def __getitem__(self,key):return Tensor(self.a[key])
    def clone(self):return Tensor(self.a.copy())
    def detach(self):return self
    def cpu(self):return self
    def numpy(self):return self.a
    def float(self):return Tensor(self.a.astype(np.float32))
    def permute(self,*axes):return Tensor(self.a.transpose(axes))
    def reshape(self,*shape):return Tensor(self.a.reshape(*shape))
    def repeat(self,n):return Tensor(np.tile(self.a,n))
    def expand(self,*shape):return Tensor(np.broadcast_to(self.a,tuple(self.shape[i] if v==-1 else v for i,v in enumerate(shape))))
    def __sub__(self,other):return Tensor(self.a-(other.a if isinstance(other,Tensor) else other))
    def __truediv__(self,other):return Tensor(self.a/(other.a if isinstance(other,Tensor) else other))


class FakeTorch:
    float32=np.float32
    get_default_dtype=staticmethod(lambda:np.float32)
    is_autocast_enabled=staticmethod(lambda:False)
    @staticmethod
    def tensor(value,device=None,dtype=None):return Tensor(np.array(value,dtype=dtype or np.float32))
    @staticmethod
    def as_tensor(value,device=None):return Tensor(value)
    @staticmethod
    def cat(values,dim=0):return Tensor(np.concatenate([v.a for v in values],axis=dim))
    @staticmethod
    def stack(values):return Tensor(np.stack([v.a for v in values]))
    @staticmethod
    def inference_mode():return nullcontext()
    testing=SimpleNamespace(assert_close=lambda a,b,rtol,atol:np.testing.assert_array_equal(a.a,b.a))


class ToyModel:
    def __init__(self):self.action_calls=[];self.prediction_calls=[];self.encode_calls=0
    def parameters(self):yield SimpleNamespace(device='synthetic_cpu')
    def encode(self,data):
        self.encode_calls+=1
        x=data['pixels'].a.mean(axis=(2,3,4),dtype=np.float32)
        return {'emb':Tensor(np.broadcast_to(x[...,None],(*x.shape,192)).copy())}
    def action_encoder(self,actions):self.action_calls.append(actions.a.copy());return actions.clone()
    def predict(self,history,actions):
        self.prediction_calls.append((history.a.copy(),actions.a.copy()))
        token=history.a[:,-1]+actions.a.sum(axis=(1,2),dtype=np.float32)[:,None]*np.float32(.01)
        return Tensor(np.repeat(token[:,None],3,axis=1))


def source_body(path,name):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
    with mock.patch.dict(sys.modules,torch=FakeTorch):spec.loader.exec_module(module)
    return module


def case(role='recipient'):
    a={k:np.zeros(shape,dtype) for k,(shape,dtype) in m.CASE_SCHEMA.items()}
    a.update(index=np.array(0,np.int64),seed=np.array(1000,np.int64),reference_route=np.array(0,np.int64),
             source_selected_index=np.array(299,np.int64),source_selected_iteration=np.array(29,np.int64))
    a['prefix']=np.arange(20,dtype=np.float32).reshape(10,2)
    pop=np.arange(300*25*2,dtype=np.float32).reshape(300,25,2)/np.float32(100)
    a['source_population_actions']=pop;a['source_first5_population']=pop[:,:5].copy();a['executed_actions']=pop[299,:5].copy()
    ids=np.r_[299,-1,np.arange(30)].astype(np.int64)
    a['source_candidate_indices']=ids
    a['suffix_actions']=np.concatenate([pop[299:300,5:],np.zeros((1,20,2),np.float32),pop[:30,5:]])
    a['history_pixels'][1]=50;a['history_pixels'][2]=100;a['current_pixels'][:]=150;a['goal_pixels'][:]=200
    if role=='donor':
        del a['source_candidate_indices'];del a['suffix_actions']
    return a


class RuntimeTests(unittest.TestCase):
    def setUp(self):self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
    def tearDown(self):self.tmp.cleanup()
    def doc(self,name,value):
        p=self.root/name;p.write_text(json.dumps(value));return dict(path=str(p),sha256=m.file_sha(p))
    def archive(self,name,value):
        p=self.root/name;np.savez_compressed(p,**value);return dict(path=str(p),sha256=m.file_sha(p))
    def admission(self,role='recipient'):
        ids=list(range(1000 if role=='recipient' else 3000,(1000 if role=='recipient' else 3000)+512))
        routes=[]
        for pool in range(3):
            routes.append(dict(pool=pool,group=2*pool,policy_index=8*pool,legacy_reference_route=32*pool,
                cases=[dict(index=i,seed=ids[i],scorer_input=dict(path=f'/synthetic/{role}/{pool}/{i}.npz',sha256='a'*64)) for i in range(512)]))
        manifest=dict(status=m.MANIFEST_STATUS,protocol_sha256='p',role=role,count=512,parent_ids=ids,routes=routes)
        receipt=dict(status=m.ADMISSION_STATUS,protocol_sha256='p',role=role,count=512,parent_ids=ids,
                     input_manifest=dict(path='/synthetic/manifest.json',sha256='a'*64))
        return receipt,manifest
    def handle(self,role='recipient',arrays=None):
        arrays=case(role) if arrays is None else arrays;record=self.archive(role+'.npz',arrays)
        norm=dict(mean=[.5,-.25],std=[2.,4.])
        h=SimpleNamespace(model=ToyModel(),boundary={},model_sha256='synthetic_model_state',normalization=norm,
            normalization_sha256=m.value_sha(norm),spec=dict(pool=0,objective='decoded_teacher',condition='T0'),
            admission=dict(role=role,count=512,parent_ids=tuple(range(1000,1512)),immutable_inputs=[],
                cases={(0,0):dict(index=0,seed=1000,scorer_input=record)}),
            access_receipt=dict(role='s3_'+role,permitted_operation='recipient_roots_and_suffixes' if role=='recipient' else
                'encode_own_executed_prefix_observation_only'))
        return h,record
    def backend(self):
        planner=source_body(m.ROOT/'scripts/visual/image_planner_cost.py','_synthetic_actual_planner').ImagePlannerCost
        suffix=source_body(m.ROOT/'scripts/visual/feedback_suffix_rollout.py','_synthetic_actual_suffix').rollout_suffixes
        return (FakeTorch,lambda *a:None,lambda:None,lambda model:'synthetic_model_state',planner,suffix)

    def test_import_does_not_load_torch_or_checkpoints(self):
        code=f"import sys;sys.path.insert(0,{str(Path(m.__file__).parent)!r});import model_runtime;assert 'torch' not in sys.modules"
        r=subprocess.run([sys.executable,'-c',code],capture_output=True,text=True)
        self.assertEqual(r.returncode,0,r.stderr)

    def test_all_twelve_real_metadata_descriptions_without_model_loading(self):
        specs=[m.describe_fixed_model(p,o,t) for p in range(3) for o in m.OBJECTIVES for t in m.CONDITIONS]
        self.assertEqual(len({s['checkpoint']['sha256'] for s in specs}),12)
        for spec in specs:
            self.assertEqual(spec['group'],2*spec['pool']);self.assertEqual(spec['architecture'],'transformer_jepa')
        for values in [(True,'decoded_teacher','T0'),(3,'decoded_teacher','T0'),(0,'latent','T0'),(0,'decoded_teacher','T2')]:
            with self.assertRaises(ValueError):m.describe_fixed_model(*values)

    def test_complete_role_manifest_and_donor_recipient_separation(self):
        for role in ['recipient','donor']:
            a,z=self.admission(role);ad=m.validate_admission(a,z,role=role,protocol_sha='p')
            self.assertEqual(len(ad['cases']),1536)
            bad=copy.deepcopy(z);bad['routes'][2]['cases'].pop()
            with self.assertRaises(ValueError):m.validate_admission(a,bad,role=role,protocol_sha='p')
            bad=copy.deepcopy(z);bad['routes'][1]['policy_index']=9
            with self.assertRaises(ValueError):m.validate_admission(a,bad,role=role,protocol_sha='p')
            bad=copy.deepcopy(z);bad['routes'][0]['cases'][0]['outcome']={'path':'/forbidden'}
            with self.assertRaises(ValueError):m.validate_admission(a,bad,role=role,protocol_sha='p')
            with self.assertRaises(ValueError):m.validate_admission(a,z,role='donor' if role=='recipient' else 'recipient',protocol_sha='p')

    def test_draft_or_incomplete_s2_cannot_load_model(self):
        p=json.loads((m.PHASE/'protocol/S3_PROTOCOL.draft.json').read_text())
        s2=self.doc('s2.json',dict(status=m.S2_STATUS))
        with self.assertRaisesRegex(ValueError,'draft'):m.validate_protocol(p,s2)
        p.update(status='S3_SCIENTIFIC_PROTOCOL_FROZEN',scientific_protocol_frozen=True)
        p['prior_stage_gate']['actual_receipt']=s2;m.validate_protocol(p,s2)
        changed=self.doc('notcomplete.json',dict(status='STILL_RUNNING'))
        p['prior_stage_gate']['actual_receipt']=changed
        with self.assertRaisesRegex(ValueError,'not completed'):m.validate_protocol(p,changed)

    def test_public_loader_uses_s3_admission_and_only_low_level_constructor(self):
        old=m.legacy();models=[dict(pool=0,objective='physical_labels',condition='T1')]
        r=self.doc('runtime.json',{})
        contract=dict(admissions=dict(recipient=r,donor=r),protocol=r,s2_completion=r)
        with mock.patch.object(m,'runtime_admission',return_value=(contract,dict(role='recipient'),models)) as gate,\
             mock.patch.object(old,'_load_bound_model',return_value='synthetic') as load,\
             mock.patch.object(old,'load_calibration_response_model',side_effect=AssertionError('no S2 public roles')):
            result=m.load_recipient_model(0,'physical_labels','T1',runtime_contract=r['path'],runtime_contract_sha256=r['sha256'])
        self.assertEqual(result,'synthetic');self.assertEqual(gate.call_args.args[1],'recipient')
        self.assertEqual(load.call_args.args[2]['role'],'s3_recipient')
        self.assertNotIn('condition',inspect.signature(m.load_donor_encoder).parameters)

    def test_exact_recipient_case_and_encoder_only_donor_case(self):
        a=case();m.validate_case(a,pool=0,index=0,seed=1000,role='recipient')
        d=case('donor');m.validate_case(d,pool=0,index=0,seed=1000,role='donor')
        bad=copy.deepcopy(a);bad['executed_actions'][0,0]+=1
        with self.assertRaisesRegex(ValueError,'selected native300'):m.validate_case(bad,pool=0,index=0,seed=1000,role='recipient')
        bad=copy.deepcopy(a);bad['suffix_actions']=bad['suffix_actions'].astype(np.float64)
        with self.assertRaisesRegex(ValueError,'field'):m.validate_case(bad,pool=0,index=0,seed=1000,role='recipient')
        for field in ['terminal_states','goal_state','physical_costs']:
            bad=dict(a,**{field:np.zeros(1)})
            with self.assertRaisesRegex(ValueError,'outcomes'):m.validate_case(bad,pool=0,index=0,seed=1000,role='recipient')
        with self.assertRaisesRegex(ValueError,'outcomes'):m.validate_case(a,pool=0,index=0,seed=1000,role='donor')

    def test_unbound_hash_or_role_rejected_before_native_backend(self):
        h,p=self.handle()
        with mock.patch.object(m,'_native_dependencies',side_effect=AssertionError('must not reach backend')):
            with self.assertRaisesRegex(ValueError,'outside accepted'):m.native_recipient_root(h,case_index=0,input_pair=dict(p,sha256='0'*64))
            h.access_receipt['role']='probe_test'
            with self.assertRaisesRegex(ValueError,'genuine S3'):m.native_recipient_root(h,case_index=0,input_pair=p)

    def test_native300_root_and_real_fixed32_suffix_source_bodies_with_fake_backend(self):
        h,p=self.handle();raw=case();back=self.backend()
        with mock.patch.object(m,'_native_dependencies',return_value=back):
            root=m.native_recipient_root(h,case_index=0,input_pair=p)
            self.assertEqual(h.model.action_calls[0].shape,(300,7,10))
            self.assertEqual(h.model.prediction_calls[0][0].shape,(300,3,192))
            self.assertEqual(len(h.model.prediction_calls),2)
            repl=dict(actual=root['arrays']['predicted'].copy()+np.float32(1),donor=root['arrays']['predicted'].copy()-np.float32(2))
            result=m.suffix_rollouts(h,case_index=0,input_pair=p,root_record=root,replacements=repl)
        self.assertEqual(set(result['arrays']),{'free','actual','donor','reset'})
        self.assertTrue(all(a.shape==(4,32,192) and a.dtype==np.float32 and not a.flags.writeable for a in result['arrays'].values()))
        self.assertFalse(result['native300_vs_fixed32_bitwise_equality_claimed'])
        self.assertEqual(len(h.model.prediction_calls),22)
        # Actual unchanged suffix implementation: every first action window
        # uses prior[-5:] + executed5 + each candidate's first suffix5.
        past=np.concatenate([raw['prefix'][-5:],raw['executed_actions']])
        blocks=np.concatenate([np.broadcast_to(past,(32,10,2)),raw['suffix_actions']],axis=1).reshape(32,6,10)
        expected=(blocks-np.tile(np.asarray(h.normalization['mean'],np.float32),5))/np.tile(np.asarray(h.normalization['std'],np.float32),5)
        np.testing.assert_array_equal(h.model.action_calls[1],expected)
        first_hist=h.model.prediction_calls[2][0]
        np.testing.assert_array_equal(first_hist[:,-1],np.broadcast_to(root['arrays']['predicted'],(32,192)))
        self.assertFalse(np.array_equal(result['arrays']['actual'],result['arrays']['donor']))

    def test_donor_calls_no_action_encoder_or_dynamics_and_cannot_rollout(self):
        h,p=self.handle('donor')
        with mock.patch.object(m,'_native_dependencies',return_value=self.backend()):
            r=m.encode_donor_current(h,case_index=0,input_pair=p)
        self.assertEqual(r['arrays']['observed'].shape,(192,));self.assertEqual(h.model.encode_calls,1)
        self.assertEqual(h.model.action_calls,[]);self.assertEqual(h.model.prediction_calls,[])
        with mock.patch.object(m,'_native_dependencies',side_effect=AssertionError('no donor prediction')):
            with self.assertRaises(ValueError):m.native_recipient_root(h,case_index=0,input_pair=p)

    def test_candidate_specific_feedback_and_changed_root_evidence_rejected(self):
        h,p=self.handle()
        with mock.patch.object(m,'_native_dependencies',return_value=self.backend()):root=m.native_recipient_root(h,case_index=0,input_pair=p)
        with mock.patch.object(m,'_native_dependencies',side_effect=AssertionError('no propagation')):
            with self.assertRaisesRegex(ValueError,'candidate-specific'):
                m.suffix_rollouts(h,case_index=0,input_pair=p,root_record=root,replacements=dict(actual=np.zeros((32,192),np.float32),donor=np.zeros(192,np.float32)))
            bad=copy.deepcopy(root);bad['arrays']['predicted'][0]+=1
            with self.assertRaisesRegex(ValueError,'identity drift'):
                m.suffix_rollouts(h,case_index=0,input_pair=p,root_record=bad,replacements=dict(actual=np.zeros(192,np.float32),donor=np.zeros(192,np.float32)))

    def test_changed_action_normalization_prevents_backend(self):
        h,p=self.handle();h.normalization['mean'][0]+=1
        with mock.patch.object(m,'_native_dependencies',side_effect=AssertionError('must not infer')):
            with self.assertRaisesRegex(ValueError,'normalization changed'):m.native_recipient_root(h,case_index=0,input_pair=p)

    def test_complete_both_role_runtime_gate_and_partial_roster_rejection(self):
        s2=self.doc('complete_s2.json',dict(status=m.S2_STATUS))
        proto=json.loads((m.PHASE/'protocol/S3_PROTOCOL.draft.json').read_text())
        proto.update(status='S3_SCIENTIFIC_PROTOCOL_FROZEN',scientific_protocol_frozen=True)
        proto['prior_stage_gate']['actual_receipt']=s2
        protocol=self.doc('protocol.json',proto);admissions={}
        for role in ['recipient','donor']:
            a,z=self.admission(role);a['protocol_sha256']=z['protocol_sha256']=protocol['sha256']
            a['input_manifest']=self.doc(role+'_manifest.json',z);a['s2_completion']=s2
            admissions[role]=self.doc(role+'_admission.json',a)
        old=m.legacy()
        required=[Path(m.__file__),m.LEGACY_PATH,*old.METADATA.values(),
            m.ROOT/'strengthening/adapters/ac_rollout.py',
            *[m.ROOT/'scripts/visual'/f for f in ['factorial_model.py','lewm_adapter.py','adaptation_freeze.py',
              'evaluation_precision.py','image_planner_cost.py','feedback_suffix_rollout.py',
              'score_feedback_ranking.py','run_adaptation_checkpoint_gate.py']],
            old.BASE/'releases/visual-v1/official/jepa.py',old.BASE/'releases/visual-v1/official/module.py']
        models=[dict(pool=p,objective=o,condition=t) for p in range(3) for o in m.OBJECTIVES for t in m.CONDITIONS]
        c=dict(status=m.RUNTIME_STATUS,role='recipient',source_root=str(m.ROOT),protocol=protocol,s2_completion=s2,
            admissions=admissions,files={str(p):'a'*64 for p in required},models=models)
        original=m.bound_path
        def source_mock(record):
            # Actual temp JSON hashes checked; unavailable runtime-source bytes mocked.
            return original(record) if Path(record['path']).is_relative_to(self.root) else Path(record['path'])
        with mock.patch.object(m,'bound_path',side_effect=source_mock),             mock.patch.object(m,'describe_fixed_model',side_effect=lambda p,o,t:dict(pool=p,objective=o,condition=t)):
            contract,admitted,roster=m.runtime_admission(self.doc('runtime_ok.json',c),'recipient')
            self.assertEqual(len(roster),12);self.assertEqual(len(admitted['cases']),1536)
            bad=copy.deepcopy(c);del bad['admissions']['donor']
            with self.assertRaisesRegex(ValueError,'Both complete'):
                m.runtime_admission(self.doc('runtime_missing_donor.json',bad),'recipient')
            bad=copy.deepcopy(c);bad['models'].pop()
            with self.assertRaisesRegex(ValueError,'twelve'):
                m.runtime_admission(self.doc('runtime_missing_model.json',bad),'recipient')
            bad=copy.deepcopy(c);bad['files']['/synthetic/head.npz']='0'*64
            with self.assertRaisesRegex(ValueError,'tensor files'):
                m.runtime_admission(self.doc('runtime_smuggled_head.json',bad),'recipient')

    def test_strict_fp32_precision_refuses_active_autocast(self):
        with mock.patch.object(FakeTorch,'is_autocast_enabled',return_value=True):
            with self.assertRaisesRegex(ValueError,'FP32'):m._precision(FakeTorch,lambda:None)


if __name__=='__main__':unittest.main()
