"""Synthetic scorer contracts; full native/raw gate is explicitly mocked.

No actual S2/head/model data is accessed. Complete count tests use an analytical
decoder stub; a separate small test calls both real old nonlinear decoders.
"""
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
import score_probe_population as s


def js(path,value):
    Path(path).write_text(json.dumps(value,sort_keys=True,allow_nan=False)+'\n')
    return s.file_pair(path)


def save(path,values):
    np.savez_compressed(path,**values)
    return s.file_pair(path)


def head(pool=0):
    p={'input.weight':np.zeros((256,192),np.float32),'input.bias':np.zeros(256,np.float32),
       'output.weight':np.zeros((6,256),np.float32),'output.bias':np.arange(6,dtype=np.float32)/16,
       'skip.weight':np.eye(6,192,dtype=np.float32),'mean':np.arange(192,dtype=np.float64)/32,
       'scale':np.full(192,2.+pool),'target_mean':np.arange(6,dtype=np.float64)*2+pool,
       'target_scale':np.arange(1,7,dtype=np.float64)}
    for block in range(2):
        for name in ('fc1','fc2'):
            p[f'blocks.{block}.{name}.weight']=np.zeros((256,256),np.float32)
            p[f'blocks.{block}.{name}.bias']=np.zeros(256,np.float32)
    return p


def fixture(root,split='calibration'):
    """Real temporary hashes, synthetic metadata/latent values, no real claims."""
    n=s.gate._size(split)
    heads=[dict(pool=p,group=2*p,file=save(root/f'head{p}.npz',head(p))) for p in range(3)]
    old=js(root/'old_protocol.json',dict(status='DESIGN_FROZEN_BEFORE_ANY_G_EVAL_FITTING_OR_EFFECTS',architecture=s.ARCHITECTURE))
    encoders=[dict(group=g,architecture='transformer_jepa',checkpoint_sha256=str(g)*64) for g in range(6)]
    selected=js(root/'selected.json',dict(status='ALL_SIX_READOUT_CHECKPOINTS_FROZEN_BEFORE_QUALIFICATION',
        protocol_sha256=old['sha256'],groups=[dict(group=g,checkpoint_sha256=heads[g//2]['file']['sha256']) for g in range(6)]))
    encoder=js(root/'encoder.json',dict(groups=encoders))
    inventory=js(root/'inventory.json',dict(source_bindings={s.SELECTED_KEY:selected['sha256'],s.ENCODER_KEY:encoder['sha256']},
        readouts=[dict(pool=p,group=2*p,measurement=heads[p]['file'],encoder=encoders[2*p],normalization_not_recomputed=True) for p in range(3)]))
    protocol=js(root/'protocol.json',dict(status='S2_SCIENTIFIC_PROTOCOL_FROZEN',assets_inventory=inventory,
        probe_population={'donor_assignment_seeds':dict(calibration=1490384197,test=1934445636)}))
    paths=[Path(s.__file__),Path(s.gate.__file__),Path(s.features.__file__),s.PRODUCTION_DECODER,s.REFERENCE_DECODER]
    files={str(path):s.gate.sha(path) for path in paths}
    probe_sources=js(root/'probe_sources.json',dict(status='S2_PROBE_POPULATION_SOURCES_FROZEN',protocol_sha256=protocol['sha256'],files=files))
    sources=js(root/'sources.json',dict(status='S2_PROBE_MEASUREMENT_SOURCES_FROZEN',protocol_sha256=protocol['sha256'],
        decoder=s.DECODER_SETTINGS,assets_inventory=inventory,selected_readouts=selected,encoder_bindings=encoder,
        readout_protocol=old,probe_sources=probe_sources,files=files,heads=heads))
    ids=np.arange(1000,1000+n,dtype=np.int64);donors=np.arange(8000,8000+n,dtype=np.int64)
    seed=dict(calibration=1490384197,test=1934445636)[split]
    perm=np.random.Generator(np.random.PCG64(seed)).permutation(n).astype(np.int64)
    assignment=save(root/'assignment.npz',dict(donor_assignment=perm,recipient_ids=ids,donor_ids=donors))
    raw=js(root/'raw.json',dict(synthetic=True));rows=[]
    for pool in range(3):
        tokens=np.zeros((2,n,4,5,192),np.float32)
        oi=np.arange(2)[:,None,None,None];goal=np.arange(n)[None,:,None,None]
        branch=np.arange(4)[None,None,:,None];horizon=np.arange(5)[None,None,None,:]
        base=pool*8+oi*4+goal/1024+branch/4+horizon/16
        for coord in range(6):tokens[...,coord]=base+coord
        # Signed G contains both signs; zero families have identical branches.
        tokens[:,1::2,2,:2]=tokens[:,1::2,1,:2]
        tokens[:,1::2,2,-1,2:4]-=2
        tokens[:,::17,1:3]=tokens[:,::17,0,None]
        history=tokens[:,:,0].copy();history[:,:,1:,:6]+=1/8
        truth=np.zeros((n,5,6),np.float64)
        truth[...,2:4]=.25
        norm=np.arange(n,dtype=np.float64)/128;norm[::17]=0
        cache=save(root/f'cache{pool}.npz',dict(free=tokens[:,:,0],reset=tokens[:,:,3],observed_history=history,truth=truth,seeds=ids))
        cache_report=js(root/f'cache_report{pool}.json',dict(arrays=cache,models=[dict(spec={'original':{'sha256':str(2*pool)*64}})]))
        rolled_arrays=save(root/f'rollout{pool}.npz',dict(tokens=tokens,observed_history=history,seeds=ids))
        rolled=js(root/f'rollout_report{pool}.json',dict(arrays=rolled_arrays))
        rows.append(dict(pool=pool,group=2*pool,recipient_cache=cache_report,rollout=rolled,
            truth_array_source=cache,effective_common_norm=norm.tolist()))
    population=js(root/'population.json',dict(protocol=protocol,sources=probe_sources))
    admission=dict(status=s.gate.STATUS,split=split,count=n,pools=rows,complete_pools=3,complete_objective_strata=6,
        complete_families=3*n,complete_directions=12*n,axes=s.gate.ROLLOUT_AXES,protocol_sha256=protocol['sha256'],
        sources_sha256=probe_sources['sha256'],source_sha256=s.gate.sha(s.gate.__file__),population_binding=population,
        raw_population_acceptance=raw,donor_assignment=assignment,parent_ids=ids.tolist(),donor_ids=donors.tolist(),
        q_g_weights_deserialized=False,q_g_outputs_computed=False,response_model_outputs_opened=False,
        outcome_statistics_computed=False,rows_removed=0,q_g_scoring_gate_passed=True)
    acceptance=js(root/'accepted.json',admission)
    binding=js(root/'binding.json',dict(status=s.INPUT_STATUS,split=split,protocol=protocol,sources=sources,
        probe_acceptance=acceptance,output_root=str(root/'output')))
    return dict(binding=binding,admission=admission,sources=sources,protocol=protocol,
        ids=ids,assigned_donors=donors[perm],output=root/'output')


def analytical(z,p,*args,batch_size):
    if batch_size!=512:raise AssertionError('Changed fixed batch')
    z=np.asarray(z,np.float64)
    return ((z[...,:6]-p['mean'][:6])/p['scale'][:6]+p['output.bias'])*p['target_scale']+p['target_mean']


class ScorerTests(unittest.TestCase):
    def admitted(self,f):
        with mock.patch.object(s.gate,'accept_population',return_value=f['admission']) as recheck:
            ctx=s.load_context(f['binding']['path'],f['binding']['sha256'])
        recheck.assert_called_once()
        self.assertFalse(ctx['loaded_heads'])
        return ctx

    def produce_analytical(self,ctx):
        module=SimpleNamespace(numpy_forward=analytical,decode=analytical)
        with mock.patch.object(s,'_load_module',return_value=module) as loader:
            report=s.produce(ctx)
        self.assertEqual(loader.call_count,30)
        return report

    def verify(self,f):
        with mock.patch.object(s.gate,'accept_population',return_value=f['admission']),mock.patch.object(s,'decode_pool_tokens') as decode:
            result=s.verify_completed_probe(s.file_pair(f['output']/'report.json'),protocol_pair=f['protocol'],
                measurement_sources_pair=f['sources'],expected_split=f['admission']['split'])
        decode.assert_not_called()
        return result

    def test_whole_calibration_signed_g_squared_dose_and_donor_once(self):
        with tempfile.TemporaryDirectory() as temp:
            f=fixture(Path(temp));ctx=self.admitted(f);report=self.produce_analytical(ctx);result=self.verify(f)
            a=result['arrays'];self.assertEqual(a['ordinary'].shape,(256,6,4))
            self.assertEqual(a['signed_g'].shape,(256,6));self.assertTrue(np.any(a['signed_g']<0));self.assertTrue(np.any(a['signed_g']>0))
            np.testing.assert_array_equal(a['donor_ids'],f['assigned_donors'])
            np.testing.assert_array_equal(a['ordinary'][...,3],np.repeat(ctx['inputs']['common_norm']**2,2,axis=1))
            self.assertTrue((f['output']/'DONE').exists());self.assertEqual(len(report['decoder_checks']),15)
            audit=s.gate._arrays(report['decoded_audit'])
            p=audit['prediction'];t=audit['truth']
            e=np.sum((p[...,2:4]-t[:,:,None,None,:,2:4])**2,axis=-1)
            np.testing.assert_array_equal(a['signed_g'],(e[:,:,:,2,-1]-e[:,:,:,1,-1]).reshape(256,6))
            for pool in range(3):
                expected=analytical(ctx['inputs']['tokens'][pool][:,:,2],head(pool),batch_size=512)
                np.testing.assert_array_equal(p[:,pool,:,2],expected.transpose(1,0,2,3))

    def test_whole_test512_and_all_six_axis_order(self):
        with tempfile.TemporaryDirectory() as temp:
            f=fixture(Path(temp),'test');ctx=self.admitted(f);r=self.produce_analytical(ctx);a=self.verify(f)['arrays']
            self.assertEqual(a['ordinary'].shape,(512,6,4));self.assertEqual(r['stratum_ids'],list(s.features.STRATA))
            self.assertEqual(sum(c['token_count'] for c in r['decoder_checks']),76800)
            np.testing.assert_array_equal(a['recipient_ids'],f['ids'])

    def test_missing_gate_and_last_pool_rejection_precede_qg_loading(self):
        with tempfile.TemporaryDirectory() as temp:
            f=fixture(Path(temp));ctx=s.authenticate_measurement_sources(f['sources'],f['protocol'])
            with mock.patch.object(s.gate,'_arrays') as arrays:
                with self.assertRaises(ValueError):s.decode_pool_tokens(ctx,0,np.zeros((2,192),np.float32))
                arrays.assert_not_called()
            with mock.patch.object(s.gate,'accept_population',side_effect=ValueError('synthetic last pool incomplete')):
                with self.assertRaises(ValueError):s.authorize_measurement_context(ctx,s.gate._json(f['binding'])['probe_acceptance'])
            self.assertFalse(ctx['probe_gate_verified']);self.assertFalse(ctx['loaded_heads'])

    def test_failed_status_and_changed_recomputed_receipt_reject(self):
        for kind in ('failed_status','changed_recomputed'):
            with tempfile.TemporaryDirectory() as temp:
                root=Path(temp);f=fixture(root);binding=s.gate._json(f['binding']);accepted=copy.deepcopy(f['admission'])
                if kind=='failed_status':accepted['status']='BLOCKED'
                binding['probe_acceptance']=js(root/'accepted.json',accepted);f['binding']=js(root/'binding.json',binding)
                repeated=copy.deepcopy(f['admission']);repeated['rows_removed']=1
                with mock.patch.object(s.gate,'accept_population',return_value=repeated),mock.patch.object(s,'decode_pool_tokens') as decode:
                    with self.assertRaises(ValueError):s.load_context(f['binding']['path'],f['binding']['sha256'])
                    decode.assert_not_called()

    def test_head_source_normalizer_and_encoder_metadata_corruption(self):
        for kind in ('head_hash','missing_head','encoder','decoder_batch','closure','architecture'):
            with tempfile.TemporaryDirectory() as temp:
                root=Path(temp);f=fixture(root);sources=s.gate._json(f['sources'])
                if kind=='head_hash':sources['heads'][2]['file']['sha256']='0'*64
                elif kind=='missing_head':sources['heads'].pop()
                elif kind=='encoder':
                    doc=s.gate._json(sources['encoder_bindings']);doc['groups'][4]['checkpoint_sha256']='f'*64
                    sources['encoder_bindings']=js(root/'encoder.json',doc)
                elif kind=='decoder_batch':sources['decoder']['batch_size']=1024
                elif kind=='closure':sources['files'].pop(str(s.REFERENCE_DECODER))
                else:
                    doc=s.gate._json(sources['readout_protocol']);doc['architecture']['blocks']=3
                    sources['readout_protocol']=js(root/'old_protocol.json',doc)
                pair=js(root/'sources.json',sources)
                with self.assertRaises(ValueError):s.authenticate_measurement_sources(pair,f['protocol'])

    def test_actual_dual_nonlinear_decoders_513_tokens_fixed_batch_and_own_norms(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);f=fixture(root);ctx=self.admitted(f)
            # Separate synthetic head path bound into this already synthetic
            # context; no real selected weights are used by this arithmetic test.
            p=head();p['input.bias'][0]=.75;p['output.weight'][0,0]=.2
            p['blocks.0.fc1.weight'][0,0]=.4;p['blocks.0.fc2.weight'][0,0]=.3
            ctx['heads'][0]['file']=save(root/'nonlinear.npz',p)
            z=(np.arange(513*192,dtype=np.float32).reshape(3,171,192)%19)/16
            result=s.decode_pool_tokens(ctx,0,z)
            self.assertEqual(result['poses'].shape,(3,171,6));self.assertTrue(result['verification']['comparison_passed'])
            self.assertLessEqual(result['verification']['maximum_tolerance_ratio'],1)
            np.testing.assert_allclose(result['poses'][...,1:],analytical(z,p,batch_size=512)[...,1:],rtol=0,atol=1e-12)
            self.assertEqual(ctx['loaded_heads'][0]['normalizer_sha256'],s._normalizer_digest(p))

    def test_dual_disagreement_and_invalid_embedded_norm_block(self):
        for kind in ('disagreement','norm'):
            with tempfile.TemporaryDirectory() as temp:
                root=Path(temp);f=fixture(root);ctx=self.admitted(f)
                if kind=='norm':
                    p=head();p['scale'][3]=0;ctx['heads'][0]['file']=save(root/'badnorm.npz',p)
                    with self.assertRaises(ValueError):s.decode_pool_tokens(ctx,0,np.zeros((2,192),np.float32))
                else:
                    module=SimpleNamespace(numpy_forward=analytical,decode=lambda *a,**kw:analytical(*a,**kw)+.01)
                    with mock.patch.object(s,'_load_module',return_value=module),self.assertRaises(AssertionError):
                        s.decode_pool_tokens(ctx,0,np.zeros((2,192),np.float32))
                self.assertFalse(ctx['decoder_checks'])

    def test_latent_and_decoded_oh5_must_be_exact_no_repair(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);f=fixture(root);ctx=self.admitted(f)
            ctx['inputs']['histories'][2][1,-1,0,0]+=np.float32(.125)
            module=SimpleNamespace(numpy_forward=analytical,decode=analytical)
            with mock.patch.object(s,'_load_module',return_value=module),self.assertRaises(ValueError):s.produce(ctx)
            self.assertFalse((f['output']/'DONE').exists());self.assertTrue((f['output']/'failure.json').exists())

    def test_rehashed_latent_oh5_mismatch_blocks_before_first_head_load(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);f=fixture(root);ctx=self.admitted(f)
            row=ctx['admission']['pools'][2];report=s.gate._json(row['rollout']);a=s.gate._arrays(report['arrays'])
            a['observed_history'][1,-1,0,0]+=np.float32(.125)
            report['arrays']=save(root/'rollout2.npz',a)
            row['rollout']=js(root/'rollout_report2.json',report)
            with self.assertRaises(ValueError):s._probe_inputs(ctx)
            self.assertFalse(ctx['loaded_heads'])

    def test_rehashed_scalar_audit_donor_normalizer_and_dual_receipt_corruption(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);f=fixture(root);ctx=self.admitted(f);report=self.produce_analytical(ctx)
            original_arrays=s.gate._arrays(report['arrays']);original_audit=s.gate._arrays(report['decoded_audit'])
            for kind in ('scalar','truth','norm','donor','normalizer','dual','missing_call','axes'):
                bad=copy.deepcopy(report);a={k:v.copy() for k,v in original_arrays.items()};d={k:v.copy() for k,v in original_audit.items()}
                if kind=='scalar':a['signed_g'][0,0]*=-1;a['signed_g'][1,0]+=.125
                elif kind=='truth':d['truth'][0,0,0,2]+=1
                elif kind=='norm':d['common_norm'][1,0]+=.5
                elif kind=='donor':a['donor_ids']=np.roll(a['donor_ids'],1)
                elif kind=='normalizer':bad['heads'][1]['normalizer_sha256']='f'*64
                elif kind=='dual':bad['decoder_checks'][14]['maximum_tolerance_ratio']=10.
                elif kind=='missing_call':bad['decoder_checks'].pop()
                else:bad['axes']['branches']=['actual','free','donor','reset']
                bad['arrays']=save(f['output']/'scalars.npz',a);bad['decoded_audit']=save(f['output']/'decoded_audit.npz',d)
                js(f['output']/'report.json',bad)
                with self.subTest(kind=kind),self.assertRaises((ValueError,AssertionError)):self.verify(f)

    def test_existing_output_or_intent_prevents_reexecution(self):
        with tempfile.TemporaryDirectory() as temp:
            f=fixture(Path(temp));ctx=self.admitted(f);self.produce_analytical(ctx)
            with self.assertRaises(ValueError):s.produce(ctx)
            with mock.patch.object(s,'authenticate_measurement_sources') as auth,self.assertRaises(ValueError):
                s.load_context(f['binding']['path'],f['binding']['sha256'])
            auth.assert_not_called()

    def test_import_does_not_load_torch_and_cli_help(self):
        cmd=[sys.executable,'-c','import sys;sys.path.insert(0,'+repr(str(Path(s.__file__).parent))+');import score_probe_population;assert "torch" not in sys.modules']
        result=subprocess.run(cmd,capture_output=True,text=True);self.assertEqual(result.returncode,0,result.stderr)
        result=subprocess.run([sys.executable,s.__file__,'--help'],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)


if __name__=='__main__':unittest.main()
