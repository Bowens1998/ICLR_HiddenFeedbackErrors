"""Generated metadata/arrays only; never read actual fitted-head results."""
import copy
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from verify_actual_selected_heads import (ATOL, RTOL, PURPOSE, STUDY, PREFIX, V8, digest,
    verify, independent_forward, validate_parameters, validate_history, compare_forward)

PHASE = Path(__file__).resolve().parents[1]


def record(path): return dict(path=str(path.resolve()), sha256=digest(path))


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(data)); return record(path)


def sparse_head(role):
    h = dict(mean=np.arange(192, dtype=np.float64)%3/10, scale=1+np.arange(192,dtype=np.float64)%5/10,
        target_mean=np.arange(6,dtype=np.float64), target_scale=1+np.arange(6,dtype=np.float64)/3)
    if role=='C': shapes={'0':(512,192),'2':(512,512),'4':(6,512)}
    else:
        shapes={'input':(256,192),'output':(6,256),'skip':(6,192)}
        for b in range(2):
            for f in ('fc1','fc2'): shapes[f'blocks.{b}.{f}']=(256,256)
    for name,shape in shapes.items():
        h[name+'.weight']=np.zeros(shape,np.float32)
        if name!='skip':h[name+'.bias']=np.zeros(shape[0],np.float32)
    if role=='C':
        h['0.weight'][0,0]=.5;h['0.bias'][0]=-.2
        h['2.weight'][0,0]=1.7;h['2.bias'][0]=.1
        h['4.weight'][:,0]=np.linspace(-.4,.4,6,dtype=np.float32);h['4.bias'][:]=.03
    else:
        h['input.weight'][0,0]=.4;h['input.bias'][0]=-.1
        for b in range(2):
            h[f'blocks.{b}.fc1.weight'][0,0]=.5+b*.2;h[f'blocks.{b}.fc1.bias'][0]=.2
            h[f'blocks.{b}.fc2.weight'][0,0]=.7;h[f'blocks.{b}.fc2.bias'][0]=-.1
        h['output.weight'][:,0]=np.linspace(-.4,.4,6,dtype=np.float32);h['output.bias'][:]=.03
        h['skip.weight'][:,1]=np.linspace(.1,.3,6,dtype=np.float32)
    return h


def scalar_oracle(tokens,h,role):
    rows=[];gelu=lambda x:.5*x*(1+math.erf(x/math.sqrt(2)))
    for z in tokens:
        x=(z.astype(np.float64)-h['mean'])/h['scale']
        if role=='C':
            v=max(float(h['0.weight'][0,0])*x[0]+float(h['0.bias'][0]),0)
            v=max(float(h['2.weight'][0,0])*v+float(h['2.bias'][0]),0)
            y=[float(h['4.weight'][j,0])*v+float(h['4.bias'][j]) for j in range(6)]
        else:
            v=gelu(float(h['input.weight'][0,0])*x[0]+float(h['input.bias'][0]))
            for b in range(2):
                v+=.5*(float(h[f'blocks.{b}.fc2.weight'][0,0])*gelu(float(h[f'blocks.{b}.fc1.weight'][0,0])*v+float(h[f'blocks.{b}.fc1.bias'][0]))+float(h[f'blocks.{b}.fc2.bias'][0]))
            y=[float(h['output.weight'][j,0])*v+float(h['output.bias'][j])+float(h['skip.weight'][j,1])*x[1] for j in range(6)]
        rows.append(np.asarray(y)*h['target_scale']+h['target_mean'])
    return np.asarray(rows,np.float64)


def generated_bundle(root):
    cfg=json.loads((PHASE/'protocol/DESIGN.lock.json').read_text()); pb=write_json(root/'design.json',cfg)
    # The test fixture authenticates source identities, but has no experiment data.
    implementation=json.loads((PHASE/'manifests/HEAD_IMPLEMENTATION.lock.json').read_text())
    implementation['protocol_sha256']=pb['sha256']; ib=write_json(root/'implementation.json',implementation)
    source=implementation['files']; tasks=[]; locked=[]; local=[]
    for g in (0,1):
        for role in ('C','D'):
            folder=root/f'group_{g}'/role;folder.mkdir(parents=True)
            tasks.append(dict(group=g,head_role=role,output=str(folder)))
            h=sparse_head(role); cp=folder/'selected.npz';np.savez(cp,**h)
            tokens=np.linspace(-3,3,64*192,dtype=np.float32).reshape(64,192)
            # Analytically generated stand-ins for saved reference output arrays.
            expected=scalar_oracle(tokens,h,role)
            fp=folder/'forward_verification.npz';np.savez(fp,tokens=tokens,torch_prediction=expected,numpy_prediction=expected)
            history=[]
            for step in range(250,8001,250):
                value=1. if step in (500,750) else 2.+step/8000
                history.append(dict(step=step,balanced_six_normalized_validation_mse=value,
                    domain_validation=dict(expert=value,planner=value),train_loss=.8))
            best=copy.deepcopy(history[1]);best.update(file='selected.npz',sha256=digest(cp))
            ns=cfg['fit']['seed_namespace'].format(role=role,group=g)
            seed=int.from_bytes(hashlib.sha256(f"{cfg['root_seed']}:{ns}".encode()).digest()[:4],'big')
            report=dict(status='PASS_S1_FIXED_HEAD_FIT',protocol_sha256=pb['sha256'],group=g,head_role=role,
                views='/synthetic/views',views_report_sha256='7'*64,architecture=cfg['head_design'][role],
                optimizer=dict(name='Adam',lr=cfg['fit']['learning_rate'],betas=[.9,.999],eps=1e-8,weight_decay=0.,amsgrad=False,maximize=False,foreach=False,fused=False),
                source_sha256=source[PREFIX+'fit_head.py'],adapter_source_sha256={Path(k).name:v for k,v in source.items() if k.startswith(PREFIX) and Path(k).name.startswith('s1_')},
                reused_gelu_source_sha256=source[V8],seed=seed,seed_namespace=ns,updates=8000,history=history,
                selected=best,forward_fixture_sha256=digest(fp),independent_forward_max_abs=0.,synthetic_fixture_only=True)
            rp=folder/'report.json';write_json(rp,report)
            locked.append(dict(group=g,head_role=role,views='/synthetic/views',views_report_sha256='7'*64,
                checkpoint=str(cp),checkpoint_sha256=digest(cp),fit_report=str(rp),fit_report_sha256=digest(rp),
                forward_fixture=str(fp),forward_fixture_sha256=digest(fp),selected_step=500))
            local.append(dict(group=g,head_role=role,checkpoint=record(cp),fit_report=record(rp),forward_fixture=record(fp)))
    fit=write_json(root/'inputs.json',dict(status='S1_FOUR_FIXED_FITS_INPUTS_FROZEN',protocol_sha256=pb['sha256'],
        implementation_lock_sha256=ib['sha256'],task_roster=tasks,views='/synthetic/views',views_report_sha256='7'*64))
    sl=write_json(root/'selected_lock.json',dict(status='ALL_FOUR_S1_HEADS_FROZEN_BEFORE_QUALIFICATION',
        protocol_sha256=pb['sha256'],formal_authorized=False,source_sha256=source[PREFIX+'freeze_selected_heads.py'],heads=locked))
    b=dict(study_id=STUDY,purpose=PURPOSE,protocol=pb,head_implementation=ib,fitting_inputs=fit,selected_lock=sl,heads=local)
    return write_json(root/'binding.json',b),b,cfg


class SelectedHeadVerification(unittest.TestCase):
    def test_both_nonlinear_architectures_against_scalar_oracle(self):
        tokens=np.linspace(-3,3,64*192,dtype=np.float32).reshape(64,192)
        for role in ('C','D'):
            h=sparse_head(role)
            np.testing.assert_allclose(independent_forward(tokens,h,role),scalar_oracle(tokens,h,role),rtol=RTOL,atol=ATOL)

    def test_normalizer_architecture_and_finite_guards(self):
        changes=[lambda h:h.update(scale=np.zeros(192,np.float64)),
            lambda h:h.update(target_mean=h['target_mean'].astype(np.float32)),
            lambda h:h.update(mean=np.zeros(191,np.float64)),
            lambda h:h['input.weight'].__setitem__((0,0),np.nan),
            lambda h:h.update(**{'skip.bias':np.zeros(6,np.float32)})]
        for change in changes:
            h=sparse_head('D');change(h)
            with self.assertRaises(ValueError):validate_parameters(h,'D')

    def test_complete_four_head_cli_and_earliest_tie(self):
        with tempfile.TemporaryDirectory(prefix='s1_selected_head_fixture_') as d:
            root=Path(d);b,_,_=generated_bundle(root);output=root/'verification'
            command=[sys.executable,str(PHASE/'verification/verify_actual_selected_heads.py'),'--binding',b['path'],'--binding-sha256',b['sha256'],'--output',str(output)]
            run=subprocess.run(command,capture_output=True,text=True)
            self.assertEqual(run.returncode,0,run.stdout+run.stderr)
            r=json.loads((output/'report.json').read_text())
            self.assertEqual(r['status'],'PASS_INDEPENDENT_ALL_FOUR_SELECTED_HEADS');self.assertEqual(r['total_fixture_rows'],256)
            self.assertEqual([h['selected_step'] for h in r['heads']],[500]*4)
            self.assertFalse(r['qualification_or_effects_opened'])
            self.assertTrue(all(not h['normalizers']['fit_moments_recomputed'] for h in r['heads']))

    def test_incomplete_roster_rejected_before_np_load(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);_,binding,_=generated_bundle(root);binding['heads'].pop()
            b=write_json(root/'bad_binding.json',binding)
            with patch('verify_actual_selected_heads.np.load',side_effect=AssertionError('Array accessed before complete roster')):
                with self.assertRaises(ValueError):verify(b['path'],b['sha256'],root/'result')

    def test_validation_schedule_balance_and_later_tie_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);_,b,cfg=generated_bundle(root);r=json.loads(Path(b['heads'][0]['fit_report']['path']).read_text())
            row=dict(selected_step=500);validate_history(r,row,cfg)
            bad=copy.deepcopy(r);bad['history'].pop()
            with self.assertRaises(ValueError):validate_history(bad,row,cfg)
            bad=copy.deepcopy(r);bad['selected'].update(bad['history'][2]);row_bad=dict(selected_step=750)
            with self.assertRaises(ValueError):validate_history(bad,row_bad,cfg)
            bad=copy.deepcopy(r);bad['history'][0]['domain_validation']['planner']+=.1
            with self.assertRaises(ValueError):validate_history(bad,row,cfg)
            bad=copy.deepcopy(r);bad['history'][0]['balanced_six_normalized_validation_mse']=float('nan')
            with self.assertRaises(ValueError):validate_history(bad,row,cfg)

    def test_frozen_forward_tolerance_and_wrong_bound_file(self):
        expected=np.zeros((64,6),np.float64)
        compare_forward(expected+ATOL*.5,expected,'within')
        with self.assertRaises(ValueError):compare_forward(expected+ATOL*2,expected,'outside')
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);b,binding,_=generated_bundle(root)
            Path(binding['heads'][3]['forward_fixture']['path']).write_bytes(b'changed bytes')
            with self.assertRaises(ValueError):verify(b['path'],b['sha256'],root/'result')
            self.assertFalse((root/'result').exists())


if __name__=='__main__':unittest.main(verbosity=2)
