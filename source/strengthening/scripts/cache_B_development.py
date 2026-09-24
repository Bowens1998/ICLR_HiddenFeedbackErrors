"""All 64 native development trajectories: official parity, action response and raw encodings."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'strengthening/adapters'),str(ROOT/'scripts/visual')]
from contracts import sha,atomic_json,require_role
from artifact_io import atomic_npz,state_hash
from native_model import load_native,encode_visual,flatten_visual
from native_rollout import prepare_native_inputs,native_feedback_rollout
from nonlinear_pose_cost import numpy_pose
from evaluation_lock import add_protocol_arguments,evaluation_context,require_output,bound_input


def main():
    p=argparse.ArgumentParser()
    for key in ['assets','environment','bank','head','head-cache','output']:
        p.add_argument('--'+key,required=True)
    add_protocol_arguments(p);a=p.parse_args();bank=Path(a.bank);headroot=Path(a.head);headcache=Path(a.head_cache)
    role=require_role(json.loads((bank/'role.json').read_text()),['diagnostic_development_B','donor_development_B','confirmation_B','donor_bank_B'])
    ctx=evaluation_context(a,role['role'],ROOT,__file__);count=ctx.count
    require_output(a.output,ctx,'B_donor_cache' if role['role'].startswith('donor_') else 'B_cache')
    if ctx.formal:assert headroot/'weights.npz'==Path(ctx.design['native_head'])
    assert role['parent_manifest_sha256']==sha(bank/'manifest.json') and (bank/'DONE').exists()
    manifest=json.loads((bank/'manifest.json').read_text());assert len(manifest['cases'])==role['count']==count
    assert [c['index'] for c in manifest['cases']]==list(range(count))
    hr=json.loads((headroot/'report.json').read_text());ha=json.loads((headroot/'acceptance.json').read_text())
    assert ha['status']=='PASS_FULL_NATIVE_HEAD_NUMPY_RECONSTRUCTION' and ha['report_sha256']==sha(headroot/'report.json')
    assert hr['weights_sha256']==sha(headroot/'weights.npz') and hr['cache_report_sha256']==sha(headcache/'report.json')
    head=dict(np.load(headroot/'weights.npz'));model,cfg,binding=load_native(a.assets,a.environment)
    if ctx.formal:assert binding==ctx.design['native_model']
    torch.set_num_threads(2);before=state_hash(model);out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    pixelsets={}
    for r in ['head_train','head_validation']:
        meta=json.loads((headcache/(r+'.json')).read_text());rp=headcache/(r+'_rows.json')
        assert sha(rp)==meta['row_manifest_sha256']
        pixelsets[r]={x['model_input_pixel_sha256'] for x in json.loads(rp.read_text())['rows']}
    rows=[];start=time.monotonic();overlaps=[];obs_all=[];pred_all=[];truth_all=[]
    with torch.inference_mode():
        for c in manifest['cases']:
            path=bank/c['file'];assert sha(path)==c['sha256'];z=dict(np.load(path))
            history,blocks,images=prepare_native_inputs(z,cfg['img_size'])
            _,official=model.rollout(history,blocks)
            free=native_feedback_rollout(model,history,blocks)
            assert free.shape==(1,8,196,404) and torch.isfinite(free).all()
            torch.testing.assert_close(free,official,rtol=0,atol=0)
            identity=native_feedback_rollout(model,history,blocks,flatten_visual(free[:,3:4]))
            torch.testing.assert_close(free,identity,rtol=0,atol=0)
            observed=torch.cat([encode_visual(model,images[j:j+1]) for j in range(3,8)])
            reset=native_feedback_rollout(model,history,blocks,observed[0])
            torch.testing.assert_close(reset[:,:3],free[:,:3],rtol=0,atol=0)
            torch.testing.assert_close(reset[:,3,:,384:],free[:,3,:,384:],rtol=0,atol=0)
            altered=z.copy();altered['actions']=z['actions'].copy();altered['actions'][10:]*=-1
            ah,ab,_=prepare_native_inputs(altered,cfg['img_size'])
            for k in history:torch.testing.assert_close(ah[k],history[k],rtol=0,atol=0)
            torch.testing.assert_close(ab[:,:2],blocks[:,:2],rtol=0,atol=0)
            response=native_feedback_rollout(model,ah,ab)
            sensitivity=(flatten_visual(response[:,3:])-flatten_visual(free[:,3:])).square().mean().item()
            if not np.isfinite(sensitivity):raise ValueError('Native model has nonfinite visual action response')
            image_hashes=[]
            for j,im in enumerate(images):
                digest=hashlib.sha256(im.contiguous().numpy().tobytes()).hexdigest();image_hashes.append(digest)
                for r,s in pixelsets.items():
                    if digest in s:overlaps.append(dict(case=c['index'],frame=5*j,split=r))
            physical=z['states'][[15,20,25,30,35]]
            truth=np.c_[physical[:,:4],np.sin(physical[:,4]),np.cos(physical[:,4])]
            prediction=flatten_visual(free[:,3:])[0].cpu().numpy();observed=observed.cpu().numpy()
            target=out/f"case_{c['index']:03d}.npz"
            atomic_npz(target,free_tokens=free[0].cpu().numpy(),observed_visual=observed,
                full_visual_reset_tokens=reset[0,3:].cpu().numpy(),truth=truth,normalized_actions=blocks[0].cpu().numpy(),
                history_proprio=history['proprio'][0].cpu().numpy(),seed=np.asarray(c['seed']))
            obs_all.append(observed);pred_all.append(prediction);truth_all.append(truth)
            rows.append(dict(index=c['index'],seed=c['seed'],input_sha256=sha(path),file=target.name,file_sha256=sha(target),
                native_parity_exact=True,identity_exact=True,earlier_context_exact=True,first_predicted_nonvisual_channels_exact=True,
                visual_action_response_mean_square=sensitivity,legitimate_zero_action_response=sensitivity==0,
                model_input_pixel_sha256=image_hashes))
    assert state_hash(model)==before
    truth=np.stack(truth_all);obs_pose=numpy_pose(np.stack(obs_all),head);pred_pose=numpy_pose(np.stack(pred_all),head)
    atomic_npz(out/'baseline_metrics.npz',truth=truth,observed_pose=obs_pose,free_pose=pred_pose,
        observed_block_error=np.square(obs_pose[...,2:4]-truth[...,2:4]).sum(-1),
        free_block_error=np.square(pred_pose[...,2:4]-truth[...,2:4]).sum(-1))
    if ctx.formal:
        frames=Path(ctx.design['base'])/'releases/feedback-strengthening-v1/artifacts/B_confirmation_input_lineage_v1/input_frames.json'
        bound_input(frames,ctx);expected=json.loads(frames.read_text())['banks'][role['role']]
        assert [[r['index'],r['seed'],r['model_input_pixel_sha256']] for r in rows]==[[r['index'],r['seed'],r['model_input_pixel_sha256']] for r in expected]
    atomic_json(out/'report.json',dict(**ctx.binding,status='PASS_NATIVE_BASELINE',role=role['role'],expected_cases=count,
        parent_manifest_sha256=sha(bank/'manifest.json'),bank_role_sha256=sha(bank/'role.json'),model_binding=binding,
        head_sha256=sha(headroot/'weights.npz'),head_acceptance_sha256=sha(headroot/'acceptance.json'),
        rows=rows,exact_selected_head_input_pixel_overlaps=overlaps,baseline_metrics_sha256=sha(out/'baseline_metrics.npz'),
        elapsed_seconds=time.monotonic()-start,peak_allocated_bytes=torch.cuda.max_memory_allocated(),gpu=torch.cuda.get_device_name(),
        frozen_tensors_unchanged=True,source_sha256=sha(__file__),adapter_sha256=sha(ROOT/'strengthening/adapters/native_rollout.py'),
        scope='All fixed native episodes. Baseline/identity/channel/action-response checks; head/model remain fixed. Zero action response is retained. Pixel isolation is against inspected readout train/val inputs, not all pretraining images.'))
    (out/'DONE').write_text('native_development_baseline_accepted\n')


if __name__=='__main__':main()
