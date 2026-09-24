"""Frozen observed-image encodings for A and continuous six-frame C segments."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'scripts/visual'),str(ROOT/'strengthening/adapters')]
from contracts import atomic_json, sha, require_role
from factorial_model import make_model
from adaptation_freeze import configure_dynamics_only, verify_frozen
from evaluation_precision import configure_evaluation_precision
from score_feedback_ranking import state_hash


def main():
    p=argparse.ArgumentParser()
    for k in ['base','manifests','output']:p.add_argument('--'+k,required=True)
    p.add_argument('--group',type=int);a=p.parse_args()
    g=int(os.environ['SLURM_ARRAY_TASK_ID']) if a.group is None else a.group
    base=Path(a.base);r=base/'releases/planner-data-adaptation-v1/runs';md=Path(a.manifests)/f'pool_{g//2}'
    if not (md/'DONE').exists():raise ValueError('Frame lineage not accepted')
    plan=json.loads((r/'fiber_confirmation_plan.json').read_text());e=plan['models'][8*g]
    td=Path(e['training_path']);tr=json.loads((td/'summary.json').read_text())
    assert sha(td/'last_weights.pt')==e['weights_sha256'] and sha(td/'summary.json')==e['training_summary_sha256']
    config=base/'assets/pusht-v1/models/config.json';assert sha(config)==tr['config_sha256']
    precision=configure_evaluation_precision();torch.set_num_threads(2)
    model=make_model(base/'releases/visual-v1/official',config,e['arm'],tr['seed'])
    model.load_state_dict(torch.load(td/'last_weights.pt',map_location='cpu',weights_only=True),strict=True)
    model=model.cuda();boundary=configure_dynamics_only(model);before=state_hash(model)
    out=Path(a.output)/f'group_{g}';out.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    im=torch.tensor([.485,.456,.406],device='cuda')[None,None,:,None,None]
    sd=torch.tensor([.229,.224,.225],device='cuda')[None,None,:,None,None]
    def encode(pixels):
        x=torch.as_tensor(np.stack(pixels),device='cuda').permute(0,3,1,2)[None].float()/255
        with torch.inference_mode():return model.encode({'pixels':(x-im)/sd})['emb'][0].cpu().numpy()
    manifests={};summaries=[];arrays={};archives={}
    for split in ['train','validation']:
        path=md/f'group_{g}_{split}.json';m=json.loads(path.read_text());require_role(m,{'head_'+split});manifests[split]=sha(path)
        assert m['group']==g
        frames=[];states=[];latent=[];domains=[];pending=[]
        for row in m['rows']:
            if row['domain']=='expert':
                for key in ['pixels_path','states_path']:
                    if row[key] not in arrays:arrays[row[key]]=np.load(row[key],mmap_mode='r')
                pix=arrays[row['pixels_path']][row['absolute_index']];state=arrays[row['states_path']][row['absolute_index']]
            else:
                fp=row['archive_path']
                if fp not in archives:
                    if sha(fp)!=row['archive_sha256']:raise ValueError('Planner archive changed')
                    with np.load(fp) as z:archives={fp:{k:z[k].copy() for k in ['pixels','states']}}
                pix=archives[fp]['pixels'][row['pixel_index']];state=archives[fp]['states'][row['frame']]
            assert hashlib.sha256(np.ascontiguousarray(pix).tobytes()).hexdigest()==row['pixel_sha256']
            assert hashlib.sha256(np.ascontiguousarray(state).tobytes()).hexdigest()==row['state_sha256']
            pending.append(np.array(pix));states.append(np.array(state));frames.append(row['pixel_sha256']);domains.append(row['domain'])
            if len(pending)==4:latent.extend(encode(pending));pending=[]
        if pending:
            n=len(pending);latent.extend(encode(pending+[pending[-1]]*(4-n))[:n])
        observed=np.asarray(latent);states=np.asarray(states);labels=np.c_[states[:,:4],np.sin(states[:,4]),np.cos(states[:,4])]
        assert observed.shape==(len(m['rows']),192) and np.isfinite(observed).all()
        fp=out/f'head_{split}.npz';np.savez_compressed(fp,observed=observed,labels=labels,domain=np.array(domains),pixel_sha256=np.array(frames))
        atomic_json(out/f'head_{split}.json',dict(role='head_'+split,group=g,parent_manifest_sha256=sha(path),file_sha256=sha(fp),
            rows=len(observed),encoder_checkpoint_sha256=e['weights_sha256'],encoder_call='one sequence of four images; last partial padded then trimmed'))
        summaries.append(dict(role='head_'+split,rows=len(observed),bytes=fp.stat().st_size,sha256=sha(fp)))
    # C is restricted to Transformer groups; raw trajectories supply all six images.
    if g%2==0:
        am=np.tile(np.asarray(tr['normalization']['mean'],np.float32),5)
        ast=np.tile(np.asarray(tr['normalization']['std'],np.float32),5)
        for split in ['train','validation']:
            obs=[];actions=[];states=[];identities=[];parents=[]
            for route in [8*g+2,8*g+3]:
                folder=r/f'trajectory_{split}/job_{route}';report=json.loads((folder/'report.json').read_text());parents.append(sha(folder/'report.json'))
                for case in report['cases']:
                    fp=folder/case['file'];assert sha(fp)==case['file_sha256']
                    with np.load(fp) as z:
                        for t in [0,5,10]:
                            pix=z['pixels'][t//5:t//5+6];assert len(pix)==6
                            obs.append(encode(pix));actions.append((z['actions'][t:t+25].astype(np.float32).reshape(5,10)-am)/ast)
                            states.append(z['states'][t+np.arange(6)*5]);identities.append([route,case['index'],case['seed'],t])
            fp=out/f'C_{split}.npz';np.savez_compressed(fp,observed=np.array(obs),normalized_actions=np.array(actions),raw_states=np.array(states),identities=np.array(identities))
            atomic_json(out/f'C_{split}.json',dict(role='continuation_'+split,group=g,
                parent_manifest_sha256=hashlib.sha256(json.dumps(parents).encode()).hexdigest(),trajectory_report_hashes=parents,
                file_sha256=sha(fp),rows=len(obs),encoder_call='one continuous sequence of six images',
                action_normalization=tr['normalization'],source_sha256=sha(__file__)))
    verify_frozen(model,boundary);assert state_hash(model)==before
    atomic_json(out/'report.json',dict(status='PASS_OBSERVED_CACHE',group=g,head_data=summaries,frame_manifests=manifests,
        initial_state_hash=before,frozen_unchanged=True,checkpoint_sha256=e['weights_sha256'],precision=precision,
        elapsed_seconds=time.monotonic()-start,gpu=torch.cuda.get_device_name(),peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        source_sha256=sha(__file__),scope='Frozen encoder observed inputs only; no head fitting or confirmation'))
    (out/'DONE').write_text('accepted\n')


if __name__=='__main__':main()
