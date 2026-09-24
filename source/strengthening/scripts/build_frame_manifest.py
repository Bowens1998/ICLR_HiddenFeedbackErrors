"""CPU-only whole-file hash, frame deduplication, and explicit role manifests."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'adapters'))
from contracts import atomic_json, sha


def digest(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--base',required=True);p.add_argument('--output',required=True);p.add_argument('--pool',type=int)
    p.add_argument('--overlap-policy',choices=['fail','exclude_training'],default='fail')
    a=p.parse_args();pool=int(os.environ['SLURM_ARRAY_TASK_ID']) if a.pool is None else a.pool
    base=Path(a.base);r=base/'releases/planner-data-adaptation-v1/runs';data=base/f'datasets/pusht-scaling-v1/replica_{pool}/n256'
    out=Path(a.output)/f'pool_{pool}';out.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    dm=json.loads((data/'manifest.json').read_text());expert={};bindings=[]
    for split in ['train','validation']:
        for name,expected in dm['splits'][split]['files'].items():
            fp=data/split/name;actual=sha(fp);assert actual==expected,(fp,actual,expected)
            bindings.append(dict(path=str(fp),sha256=actual))
        pixels=np.load(data/split/'pixels.npy',mmap_mode='r');states=np.load(data/split/'state.npy',mmap_mode='r')
        ep=np.load(data/split/'episodes.npz');rows=[]
        for offset,length,episode in zip(ep['offsets'],ep['lengths'],ep['source_episode_ids']):
            for frame in range(int(length)):
                i=int(offset)+frame
                rows.append(dict(domain='expert',parent_episode=int(episode),frame=frame,absolute_index=i,
                    pixels_path=str(data/split/'pixels.npy'),states_path=str(data/split/'state.npy'),
                    pixel_sha256=digest(pixels[i]),state_sha256=digest(states[i])))
        expert[split]=rows
    summaries=[]
    for g in [2*pool,2*pool+1]:
        unique={};all_rows={};duplicates={}
        for split in ['train','validation']:
            rows=list(expert[split])
            for route in [8*g+2,8*g+3]:
                folder=r/f'trajectory_{split}/job_{route}';report=json.loads((folder/'report.json').read_text())
                assert report['status']=='PASS_REPLAYED_COMPLETE_SELECTED_TRAJECTORIES' and report['split']==split
                bindings.append(dict(path=str(folder/'report.json'),sha256=sha(folder/'report.json')))
                for case in report['cases']:
                    fp=folder/case['file'];actual=sha(fp);assert actual==case['file_sha256']
                    bindings.append(dict(path=str(fp),sha256=actual))
                    with np.load(fp) as z:
                        assert z['pixels'].shape==(8,224,224,3) and z['states'].shape==(36,7)
                        for j in range(8):
                            rows.append(dict(domain='planner',route=route,goal_index=case['index'],seed=case['seed'],
                                frame=5*j,pixel_index=j,archive_path=str(fp),archive_sha256=actual,
                                pixel_sha256=digest(z['pixels'][j]),state_sha256=digest(z['states'][5*j])))
            first={};dups=[]
            for row in rows:
                h=row['pixel_sha256']
                if h not in first:first[h]=row
                else:dups.append(dict(pixel_sha256=h,retained_domain=first[h]['domain'],duplicate_domain=row['domain'],
                                      state_payload_equal=first[h]['state_sha256']==row['state_sha256']))
            unique[split]=first;all_rows[split]=rows;duplicates[split]=dups
        overlap=sorted(set(unique['train'])&set(unique['validation']))
        exclusions=[dict(training=unique['train'][h],validation=unique['validation'][h]) for h in overlap]
        if overlap and a.overlap_policy=='exclude_training':
            for h in overlap:del unique['train'][h]
        report=dict(group=g,counts={s:dict(raw_frames=len(all_rows[s]),unique_frames=len(unique[s]),duplicates=len(duplicates[s]),
            by_domain={domain:sum(row['domain']==domain for row in unique[s].values()) for domain in ['expert','planner']}) for s in unique},
            train_validation_exact_pixel_overlap_before_exclusion=overlap,
            train_validation_exact_pixel_overlap=sorted(set(unique['train'])&set(unique['validation'])),
            overlap_policy=a.overlap_policy,excluded_training_frames=exclusions,
            duplicate_rule='Keep first observed frame in deterministic expert-then-planner order; no label averaging',
            duplicates=duplicates)
        atomic_json(out/f'group_{g}_deduplication.json',report)
        if overlap and a.overlap_policy=='fail':
            atomic_json(out/'BLOCKED.json',dict(reason='Exact train/validation pixel overlap',group=g,count=len(overlap)))
            raise ValueError('Exact training/validation frame overlap; no frame manifests released')
        for split in ['train','validation']:
            atomic_json(out/f'group_{g}_{split}.json',dict(role='head_'+split,group=g,pool=pool,
                parent_manifest_sha256=sha(data/'manifest.json'),rows=list(unique[split].values()),
                source_sha256=sha(__file__),deduplication_sha256=sha(out/f'group_{g}_deduplication.json'),
                scope='Observed raw images only; no prediction tokens or confirmation trajectories'))
        summaries.append({k:v for k,v in report.items() if k!='duplicates'})
    atomic_json(out/'report.json',dict(status='PASS_FULL_RAW_FRAME_SPLIT_HASHES',pool=pool,groups=summaries,bindings=bindings,
        elapsed_seconds=time.monotonic()-start,source_sha256=sha(__file__)))
    (out/'DONE').write_text('accepted\n')


if __name__=='__main__':main()
