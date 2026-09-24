"""Deterministic raw-window iterators for the frozen formal continuation protocol."""
import json, hashlib
from pathlib import Path
import numpy as np
from adaptation_windows import starts, dense_window, recorded_window

def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):digest.update(block)
    return digest.hexdigest()



def expert_windows(data, freeze, replica, split):
    """Yield (identity, images/actions/states), checking episode boundaries first."""
    assert split in ('train','validation')
    data=Path(data)/f'replica_{replica}'/'n256';manifest=json.loads((data/'manifest.json').read_text())
    freeze=Path(freeze);report=json.loads((freeze/'report.json').read_text());pool=report['pools'][replica]
    assert pool['replica']==replica and pool['data_manifest_sha256']==sha(data/'manifest.json')
    for name in ('pixels.npy','state.npy','action.npy','episodes.npz'):
        assert sha(data/split/name)==manifest['splits'][split]['files'][name]
    ep=np.load(data/split/'episodes.npz');ar={k:np.load(data/split/f'{k}.npy',mmap_mode='r') for k in ('pixels','state','action')}
    assert len(ep['offsets'])==len(ep['lengths'])==len(ep['source_episode_ids'])=={'train':256,'validation':64}[split]
    indices=[];lookup={}
    for offset,length,identity in zip(ep['offsets'],ep['lengths'],ep['source_episode_ids']):
        offset,length,identity=int(offset),int(length),int(identity)
        assert identity not in lookup;lookup[identity]=(offset,length)
        for start in starts(length,length):indices.append([identity,start,offset+start])
    if split=='train':
        assert len(indices)==pool['available_windows']
        selected=np.asarray(indices,dtype=np.int64)[pool['selected_indices']].tolist()
        assert selected==pool['identities'];indices=selected
    for identity,start,absolute in indices:
        offset,length=lookup[identity];assert absolute==offset+start
        yield [identity,start],dense_window(ar['pixels'][offset:offset+length],ar['state'][offset:offset+length],ar['action'][offset:offset+length],start)


def planner_windows(trajectories, plan, route):
    trajectories=Path(trajectories);plan=Path(plan);p=json.loads(plan.read_text());split=p['adaptation_formal']['split']
    assert route in p['adaptation_formal']['execute_routes'] and route%8 in (2,6)
    count={'train':128,'validation':32}[split]
    for rid in (route,route+1):
        folder=trajectories/f'job_{rid}';report=json.loads((folder/'report.json').read_text())
        assert report['status']=='PASS_REPLAYED_COMPLETE_SELECTED_TRAJECTORIES' and report['route']==rid
        assert report['split']==split and report['plan_sha256']==sha(plan) and len(report['cases'])==count
        for i,c in enumerate(report['cases']):
            assert c['index']==i;fp=folder/c['file'];assert sha(fp)==c['file_sha256']
            with np.load(fp) as z:
                assert int(z['seed'])==c['seed'] and starts(len(z['states']),len(z['actions']))==[0,5,10,15,20]
                for start in (0,5,10,15,20):
                    yield [rid,i,c['seed'],start],recorded_window(z['pixels'],z['states'],z['actions'],start)
