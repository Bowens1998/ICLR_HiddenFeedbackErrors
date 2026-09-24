"""Independently verify nested sample provenance and materialized data."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np

def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        while c:=f.read(8*1024*1024):h.update(c)
    return h.hexdigest()

p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();root=Path(a.root);s=json.loads((root/'selection.json').read_text());rep=s['replica'];seen=set()
for pool in s['all_pools']:
    small,large=set(pool['256']),set(pool['1024']);assert len(small)==256 and len(large)==1024 and small<=large
    assert not large&(seen|set(s['excluded_pilot_episode_ids']));seen|=large
arrays={};episodes={};validation=None
for n in [256,1024]:
    d=root/f'n{n}';m=json.loads((d/'manifest.json').read_text());assert m['selection_sha256']==digest(root/'selection.json')
    ids=s['all_pools'][rep][str(n)];assert m['splits']['train']['episode_ids']==ids
    e=np.load(d/'train/episodes.npz');np.testing.assert_array_equal(e['source_episode_ids'],ids);assert (e['lengths']>=40).all();np.testing.assert_array_equal(e['offsets'],np.r_[0,np.cumsum(e['lengths'])[:-1]])
    assert int(e['lengths'].sum())==m['splits']['train']['frames'];episodes[n]={int(i):(int(o),int(l)) for i,o,l in zip(e['source_episode_ids'],e['offsets'],e['lengths'])}
    for f,h in m['splits']['train']['files'].items():assert digest(d/'train'/f)==h
    assert digest(d/'normalization.json')==m['normalization_sha256']
    if validation is None:
        for f,h in m['splits']['validation']['files'].items():assert digest(d/'validation'/f)==h
        validation=m['splits']['validation']
    else:assert m['splits']['validation']==validation
    assert not set(ids)&set(validation['episode_ids'])
    arrays[n]={k:np.load(d/'train'/f'{k}.npy',mmap_mode='r') for k in ['pixels','action','state','proprio']}
    norm=json.loads((d/'normalization.json').read_text())
    for k in ['action','state','proprio']:
        v=np.array(arrays[n][k],dtype=np.float64);assert np.isfinite(v).all()
        np.testing.assert_allclose(v.mean(0),norm[k]['mean'],rtol=1e-12,atol=1e-12);np.testing.assert_allclose(np.maximum(v.std(0,ddof=1),1e-6),norm[k]['std'],rtol=1e-12,atol=1e-12)
for ep,(o,l) in episodes[256].items():
    q,ll=episodes[1024][ep];assert l==ll
    for k in arrays[256]:np.testing.assert_array_equal(arrays[256][k][o:o+l],arrays[1024][k][q:q+l])
r={'replica':rep,'all_three_training_pools_disjoint':True,'prior_pilot_excluded':True,'nested_episode_arrays_exact':256,'all_materialized_file_hashes_and_train_normalization':'verified','shared_validation_hashes':'verified','scope':'materialization consistency, not scientific outcome confirmation'}
(root/'acceptance.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r),flush=True)
