"""Independently reconstruct all identities/labels and first-batch native tokens."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from factorial_model import make_model
from evaluation_precision import configure_evaluation_precision


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser()
    for k in ['run','manifest','training','data','official','config']:p.add_argument('--'+k,required=True)
    a=p.parse_args();out=Path(a.run);r=json.loads((out/'report.json').read_text())
    assert (out/'COMPLETE').exists() and r['protocol']=='pusht_nonlinear_transfer_features_v1'
    m=json.loads(Path(a.manifest).read_text());assert m['layout']=='action_auxiliary'
    assert sha(a.manifest)==r['manifest_sha256']
    rep,arm=r['replica'],r['arm'];assert r['index']==rep*2+['transformer_jepa','gru_jepa'].index(arm)
    entries=[e for e in m['models'] if (e['replica'],e['arm'],e['mode'],e['checkpoint'])==(rep,arm,'none','last')]
    assert len(entries)==1;e=entries[0];training=Path(a.training)/e['training_path']
    assert r['mode']=='none' and r['checkpoint']=='last'
    assert sha(training/'summary.json')==e['training_summary_sha256']==r['training_summary_sha256']
    assert sha(training/'last_weights.pt')==e['weights_sha256']==r['weights_sha256']
    tr=json.loads((training/'summary.json').read_text())
    assert (tr['completed_updates'],tr['seed'],tr['arm'],tr['mode'])==(21000,3072,arm,'none')
    assert sha(a.config)==tr['config_sha256']==r['config_sha256']
    for name,h in r['source_sha256'].items():
        path=Path(a.official)/name if name in ['jepa.py','module.py'] else Path(__file__).with_name(name)
        assert sha(path)==h,name
    data=Path(a.data)/f'replica_{rep}'/'n256';dm=json.loads((data/'manifest.json').read_text())
    assert sha(data/'manifest.json')==r['data_manifest_sha256']==e['data_manifest_sha256']==tr['data_manifest_sha256']
    assert set(dm['splits']['train']['episode_ids']).isdisjoint(dm['splits']['validation']['episode_ids'])
    assert sha(data/'normalization.json')==dm['normalization_sha256']
    assert json.loads((data/'normalization.json').read_text())['action']==tr['normalization']
    precision=configure_evaluation_precision();assert precision==r['precision'];torch.set_num_threads(2)
    model=make_model(a.official,a.config,arm,tr['seed'])
    model.load_state_dict(torch.load(training/'last_weights.pt',map_location='cpu',weights_only=True),strict=True)
    model=model.cuda().eval();rows=[]
    for split in ['train','validation']:
        folder=data/split
        for name in ['pixels.npy','action.npy','state.npy','episodes.npz']:
            assert sha(folder/name)==dm['splits'][split]['files'][name]
        assert sha(out/f'{split}_features.npz')==r['files_sha256'][f'{split}_features.npz']
        z=np.load(out/f'{split}_features.npz');episodes=np.load(folder/'episodes.npz')
        ids=episodes['source_episode_ids'];lengths=episodes['lengths'];offsets=episodes['offsets']
        np.testing.assert_array_equal(ids,dm['splits'][split]['episode_ids'])
        np.testing.assert_array_equal(offsets,np.r_[0,np.cumsum(lengths)[:-1]])
        identities=[];starts=[]
        for ep,n,offset in zip(ids,lengths,offsets):
            for t in range(int(n)):
                if t%5==0 and t+35<int(n):identities.append([ep,t]);starts.append(int(offset)+t)
        if r['engineering']:identities=identities[:64];starts=starts[:64]
        n=len(starts);assert n==r['counts'][split]
        assert set(z.files)=={'encoded','predicted','target','identity'}
        for key,dim in [('encoded',192),('predicted',192),('target',6),('identity',2)]:
            assert z[key].shape==(n,dim) and np.isfinite(z[key]).all()
        assert z['encoded'].dtype==z['predicted'].dtype==np.float32
        np.testing.assert_array_equal(z['identity'],identities)
        state=np.load(folder/'state.npy',mmap_mode='r');pixels=np.load(folder/'pixels.npy',mmap_mode='r');actions=np.load(folder/'action.npy',mmap_mode='r')
        assert len(state)==len(pixels)==len(actions)==int(lengths.sum())
        raw=np.asarray(state[np.array(starts)+35],dtype=np.float64)
        targets=np.column_stack([raw[:,:4],np.sin(raw[:,4]),np.cos(raw[:,4])])
        np.testing.assert_array_equal(z['target'],targets)
        sample=starts[:64]
        ims=np.stack([pixels[s+np.array([0,5,10,35])] for s in sample])
        ac=np.stack([actions[s:s+35] for s in sample]).astype(np.float32)
        x=torch.tensor(ims,device='cuda').permute(0,1,4,2,3).float()/255
        x=(x-torch.tensor([.485,.456,.406],device='cuda')[None,None,:,None,None])/torch.tensor([.229,.224,.225],device='cuda')[None,None,:,None,None]
        ac=torch.tensor(ac,device='cuda').reshape(len(sample),7,10)
        ac=(ac-torch.tensor(tr['normalization']['mean'],device='cuda').repeat(5))/torch.tensor(tr['normalization']['std'],device='cuda').repeat(5)
        with torch.inference_mode():
            encoded=model.encode({'pixels':x[:,3:]})['emb'][:,0]
            predicted=model.rollout({'pixels':x[:,None,:3]},ac[:,None])['predicted_emb'][:,0,-1]
        errors={}
        for key,value in [('encoded',encoded),('predicted',predicted)]:
            value=value.cpu().numpy();np.testing.assert_allclose(value,z[key][:len(sample)],rtol=2e-5,atol=2e-5)
            errors[key]=float(np.max(np.abs(value-z[key][:len(sample)])))
        rows.append(dict(split=split,all_identities_and_targets=n,native_sample=len(sample),max_abs=errors))
    result=dict(status='PASS',report_sha256=sha(out/'report.json'),verifier_sha256=sha(__file__),engineering=r['engineering'],rows=rows,
                scope='All file hashes, all identities/labels, first64 encoded and native predicted tokens per split. Not a replay of every feature or a planning result.')
    (out/'acceptance.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':main()
