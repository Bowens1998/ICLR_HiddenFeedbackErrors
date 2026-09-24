"""Verify training provenance, paired initialization and native recursive interfaces."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import torch
from latent_state_model import ARMS,make_model
from train_latent_state import tensor_hash

def main():
    p=argparse.ArgumentParser()
    for key in ['run','official','config','data']:p.add_argument('--'+key,required=True)
    p.add_argument('--updates',type=int,required=True);p.add_argument('--validation-every',type=int,default=210);a=p.parse_args();torch.set_num_threads(2);run=Path(a.run);rows=[]
    data_hash=hashlib.sha256((Path(a.data)/'manifest.json').read_bytes()).hexdigest()
    for arm in ARMS:
        out=run/arm;assert (out/'COMPLETE').exists();r=json.loads((out/'summary.json').read_text());assert r['arm']==arm and r['data_manifest_sha256']==data_hash
        expected=[(v,min(v+a.validation_every,a.updates)) for v in range(0,a.updates,a.validation_every)]
        assert r['completed_updates']==r['expected_updates']==a.updates
        assert r['validation_every']==a.validation_every and len(r['evaluations'])==len(expected)
        for j,(row,(begin,end)) in enumerate(zip(r['evaluations'],expected)):
            assert row['evaluation']==j+1 and (row['update_start'],row['update_end'])==(begin,end)
            assert row['train']['sequences']==128*(end-begin)
            assert row['validation']['sequences']==r['clip_counts']['validation']
            for split in ['train','validation']:
                assert all(np.isfinite(row[split][k]) for k in ['loss','prediction_loss','observation_loss','sigreg_loss'])
        assert r['best_evaluation']==1+int(np.argmin([v['validation']['loss'] for v in r['evaluations']]))
        assert r['best_update']==r['evaluations'][r['best_evaluation']-1]['update_end']
        assert r['evaluations'][-1]['lr_last']==0
        for filename,h in json.loads((out/'artifact_manifest.json').read_text()).items():assert hashlib.sha256((out/filename).read_bytes()).hexdigest()==h,filename
        for filename,h in json.loads((out/'source/manifest.json').read_text()).items():assert hashlib.sha256((out/'source'/filename).read_bytes()).hexdigest()==h,filename
        model=make_model(a.official,a.config,arm,r['seed'])
        for key,module in [('encoder',model.encoder),('action_encoder',model.action_encoder),('temporal_core',model.predictor)]:assert tensor_hash(module)==r['initial_hashes'][key]
        model.load_state_dict(torch.load(out/'best_weights.pt',map_location='cpu',weights_only=True),strict=True);model.eval()
        # Full native rollout and a separately indexed recursion must agree in both token spaces.
        g=torch.Generator().manual_seed(940001);pixels=torch.rand(1,1,3,3,224,224,generator=g).expand(1,2,-1,-1,-1,-1);act=torch.randn(1,2,7,10,generator=g)
        with torch.inference_mode():
            native=model.rollout({'pixels':pixels},act)['predicted_emb'][0]
            emb=model.encode({'pixels':pixels[:,0]})['emb'].expand(2,-1,-1).clone()
            for k in range(2,7):emb=torch.cat([emb,model.predict(emb[:,-3:],model.action_encoder(act[0,:,k-2:k+1]))[:,-1:]],1)
            torch.testing.assert_close(native,emb,rtol=2e-5,atol=2e-5);assert torch.isfinite(native).all()
            assert native.shape==(2,8,192)
            decoded=model.state_head(native);assert decoded.shape==(2,8,6) and torch.isfinite(decoded).all()
        model.load_state_dict(torch.load(out/'last_weights.pt',map_location='cpu',weights_only=True),strict=True)
        rows.append(r)
    assert len({r['initial_hashes']['encoder'] for r in rows})==1
    assert len({r['initial_hashes']['action_encoder'] for r in rows})==1
    for prefix in ['gru','transformer']:assert len({r['initial_hashes']['temporal_core'] for r in rows if r['arm'].startswith(prefix)})==1
    assert len({json.dumps(r['target_normalization'],sort_keys=True) for r in rows})==1
    result={'accepted_arms':ARMS,'updates':a.updates,'validation_every':a.validation_every,'shared_initializations':'encoder, action encoder; temporal core within architecture verified',
            'strict_best_and_last_load':'passed','native_recursive_alignment':'passed','scope':'training and interface acceptance; task performance requires separate saved-prediction rescoring'}
    (run/'acceptance.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
