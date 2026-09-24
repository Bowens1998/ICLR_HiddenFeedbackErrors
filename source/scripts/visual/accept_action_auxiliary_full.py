"""Accept fixed-budget auxiliary training and base-only recursive inference."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import torch
from factorial_model import make_model
from action_auxiliary import ActionAuxiliary
from train_visual_updates import tensor_hash,Clips
from evaluation_precision import configure_evaluation_precision


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

p=argparse.ArgumentParser()
for k in ['root','data','official','config']:p.add_argument('--'+k,required=True)
a=p.parse_args();root=Path(a.root);data=Path(a.data);torch.set_num_threads(2);precision=configure_evaluation_precision()
reports=[];dataset=Clips(data/'validation');pixels=torch.stack([dataset[i][0][:3] for i in [0,1]]).cuda().float()/255
pixels=(pixels-torch.tensor([.485,.456,.406],device='cuda')[None,None,:,None,None])/torch.tensor([.229,.224,.225],device='cuda')[None,None,:,None,None]
for arm in ['transformer_jepa','gru_jepa']:
    base=None
    for mode in ['none','inverse','inverse_goal']:
        folder=root/arm/mode;r=json.loads((folder/'summary.json').read_text());assert (folder/'COMPLETE').exists()
        for name,h in json.loads((folder/'artifact_manifest.json').read_text()).items():assert sha(folder/name)==h
        for name,h in json.loads((folder/'source/manifest.json').read_text()).items():assert sha(folder/'source'/name)==h
        assert (r['arm'],r['mode'],r['seed'],r['completed_updates'],r['expected_updates'],r['batch_size'],r['validation_every'])==(arm,mode,3072,21000,21000,128,210)
        assert r['data_manifest_sha256']==sha(data/'manifest.json') and r['config_sha256']==sha(Path(a.config))
        assert r['normalization']==json.loads((data/'normalization.json').read_text())['action']
        assert r['clip_counts']=={s:len(Clips(data/s)) for s in ['train','validation']}
        assert [(e['update_start'],e['update_end']) for e in r['evaluations']]==[(i,i+210) for i in range(0,21000,210)]
        assert r['best_update']==min(r['evaluations'],key=lambda e:e['validation']['loss'])['update_end']
        if base is None:base=r
        for k in ['initial_hashes','data_manifest_sha256','normalization','parameters','auxiliary_parameters']:assert r[k]==base[k]
        for e in r['evaluations']:
            for split in ['train','validation']:
                m=e[split];assert m['sequences']==(210*128 if split=='train' else r['clip_counts']['validation'])
                expected=m['prediction_loss']+m['weighted_sigreg_loss']+m['weighted_auxiliary_loss']
                if mode=='none':assert m['weighted_auxiliary_loss']==0
                np.testing.assert_allclose(m['loss'],expected,rtol=1e-6,atol=1e-6)
        for checkpoint in ['best','last']:
            model=make_model(a.official,a.config,arm,3072)
            for name,key in [('encoder','encoder'),('action_encoder','action_encoder'),('predictor','temporal_core')]:assert tensor_hash(getattr(model,name))==r['initial_hashes'][key]
            model.load_state_dict(torch.load(folder/f'{checkpoint}_weights.pt',map_location='cpu',weights_only=True),strict=True);model=model.cuda().eval()
            heads=ActionAuxiliary();heads.load_state_dict(torch.load(folder/f'{checkpoint}_auxiliary.pt',map_location='cpu',weights_only=True),strict=True)
            if mode=='none':assert tensor_hash(heads)==r['initial_hashes']['auxiliary']
            # Normalized zero actions: this is inference consistency, not a task result.
            actions=torch.zeros(2,7,10,device='cuda')
            with torch.inference_mode():
                native=model.rollout({'pixels':pixels[:,None]},actions[:,None])['predicted_emb'][:,0,-1]
                encoded=model.encode({'pixels':pixels,'action':actions});history=encoded['emb'];act=encoded['act_emb']
                for step in range(2,7):history=torch.cat([history,model.predict(history[:,-3:],act[:,step-2:step+1])[:,-1:]],1)
                torch.testing.assert_close(native,history[:,-1],rtol=2e-5,atol=2e-5)
            del model,heads;torch.cuda.empty_cache()
        reports.append(dict(arm=arm,mode=mode,summary_sha256=sha(folder/'summary.json')))
result=dict(rows=reports,updates=21000,precision=precision,base_only_recursive_rollout='native/manual checked for all12 checkpoints',scope='training and inference acceptance; not planning performance',verifier_sha256=sha(Path(__file__)))
(root/'acceptance.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
