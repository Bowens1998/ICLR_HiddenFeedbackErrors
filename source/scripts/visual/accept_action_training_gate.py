"""Fixed-budget artifact checks and exact no-auxiliary trainer equivalence."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import torch
from action_auxiliary import ActionAuxiliary
from train_visual_updates import tensor_hash


def read(folder):
    r=json.loads((folder/'summary.json').read_text());assert (folder/'COMPLETE').exists()
    for f,h in json.loads((folder/'artifact_manifest.json').read_text()).items():assert hashlib.sha256((folder/f).read_bytes()).hexdigest()==h
    for f,h in json.loads((folder/'source/manifest.json').read_text()).items():assert hashlib.sha256((folder/'source'/f).read_bytes()).hexdigest()==h
    assert r['completed_updates']==r['expected_updates']==213 and r['validation_every']==210
    assert [(x['update_start'],x['update_end']) for x in r['evaluations']]==[(0,210),(210,213)]
    assert [x['train']['sequences'] for x in r['evaluations']]==[210*128,3*128]
    assert r['best_update']==min(r['evaluations'],key=lambda x:x['validation']['loss'])['update_end']
    return r

p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();root=Path(a.root);rows=[]
for arm in ['transformer_jepa','gru_jepa']:
    original=root/'original'/arm;base=read(original)
    for mode in ['none','inverse','inverse_goal']:
        folder=root/'auxiliary'/arm/mode;r=read(folder)
        for k in ['seed','clip_counts','normalization','data_manifest_sha256','config_sha256','parameters']:assert r[k]==base[k]
        assert all(r['initial_hashes'][k]==v for k,v in base['initial_hashes'].items())
        assert r['initial_hashes']['auxiliary']==tensor_hash(ActionAuxiliary()) and r['batch_size']==128
        for ev in r['evaluations']:
            for split in ['train','validation']:
                m=ev[split];expected=m['prediction_loss']+m['weighted_sigreg_loss']+m['weighted_auxiliary_loss']
                if mode=='none':assert m['weighted_auxiliary_loss']==0
                np.testing.assert_allclose(m['loss'],expected,rtol=1e-6,atol=1e-6)
        for ck in ['best','last']:
            head=ActionAuxiliary();head.load_state_dict(torch.load(folder/f'{ck}_auxiliary.pt',map_location='cpu',weights_only=True),strict=True)
            if mode=='none':
                assert tensor_hash(head)==r['initial_hashes']['auxiliary']
                left=torch.load(folder/f'{ck}_weights.pt',map_location='cpu',weights_only=True);right=torch.load(original/f'{ck}_weights.pt',map_location='cpu',weights_only=True)
                assert left.keys()==right.keys()
                for key in left:torch.testing.assert_close(left[key],right[key],rtol=0,atol=0,msg=lambda msg:f'{arm}/{ck}/{key}: {msg}')
        rows.append(dict(arm=arm,mode=mode,summary_sha256=hashlib.sha256((folder/'summary.json').read_bytes()).hexdigest()))
result=dict(rows=rows,no_auxiliary_checkpoints='bitwise identical to original trainer for both architectures and best/last',scope='213-update engineering gate only; no scientific outcome',verifier_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
(root/'acceptance.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
