"""Native input-only isolation and preprocessing inventory; no world model is loaded."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import sys
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'strengthening/adapters'),str(ROOT/'strengthening/external/dino_wm')]
from contracts import sha,atomic_json
from input_lock import add_design_arguments,read_design,check_parent_design,verify_environment
from datasets.img_transforms import default_transform


def signature(x):
    return hashlib.sha256(np.ascontiguousarray(x,dtype=np.float32).tobytes()).hexdigest()


def main():
    p=argparse.ArgumentParser();add_design_arguments(p)
    for k in ['native-audit','head-cache','output']:p.add_argument('--'+k,required=True)
    a=p.parse_args();design=read_design(a.design_lock,a.design_sha256,ROOT)
    verify_environment(design,'native-dinowm-v1');torch.set_num_threads(1)
    binding=dict(scientific_design_sha256=a.design_sha256,phase='confirmation_input_isolation')
    release=Path(design['base'])/'releases/feedback-strengthening-v1'
    previous=release/'artifacts/B_development_lineage_v1/report.json'
    assert sha(previous)==design['verified_assets'][str(previous)];previous=json.loads(previous.read_text())
    audit=Path(a.native_audit);old=json.loads((audit/'report.json').read_text())
    assert sha(audit/'report.json')==previous['native_metadata_audit_sha256']
    datasets={}
    for split in ['train','val']:
        row=old['splits'][split];root=Path(row['root'])
        assert sha(root/'states.pth')==row['metadata_sha256']['states.pth']
        states=torch.load(root/'states.pth',map_location='cpu',weights_only=True).numpy()
        datasets[split]=dict(initial_states={signature(x[0,:5]):i for i,x in enumerate(states)},
            first_36_state_paths={signature(x[:36,:5]):i for i,x in enumerate(states)})
        del states
    headcache=Path(a.head_cache);head=Path(design['native_head']).parent
    hr=json.loads((head/'report.json').read_text());assert sha(head/'report.json')==design['verified_assets'][str(head/'report.json')]
    assert hr['cache_report_sha256']==sha(headcache/'report.json')
    pixels={};head_bindings={}
    for split in ['head_train','head_validation']:
        meta=json.loads((headcache/(split+'.json')).read_text());rp=headcache/(split+'_rows.json')
        assert sha(rp)==meta['row_manifest_sha256']
        pixels[split]={x['model_input_pixel_sha256'] for x in json.loads(rp.read_text())['rows']}
        head_bindings[split]=sha(rp)
    banks={r:Path(design['banks'][r]['output']) for r in ['confirmation_B','donor_bank_B']}
    banks.update(diagnostic_development_B=release/'artifacts/B_development_bank_v2',
                 donor_development_B=release/'artifacts/B_donor_development_bank_v2')
    content={};bindings={};dataset_hits=[];head_hits=[];frame_rows={}
    transform=default_transform(224)
    for role,bank in banks.items():
        m=json.loads((bank/'manifest.json').read_text());rm=json.loads((bank/'role.json').read_text())
        assert (bank/'DONE').exists() and rm['parent_manifest_sha256']==sha(bank/'manifest.json') and rm['role']==role
        formal=role in design['banks'];count=256 if formal else 64
        assert len(m['cases'])==count and [c['index'] for c in m['cases']]==list(range(count))
        if formal:check_parent_design(rm,binding)
        else:
            key='recipient' if role=='diagnostic_development_B' else 'donor'
            assert sha(bank/'manifest.json')==previous['bindings'][key]['bank_manifest_sha256']
        fields={k:set() for k in ['seeds','initial_states','history_states','state_paths','history_pixels','input_pixels']}
        frame_rows[role]=[]
        for c in m['cases']:
            path=bank/c['file'];assert sha(path)==c['sha256']
            with np.load(path) as z:
                first=signature(z['states'][0,:5]);full=signature(z['states'][:,:5])
                fields['seeds'].add(c['seed']);fields['initial_states'].add(first);fields['state_paths'].add(full)
                fields['history_states'].add(signature(z['states'][:11,:5]))
                fields['history_pixels'].add(hashlib.sha256(z['pixels'][:3].tobytes()).hexdigest())
                ims=transform(torch.as_tensor(z['pixels']).permute(0,3,1,2).float()/255.)
                hashes=[hashlib.sha256(im.contiguous().numpy().tobytes()).hexdigest() for im in ims]
                fields['input_pixels'].update(hashes)
                frame_rows[role].append(dict(index=c['index'],seed=c['seed'],input_sha256=sha(path),model_input_pixel_sha256=hashes))
                if formal:
                    for split,lookup in datasets.items():
                        for key,value in [('initial_states',first),('first_36_state_paths',full)]:
                            if value in lookup[key]:dataset_hits.append(dict(role=role,case=c['index'],split=split,kind=key,episode=lookup[key][value]))
                    for j,h in enumerate(hashes):
                        for split,lookup in pixels.items():
                            if h in lookup:head_hits.append(dict(role=role,case=c['index'],frame=5*j,split=split,pixel_sha256=h))
        content[role]=fields;bindings[role]=dict(bank_manifest_sha256=sha(bank/'manifest.json'),role_sha256=sha(bank/'role.json'))
    pairs=[];blocked=bool(dataset_hits)
    for left,right in itertools.combinations(content,2):
        if left not in design['banks'] and right not in design['banks']:continue
        matches={k:sorted(content[left][k]&content[right][k]) for k in content[left]}
        blocked |= any(matches[k] for k in ['seeds','initial_states','history_states','state_paths','history_pixels'])
        pairs.append(dict(left=left,right=right,intersections=matches))
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    atomic_json(out/'input_frames.json',dict(**binding,banks=frame_rows,
        transform_source_sha256=sha(ROOT/'strengthening/external/dino_wm/datasets/img_transforms.py')))
    atomic_json(out/'report.json',dict(**binding,status='BLOCKED_NATIVE_INPUT_PARENT_OVERLAP' if blocked else 'PASS_NATIVE_CONFIRMATION_INPUT_ISOLATION',
        bindings=bindings,pairwise=pairs,released_dataset_state_matches=dataset_hits,head_input_pixel_matches=head_hits,
        head_row_manifests=head_bindings,native_metadata_audit_sha256=sha(audit/'report.json'),
        input_frames_sha256=sha(out/'input_frames.json'),source_sha256=sha(__file__),models_loaded=0,
        scope='Original dataset initial state and first36 state-path, both independent development banks, and inspected readout images. Preprocessing only; no model prediction or intervention effect.',
        unresolved=['All upstream seeds','All interior state windows','All foundation-pretraining images']))
    if blocked:raise ValueError('Parent overlap; preserve all cases and stop, no reseeding')
    (out/'DONE').write_text('input_isolation_verified\n')


if __name__=='__main__':main()
