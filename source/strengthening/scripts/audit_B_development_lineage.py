"""Native input-only parent checks against released dataset metadata and inspected head images."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'strengthening/adapters'))
from contracts import sha,atomic_json


def signature(x):return hashlib.sha256(np.ascontiguousarray(x,dtype=np.float32).tobytes()).hexdigest()


def main():
    p=argparse.ArgumentParser()
    for k in ['recipient','donor','recipient-cache','donor-cache','native-audit','output']:p.add_argument('--'+k,required=True)
    a=p.parse_args();audit=Path(a.native_audit);old=json.loads((audit/'report.json').read_text())
    datasets={}
    for split in ['train','val']:
        row=old['splits'][split];root=Path(row['root'])
        assert sha(root/'states.pth')==row['metadata_sha256']['states.pth']
        states=torch.load(root/'states.pth',map_location='cpu',weights_only=True).numpy()
        datasets[split]=dict(initial_states={signature(x[0,:5]):i for i,x in enumerate(states)},
            first_36_state_paths={signature(x[:36,:5]):i for i,x in enumerate(states)})
    content={};bindings={};dataset_hits=[];head_hits=[]
    for role,bank,cache in [('recipient',Path(a.recipient),Path(a.recipient_cache)),('donor',Path(a.donor),Path(a.donor_cache))]:
        m=json.loads((bank/'manifest.json').read_text());r=json.loads((cache/'report.json').read_text())
        assert (bank/'DONE').exists() and (cache/'DONE').exists() and len(m['cases'])==len(r['rows'])==64
        assert r['parent_manifest_sha256']==sha(bank/'manifest.json')
        content[role]=dict(seeds=set(),initial_states=set(),history_states=set(),state_paths=set(),history_pixels=set(),input_pixels=set())
        for c,cr in zip(m['cases'],r['rows'],strict=True):
            assert c['index']==cr['index'] and c['seed']==cr['seed']
            path=bank/c['file'];assert sha(path)==c['sha256']
            with np.load(path) as z:
                first=signature(z['states'][0,:5]);full=signature(z['states'][:,:5])
                content[role]['seeds'].add(c['seed']);content[role]['initial_states'].add(first)
                content[role]['state_paths'].add(full);content[role]['history_states'].add(signature(z['states'][:11,:5]))
                content[role]['history_pixels'].add(hashlib.sha256(z['pixels'][:3].tobytes()).hexdigest())
                content[role]['input_pixels'].update(cr['model_input_pixel_sha256'])
                for split,lookup in datasets.items():
                    for key,value in [('initial_states',first),('first_36_state_paths',full)]:
                        if value in lookup[key]:dataset_hits.append(dict(role=role,case=c['index'],split=split,kind=key,episode=lookup[key][value]))
        head_hits.extend(dict(role=role,**x) for x in r['exact_selected_head_input_pixel_overlaps'])
        bindings[role]=dict(bank_manifest_sha256=sha(bank/'manifest.json'),cache_report_sha256=sha(cache/'report.json'))
    pair={k:sorted(content['recipient'][k]&content['donor'][k]) for k in content['recipient']}
    blocked=bool(dataset_hits or any(pair[k] for k in ['seeds','initial_states','history_states','state_paths','history_pixels']))
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    atomic_json(out/'report.json',dict(status='BLOCKED_NATIVE_PARENT_OVERLAP' if blocked else 'PASS_NATIVE_DEVELOPMENT_PARENT_ISOLATION',
        bindings=bindings,native_metadata_audit_sha256=sha(audit/'report.json'),recipient_donor_intersections=pair,
        released_dataset_state_matches=dataset_hits,inspected_head_input_pixel_matches=head_hits,source_sha256=sha(__file__),
        state_comparison='First five physical coordinates canonicalized to float32; initial episode state and first36-frame path. No claim about every possible interior window.',
        upstream_generator_seeds='UNRESOLVED: released dataset metadata does not establish the complete upstream seed lineage',
        pretraining_scope='DINO foundation pretraining image overlap is UNRESOLVED; no OOD or all-pretraining-disjoint claim.',
        policy='No future cases removed. Exact individual future image coincidences are recorded separately from blocking parent/whole-history duplication.'))
    if blocked:raise ValueError('Native parent overlap requires explicit disposition; no post-outcome filtering')
    (out/'DONE').write_text('audited\n')


if __name__=='__main__':main()
