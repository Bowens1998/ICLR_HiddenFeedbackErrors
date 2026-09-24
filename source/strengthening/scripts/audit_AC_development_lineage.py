"""Input-only parent/content isolation audit; never remove cases based on future content."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'strengthening/adapters'))
from contracts import sha,atomic_json
from input_lock import add_design_arguments,read_design,check_parent_design


def digest(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def main():
    p=argparse.ArgumentParser()
    for k in ['base','recipient','donor','recipient-physics','donor-physics','frame-manifests','output']:p.add_argument('--'+k,required=True)
    add_design_arguments(p)
    a=p.parse_args();old=json.loads((ROOT/'strengthening/manifests/split_lineage.json').read_text())['banks']
    design=None;count=64;phase_binding={}
    if a.design_lock or a.design_sha256:
        design=read_design(a.design_lock,a.design_sha256,ROOT);count=256
        phase_binding=dict(scientific_design_sha256=a.design_sha256,phase='confirmation_input_isolation')
        for role,path in [('confirmation_A_C',a.recipient),('donor_bank_A_C',a.donor)]:
            assert Path(path)==Path(design['banks'][role]['output'])
            rm=json.loads((Path(path)/'role.json').read_text());assert rm['role']==role
            check_parent_design(rm,phase_binding)
        release=Path(a.base)/'releases/feedback-strengthening-v1'
        previous=release/'artifacts/AC_development_lineage_v1/report.json'
        assert sha(previous)==design['verified_assets'][str(previous)]
        previous=json.loads(previous.read_text())['bank_manifest_sha256']
        for label,bank,oldkey in [('diagnostic_development','AC_development_bank_v1','new_recipient'),
                                 ('independent_development_donor','AC_donor_development_bank_v1','new_donor')]:
            old[label]=dict(path=str(release/'artifacts'/bank),manifest_sha256=previous[oldkey])
    banks={name:Path(v['path']) for name,v in old.items()};banks.update(new_recipient=Path(a.recipient),new_donor=Path(a.donor))
    records={};bindings={};pixels={};states={}
    for name,bank in banks.items():
        manifest=json.loads((bank/'manifest.json').read_text());h=sha(bank/'manifest.json')
        if name in old:assert h==old[name]['manifest_sha256']
        else:
            role=json.loads((bank/'role.json').read_text());assert role['parent_manifest_sha256']==h and len(manifest['cases'])==count
            assert [c['index'] for c in manifest['cases']]==list(range(count))
        fields={k:set() for k in ['seed','initial_state','history_states','history_pixels','prefix','goal_actions','goal_state']}
        pix={};st={}
        for c in manifest['cases']:
            file=bank/f"case_{c['index']:03d}.npz";assert sha(file)==c['sha256']
            with np.load(file) as z:
                fields['seed'].add(int(z['seed']));fields['initial_state'].add(digest(z['history_states'][0]))
                for k in ['history_states','history_pixels','prefix','goal_actions','goal_state']:fields[k].add(digest(z[k]))
                for j,im in enumerate(z['history_pixels']):pix.setdefault(digest(im),[]).append(dict(case=c['index'],frame=j*5,source='history'))
                pix.setdefault(digest(z['goal_pixels']),[]).append(dict(case=c['index'],source='goal'))
                for j,state in enumerate(z['history_states']):st.setdefault(digest(state),[]).append(dict(case=c['index'],frame=j*5,source='history'))
        records[name]=fields;bindings[name]=h;pixels[name]=pix;states[name]=st
    pairs=[];blocked=[]
    for left,right in itertools.combinations(records,2):
        if not left.startswith('new_') and not right.startswith('new_'):continue
        overlaps={k:sorted(records[left][k]&records[right][k]) for k in records[left]}
        pairs.append(dict(left=left,right=right,overlaps=overlaps))
        if any(overlaps[k] for k in ['seed','initial_state','history_states','history_pixels','goal_state']):blocked.append([left,right])
    train={};validation={};frame_bindings=[]
    for g in range(6):
        directory=Path(a.frame_manifests)/f'pool_{g//2}'
        cache=Path(a.base)/f'releases/feedback-strengthening-v1/artifacts/observed_cache_v1/group_{g}'
        cr=json.loads((cache/'report.json').read_text())
        for split,target in [('train',train),('validation',validation)]:
            file=directory/f'group_{g}_{split}.json';assert sha(file)==cr['frame_manifests'][split]
            m=json.loads(file.read_text());frame_bindings.append(dict(group=g,split=split,sha256=sha(file)))
            for r in m['rows']:target.setdefault(r['pixel_sha256'],set()).add((g,r['domain']))
    physics_bindings={};hits=[]
    for role in ['new_recipient','new_donor']:
        for h,locations in pixels[role].items():
            for split,lookup in [('head_train',train),('head_validation',validation)]:
                if h in lookup:
                    for location in locations:hits.append(dict(role=role,**location,matched_split=split,
                        groups_domains=sorted(lookup[h]),pixel_sha256=h))
    for role,root in [('new_recipient',Path(a.recipient_physics)),('new_donor',Path(a.donor_physics))]:
        for route in range(24):
            folder=root/f'route_{route}';r=json.loads((folder/'report.json').read_text());assert (folder/'DONE').exists() and len(r['cases'])==count
            assert r['binding']['parent_manifest_sha256']==bindings[role]
            check_parent_design(r['binding'],phase_binding)
            physics_bindings[f'{role}/{route}']=sha(folder/'report.json')
            for c in r['cases']:
                file=folder/c['file'];assert sha(file)==c['file_sha256']
                with np.load(file) as z:
                    for j,im in enumerate(z['pixels']):
                        h=digest(im);location=dict(role=role,route=route,case=c['index'],frame=5*j)
                        pixels[role].setdefault(h,[]).append(location)
                        for split,lookup in [('head_train',train),('head_validation',validation)]:
                            if h in lookup:hits.append(dict(**location,matched_split=split,groups_domains=sorted(lookup[h]),pixel_sha256=h))
    individual=[]
    for left,right in itertools.combinations(pixels,2):
        if not left.startswith('new_') and not right.startswith('new_'):continue
        overlap=sorted(set(pixels[left])&set(pixels[right]))
        individual.append(dict(left=left,right=right,unique_pixel_overlap=len(overlap),
            overlaps=[dict(sha256=h,left_locations=pixels[left][h],right_locations=pixels[right][h]) for h in overlap]))
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    atomic_json(out/'report.json',dict(**phase_binding,status='BLOCKED_PARENT_CONTENT_OVERLAP' if blocked else 'PASS_NEW_INPUT_PARENT_ISOLATION',
        parent_pairwise=pairs,blocked_parent_pairs=blocked,bank_manifest_sha256=bindings,frame_bindings=frame_bindings,
        physics_report_sha256=physics_bindings,head_input_pixel_matches=hits,individual_image_pairwise=individual,
        source_sha256=sha(__file__),expected_cases=count,
        scope='New recipients and independent donors compared with all listed prior bank roles and inspected head train/validation images. Formal mode additionally includes both new-development banks. Exact future-frame coincidences are reported, never removed after seeing trajectories; parent overlap is blocking. Not an audit of all foundation-model pretraining images.'))
    if blocked:raise ValueError('Parent lineage overlap; no case deletion or reseeding allowed')
    (out/'DONE').write_text('audited\n')


if __name__=='__main__':main()
