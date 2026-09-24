"""Freeze fixed-last noise comparisons from accepted action and state manifests."""
import argparse,hashlib,json
from pathlib import Path

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--remote-base',required=True);a=p.parse_args()
root=Path(__file__).resolve().parents[2];out=Path(a.output);out.mkdir(parents=True,exist_ok=False);remote=a.remote_base.rstrip('/')
parents={'action':root/'runs/hpg/action_planning_v1/scientific_manifest.json','state':root/'runs/hpg/visual_updates_v1/scaling_manifest.json'}
clean=root/'data/pusht_prospective_v1/heldout/manifest.json';engineering=[];scientific=[];manifests={}
for sigma in [0,8,24]:
    if sigma:
        bank=root/f'data/pusht_observation_noise_v1/sigma{sigma}';accept=json.loads((bank/'acceptance.json').read_text())
        assert accept['manifest_sha256']==sha(bank/'manifest.json') and accept['parent_manifest_sha256']==sha(clean)
        assert accept['verifier_sha256']==sha(Path(__file__).with_name('accept_observation_noise.py'))
        bank_hash=sha(bank/'manifest.json');bank_remote=f'{remote}/datasets/pusht-observation-noise-v1/sigma{sigma}'
    else:bank_hash=sha(clean);bank_remote=f'{remote}/datasets/pusht-prospective-v1/heldout'
    for family,path in parents.items():
        parent=json.loads(path.read_text())
        evidence=root/('outputs/maintrack/action_auxiliary_summary.json' if family=='action' else 'outputs/maintrack/scaling_summary.json')
        accepted=json.loads(evidence.read_text());assert accepted['manifest_sha256']==sha(path) and accepted['bank_manifest_sha256']==sha(clean)
        models=[];routes=[];mapping={}
        for parent_route,r in enumerate(parent['routes']):
            e=parent['models'][r['model_index']]
            if e['checkpoint']!='last':continue
            if family=='state' and not (e['arm'].endswith('state') and e['episodes']==256 and e['updates']==21000):continue
            index=mapping.get(r['model_index'])
            if index is None:
                index=len(models);mapping[r['model_index']]=index;models.append(dict(e,parent_model_index=r['model_index']))
            routes.append(dict(r,model_index=index,parent_route_index=parent_route))
        assert len(models)==(18 if family=='action' else 6) and len(routes)==2*len(models)
        name=f'{family}_sigma{sigma}.json';manifest=dict(parent,models=models,routes=routes,bank_manifest_sha256=bank_hash,
            parent_manifest_sha256=sha(path),clean_bank_manifest_sha256=sha(clean),noise_sigma=sigma,
            scope='Frozen fixed-last observation-noise stress comparison; engineering first2 or scientific all128 development goals; same physical targets, no training adaptation')
        (out/name).write_text(json.dumps(manifest,indent=2)+'\n');manifests[name]=sha(out/name)
        training=f"{remote}/releases/{'action-auxiliary-v3' if family=='action' else 'visual-updates-v1'}/runs/full"
        original=f"{remote}/releases/{'action-planning-v1/runs/planner_full' if family=='action' else 'manifest-planner-v1/runs/scaling'}"
        for index,r in enumerate(routes):
            e=models[r['model_index']]
            job=dict(family=family,sigma=sigma,manifest=name,manifest_sha256=sha(out/name),route_index=index,bank=bank_remote,training=training,
                replica=e['replica'],arm=e['arm'],mode=e.get('mode','direct_state'),algorithm=r['algorithm'],original_run=f"{original}/job_{r['parent_route_index']}")
            if e['replica']==0:engineering.append(dict(job,index=len(engineering)))
            if sigma:scientific.append(dict(job,index=len(scientific)))
assert len(engineering)==48 and len(scientific)==96
plan=dict(engineering=engineering,scientific=scientific,manifests=manifests,source_sha256=sha(Path(__file__)),
    scope='Engineering:pool0,all8 model branches,two planners,three noise conditions,two goals. Scientific:all3 pools,all8 branches,two planners,sigma8/24,128 existing development goals;96 routes. Sigma0 uses the unchanged clean bank.')
(out/'plan.json').write_text(json.dumps(plan,indent=2)+'\n');print('FROZEN48 ENGINEERING AND96 SCIENTIFIC NOISE ROUTES')
