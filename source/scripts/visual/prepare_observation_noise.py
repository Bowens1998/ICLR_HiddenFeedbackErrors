"""Derive noisy input banks; all physical arrays and target states stay bitwise fixed."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from observation_noise import corrupt_rgb

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
p=argparse.ArgumentParser();p.add_argument('--bank',required=True);p.add_argument('--output',required=True);a=p.parse_args()
bank=Path(a.bank);mp=bank/'manifest.json';base=json.loads(mp.read_text());assert len(base['cases'])==128
root=Path(a.output);root.mkdir(parents=True,exist_ok=False);reports=[]
for sigma in [8,24]:
    out=root/f'sigma{sigma}';out.mkdir();manifest=json.loads(mp.read_text());checks=[]
    for item in manifest['cases']:
        source=bank/f"case_{item['index']:03d}.npz";original_hash=item['sha256'];assert sha(source)==original_hash
        with np.load(source,allow_pickle=False) as z:original={k:z[k].copy() for k in z.files}
        changed=dict(original);seed=int(original['seed']);assert seed==item['seed']
        changed['history_pixels']=np.stack([corrupt_rgb(original['history_pixels'][i],seed=seed,slot=i,sigma=sigma) for i in range(3)])
        changed['goal_pixels']=corrupt_rgb(original['goal_pixels'],seed=seed,slot=3,sigma=sigma)
        target=out/source.name;np.savez_compressed(target,**changed)
        with np.load(target,allow_pickle=False) as z:
            assert set(z.files)==set(original)
            for key in original:
                np.testing.assert_array_equal(z[key],changed[key])
                if key not in ['history_pixels','goal_pixels']:np.testing.assert_array_equal(z[key],original[key])
        item['sha256']=sha(target);item['clean_archive_sha256']=original_hash
        checks.append(dict(index=item['index'],sha256=item['sha256'],clean_archive_sha256=original_hash))
    manifest.update(scope='Observation-only posthoc development stress bank; same physical targets, states, actions and candidate outcomes as clean parent. Not new independent goals.',
        clean_manifest_sha256=sha(mp),observation_corruption=dict(kind='additive_iid_RGB_Gaussian_then_clip_round_uint8',sigma_pixel_units=sigma,
            seed_namespace=1112001,slots='history0/1/2 and goal3; shared across models/planners; same standard noise across severities'),
        noise_builder_sha256=sha(Path(__file__)),noise_helper_sha256=sha(Path(__file__).with_name('observation_noise.py')))
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    reports.append(dict(sigma=sigma,cases=checks,manifest_sha256=sha(out/'manifest.json')))
(root/'derivation.json').write_text(json.dumps(dict(parent_manifest_sha256=sha(mp),conditions=reports,
    scope='All256 derived archives roundtrip checked; all non-input-image arrays bitwise preserved. Independent transformation acceptance still required.'),indent=2)+'\n')
print('DERIVED256 NOISY INPUT ARCHIVES; PHYSICAL ARRAYS UNCHANGED')
