"""Independently reconstruct perturbations and check physical-array preservation."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
p=argparse.ArgumentParser();p.add_argument('--bank',required=True);p.add_argument('--noisy',required=True);a=p.parse_args()
bank=Path(a.bank);root=Path(a.noisy);bp=bank/'manifest.json';base=json.loads(bp.read_text());reports=[]
for sigma in [8,24]:
    folder=root/f'sigma{sigma}';mp=folder/'manifest.json';m=json.loads(mp.read_text())
    assert m['clean_manifest_sha256']==sha(bp) and len(m['cases'])==len(base['cases'])==128
    assert m['observation_corruption']['sigma_pixel_units']==sigma and m['observation_corruption']['seed_namespace']==1112001
    assert m['noise_builder_sha256']==sha(Path(__file__).with_name('prepare_observation_noise.py'))
    assert m['noise_helper_sha256']==sha(Path(__file__).with_name('observation_noise.py'))
    for original,row in zip(base['cases'],m['cases']):
        cp=bank/f"case_{original['index']:03d}.npz";npz=folder/cp.name
        assert sha(cp)==original['sha256']==row['clean_archive_sha256'] and sha(npz)==row['sha256']
        assert row==dict(original,sha256=sha(npz),clean_archive_sha256=sha(cp))
        with np.load(cp,allow_pickle=False) as clean,np.load(npz,allow_pickle=False) as noisy:
            assert set(clean.files)==set(noisy.files)
            for key in clean.files:
                assert clean[key].dtype==noisy[key].dtype and clean[key].shape==noisy[key].shape
                if key not in ['history_pixels','goal_pixels']:np.testing.assert_array_equal(clean[key],noisy[key])
            frames=list(clean['history_pixels'])+[clean['goal_pixels']]
            altered=list(noisy['history_pixels'])+[noisy['goal_pixels']]
            for slot,(image,result) in enumerate(zip(frames,altered)):
                generator=np.random.Generator(np.random.PCG64(np.random.SeedSequence([1112001,int(clean['seed']),slot])))
                expected=np.round(np.minimum(255,np.maximum(0,image.astype(float)+sigma*generator.standard_normal(image.shape)))).astype(np.uint8)
                np.testing.assert_array_equal(result,expected)
    report=dict(sigma=sigma,cases=128,images=512,manifest_sha256=sha(mp),parent_manifest_sha256=sha(bp),verifier_sha256=sha(Path(__file__)),
        scope='Independent exact perturbation reconstruction and all remaining arrays bitwise preserved; observation-only development stress, not independent goals or new physical replay.')
    (folder/'acceptance.json').write_text(json.dumps(report,indent=2)+'\n');reports.append(report)
print('ACCEPTED256 ARCHIVES AND1024 INPUT-IMAGE PERTURBATIONS')
