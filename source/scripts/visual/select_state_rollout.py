"""Freeze validation-only LR selection and verify saved prediction losses and hashes."""
import argparse, hashlib, json
from pathlib import Path
import numpy as np

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();root=Path(a.root);rows=[]
    for i in range(6):
        run=root/f'job_{i}';assert (run/'COMPLETE').exists()
        for name,h in json.loads((run/'artifact_manifest.json').read_text()).items():assert hashlib.sha256((run/name).read_bytes()).hexdigest()==h
        r=json.loads((run/'summary.json').read_text());assert len(r['epochs'])==r['expected_epochs']==100
        assert r['base_lr']==[5e-5,3e-4,1e-3][i%3] and r['seed']==3072
        losses=[e['validation_prediction_mse'] for e in r['epochs']];assert np.argmin(losses)+1==r['best_epoch']
        z=np.load(run/'validation_best_predictions.npz');mse=float(np.mean((z['predicted'].astype(np.float64)-z['target'])**2))
        np.testing.assert_allclose(mse,r['best_validation_mse'],rtol=1e-6,atol=1e-9)
        rows.append({'index':i,'lr':r['base_lr'],'best_epoch':r['best_epoch'],'validation_mse':mse,'weights_sha256':hashlib.sha256((run/'best_weights.pt').read_bytes()).hexdigest()})
    selected={mode:min(rows[start:start+3],key=lambda v:(v['validation_mse'],v['index'])) for mode,start in [('teacher_forced',0),('recursive',3)]}
    reports=[json.loads((root/f'job_{i}'/'summary.json').read_text()) for i in range(6)]
    assert all(r['initial_hashes']==reports[0]['initial_hashes'] and r['clip_counts']==reports[0]['clip_counts'] for r in reports)
    assert all(r['mode']==['teacher_forced','recursive'][i//3] for i,r in enumerate(reports))
    result={'selection':'validation only, before candidate-bank evaluation','selected':selected,'all_candidates':rows,'training_artifacts_and_saved_validation_mse_verified':True}
    with (root/'selection.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
