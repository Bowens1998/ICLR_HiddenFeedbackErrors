"""Independently reconstruct horizon audit errors from saved prediction tensors."""
import hashlib,json
from pathlib import Path
import numpy as np
import argparse
p=argparse.ArgumentParser();p.add_argument('--run',required=True);a=p.parse_args();root=Path(a.run);r=json.loads((root/'summary.json').read_text())
for name,h in json.loads((root/'artifact_manifest.json').read_text()).items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==h
for row in r['runs']:
 z=np.load(root/f"job_{row['index']}_predictions.npz");states=z['states'];eps=z['episodes'];mn=np.array(row['normalization']['mean']);sd=np.array(row['normalization']['std']);truth=np.concatenate([states[:,3:,:4],np.sin(states[:,3:,4:5]),np.cos(states[:,3:,4:5])],-1);truth=(truth-mn)/sd
 np.testing.assert_allclose(truth,z['truth'],rtol=1e-5,atol=1e-6)
 for mode,metrics in row['metrics'].items():
  pred=z[mode].astype(float);decoded=pred*sd+mn;angle=np.arctan2(decoded[...,4],decoded[...,5])-states[:,3:,4];angle=np.angle(np.exp(1j*angle))
  values={'normalized_mse':np.mean((pred-z['truth'].astype(float))**2,-1),'block_pose_error':np.sum((decoded[...,2:4]-states[:,3:,2:4])**2,-1)+900*angle**2,'agent_xy_error':np.sum((decoded[...,:2]-states[:,3:,:2])**2,-1)}
  for key,v in values.items():
   np.testing.assert_allclose(metrics[key]['clip_mean'],v.mean(0),rtol=3e-6,atol=1e-7)
   np.testing.assert_allclose(metrics[key]['episode_mean'],np.mean([v[eps==i].mean(0) for i in np.unique(eps)],0),rtol=3e-6,atol=1e-7)
 result={'runs':len(r['runs']),'clips':r['clips'],'episodes':r['episodes'],'prediction_metrics_and_hashes':'independently reconstructed'}
(root/'acceptance.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
