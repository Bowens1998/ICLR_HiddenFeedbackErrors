"""Episode-disjoint small development dataset; contiguous uint8 memmaps for I/O.

This prepares a pipeline pilot, not a confirmation or pretrained-model test set.
"""
import argparse,hashlib,json,time
from pathlib import Path
import h5py,hdf5plugin
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--output',required=True);a=p.parse_args()
out=Path(a.output);out.mkdir(parents=True,exist_ok=False);rng=np.random.default_rng(800001)
with h5py.File(a.input,'r') as f:
 lengths=f['ep_len'][:];offsets=f['ep_offset'][:];eligible=np.flatnonzero(lengths>=40);order=rng.permutation(eligible)
 counts={'train':256,'validation':64,'calibration':128,'development':128};cursor=0;manifest={'seed':800001,'purpose':'pipeline_development_only','input':a.input,'splits':{}}
 assert len(order)>=sum(counts.values())
 for split,n in counts.items():
  ids=np.sort(order[cursor:cursor+n]);cursor+=n;size=int(lengths[ids].sum());dest=out/split;dest.mkdir()
  arrays={key:np.lib.format.open_memmap(dest/(key+'.npy'),mode='w+',dtype=f[key].dtype,shape=(size,*f[key].shape[1:])) for key in ['pixels','action','state','proprio']}
  pos=0;new_offsets=[]
  for ep in ids:
   start=int(offsets[ep]);length=int(lengths[ep]);new_offsets.append(pos)
   for key,array in arrays.items():array[pos:pos+length]=f[key][start:start+length]
   pos+=length
  for array in arrays.values():array.flush()
  np.savez(dest/'episodes.npz',source_episode_ids=ids,lengths=lengths[ids],offsets=np.array(new_offsets))
  if split=='train':
   norm={}
   for key in ['action','state','proprio']:
    values=arrays[key];valid=np.isfinite(values).all(1);v=values[valid].astype('float64')
    norm[key]={'mean':v.mean(0).tolist(),'std':np.maximum(v.std(0,ddof=1),1e-6).tolist(),'finite_rows':int(valid.sum()),'total_rows':len(values)}
   (out/'normalization.json').write_text(json.dumps(norm,indent=2,allow_nan=False)+'\n')
  hashes={}
  for path in dest.glob('*'):
   h=hashlib.sha256()
   with path.open('rb') as src:
    while chunk:=src.read(8*1024*1024):h.update(chunk)
   hashes[path.name]=h.hexdigest()
  manifest['splits'][split]={'episode_ids':ids.tolist(),'frames':size,'files':hashes}
  del arrays
  print('PREPARED',split,n,size,flush=True)
(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print('PUSHT_PILOT_PREPARED',flush=True)
