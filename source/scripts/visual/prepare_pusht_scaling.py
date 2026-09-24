"""Materialize independent nested train pools; retain a shared fixed validation set."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from nested_training_samples import select_pools

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        while chunk:=f.read(8*1024*1024):h.update(chunk)
    return h.hexdigest()

def main():
    import h5py,hdf5plugin
    p=argparse.ArgumentParser()
    for k in ['input','pilot','output']:p.add_argument('--'+k,required=True)
    p.add_argument('--replica',type=int,choices=range(3),required=True);a=p.parse_args()
    pilot=Path(a.pilot).resolve();old=json.loads((pilot/'manifest.json').read_text());excluded={i for s in old['splits'].values() for i in s['episode_ids']}
    root=Path(a.output)/f'replica_{a.replica}';root.mkdir(parents=True,exist_ok=False)
    with h5py.File(a.input,'r') as f:
        lengths=f['ep_len'][:];offsets=f['ep_offset'][:];pools=select_pools(lengths,excluded)
        selection={'seed':956001,'replica':a.replica,'all_pools':[{str(k):v.tolist() for k,v in pool.items()} for pool in pools],'excluded_pilot_episode_ids':sorted(excluded),'pilot_manifest_sha256':digest(pilot/'manifest.json')}
        (root/'selection.json').write_text(json.dumps(selection,indent=2)+'\n')
        for size,ids in pools[a.replica].items():
            dest=root/f'n{size}';train=dest/'train';train.mkdir(parents=True)
            (dest/'validation').symlink_to(pilot/'validation',target_is_directory=True)
            count=int(lengths[ids].sum());arrays={k:np.lib.format.open_memmap(train/(k+'.npy'),mode='w+',dtype=f[k].dtype,shape=(count,*f[k].shape[1:])) for k in ['pixels','action','state','proprio']}
            pos=0;new_offsets=[]
            for ep in ids:
                start=int(offsets[ep]);n=int(lengths[ep]);new_offsets.append(pos)
                for k,arr in arrays.items():arr[pos:pos+n]=f[k][start:start+n]
                pos+=n
            for arr in arrays.values():arr.flush()
            np.savez(train/'episodes.npz',source_episode_ids=ids,lengths=lengths[ids],offsets=np.array(new_offsets))
            norm={}
            for k in ['action','state','proprio']:
                v=np.asarray(arrays[k],dtype=np.float64);assert np.isfinite(v).all()
                norm[k]={'mean':v.mean(0).tolist(),'std':np.maximum(v.std(0,ddof=1),1e-6).tolist(),'finite_rows':len(v),'total_rows':len(v)}
            (dest/'normalization.json').write_text(json.dumps(norm,indent=2)+'\n')
            files={p.name:digest(p) for p in train.iterdir()}
            manifest={'seed':956001,'replica':a.replica,'purpose':'independent_nested_training_data_scaling_development','input':a.input,'selection_sha256':digest(root/'selection.json'),'pilot_manifest_sha256':selection['pilot_manifest_sha256'],'splits':{'train':{'episode_ids':ids.tolist(),'frames':count,'files':files},'validation':old['splits']['validation']},'validation_path':str(pilot/'validation'),'normalization_sha256':digest(dest/'normalization.json')}
            (dest/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');del arrays
            print('PREPARED',a.replica,size,count,flush=True)
    (root/'COMPLETE').write_text('data preparation; independent validation still required\n')
if __name__=='__main__':main()
