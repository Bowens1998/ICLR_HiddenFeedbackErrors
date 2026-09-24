"""Match official evaluation StandardScaler action statistics on the pinned dataset."""
import argparse,hashlib,json
from pathlib import Path
import h5py,hdf5plugin
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--output',required=True);a=p.parse_args()
with h5py.File(a.input,'r') as f:
    values=f['action'][:].astype('float64');total=len(values);values=values[np.isfinite(values).all(1)]
mean=values.mean(0);std=values.std(0,ddof=0);std=np.where(std==0,1.,std)
out={'action':{'mean':mean.tolist(),'std':std.tolist(),'finite_rows':len(values),'total_rows':total},'ddof':0,
     'scope':'released-model implementation reference only; full pinned source data, matching official eval StandardScaler',
     'input':a.input,'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
Path(a.output).write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
