"""Label-blind fixed expert identities and common continuation batch schedule."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
from adaptation_windows import starts

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--plan',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    plan=json.loads(Path(a.plan).read_text());rows=[]
    for rep in range(3):
        d=Path(a.data)/f'replica_{rep}'/'n256';dm=json.loads((d/'manifest.json').read_text());path=d/'train/episodes.npz'
        entries=[e for e in plan['models'] if e['replica']==rep]
        assert entries and all(e['data_manifest_sha256']==sha(d/'manifest.json') for e in entries)
        assert sha(path)==dm['splits']['train']['files']['episodes.npz'];ep=np.load(path)
        assert len(ep['offsets'])==len(ep['lengths'])==len(ep['source_episode_ids'])==256
        windows=[]
        for offset,length,identity in zip(ep['offsets'],ep['lengths'],ep['source_episode_ids']):
            for start in starts(int(length),int(length)):
                windows.append([int(identity),start,int(offset)+start])
        assert len(windows)>=1280
        rng=np.random.default_rng(1293001+rep);indices=np.sort(rng.choice(len(windows),1280,replace=False));selected=np.asarray(windows,dtype=np.int64)[indices]
        assert len({tuple(v[:2]) for v in selected})==1280
        rows.append(dict(replica=rep,available_windows=len(windows),selection_seed=1293001+rep,selected_indices=indices.tolist(),identities=selected.tolist(),data_manifest_sha256=sha(d/'manifest.json'),episodes_sha256=sha(path)))
    rng=np.random.default_rng(1294001);schedule=np.concatenate([rng.permutation(1280).reshape(10,128) for _ in range(210)]).astype(np.int32)
    assert schedule.shape==(2100,128)
    for epoch in schedule.reshape(210,1280):np.testing.assert_array_equal(np.sort(epoch),np.arange(1280))
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False);np.savez_compressed(out/'update_indices.npz',indices=schedule)
    report=dict(status='FROZEN3_EXPERT_SUBSETS_AND2100_BATCH_SCHEDULE',pools=rows,plan_sha256=sha(a.plan),source_sha256=sha(__file__),window_source_sha256=sha(Path(__file__).with_name('adaptation_windows.py')),numpy_version=np.__version__,schedule_seed=1294001,schedule_sha256=sha(out/'update_indices.npz'),scope='Reads original episode metadata and manifests only; no labels, costs, images or validation/test outcomes used for sample selection.1280unique windows per pool; common epoch permutation schedule. Not fitted model results.')
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print('FROZEN',[(r['replica'],r['available_windows']) for r in rows])

if __name__=='__main__':main()
