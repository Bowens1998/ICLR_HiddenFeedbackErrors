"""Native dataset input-only inventory and exact physical-trajectory overlap audit."""
import argparse
import hashlib
import json
from pathlib import Path
import pickle
import sys
import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'adapters'))
from contracts import atomic_json, sha


def main():
    p=argparse.ArgumentParser();p.add_argument('--assets',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    assets=Path(a.assets);download=json.loads((assets/'download_report.json').read_text())
    assert download['status']=='PASS_OFFICIAL_ARCHIVE_HASHES'
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False);rows={};sets={}
    for split in ['train','val']:
        candidates=[x.parent for x in (assets/'pusht_noise').rglob('states.pth') if x.parent.name==split]
        if len(candidates)!=1:raise ValueError('Ambiguous native split')
        root=candidates[0];states=torch.load(root/'states.pth',map_location='cpu',weights_only=True).numpy()
        actions=torch.load(root/'rel_actions.pth',map_location='cpu',weights_only=True).numpy()
        with (root/'seq_lengths.pkl').open('rb') as f:lengths=np.asarray(pickle.load(f),int)
        assert len(states)==len(actions)==len(lengths) and np.all(lengths<=states.shape[1]) and np.all(lengths>0)
        episodes=[]
        for i,n in enumerate(lengths):
            movie=root/'obses'/f'episode_{i:03d}.mp4';assert movie.exists()
            signature=hashlib.sha256(np.ascontiguousarray(states[i,:n]).tobytes()+np.ascontiguousarray(actions[i,:n]).tobytes()).hexdigest()
            episodes.append(dict(index=i,frames=int(n),trajectory_sha256=signature,video=str(movie),video_bytes=movie.stat().st_size))
        sets[split]={x['trajectory_sha256'] for x in episodes}
        metadata={str(p.relative_to(root)):sha(p) for p in root.iterdir() if p.is_file() and not p.name.startswith('._')}
        rows[split]=dict(root=str(root),episodes=len(episodes),total_frames=int(sum(lengths)),
            states_shape=list(states.shape),actions_shape=list(actions.shape),min_frames=int(min(lengths)),max_frames=int(max(lengths)),
            episodes_supporting_35_actions=int(sum(lengths>=36)),physical_trajectory_unique_count=len(sets[split]),
            video_bytes=sum(x['video_bytes'] for x in episodes),metadata_sha256=metadata)
        atomic_json(out/(split+'_episodes.json'),dict(split=split,rows=episodes))
        rows[split]['episode_manifest_sha256']=sha(out/(split+'_episodes.json'))
    overlap=sets['train']&sets['val']
    atomic_json(out/'report.json',dict(status='PASS_NATIVE_METADATA_INVENTORY' if not overlap else 'BLOCKED_NATIVE_EXACT_TRAJECTORY_OVERLAP',
        splits=rows,exact_train_val_trajectory_intersection=sorted(overlap),
        image_overlap='UNRESOLVED: must decode and hash selected readout/basis frames before fitting',
        source_sha256=sha(__file__),archive_receipt_sha256=sha(assets/'download_report.json'),
        scope='Input metadata only; no model accuracy, new confirmation selection or intervention outcomes.'))
    (out/'DONE').write_text('inventoried\n')


if __name__=='__main__':main()
