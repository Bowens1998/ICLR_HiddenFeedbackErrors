"""Native observed visual caches and training-only residuals, with input-image isolation."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'strengthening/adapters'),str(ROOT/'strengthening/external/dino_wm')]
from contracts import atomic_json, sha, namespace_seed
from native_model import load_native, encode_visual, flatten_visual, replace_visual


def main():
    p=argparse.ArgumentParser()
    for k in ['assets','environment','audit','output']:p.add_argument('--'+k,required=True)
    a=p.parse_args();audit=Path(a.audit);ar=json.loads((audit/'report.json').read_text())
    assert ar['status']=='PASS_NATIVE_METADATA_INVENTORY'
    specpath=ROOT/'strengthening/configs/B_development.json';spec=json.loads(specpath.read_text());out=Path(a.output)
    out.mkdir(parents=True,exist_ok=False);start=time.monotonic();torch.set_num_threads(2)
    model,cfg,binding=load_native(a.assets,a.environment)
    from datasets.pusht_dset import PushTDataset
    from datasets.img_transforms import default_transform
    validation_hashes=set();reports=[];datasets={};train_pixels=set();norm_sum=np.zeros(75264);norm_square=np.zeros(75264)
    for split in ['val','train']:
        info=ar['splits'][split];root=Path(info['root'])
        for name,h in info['metadata_sha256'].items():assert sha(root/name)==h
        data=PushTDataset(data_path=str(root),transform=default_transform(cfg['img_size']),n_rollout=None,with_velocity=True);datasets[split]=data
        if split=='train':
            rng=np.random.default_rng(namespace_seed(spec['root_seed'],'B_head_train_episodes'))
            episode_order=rng.permutation(len(data))[:spec['head_train_episodes']]
        else:episode_order=np.arange(len(data))
        role='head_validation' if split=='val' else 'head_train';rows=[];labels=[];excluded=0;intra=0;own=set();videos={}
        features=out/(role+'.f32')
        with features.open('wb') as f,torch.inference_mode():
            for episode in episode_order:
                episode=int(episode);n=data.get_seq_length(episode)
                frames=np.arange(n) if split=='val' else np.linspace(0,n-1,spec['frames_per_train_episode'],dtype=int)
                obs,actions,states,_=data.get_frames(episode,frames)
                video=root/'obses'/f'episode_{episode:03d}.mp4';videos[str(video)]=sha(video);keep=[]
                for j,frame in enumerate(frames):
                    pix=hashlib.sha256(obs['visual'][j].contiguous().numpy().tobytes()).hexdigest()
                    if split=='train' and pix in validation_hashes:excluded+=1;continue
                    if pix in own:intra+=1;continue
                    own.add(pix);keep.append(j);st=states[j].numpy()
                    rows.append(dict(split=split,episode=episode,frame=int(frame),model_input_pixel_sha256=pix))
                    labels.append(np.r_[st[:4],np.sin(st[4]),np.cos(st[4])].astype(np.float64))
                for offset in range(0,len(keep),spec['encoder_batch']):
                    value=encode_visual(model,obs['visual'][keep[offset:offset+spec['encoder_batch']]]).cpu().numpy()
                    if not np.isfinite(value).all():raise ValueError('Nonfinite observed native feature')
                    value.tofile(f)
                    if split=='train':norm_sum+=value.astype(float).sum(0);norm_square+=(value.astype(float)**2).sum(0)
        if split=='val':validation_hashes=own
        else:train_pixels=own
        y=np.asarray(labels);np.save(out/(role+'_labels.npy'),y)
        atomic_json(out/(role+'_rows.json'),dict(role=role,rows=rows,video_sha256=videos))
        meta=dict(role=role,parent_manifest_sha256=sha(audit/'report.json'),file=str(features),file_sha256=sha(features),
            rows=len(rows),shape=[len(rows),75264],dtype='float32',labels_sha256=sha(out/(role+'_labels.npy')),
            row_manifest_sha256=sha(out/(role+'_rows.json')),within_split_duplicate_images=intra,excluded_validation_images=excluded,
            model_binding=binding,development_spec_sha256=sha(specpath))
        assert features.stat().st_size==len(rows)*75264*4
        atomic_json(out/(role+'.json'),meta);reports.append(meta)
        if split=='train':
            mean=norm_sum/len(rows);scale=np.sqrt(np.maximum(0,norm_square/len(rows)-mean**2)).clip(1e-6)
            np.savez_compressed(out/'normalizers.npz',mean=mean,scale=scale,target_mean=y.mean(0),target_scale=y.std(0).clip(1e-3))
        print('NATIVE_OBSERVED_CACHE',role,len(rows),flush=True)
    assert not train_pixels&validation_hashes
    # Residual basis may use training inputs only. No future intervention utility is computed.
    basis=[];preds=[];observed=[];rows=[];data=datasets['train']
    rng=np.random.default_rng(namespace_seed(spec['root_seed'],'B_basis_windows'))
    with torch.inference_mode():
        for ep in episode_order[:spec['basis_train']['episodes']]:
            ep=int(ep);start_frame=int(rng.integers(0,data.get_seq_length(ep)-35));frames=np.arange(start_frame,start_frame+36)
            obs,actions,states,_=data.get_frames(ep,frames);history={k:v[[0,5,10]][None].cuda() for k,v in obs.items()}
            blocks=actions[:35].reshape(1,7,10).cuda();z=model.encode(history,blocks[:,:3])
            root=model.replace_actions_from_z(model.predict(z)[:,-1:].clone(),blocks[:,3:4]);visual=flatten_visual(root)
            torch.testing.assert_close(replace_visual(root,visual),root,rtol=0,atol=0)
            actual=encode_visual(model,obs['visual'][15:16]);pred=visual[0,0].cpu().numpy();actual=actual[0].cpu().numpy()
            basis.append(((actual.astype(float)-pred)/scale).astype(np.float32));rows.append(dict(episode=ep,start_frame=start_frame))
            if len(preds)<4:preds.append(pred);observed.append(actual)
    np.save(out/'basis_training_residuals.npy',np.stack(basis))
    np.savez_compressed(out/'QP_profile_inputs.npz',predicted=np.stack(preds),observed=np.stack(observed))
    atomic_json(out/'basis_train.json',dict(role='basis_train',parent_manifest_sha256=sha(audit/'report.json'),rows=rows,
        file_sha256=sha(out/'basis_training_residuals.npy'),normalizers_sha256=sha(out/'normalizers.npz'),
        QP_inputs_sha256=sha(out/'QP_profile_inputs.npz'),development_spec_sha256=sha(specpath)))
    atomic_json(out/'report.json',dict(status='PASS_NATIVE_OBSERVED_CACHE_REQUIRES_READOUT_FIT',head_splits=reports,
        basis_train_sha256=sha(out/'basis_train.json'),normalizers_sha256=sha(out/'normalizers.npz'),
        train_validation_pixel_overlap=0,model_binding=binding,development_spec_sha256=sha(specpath),
        elapsed_seconds=time.monotonic()-start,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        gpu=torch.cuda.get_device_name(),source_sha256=sha(__file__),scope='Native train/val only; no new confirmation or intervention effects.'))
    (out/'DONE').write_text('accepted\n')


if __name__=='__main__':main()
