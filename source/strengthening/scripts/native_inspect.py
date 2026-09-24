"""Load official native checkpoint; observed-training example and exact adapter parity."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
import torch
import yaml

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'strengthening/external/dino_wm'),str(ROOT/'strengthening/adapters')]
from contracts import atomic_json, sha


def main():
    p=argparse.ArgumentParser();p.add_argument('--assets',required=True);p.add_argument('--environment',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    start=time.monotonic();assets=Path(a.assets);env=json.loads(Path(a.environment).read_text())
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    report=json.loads((assets/'download_report.json').read_text());assert report['status']=='PASS_OFFICIAL_ARCHIVE_HASHES'
    model_dir=assets/'outputs/outputs/pusht';ckpt=model_dir/'checkpoints/model_latest.pth';cfgpath=model_dir/'hydra.yaml'
    cfg=yaml.safe_load(cfgpath.read_text());assert cfg['frameskip']==5 and cfg['num_hist']==3
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.hub.set_dir(env['torch_hub']);os.environ['XFORMERS_DISABLED']='1'
    # Trusted official archive verified before explicit legacy module deserialization.
    payload=torch.load(ckpt,map_location='cuda',weights_only=False)
    from models.visual_world_model import VWorldModel
    from models.dino import DinoV2Encoder
    saved_loader=torch.hub.load
    def pinned_loader(repository,name,*args,**kwargs):
        if repository!='facebookresearch/dinov2':raise ValueError('Unexpected hub repository')
        return saved_loader(env['backbone_source'],name,source='local',pretrained=True)
    torch.hub.load=pinned_loader
    try:encoder=payload.get('encoder') or DinoV2Encoder(cfg['encoder']['name'],cfg['encoder']['feature_key'])
    finally:torch.hub.load=saved_loader
    for key in ['predictor','proprio_encoder','action_encoder']:
        if key not in payload:raise ValueError('Missing native checkpoint component '+key)
    wm=VWorldModel(image_size=cfg['img_size'],num_hist=cfg['num_hist'],num_pred=cfg['num_pred'],encoder=encoder,
        predictor=payload['predictor'],decoder=payload.get('decoder'),proprio_encoder=payload['proprio_encoder'],
        action_encoder=payload['action_encoder'],proprio_dim=cfg['proprio_emb_dim'],action_dim=cfg['action_emb_dim'],
        concat_dim=cfg['concat_dim'],num_action_repeat=cfg['num_action_repeat'],num_proprio_repeat=cfg['num_proprio_repeat'],
        train_encoder=False,train_predictor=False,train_decoder=False).cuda()
    wm.eval()  # Official override mutates in place and returns None.
    from datasets.pusht_dset import PushTDataset
    from datasets.img_transforms import default_transform
    paths=[p for p in (assets/'pusht_noise').rglob('states.pth') if p.parent.name=='train']
    if len(paths)!=1:raise ValueError('Ambiguous native training dataset path')
    data=PushTDataset(data_path=str(paths[0].parent),transform=default_transform(cfg['img_size']),n_rollout=1,with_velocity=True)
    frames=np.arange(0,36);obs,actions,states,_=data.get_frames(0,frames)
    history={k:v[[0,5,10]][None].cuda() for k,v in obs.items()}
    blocks=actions[:35].reshape(1,7,10).cuda()  # Native history at 0/5/10; successor targets 15..35.
    with torch.inference_mode():
        native_obs,native=wm.rollout(history,blocks)
        z=wm.encode(history,blocks[:,:3]);inserted=[]
        for t in range(3,7):
            nxt=wm.predict(z[:,-wm.num_hist:])[:,-1:]
            nxt=wm.replace_actions_from_z(nxt,blocks[:,t:t+1])
            if t==3:
                original=nxt.clone();visual=nxt[...,:wm.encoder.emb_dim].clone()
                identity=nxt.clone();identity[...,:wm.encoder.emb_dim]=visual
                torch.testing.assert_close(identity,original,rtol=0,atol=0)
                # Nontrivial visual edit must leave every predicted proprio/action channel unchanged.
                probe=original.clone();probe[...,:wm.encoder.emb_dim]+=1e-3
                torch.testing.assert_close(probe[...,wm.encoder.emb_dim:],original[...,wm.encoder.emb_dim:],rtol=0,atol=0)
                nxt=identity
            z=torch.cat([z,nxt],dim=1);inserted.append(t)
        z=torch.cat([z,wm.predict(z[:,-wm.num_hist:])[:,-1:]],dim=1)
        torch.testing.assert_close(z,native,rtol=0,atol=0)
    attrs=dict(num_hist=wm.num_hist,frameskip=cfg['frameskip'],concat_dim=wm.concat_dim,
        visual_patches=int(native.shape[2]),visual_dim=wm.encoder.emb_dim,total_token_dim=int(native.shape[3]),
        native_encoder_image_size=wm.encoder_image_size,proprio_feature_dim=wm.proprio_dim,action_feature_dim=wm.action_dim,
        output_frames=int(native.shape[1]),first_predicted_index=3,block5_endpoint_index=7,
        primitive_relative_times=[-10,-5,0,5,10,15,20,25],extra_terminal_prediction_index=7)
    atomic_json(out/'report.json',dict(status='PASS_NATIVE_CHECKPOINT_IDENTITY_ADAPTER',checkpoint_sha256=sha(ckpt),
        checkpoint_path=str(ckpt),config_sha256=sha(cfgpath),checkpoint_keys=list(payload),epoch=int(payload['epoch']),
        config=cfg,interface=attrs,training_episode_for_smoke=0,training_dataset=str(paths[0].parent),
        visual_only_channel_probe_pass=True,native_rollout_bitwise_equal=True,
        image_normalization={'mean':[.5,.5,.5],'std':[.5,.5,.5],'source':'native datasets/img_transforms.py'},
        environment_binding=env,gpu=torch.cuda.get_device_name(),elapsed_seconds=time.monotonic()-start,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),source_sha256=sha(__file__),
        scope='Observed training example only; identity and channel isolation. No readout fit, projection, or confirmation benefit measured.'))
    (out/'DONE').write_text('accepted\n')


if __name__=='__main__':main()
