"""Official native DINO-WM loader and visual-only layout operations."""
import json
import os
from pathlib import Path
import sys
import numpy as np
import torch
import yaml
from contracts import sha

ROOT=Path(__file__).resolve().parents[2]


def load_native(assets,environment):
    assets=Path(assets);env=json.loads(Path(environment).read_text())
    sys.path.insert(0,str(ROOT/'strengthening/external/dino_wm'))
    ckpt=assets/'outputs/outputs/pusht/checkpoints/model_latest.pth';cfgpath=ckpt.parent.parent/'hydra.yaml'
    assert sha(ckpt)=='e909f0cec958fc0b49a79f2e85730ae6b5f83dee61a224705f91883eb3bb74c5'
    assert sha(env['backbone_weight'])==env['backbone_weight_sha256']
    cfg=yaml.safe_load(cfgpath.read_text());torch.hub.set_dir(env['torch_hub']);os.environ['XFORMERS_DISABLED']='1'
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
    payload=torch.load(ckpt,map_location='cuda',weights_only=False)
    from models.visual_world_model import VWorldModel
    from models.dino import DinoV2Encoder
    original=torch.hub.load
    def pinned(repository,name,*args,**kwargs):
        if repository!='facebookresearch/dinov2':raise ValueError('Unexpected hub source')
        return original(env['backbone_source'],name,source='local',pretrained=True)
    torch.hub.load=pinned
    try:encoder=DinoV2Encoder(cfg['encoder']['name'],cfg['encoder']['feature_key'])
    finally:torch.hub.load=original
    model=VWorldModel(image_size=cfg['img_size'],num_hist=cfg['num_hist'],num_pred=cfg['num_pred'],encoder=encoder,
        predictor=payload['predictor'],decoder=payload.get('decoder'),proprio_encoder=payload['proprio_encoder'],
        action_encoder=payload['action_encoder'],proprio_dim=cfg['proprio_emb_dim'],action_dim=cfg['action_emb_dim'],
        concat_dim=cfg['concat_dim'],num_action_repeat=cfg['num_action_repeat'],num_proprio_repeat=cfg['num_proprio_repeat'],
        train_encoder=False,train_predictor=False,train_decoder=False).cuda()
    model.eval()
    for p in model.parameters():p.requires_grad_(False)
    assert model.concat_dim==1 and model.encoder.emb_dim==384 and model.num_hist==3 and cfg['frameskip']==5
    return model,cfg,dict(checkpoint_sha256=sha(ckpt),config_sha256=sha(cfgpath),epoch=int(payload['epoch']),
        environment_sha256=sha(environment),encoder_weight_sha256=env['backbone_weight_sha256'],
        encoder_source_commit=env['backbone_commit'],source_sha256=sha(__file__))


def flatten_visual(token):
    if token.shape[-2:]!=(196,404):raise ValueError('Unexpected native patch layout')
    return token[...,:384].contiguous().reshape(*token.shape[:-2],75264)


def replace_visual(token,flat):
    if tuple(flat.shape)!=tuple(token.shape[:-2])+(75264,):raise ValueError('Visual shape mismatch')
    value=token.clone();value[...,:384]=flat.reshape(*token.shape[:-2],196,384)
    torch.testing.assert_close(value[...,384:],token[...,384:],rtol=0,atol=0)
    return value


def encode_visual(model,images):
    result=model.encoder(model.encoder_transform(images.cuda()))
    assert result.shape[1:]==(196,384) and result.dtype==torch.float32
    return result.reshape(len(images),75264)
