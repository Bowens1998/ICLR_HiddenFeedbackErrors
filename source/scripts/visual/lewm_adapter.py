"""Minimal constructor equivalent to official ViT-tiny factory; official JEPA modules."""
import json,sys
from pathlib import Path
import torch
from transformers import ViTConfig,ViTModel
def build(official,config,weights=None):
 sys.path.insert(0,str(official))
 from jepa import JEPA
 from module import ARPredictor,Embedder,MLP
 cfg=json.loads(Path(config).read_text())
 encoder=ViTModel(ViTConfig(hidden_size=192,num_hidden_layers=12,num_attention_heads=3,intermediate_size=768,
  image_size=cfg['encoder']['image_size'],patch_size=cfg['encoder']['patch_size']),add_pooling_layer=False,use_mask_token=False)
 encoder.config.interpolate_pos_encoding=True
 args=lambda k:{key:value for key,value in cfg[k].items() if key not in ['_target_','norm_fn']}
 mlp=lambda k:MLP(**args(k),norm_fn=torch.nn.BatchNorm1d)
 model=JEPA(encoder,ARPredictor(**args('predictor')),Embedder(**args('action_encoder')),mlp('projector'),mlp('pred_proj'))
 if weights is not None:model.load_state_dict(torch.load(weights,map_location='cpu',weights_only=True),strict=True)
 return model
