"""Hardware-only benchmark: official LeWM modules, pinned weights, synthetic batches.

The ViT constructor is equivalent to stable_pretraining.backbone.utils.vit_hf(tiny),
without importing its training framework. Strict checkpoint loading checks topology.
This measures compute and memory, never task performance or data-loader throughput.
"""
import argparse,gc,json,os,platform,sys,time
from pathlib import Path
import torch
from transformers import ViTConfig,ViTModel
p=argparse.ArgumentParser();p.add_argument('--assets',required=True);p.add_argument('--official',required=True);p.add_argument('--output',required=True);a=p.parse_args()
sys.path.insert(0,a.official)
from jepa import JEPA
from module import ARPredictor,Embedder,MLP,SIGReg
cfg=json.loads((Path(a.assets)/'models/config.json').read_text())
def build():
 encoder=ViTModel(ViTConfig(hidden_size=192,num_hidden_layers=12,num_attention_heads=3,intermediate_size=768,
  image_size=cfg['encoder']['image_size'],patch_size=cfg['encoder']['patch_size']),add_pooling_layer=False,use_mask_token=False)
 encoder.config.interpolate_pos_encoding=True
 args=lambda k:{key:value for key,value in cfg[k].items() if key not in ['_target_','norm_fn']}
 mlp=lambda k:MLP(**args(k),norm_fn=torch.nn.BatchNorm1d)
 model=JEPA(encoder,ARPredictor(**args('predictor')),Embedder(**args('action_encoder')),mlp('projector'),mlp('pred_proj'))
 model.load_state_dict(torch.load(Path(a.assets)/'models/weights.pt',map_location='cpu',weights_only=True),strict=True)
 return model.cuda().train()
torch.manual_seed(3072);torch.set_num_threads(4)
report={'purpose':'hardware_compute_only','gpu':torch.cuda.get_device_name(),'torch':torch.__version__,'cuda':torch.version.cuda,'host':platform.node(),
 'job_id':os.environ.get('SLURM_JOB_ID'),'partition':os.environ.get('SLURM_JOB_PARTITION'),'rows':[]}
for batch in [16,32,64,128]:
 try:
  model=build();reg=SIGReg().cuda();opt=torch.optim.AdamW(model.parameters(),lr=5e-5,weight_decay=1e-3)
  pixels=torch.randn(batch,4,3,224,224,device='cuda');actions=torch.randn(batch,4,cfg['action_encoder']['input_dim'],device='cuda')
  def step():
   opt.zero_grad(set_to_none=True)
   with torch.autocast('cuda',dtype=torch.bfloat16):
    z=model.encode({'pixels':pixels,'action':actions});pred=model.predict(z['emb'][:,:3],z['act_emb'][:,:3])
    loss=(pred-z['emb'][:,1:]).square().mean()+.09*reg(z['emb'].transpose(0,1))
   assert torch.isfinite(loss)
   loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
   return loss.detach()
  for _ in range(3):step()
  torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
  for _ in range(10):loss=step()
  torch.cuda.synchronize();seconds=time.perf_counter()-start
  report['rows'].append({'batch':batch,'status':'ok','steps':10,'seconds':seconds,'sequences_per_second':batch*10/seconds,
   'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,'peak_reserved_gib':torch.cuda.max_memory_reserved()/2**30,'last_loss':float(loss)})
 except torch.cuda.OutOfMemoryError:
  report['rows'].append({'batch':batch,'status':'OOM'})
 finally:
  for name in ['model','reg','opt','pixels','actions','loss']:
   globals().pop(name,None)
  gc.collect();torch.cuda.empty_cache()
 print(report['rows'][-1],flush=True)
 Path(a.output).write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
