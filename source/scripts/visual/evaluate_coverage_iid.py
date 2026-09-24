"""Evaluate accepted frozen goal heads on the independent static-pose holdout."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import torch
from factorial_model import make_model
from nonlinear_pose_cost import NonlinearPoseCost,numpy_pose
from evaluation_precision import configure_evaluation_precision


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
 p=argparse.ArgumentParser()
 for k in ['config','holdout-config','plan','fits','bank','official','model-config','output']:p.add_argument('--'+k,required=True)
 p.add_argument('--index',type=int,choices=range(6),required=True);a=p.parse_args()
 cfg=json.loads(Path(a.config).read_text());plan=json.loads(Path(a.plan).read_text());entry=plan['models'][plan['routes'][a.index*8+2]['model_index']]
 fits=Path(a.fits)/f'job_{a.index}';fr=json.loads((fits/'report.json').read_text());fa=json.loads((fits/'acceptance.json').read_text())
 assert fr['index']==a.index and not fr['engineering'] and not fa['engineering'] and fa['status']=='PASS'
 assert fa['report_sha256']==sha(fits/'report.json') and fa['verifier_sha256']==sha(Path(__file__).with_name('accept_coverage_readouts.py'))
 assert fr['plan_sha256']==sha(a.plan) and fr['config_sha256']==sha(a.config) and fr['backbone_sha256']==entry['weights_sha256']
 bank=Path(a.bank);bm=json.loads((bank/'report.json').read_text());ba=json.loads((bank/'acceptance.json').read_text())
 hc=json.loads(Path(a.holdout_config).read_text())
 assert hc['broad_sampling_seeds']==[1231001,1231002,1231003]
 assert not set(hc['broad_sampling_seeds']) & set(cfg['broad_sampling_seeds'])
 assert ba['status']=='PASS' and not ba['engineering'] and ba['report_sha256']==sha(bank/'report.json')
 assert ba['verifier_sha256']==sha(Path(__file__).with_name('accept_coverage_poses.py'))
 assert bm['config_sha256']==sha(a.holdout_config) and bm['replica']==0 and not bm['engineering']
 row=next(v for v in bm['rows'] if v['arm']=='broad');assert row['count']==512
 for name,expected in row['files_sha256'].items():assert sha(bank/'broad'/name)==expected
 training=Path(entry['training_path']);assert sha(training/'last_weights.pt')==entry['weights_sha256'] and sha(training/'summary.json')==entry['training_summary_sha256']
 tr=json.loads((training/'summary.json').read_text());assert sha(a.model_config)==tr['config_sha256']
 precision=configure_evaluation_precision();torch.set_num_threads(2)
 model=make_model(a.official,a.model_config,entry['arm'],tr['seed']);model.load_state_dict(torch.load(training/'last_weights.pt',map_location='cpu',weights_only=True),strict=True);model=model.cuda().eval()
 images=np.load(bank/'broad/pixels.npy');z=np.load(bank/'broad/poses.npz');targets=z['target'];source_index=z['source_index'];tokens=[]
 assert images.shape==(512,224,224,3) and targets.shape==(512,6) and len(set(source_index))==512
 with torch.inference_mode():
  for start in range(0,512,64):
   x=torch.tensor(images[start:start+64],device='cuda').permute(0,3,1,2).float()/255
   x=(x-torch.tensor([.485,.456,.406],device='cuda')[None,:,None,None])/torch.tensor([.229,.224,.225],device='cuda')[None,:,None,None]
   tokens.append(model.encode({'pixels':x[:,None]})['emb'][:,0].cpu().numpy())
 tokens=np.concatenate(tokens);out=Path(a.output)/f'job_{a.index}';out.mkdir(parents=True,exist_ok=False)
 np.savez_compressed(out/'goals.npz',tokens=tokens,target=np.array(targets),source_index=source_index);rows=[]
 for arm in ['expert','broad','original_reference']:
  if arm=='original_reference':hp=Path(entry['goal_head']['path']);expected=entry['goal_head']['sha256']
  else:
   hp=fits/arm/'weights.npz';row=next(v for v in fr['rows'] if v['arm']==arm);assert row['updates']==2000 and row['labels']==512;expected=row['files_sha256']['weights.npz']
  assert sha(hp)==expected;head=dict(np.load(hp));h={k:torch.tensor(v,device='cuda',dtype=torch.float64) for k,v in head.items()}
  with torch.inference_mode():pred=NonlinearPoseCost.forward(torch.tensor(tokens,device='cuda'),h).cpu().numpy()
  np.testing.assert_allclose(pred,numpy_pose(tokens,head),rtol=1e-10,atol=1e-9);np.savez_compressed(out/f'{arm}.npz',prediction=pred)
  rows.append(dict(arm=arm,head_sha256=expected,file=f'{arm}.npz',file_sha256=sha(out/f'{arm}.npz')))
 result=dict(status='COMPLETE_FROZEN_IID_EVALUATION',index=a.index,rows=rows,goal_file_sha256=sha(out/'goals.npz'),fit_report_sha256=sha(fits/'report.json'),fit_acceptance_sha256=sha(fits/'acceptance.json'),bank_report_sha256=sha(bank/'report.json'),holdout_config_sha256=sha(a.holdout_config),bank_acceptance_sha256=sha(bank/'acceptance.json'),plan_sha256=sha(a.plan),config_sha256=sha(a.config),precision=precision,source_sha256=sha(__file__),scope='All512 independent static poses, two matched512-label heads and larger-label original reference. Torch float64 outputs match NumPy; independent complete-matrix metric analysis required before interpreting manipulation gate.')
 (out/'report.json').write_text(json.dumps(result,indent=2)+'\n');print('EVALUATED_THREE_FROZEN_HEADS',a.index)


if __name__=='__main__':main()
