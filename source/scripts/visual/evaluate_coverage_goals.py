"""Evaluate accepted frozen goal heads on the independent prospective bank."""
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
 for k in ['config','plan','fits','bank','official','model-config','output']:p.add_argument('--'+k,required=True)
 p.add_argument('--index',type=int,choices=range(6),required=True);a=p.parse_args()
 cfg=json.loads(Path(a.config).read_text());plan=json.loads(Path(a.plan).read_text());entry=plan['models'][plan['routes'][a.index*8+2]['model_index']]
 fits=Path(a.fits)/f'job_{a.index}';fr=json.loads((fits/'report.json').read_text());fa=json.loads((fits/'acceptance.json').read_text())
 assert fr['index']==a.index and not fr['engineering'] and not fa['engineering'] and fa['status']=='PASS'
 assert fa['report_sha256']==sha(fits/'report.json') and fa['verifier_sha256']==sha(Path(__file__).with_name('accept_coverage_readouts.py'))
 assert fr['plan_sha256']==sha(a.plan) and fr['config_sha256']==sha(a.config) and fr['backbone_sha256']==entry['weights_sha256']
 bank=Path(a.bank);bm=json.loads((bank/'manifest.json').read_text());ba=json.loads((bank/'acceptance.json').read_text())
 assert ba['status']=='PASS' and ba['manifest_sha256']==sha(bank/'manifest.json') and ba['config_sha256']==sha(a.config)
 assert ba['verifier_sha256']==sha(Path(__file__).with_name('accept_coverage_goals.py')) and ba['cases']==len(bm['cases'])==cfg['evaluation_goals']==256
 assert bm['seed_start']==cfg['evaluation_seed_start']
 training=Path(entry['training_path']);assert sha(training/'last_weights.pt')==entry['weights_sha256'] and sha(training/'summary.json')==entry['training_summary_sha256']
 tr=json.loads((training/'summary.json').read_text());assert sha(a.model_config)==tr['config_sha256']
 precision=configure_evaluation_precision();torch.set_num_threads(2)
 model=make_model(a.official,a.model_config,entry['arm'],tr['seed']);model.load_state_dict(torch.load(training/'last_weights.pt',map_location='cpu',weights_only=True),strict=True);model=model.cuda().eval()
 images=[];targets=[];seeds=[]
 for item in bm['cases']:
  f=bank/f"case_{item['index']:03d}.npz";assert sha(f)==item['sha256'];z=np.load(f);s=z['goal_state']
  images.append(z['goal_pixels']);targets.append(np.r_[s[:4],np.sin(s[4]),np.cos(s[4])]);seeds.append(item['seed'])
 images=np.array(images);tokens=[]
 with torch.inference_mode():
  for start in range(0,256,64):
   x=torch.tensor(images[start:start+64],device='cuda').permute(0,3,1,2).float()/255
   x=(x-torch.tensor([.485,.456,.406],device='cuda')[None,:,None,None])/torch.tensor([.229,.224,.225],device='cuda')[None,:,None,None]
   tokens.append(model.encode({'pixels':x[:,None]})['emb'][:,0].cpu().numpy())
 tokens=np.concatenate(tokens);out=Path(a.output)/f'job_{a.index}';out.mkdir(parents=True,exist_ok=False)
 np.savez_compressed(out/'goals.npz',tokens=tokens,target=np.array(targets),seed=np.array(seeds));rows=[]
 for arm in ['expert','broad','original_reference']:
  if arm=='original_reference':hp=Path(entry['goal_head']['path']);expected=entry['goal_head']['sha256']
  else:
   hp=fits/arm/'weights.npz';row=next(v for v in fr['rows'] if v['arm']==arm);assert row['updates']==2000 and row['labels']==512;expected=row['files_sha256']['weights.npz']
  assert sha(hp)==expected;head=dict(np.load(hp));h={k:torch.tensor(v,device='cuda',dtype=torch.float64) for k,v in head.items()}
  with torch.inference_mode():pred=NonlinearPoseCost.forward(torch.tensor(tokens,device='cuda'),h).cpu().numpy()
  np.testing.assert_allclose(pred,numpy_pose(tokens,head),rtol=1e-10,atol=1e-9);np.savez_compressed(out/f'{arm}.npz',prediction=pred)
  rows.append(dict(arm=arm,head_sha256=expected,file=f'{arm}.npz',file_sha256=sha(out/f'{arm}.npz')))
 result=dict(status='COMPLETE_FROZEN_GOAL_EVALUATION',index=a.index,rows=rows,goal_file_sha256=sha(out/'goals.npz'),fit_report_sha256=sha(fits/'report.json'),fit_acceptance_sha256=sha(fits/'acceptance.json'),bank_manifest_sha256=sha(bank/'manifest.json'),bank_acceptance_sha256=sha(bank/'acceptance.json'),plan_sha256=sha(a.plan),config_sha256=sha(a.config),precision=precision,source_sha256=sha(__file__),scope='All256 fresh goals, two matched512-label heads and larger-label original reference. Torch float64 outputs match NumPy; independent complete-matrix metric analysis required before interpreting manipulation gate.')
 (out/'report.json').write_text(json.dumps(result,indent=2)+'\n');print('EVALUATED_THREE_FROZEN_HEADS',a.index)


if __name__=='__main__':main()
