"""Reconstruct coverage-head normalization, raw predictions and native features."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import torch
from factorial_model import make_model
from evaluation_precision import configure_evaluation_precision


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
 p=argparse.ArgumentParser()
 for k in ['run','data','plan','config','official','model-config']:p.add_argument('--'+k,required=True)
 a=p.parse_args();out=Path(a.run);r=json.loads((out/'report.json').read_text());cfg=json.loads(Path(a.config).read_text());plan=json.loads(Path(a.plan).read_text())
 assert r['schema']=='pusht_coverage_goal_heads_v1' and r['config_sha256']==sha(a.config) and r['plan_sha256']==sha(a.plan)
 for f,h in r['source_sha256'].items():assert sha(Path(__file__).with_name(f))==h
 entry=plan['models'][plan['routes'][r['index']*8+2]['model_index']];assert entry['weights_sha256']==r['backbone_sha256']
 data=Path(a.data)/f"replica_{r['replica']}";assert sha(data/'report.json')==r['data_report_sha256'] and sha(data/'acceptance.json')==r['data_acceptance_sha256']
 dr=json.loads((data/'report.json').read_text());da=json.loads((data/'acceptance.json').read_text());assert da['status']=='PASS' and not dr['engineering'] and not da['engineering']
 hp=Path(entry['goal_head']['path']);assert sha(hp)==r['normalization_reference_sha256']==entry['goal_head']['sha256'];reference=np.load(hp)
 trpath=Path(entry['training_path']);assert sha(trpath/'last_weights.pt')==r['backbone_sha256'] and sha(trpath/'summary.json')==entry['training_summary_sha256']
 tr=json.loads((trpath/'summary.json').read_text());assert sha(a.model_config)==tr['config_sha256']
 assert configure_evaluation_precision()==r['precision'];torch.set_num_threads(2)
 model=make_model(a.official,a.model_config,entry['arm'],tr['seed']);model.load_state_dict(torch.load(trpath/'last_weights.pt',map_location='cpu',weights_only=True),strict=True);model=model.cuda().eval()
 initial_reference=None;rows=[]
 assert [x['arm'] for x in r['rows']]==['expert','broad']
 for row in r['rows']:
  arm=row['arm'];sub=out/arm
  assert row['updates']==(20 if r['engineering'] else cfg['fit_updates']) and row['labels']==512 and row['batch_size']==cfg['batch_size'] and row['learning_rate']==cfg['learning_rate'] and row['seed']==430001+r['index']
  for name,h in row['files_sha256'].items():assert sha(sub/name)==h
  datarow=next(v for v in dr['rows'] if v['arm']==arm)
  for name,h in datarow['files_sha256'].items():assert sha(data/arm/name)==h
  h=np.load(sub/'weights.npz');initial=np.load(sub/'initial.npz');f=np.load(sub/'features.npz');poses=np.load(data/arm/'poses.npz');pixels=np.load(data/arm/'pixels.npy',mmap_mode='r')
  np.testing.assert_array_equal(f['target'],poses['target']);np.testing.assert_array_equal(f['source_index'],poses['source_index'])
  for key in ['mean','scale']:np.testing.assert_array_equal(h[key],reference[key])
  np.testing.assert_array_equal(h['target_mean'],[256,256,256,256,0,0]);np.testing.assert_array_equal(h['target_scale'],[256,256,256,256,1,1])
  if initial_reference is None:initial_reference={k:initial[k] for k in initial.files}
  else:
   assert set(initial.files)==set(initial_reference)
   for k in initial.files:np.testing.assert_array_equal(initial[k],initial_reference[k])
  y=(f['encoded'].astype(float)-h['mean'])/h['scale']
  assert f['encoded'].shape==(512,192)
  for layer,nin,nout in [(0,192,256),(2,256,256),(4,256,6)]:
   assert h[f'{layer}.weight'].shape==initial[f'{layer}.weight'].shape==(nout,nin)
   assert h[f'{layer}.bias'].shape==initial[f'{layer}.bias'].shape==(nout,)
   y=y@h[f'{layer}.weight'].astype(float).T+h[f'{layer}.bias'].astype(float)
   if layer<4:y=np.maximum(y,0)
  pred=y*h['target_scale']+h['target_mean'];saved=np.load(sub/'train_prediction.npz')
  np.testing.assert_allclose(pred,saved['prediction'],rtol=1e-10,atol=1e-9);mse=float(np.square((pred-f['target'])/h['target_scale']).mean());np.testing.assert_allclose(mse,row['train_standardized_mse'],rtol=1e-10)
  maxdiff=0.
  with torch.inference_mode():
   for start in range(0,512,64):
    x=torch.tensor(np.array(pixels[start:start+64]),device='cuda').permute(0,3,1,2).float()/255
    x=(x-torch.tensor([.485,.456,.406],device='cuda')[None,:,None,None])/torch.tensor([.229,.224,.225],device='cuda')[None,:,None,None]
    encoded=model.encode({'pixels':x[:,None]})['emb'][:,0].cpu().numpy();np.testing.assert_allclose(encoded,f['encoded'][start:start+64],rtol=2e-5,atol=2e-5);maxdiff=max(maxdiff,float(np.max(np.abs(encoded-f['encoded'][start:start+64]))))
  rows.append(dict(arm=arm,labels=512,all_native_features_verified=True,max_feature_difference=maxdiff,train_mse=mse))
 result=dict(status='PASS',engineering=r['engineering'],report_sha256=sha(out/'report.json'),verifier_sha256=sha(__file__),rows=rows,scope='All encoded features recomputed from accepted images; same initialization/normalization, labels and raw-head predictions reconstructed. Does not rerun optimization or evaluate held-out goals.')
 (out/'acceptance.json').write_text(json.dumps(result,indent=2)+'\n');print('PASS_TWO_COVERAGE_HEADS')


if __name__=='__main__':main()
