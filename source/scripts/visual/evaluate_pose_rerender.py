"""Same-state original/rerender validation intervention, frozen encoder and head."""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
import torch
from factorial_model import make_model
from nonlinear_pose_cost import numpy_pose
from evaluation_precision import configure_evaluation_precision


def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()


def main():
 p=argparse.ArgumentParser()
 for k in ['release','validation','simulator','official','config','output']:p.add_argument('--'+k,required=True)
 p.add_argument('--index',type=int,choices=range(6),required=True);a=p.parse_args();root=Path(a.release)
 plan=json.loads((root/'planning_plan.json').read_text());entry=plan['models'][plan['routes'][a.index*8+2]['model_index']];assert entry['score']=='pose_encoded'
 training=Path(entry['training_path']);assert sha(training/'last_weights.pt')==entry['weights_sha256']
 assert sha(training/'summary.json')==entry['training_summary_sha256'];tr=json.loads((training/'summary.json').read_text());assert sha(a.config)==tr['config_sha256']
 hp=Path(entry['goal_head']['path']);assert sha(hp)==entry['goal_head']['sha256'];head=dict(np.load(hp))
 ff=root/'runs/features'/f'job_{a.index}';fr=json.loads((ff/'report.json').read_text());assert sha(ff/'validation_features.npz')==fr['files_sha256']['validation_features.npz'];features=np.load(ff/'validation_features.npz')
 assert fr['weights_sha256']==entry['weights_sha256']
 folder=Path(a.validation);episodes=np.load(folder/'episodes.npz');pixels=np.load(folder/'pixels.npy',mmap_mode='r');states=np.load(folder/'state.npy',mmap_mode='r')
 mapping={int(e):(int(o),int(n)) for e,o,n in zip(episodes['source_episode_ids'],episodes['offsets'],episodes['lengths'])}
 indices=[]
 for e,t in features['identity']:
  o,n=mapping[int(e)];assert int(t)+35<n;indices.append(o+int(t)+35)
 assert len(indices)==1185
 sys.path.insert(0,a.simulator)
 from stable_worldmodel.envs.pusht.env import PushT
 precision=configure_evaluation_precision();torch.set_num_threads(2)
 model=make_model(a.official,a.config,entry['arm'],tr['seed']);model.load_state_dict(torch.load(training/'last_weights.pt',map_location='cpu',weights_only=True),strict=True);model=model.cuda().eval()
 env=PushT(resolution=224);env.reset(seed=0)
 im=torch.tensor([.485,.456,.406],device='cuda')[None,:,None,None];sd=torch.tensor([.229,.224,.225],device='cuda')[None,:,None,None]
 outputs={k:[] for k in ['original','restored','pixel_mae']};pixel_hashes=[]
 with torch.inference_mode():
  for begin in range(0,len(indices),64):
   ix=indices[begin:begin+64];original=np.array(pixels[ix]);restored=[]
   for j in ix:
    state=states[j];env.agent.position=state[:2].tolist();env.agent.velocity=state[-2:].tolist();env.block.angle=float(state[4]);env.block.position=state[2:4].tolist()
    env.space.reindex_shapes_for_body(env.agent);env.space.reindex_shapes_for_body(env.block)
    actual=env._get_obs();np.testing.assert_allclose(actual[:4],state[:4],rtol=0,atol=1e-10)
    assert abs(np.angle(np.exp(1j*(actual[4]-state[4]))))<1e-10
    restored.append(env.render().copy())
   restored=np.array(restored);outputs['pixel_mae'].append(np.abs(original.astype(float)-restored.astype(float)).mean((1,2,3)))
   pixel_hashes.append(hashlib.sha256(restored.tobytes()).hexdigest())
   for kind,ims in [('original',original),('restored',restored)]:
    x=(torch.tensor(ims,device='cuda').permute(0,3,1,2).float()/255-im)/sd
    token=model.encode({'pixels':x[:,None]})['emb'][:,0].cpu().numpy()
    if kind=='original':np.testing.assert_allclose(token,features['encoded'][begin:begin+len(ix)],rtol=2e-5,atol=2e-5)
    outputs[kind].append(numpy_pose(token,head))
 env.close();out=Path(a.output)/f'job_{a.index}';out.mkdir(parents=True,exist_ok=False)
 values={k:np.concatenate(v) for k,v in outputs.items()};values.update(target=features['target'],identity=features['identity'])
 np.savez_compressed(out/'predictions.npz',**values)
 result=dict(status='COMPLETE_PAIRED_RENDER',index=a.index,examples=1185,weights_sha256=entry['weights_sha256'],head_sha256=sha(hp),plan_sha256=sha(root/'planning_plan.json'),feature_report_sha256=sha(ff/'report.json'),files_sha256={'predictions.npz':sha(out/'predictions.npz')},rerender_batch_hashes=pixel_hashes,precision=precision,source_sha256=sha(__file__),simulator_sha256=sha(Path(a.simulator)/'stable_worldmodel/envs/pusht/env.py'),scope='Same1185 validation physical poses, original image versus current renderer, frozen model/head; original tokens checked against accepted extraction. No physics step, labels or retraining. Independent output/metric acceptance required.')
 (out/'report.json').write_text(json.dumps(result,indent=2)+'\n');print('COMPLETE',a.index)


if __name__=='__main__':main()
