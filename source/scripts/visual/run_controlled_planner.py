"""Strong privileged-state planner preflight, with full traces and realized trajectories."""
import argparse,hashlib,json,sys,time
from pathlib import Path
import numpy as np
import torch
from controlled_search import search,expand_actions
from state_input_core import make_state_core
from factorial_model import state_features
from evaluation_precision import configure_evaluation_precision
p=argparse.ArgumentParser()
for k in ['training','bank','official','config','simulator','selection','output']:p.add_argument('--'+k,required=True)
p.add_argument('--index',type=int,required=True);p.add_argument('--cases',type=int,default=128);a=p.parse_args();sys.path.insert(0,a.simulator)
from stable_worldmodel.envs.pusht.env import PushT
model_index=[1,3,4][a.index//4];algorithm,parameterization=[('random','held'),('random','full'),('cem','held'),('cem','full')][a.index%4]
run=Path(a.training)/f'job_{model_index}';r=json.loads((run/'summary.json').read_text());frozen=json.loads(Path(a.selection).read_text());weights=run/'best_weights.pt';wh=hashlib.sha256(weights.read_bytes()).hexdigest();assert wh==frozen['all_validation_candidates'][model_index]['weights_sha256']
out=Path(a.output);out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4);precision=configure_evaluation_precision();model=make_state_core(a.official,a.config,r['seed']);model.load_state_dict(torch.load(weights,map_location='cpu',weights_only=True));model=model.cuda().eval()
mn=torch.tensor(r['target_normalization']['mean'],device='cuda');sd=torch.tensor(r['target_normalization']['std'],device='cuda');am=torch.tensor(r['normalization']['mean'],device='cuda').repeat(5);asd=torch.tensor(r['normalization']['std'],device='cuda').repeat(5)
source=out/'source';source.mkdir()
for path in [Path(__file__),Path(__file__).with_name('controlled_search.py'),Path(__file__).with_name('state_input_core.py'),Path(__file__).with_name('factorial_model.py'),Path(__file__).with_name('evaluation_precision.py'),Path(a.simulator)/'stable_worldmodel/envs/pusht/env.py']:(source/path.name).write_bytes(path.read_bytes())
(source/'manifest.json').write_text(json.dumps({f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in source.iterdir()},indent=2)+'\n')
bank=Path(a.bank);manifest=json.loads((bank/'manifest.json').read_text());rows=[];started=time.perf_counter()
def physical_cost(states,goal):
 delta=np.angle(np.exp(1j*(states[...,4]-goal[4])));distance=np.linalg.norm(states[...,2:4]-goal[2:4],axis=-1)
 return distance**2+900*delta**2,(distance<20)&(abs(delta)<np.pi/9)
with torch.inference_mode():
 for item in manifest['cases'][:a.cases]:
  path=bank/f"case_{item['index']:03d}.npz";assert hashlib.sha256(path.read_bytes()).hexdigest()==item['sha256'];z=np.load(path);seed=int(z['seed']);initial=(state_features(torch.tensor(z['history_states'],device='cuda').float())-mn)/sd;prefix=torch.tensor(z['prefix'],device='cuda').reshape(1,2,10);goal=torch.tensor(z['goal_state'],device='cuda').float()
  def cost_function(actions):
   n=len(actions);act=torch.cat([prefix.expand(n,-1,-1),actions.reshape(n,5,10)],1);encoded=model.action_encoder((act-am)/asd);history=initial[None].expand(n,-1,-1).clone()
   for k in range(2,7):history=torch.cat([history,model.predict(history[:,-3:],encoded[:,k-2:k+1])[:,-1:]],1)
   token=history[:,-1];state=token*sd+mn;angle=torch.atan2(state[:,4],state[:,5])-goal[4];angle=torch.atan2(angle.sin(),angle.cos());cost=(state[:,2:4]-goal[2:4]).square().sum(1)+900*angle.square()
   return cost,token
  torch_seed=int(np.random.SeedSequence([seed,955001]).generate_state(1)[0]);torch.cuda.synchronize();t=time.perf_counter();best,trace,final_mean=search(cost_function,algorithm=algorithm,parameterization=parameterization,device='cuda',seed=torch_seed);torch.cuda.synchronize();seconds=time.perf_counter()-t
  actions=expand_actions(best['parameters'][None]);check_cost,check_tokens=cost_function(actions);torch.testing.assert_close(check_tokens[0],best['tokens'],rtol=2e-5,atol=2e-5);actions=actions[0].cpu().numpy()
  env=PushT(resolution=224);obs,_=env.reset(seed=seed);np.testing.assert_array_equal(obs['state'],z['history_states'][0]);history_index=1
  for t,action in enumerate(z['prefix']):
   obs,*_=env.step(action)
   if (t+1)%5==0:np.testing.assert_array_equal(obs['state'],z['history_states'][history_index]);history_index+=1
  states=[obs['state'].copy()];boundary=[]
  for action in actions:
   obs,*_=env.step(action);states.append(obs['state'].copy());outside=False
   for body in [env.agent,env.block]:
    for shape in body.shapes:
     bb=shape.cache_bb();outside|=min(bb.left,bb.bottom)<8 or max(bb.right,bb.top)>504
   boundary.append(outside)
  terminal_pixels=env.render().copy();env.close();states=np.array(states);cost,success=physical_cost(states[-1],z['goal_state']);zero_cost,zero_success=physical_cost(z['terminal_states'][0],z['goal_state'])
  row={'index':item['index'],'seed':seed,'torch_seed':torch_seed,'selected_iteration':best['iteration'],'selected_candidate':best['candidate'],'predicted_cost':best['cost'],'realized_cost':float(cost),'success':bool(success),'zero_cost':float(zero_cost),'zero_success':bool(zero_success),'boundary_steps':int(sum(boundary)),'planning_seconds':seconds,'scored_candidates':9000,'additional_verification_forwards':1};rows.append(row)
  np.savez_compressed(out/f"case_{item['index']:03d}_predictions.npz",**{k:np.stack([v[k] for v in trace]) for k in ['parameters','costs','tokens','mean','std']},selected_actions=actions,selected_states=states,boundary=np.array(boundary),terminal_pixels=terminal_pixels,final_distribution_mean=final_mean.cpu().numpy())
  print(json.dumps(row),flush=True)
report={'model_index':model_index,'algorithm':algorithm,'parameterization':parameterization,'precision':precision,'gpu':torch.cuda.get_device_name(),'cases':rows,'target_normalization':r['target_normalization'],'scope':'privileged-state open-loop development; controlled clipped/best-scored CEM variant, not a published LeWM reproduction','hashes':{'weights':wh,'bank_manifest':hashlib.sha256((bank/'manifest.json').read_bytes()).hexdigest()},'wall_seconds':time.perf_counter()-started}
(out/'summary.json').write_text(json.dumps(report,indent=2)+'\n');(out/'artifact_manifest.json').write_text(json.dumps({f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in out.iterdir() if f.is_file()},indent=2)+'\n');(out/'COMPLETE').write_text('requires independent trace and simulator acceptance\n')
