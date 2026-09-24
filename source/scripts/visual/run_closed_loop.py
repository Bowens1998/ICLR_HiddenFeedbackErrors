"""Image-only receding-horizon control with auditable candidate banks and action execution."""
import argparse,hashlib,json,sys,time
from pathlib import Path
import numpy as np
import torch
from factorial_model import make_model
from factorial_task_scores import token_scores
from lewm_adapter import build
from evaluation_precision import configure_evaluation_precision

def candidates(seed,replan):
    rng=np.random.default_rng(np.random.SeedSequence([seed,951001,replan]));plans=rng.uniform(-.35,.35,(64,5,2)).astype('float32');plans[0]=0;return plans

def choose(scores,seed,replan,policy):
    if policy=='zero':return 0
    if policy=='uniform':return int(np.random.default_rng(np.random.SeedSequence([seed,951002,replan])).integers(64))
    ties=np.flatnonzero(scores==scores.min())
    return int(np.random.default_rng(np.random.SeedSequence([seed,951003,replan])).choice(ties))

def outcomes(states,goal):
    pos=np.linalg.norm(states[:,2:4]-goal[2:4],axis=1);angle=np.abs(np.arctan2(np.sin(states[:,4]-goal[4]),np.cos(states[:,4]-goal[4])))
    allpos=np.linalg.norm(states[:,:4]-goal[:4],axis=1);cost=pos**2+900*angle**2;success=(pos<20)&(angle<np.pi/9)
    return {'terminal_block_cost':float(cost[-1]),'initial_block_cost':float(cost[0]),'terminal_block_success':bool(success[-1]),'initial_block_success':bool(success[0]),'ever_block_success':bool(success.any()),
            'terminal_agent_block_cost':float(allpos[-1]**2+900*angle[-1]**2),'terminal_agent_block_success':bool((allpos[-1]<20)&(angle[-1]<np.pi/9))}

def main():
    p=argparse.ArgumentParser()
    for key in ['bank','source','output']:p.add_argument('--'+key,required=True)
    p.add_argument('--policy',choices=['zero','uniform','model'],required=True);p.add_argument('--training-run');p.add_argument('--weights');p.add_argument('--normalization');p.add_argument('--official');p.add_argument('--config');p.add_argument('--checkpoint',choices=['best','last'],default='best');p.add_argument('--cases',type=int,default=32);p.add_argument('--replans',type=int,default=10);a=p.parse_args()
    sys.path.insert(0,a.source);from stable_worldmodel.envs.pusht.env import PushT
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4);bank=Path(a.bank);manifest=json.loads((bank/'manifest.json').read_text());assert a.cases<=len(manifest['cases'])
    precision=configure_evaluation_precision();model=None;hashes={'bank_manifest':hashlib.sha256((bank/'manifest.json').read_bytes()).hexdigest()};label=a.policy;norm_target=None
    if a.policy=='model':
        if a.training_run:
            run=Path(a.training_run);assert (run/'COMPLETE').exists();r=json.loads((run/'summary.json').read_text());label=r['arm'];norm=r['normalization'];norm_target=r['target_normalization'];model=make_model(a.official,a.config,label,r['seed']);weights=run/f'{a.checkpoint}_weights.pt'
            model.load_state_dict(torch.load(weights,map_location='cpu',weights_only=True),strict=True);hashes['training_summary']=hashlib.sha256((run/'summary.json').read_bytes()).hexdigest()
        else:
            label='released_jepa_reference';weights=Path(a.weights);model=build(a.official,a.config,weights);norm=json.loads(Path(a.normalization).read_text())['action'];hashes['normalization']=hashlib.sha256(Path(a.normalization).read_bytes()).hexdigest()
        hashes['weights']=hashlib.sha256(weights.read_bytes()).hexdigest();model=model.cuda().eval();am=torch.tensor(norm['mean'],device='cuda').repeat(5);sd=torch.tensor(norm['std'],device='cuda').repeat(5)
        im=torch.tensor([.485,.456,.406],device='cuda')[None,:,None,None];isd=torch.tensor([.229,.224,.225],device='cuda')[None,:,None,None]
        def preprocess(frames):return (torch.from_numpy(np.array(frames)).cuda().permute(0,3,1,2).float()/255-im)/isd
        def encode(frames):return model.encode({'pixels':preprocess(frames)[:,None]})['emb'][:,0]
    source=out/'source';source.mkdir()
    for file in [Path(__file__),Path(__file__).with_name('factorial_model.py'),Path(__file__).with_name('factorial_task_scores.py'),Path(__file__).with_name('lewm_adapter.py'),Path(__file__).with_name('evaluation_precision.py'),Path(a.source)/'stable_worldmodel/envs/pusht/env.py']:
        (source/file.name).write_bytes(file.read_bytes())
    if model:
        for file in [Path(a.official)/'jepa.py',Path(a.official)/'module.py',Path(a.config)]:(source/file.name).write_bytes(file.read_bytes())
    (source/'manifest.json').write_text(json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in source.iterdir()},indent=2)+'\n')
    rows=[];started=time.perf_counter()
    for item in manifest['cases'][:a.cases]:
        path=bank/f"case_{item['index']:03d}.npz";assert hashlib.sha256(path.read_bytes()).hexdigest()==item['sha256'];z=np.load(path);seed=int(z['seed']);env=PushT(resolution=224);obs,_=env.reset(seed=seed);frames=[env.render().copy()];history_states=[obs['state'].copy()]
        for i,action in enumerate(z['prefix']):
            obs,*_=env.step(action)
            if (i+1)%5==0:frames.append(env.render().copy());history_states.append(obs['state'].copy())
        np.testing.assert_array_equal(frames,z['history_pixels']);np.testing.assert_array_equal(history_states,z['history_states'])
        actions=[];states=[obs['state'].copy()];plans_log=[];scores_log=[];chosen=[];predicted=[];boundary=[];past=z['prefix'].copy()
        with torch.inference_mode():
            goal=encode(z['goal_pixels'][None])[0].cpu().numpy() if model else np.empty(0)
            for replan in range(a.replans):
                plans=candidates(seed,replan);scores=np.zeros(64,dtype='float64')
                if model:
                    emb=encode(frames[-3:])[None].expand(64,-1,-1).clone();act=np.concatenate([np.broadcast_to(past.reshape(1,2,10),(64,2,10)),np.repeat(plans,5,axis=1).reshape(64,5,10)],1);act=(torch.from_numpy(act).cuda()-am)/sd
                    for k in range(2,7):emb=torch.cat([emb,model.predict(emb[:,-3:],model.action_encoder(act[:,k-2:k+1]))[:,-1:]],1)
                    if item['index']==0 and replan==0:
                        native=model.rollout({'pixels':preprocess(frames[-3:])[None,None].expand(1,64,-1,-1,-1,-1)},act[None])['predicted_emb'][0]
                        torch.testing.assert_close(emb,native,rtol=2e-5,atol=2e-5)
                    pred=emb[:,-1].cpu().numpy();scoring_arm=label if label.endswith('state') else 'jepa';scores=token_scores(pred,goal,scoring_arm,norm_target)['block_pose'];predicted.append(pred)
                assert np.isfinite(scores).all();selected=choose(scores,seed,replan,a.policy);plans_log.append(plans);scores_log.append(scores);chosen.append(selected)
                execute=np.repeat(plans[selected,0][None],5,axis=0)
                for action in execute:
                    obs,*_=env.step(action);actions.append(action.copy());states.append(obs['state'].copy())
                    outside=False
                    for body in [env.agent,env.block]:
                        for shape in body.shapes:
                            bb=shape.cache_bb();outside|=min(bb.left,bb.bottom)<0 or max(bb.right,bb.top)>512
                    boundary.append(bool(outside))
                past=np.concatenate([past,execute])[-10:];frames.append(env.render().copy())
        env.close();metrics=outcomes(np.array(states),z['goal_state']);metrics['boundary_exit_steps']=sum(boundary)
        np.savez_compressed(out/f"case_{item['index']:03d}_predictions.npz",seed=seed,states=np.array(states),actions=np.array(actions),frames=np.array(frames),plans=np.array(plans_log),scores=np.array(scores_log),selected=np.array(chosen),predicted_tokens=np.array(predicted),goal_tokens=goal,boundary=np.array(boundary),goal_state=z['goal_state'])
        row={'index':item['index'],'seed':seed,'prefix_replay_verified':True,**metrics};rows.append(row);print(json.dumps(row),flush=True)
    summary={'policy':a.policy,'model':label,'checkpoint':a.checkpoint if a.training_run else None,'replans':a.replans,'candidates':64,'executed_steps':5*a.replans,'lookahead_steps':25,'cases':rows,'target_normalization':norm_target,'precision':precision,'hashes':hashes,'wall_seconds':time.perf_counter()-started,'gpu':torch.cuda.get_device_name() if model else None,'scope':'developmental closed loop; no true-state candidate scoring or stopping; no independent training-seed claim'}
    summary['aggregate']={k:float(np.mean([r[k] for r in rows])) for k in metrics}
    (out/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n');(out/'artifact_manifest.json').write_text(json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir() if p.is_file()},indent=2)+'\n');(out/'COMPLETE').write_text('requires independent numeric and simulator replay acceptance\n')
if __name__=='__main__':main()
