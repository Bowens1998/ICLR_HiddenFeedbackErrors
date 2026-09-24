"""P0: reload all 18 models and replay fixed old cases at original arithmetic."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'scripts/visual'),str(ROOT/'strengthening/adapters')]
from contracts import atomic_json, sha
from factorial_model import make_model
from adaptation_freeze import configure_dynamics_only, verify_frozen
from evaluation_precision import configure_evaluation_precision
from image_planner_cost import ImagePlannerCost
from feedback_suffix_rollout import rollout_suffixes
from score_feedback_ranking import state_hash


def main():
    p=argparse.ArgumentParser();p.add_argument('--base',required=True);p.add_argument('--output',required=True)
    p.add_argument('--group',type=int,default=None);a=p.parse_args()
    g=int(os.environ['SLURM_ARRAY_TASK_ID']) if a.group is None else a.group
    base=Path(a.base);r=base/'releases/planner-data-adaptation-v1/runs'
    bank=r/f'feedback_ranking_confirmation_v1/bank_{g}'
    scores=r/f'feedback_ranking_confirmation_v1/scores_{g}'
    sr=json.loads((scores/'report.json').read_text())
    bm=json.loads((bank/'manifest.json').read_text())
    plan=json.loads((r/'fiber_confirmation_plan.json').read_text())
    official=base/'releases/visual-v1/official';config=base/'assets/pusht-v1/models/config.json'
    out=Path(a.output)/f'group_{g}';out.mkdir(parents=True,exist_ok=False)
    precision=configure_evaluation_precision();torch.set_num_threads(4);start=time.monotonic();rows=[]
    for mi in [8*g+2,8*g+3,8*g+4]:
        e=plan['models'][mi];td=Path(e['training_path']);tr=json.loads((td/'summary.json').read_text())
        assert sha(td/'last_weights.pt')==e['weights_sha256'] and sha(config)==tr['config_sha256']
        model=make_model(official,config,e['arm'],tr['seed'])
        model.load_state_dict(torch.load(td/'last_weights.pt',map_location='cpu',weights_only=True),strict=True)
        model=model.cuda();boundary=configure_dynamics_only(model);before=state_hash(model)
        mr=next(x for x in sr['models'] if x['model_index']==mi)
        checks=[]
        with torch.inference_mode():
            for i in [0,1]:
                c=bm['cases'][i];fp=bank/c['input_file'];sp=scores/mr['cases'][i]['file']
                assert sha(fp)==c['input_sha256'] and sha(sp)==mr['cases'][i]['sha256']
                z=dict(np.load(fp));saved=dict(np.load(sp))
                cost=ImagePlannerCost(model,z['history_pixels'],z['goal_pixels'],z['prefix'],tr['normalization'],None,'latent')
                pop=torch.as_tensor(z['source_population_actions'],device='cuda')
                enc=model.action_encoder(cost.normalized_actions(pop))
                h=cost.initial[None].expand(300,-1,-1).clone()
                pred=model.predict(h,enc[:,:3])[:,-1]
                selected=int(z['source_selected_index']);root=pred[selected]
                np.testing.assert_array_equal(root.cpu().numpy(),saved['initial_tokens'][0])
                past=torch.as_tensor(np.concatenate([z['prefix'][-5:],z['executed_actions']]),device='cuda')
                suffix=torch.as_tensor(z['suffix_actions'],device='cuda')
                am=torch.tensor(tr['normalization']['mean'],device='cuda');sd=torch.tensor(tr['normalization']['std'],device='cuda')
                replay=rollout_suffixes(model,cost.initial,root,past,suffix,am,sd)
                identity=rollout_suffixes(model,cost.initial,root.clone(),past.clone(),suffix.clone(),am,sd)
                np.testing.assert_array_equal(replay.cpu().numpy(),saved['tokens'][0])
                torch.testing.assert_close(replay,identity,rtol=0,atol=0)
                # Independent explicit rolling calculation at the same candidate batch size.
                raw=torch.cat([past[None].expand(32,-1,-1),suffix],1).reshape(32,6,10)
                ae=model.action_encoder((raw-am.repeat(5))/sd.repeat(5))
                hh=torch.cat([cost.initial[1:],root[None]])[None].expand(32,-1,-1).clone();steps=[]
                for j in range(4):
                    nxt=model.predict(hh[:,-3:],ae[:,j:j+3])[:,-1:];steps.append(nxt[:,0]);hh=torch.cat([hh,nxt],1)
                torch.testing.assert_close(replay,torch.stack(steps),rtol=0,atol=0)
                checks.append(dict(case=i,seed=c['seed'],input_sha256=sha(fp),saved_score_sha256=sha(sp),
                    root_exact=True,all_four_horizons_exact=True,identity_exact=True,explicit_rollout_exact=True,
                    candidate_count=32,first_prediction_batch=300))
        verify_frozen(model,boundary);assert state_hash(model)==before
        rows.append(dict(model_index=mi,weights_sha256=e['weights_sha256'],checks=checks,frozen_unchanged=True))
        del model,boundary;torch.cuda.empty_cache()
    atomic_json(out/'report.json',dict(status='PASS_P0_MODEL_RELOAD_AND_RAW_IMAGE_REPLAY',group=g,models=rows,
        precision=precision,gpu=torch.cuda.get_device_name(),elapsed_seconds=time.monotonic()-start,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),source_sha256=sha(__file__),
        scope='18 model roster across six shards; fixed old cases 0/1, raw-image encoding and original 300/32 arithmetic. Not new confirmation.'))
    (out/'DONE').write_text('accepted\n')


if __name__=='__main__':main()
