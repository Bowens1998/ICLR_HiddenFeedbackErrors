"""Isolate construction/freeze flags in the failed original-actor anchor."""
import argparse,json,contextlib
from pathlib import Path
import numpy as np
import torch
from factorial_model import make_model
from image_planner_cost import ImagePlannerCost
from adaptation_freeze import configure_dynamics_only
from evaluation_precision import configure_evaluation_precision
from adaptation_streams import sha


def main():
    p=argparse.ArgumentParser()
    for k in ('plan','runs','bank','official','config','output'):p.add_argument('--'+k,required=True)
    a=p.parse_args();plan=json.loads(Path(a.plan).read_text());route=16;entry=plan['models'][plan['routes'][route]['model_index']];td=Path(entry['training_path']);tr=json.loads((td/'summary.json').read_text());assert sha(td/'last_weights.pt')==entry['weights_sha256']
    d=Path(a.runs)/'job_16';r=json.loads((d/'summary.json').read_text());am=json.loads((d/'artifact_manifest.json').read_text());fp=d/'case_000_predictions.npz';assert sha(fp)==am[fp.name]
    with np.load(fp) as z:
        c=r['cases'][0];parameters=z['parameters'][c['selected_iteration']].copy();expected=z['tokens'][c['selected_iteration'],c['selected_candidate']].copy();expected_goal=z['goal_tokens'].copy()
    bp=Path(a.bank)/'case_000.npz';b=dict(np.load(bp));configure_evaluation_precision();torch.set_num_threads(4)
    weights=torch.load(td/'last_weights.pt',map_location='cpu',weights_only=True);arrays={};rows=[]
    def evaluate(model,label):
        with torch.inference_mode():
            cost=ImagePlannerCost(model,b['history_pixels'],b['goal_pixels'],b['prefix'],tr['normalization'],entry['target_normalization'],'latent');_,tokens=cost(torch.tensor(parameters,device='cuda').reshape(300,25,2));token=tokens[c['selected_candidate']].cpu().numpy();goal=cost.goal.cpu().numpy();initial=cost.initial.cpu().numpy()
        arrays[label+'_tokens']=token;arrays[label+'_goal']=goal;arrays[label+'_initial']=initial
        rows.append(dict(label=label,max_token_difference=float(np.max(abs(token-expected))),max_goal_difference=float(np.max(abs(goal-expected_goal))),max_initial_difference_from_original=float(np.max(abs(initial-arrays.get('original_initial',initial)))),parameter_flags={k:dict(requires_grad=v.requires_grad,is_inference=v.is_inference()) for k,v in list(model.named_parameters())[:1]}))
    model=make_model(a.official,a.config,entry['arm'],tr['seed']);model.load_state_dict(weights);model=model.cuda().eval();evaluate(model,'original')
    boundary=configure_dynamics_only(model);evaluate(model,'same_model_frozen_flags')
    model.load_state_dict(weights);evaluate(model,'same_model_second_gpu_load');del model,boundary;torch.cuda.empty_cache()
    with torch.inference_mode():
        model=make_model(a.official,a.config,entry['arm'],tr['seed']);model.load_state_dict(weights);model=model.cuda();boundary=configure_dynamics_only(model);model.load_state_dict(weights);evaluate(model,'constructed_in_inference')
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False);np.savez_compressed(out/'arrays.npz',expected_tokens=expected,expected_goal=expected_goal,**arrays)
    report=dict(status='FORENSIC_CONSTRUCTION_COMPARISON_ONLY',rows=rows,route=route,case=0,source_sha256=sha(__file__),arrays_sha256=sha(out/'arrays.npz'),plan_sha256=sha(a.plan),archive_sha256=sha(fp),bank_case_sha256=sha(bp),weights_sha256=entry['weights_sha256'],torch_version=torch.__version__,cuda_version=torch.version.cuda,cudnn_version=torch.backends.cudnn.version(),gpu=torch.cuda.get_device_name(),scope='Single-context diagnostic of original construction, frozen requires_grad flags, repeated checkpoint load and inference-mode construction. No amended success or full matched-action result.')
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)

if __name__=='__main__':main()
