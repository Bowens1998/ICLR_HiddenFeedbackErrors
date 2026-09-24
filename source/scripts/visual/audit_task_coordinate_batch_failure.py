"""Forensic replay of a failed batch/single-token check; no policy replacement."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from factorial_model import make_model
from image_planner_cost import ImagePlannerCost
from controlled_search import search,expand_actions
from evaluation_precision import configure_evaluation_precision
from adaptation_streams import sha


def main():
    p=argparse.ArgumentParser()
    for k in ('plan','bank','official','config','output'):p.add_argument('--'+k,required=True)
    p.add_argument('--route',type=int,default=31);p.add_argument('--case',type=int,default=22);a=p.parse_args()
    plan=json.loads(Path(a.plan).read_text());route=plan['routes'][a.route];e=plan['models'][route['model_index']];assert e['score']=='state'
    td=Path(e['training_path']);tr=json.loads((td/'summary.json').read_text());assert sha(td/'summary.json')==e['training_summary_sha256'] and sha(td/'last_weights.pt')==e['weights_sha256']
    configure_evaluation_precision();torch.set_num_threads(4)
    model=make_model(a.official,a.config,e['arm'],tr['seed']);model.load_state_dict(torch.load(td/'last_weights.pt',map_location='cpu',weights_only=True));model=model.cuda().eval()
    bank=Path(a.bank);manifest=json.loads((bank/'manifest.json').read_text());assert sha(bank/'manifest.json')==plan['bank_manifest_sha256'];item=manifest['cases'][a.case];fp=bank/f'case_{a.case:03d}.npz';assert sha(fp)==item['sha256'];z=np.load(fp)
    cost=ImagePlannerCost(model,z['history_pixels'],z['goal_pixels'],z['prefix'],tr['normalization'],e['target_normalization'],'state')
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    with torch.inference_mode():
        native=torch.zeros(2,25,2,device='cuda');native[1]=.1;cost.verify_native(native)
        seed=int(np.random.SeedSequence([int(z['seed']),955001]).generate_state(1)[0]);best,trace,_=search(cost,algorithm=route['algorithm'],parameterization=route['parameterization'],device='cuda',seed=seed)
        population=torch.tensor(trace[best['iteration']]['parameters'],device='cuda');actions=expand_actions(best['parameters'][None]);checks={};arrays={}
        for label,action_batch,index in [('single',actions,0),('original_population',expand_actions(population),best['candidate']),('repeated_selected_300',actions.expand(300,-1,-1).clone(),best['candidate'])]:
            scores,tokens=cost(action_batch);token=tokens[index];diff=abs(token-best['tokens']);tol=2e-5+2e-5*abs(best['tokens'])
            checks[label]=dict(max_absolute_token_difference=float(diff.max()),failed_elements=int((diff>tol).sum()),matches_original_tolerance=bool((diff<=tol).all()),cost=float(scores[index]),cost_difference=float(scores[index])-best['cost'])
            arrays[label+'_tokens']=tokens.cpu().numpy();arrays[label+'_costs']=scores.cpu().numpy()
        arrays.update(selected_tokens=best['tokens'].cpu().numpy(),selected_parameters=best['parameters'].cpu().numpy(),population_parameters=population.cpu().numpy(),population_original_tokens=trace[best['iteration']]['tokens'],population_original_costs=trace[best['iteration']]['costs'])
        np.savez_compressed(out/'replay.npz',**arrays)
    report=dict(status='FORENSIC_ONLY_NO_EVALUATION_ACCEPTANCE',route=a.route,case=a.case,seed=int(z['seed']),selected_iteration=best['iteration'],selected_candidate=best['candidate'],original_search_cost=best['cost'],checks=checks,plan_sha256=sha(a.plan),weights_sha256=e['weights_sha256'],case_sha256=sha(fp),source_sha256=sha(__file__),replay_sha256=sha(out/'replay.npz'),gpu=torch.cuda.get_device_name(),scope='Replays unchanged9000-candidate search on one failed context; compares selected-token arithmetic at batch1, original population300 and repeated action300. No tolerance change, full-route success, policy replacement or physical-outcome inference.')
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)

if __name__=='__main__':main()
