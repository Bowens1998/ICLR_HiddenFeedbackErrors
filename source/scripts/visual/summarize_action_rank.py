"""Accepted visual pilot/reference comparison, with explicitly post-hoc controls."""
import argparse,json
from pathlib import Path
import numpy as np

def main():
    p=argparse.ArgumentParser();p.add_argument('--pilot',required=True);p.add_argument('--released',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    result={'scope':'development only; released and pilot model training budgets differ; not a fair algorithm comparison','models':{},
            'posthoc_controls':'Zero action and cross-scenario goal shuffle were added after inspecting first pilot ranks. They are exploratory controls, not confirmation.'}
    lines=['# Native visual action ranking: accepted development diagnostic','',result['scope'],'',
           '| Model / score | Block rank ρ | Block selected cost | Agent + block rank ρ | Agent + block selected cost |','|---|---:|---:|---:|---:|']
    for label,folder in [('eight_epoch_pilot',a.pilot),('official_released_reference',a.released)]:
        folder=Path(folder);s=json.loads((folder/'summary.json').read_text());accept=json.loads((folder/'acceptance.json').read_text());assert accept['accepted_cases']==32
        files=[np.load(folder/f'case_{r["index"]:03d}_predictions.npz') for r in s['cases']];rng=np.random.default_rng(872009)
        while True:
            permutation=rng.permutation(len(files))
            if np.all(permutation!=np.arange(len(files))):break
        zero={task:float(np.mean([z['cost_'+task][0] for z in files])) for task in s['aggregate']};controls={}
        for key in ['predicted_embedding','realized_embedding']:
            rows=[]
            for i,z in enumerate(files):
                # Recompute both scores in float64 so arithmetic is matched in this control.
                emb=z[key].astype('float64');true=np.square(emb-z['goal_embedding']).sum(1);shuffled=np.square(emb-files[permutation[i]]['goal_embedding']).sum(1)
                sel=np.flatnonzero(true==true.min());other=np.flatnonzero(shuffled==shuffled.min())
                rows.append({'index':i,'other_goal_index':int(permutation[i]),'identical_selected_set':np.array_equal(sel,other),
                             'tasks':{task:{'true_goal':float(z['cost_'+task][sel].mean()),'shuffled_goal':float(z['cost_'+task][other].mean())} for task in zero}})
            controls[key]={'identical_selected_sets':sum(r['identical_selected_set'] for r in rows),
                           'mean_costs':{task:{condition:float(np.mean([r['tasks'][task][condition] for r in rows])) for condition in ['true_goal','shuffled_goal']} for task in zero},'cases':rows}
        result['models'][label]={'aggregate':s['aggregate'],'zero_action_mean_cost':zero,'goal_shuffle_float64':controls,'scenario_quality':accept['scenario_quality']}
        for method in s['aggregate']['block_pose']:
            b=s['aggregate']['block_pose'][method];c=s['aggregate']['agent_and_block_pose'][method]
            fmt=lambda x:'—' if x is None else f'{x:.3f}'
            lines.append(f'| {label} / {method} | {fmt(b["spearman"])} | {b["selected_cost"]:.1f} | {fmt(c["spearman"])} | {c["selected_cost"]:.1f} |')
        lines.append(f'| {label} / zero action (post hoc) | — | {zero["block_pose"]:.1f} | — | {zero["agent_and_block_pose"]:.1f} |')
    lines+=['','Costs are the declared squared-pixel diagnostic costs. Realized-latent and realized-pixel scores use true future images and are oracle ablations, not deployed baselines.',
            'Both models use exactly the same 32 cases × 32 plans. All cases retained. No independent training-seed uncertainty is available.','',result['posthoc_controls']]
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True);out.with_suffix('.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');out.with_suffix('.md').write_text('\n'.join(lines)+'\n')
if __name__=='__main__':main()
