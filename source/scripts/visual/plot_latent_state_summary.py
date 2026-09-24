"""Complete paired planner contrasts, retaining independent training sets."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
r=json.loads(Path('outputs/maintrack/latent_state_summary.json').read_text())
fig,axes=plt.subplots(1,2,figsize=(10,4.6),sharey=True)
for ax,arm,label in zip(axes,['transformer_latent_state','gru_latent_state'],['Transformer, 192D state-supervised','GRU, 192D state-supervised']):
    for checkpoint,offset,color,marker in [('best',-.10,'#2166ac','o'),('last',.10,'#b35806','s')]:
        rows=[v for v in r['rows'] if v['arm']==arm and v['checkpoint']==checkpoint and v['algorithm']=='cem']
        cs=[next(c for c in r['contrasts'] if c['cem_route']==v['route']) for v in rows]
        y=np.array([c['mean_cost_difference'] for c in cs]);ci=np.array([c['conditional_scenario_95_percentile_interval'] for c in cs])
        ax.errorbar(np.arange(3)+offset,y,yerr=np.stack([y-ci[:,0],ci[:,1]-y]),fmt=marker,color=color,capsize=3,label=checkpoint)
    ax.axhline(0,color='#555555',lw=1);ax.set_xticks(range(3),['Pool 0','Pool 1','Pool 2']);ax.set_title(label,fontsize=11);ax.grid(axis='y',alpha=.2);ax.legend(frameon=False)
axes[0].set_ylabel('Realized cost: CEM − random (lower is better)')
fig.suptitle('Wider recurrent state does not remove the observed CEM cost increase',fontsize=12)
fig.text(.5,.015,'128 shared development cases; 9,000 scores/planner. Intervals: nominal paired scenario bootstrap.\nThree independent training pools; best/last are correlated checkpoints, not independent repetitions.',ha='center',fontsize=8)
fig.tight_layout(rect=[0,.09,1,.94]);out=Path('outputs/figures/maintrack');out.mkdir(parents=True,exist_ok=True)
for ext in ['png','pdf']:fig.savefig(out/f'latent_state_planner_contrasts.{ext}',dpi=180)
