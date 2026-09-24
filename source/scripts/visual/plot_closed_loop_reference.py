"""Descriptive task-cost trajectories from accepted, replayed control runs."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parents[2];base=root/'runs/closed_loop_full_v1';fig,ax=plt.subplots(figsize=(7.5,4.2))
for name,label,color in [('zero_full','Zero action','#9b8d72'),('uniform_full','Uniform random','#9ba5af'),('released_full','Released LeWM','#24547c')]:
    run=base/name;s=json.loads((run/'summary.json').read_text());assert json.loads((run/'numeric_acceptance.json').read_text())['accepted_cases']==32 and json.loads((run/'replay_acceptance.json').read_text())['accepted_cases']==32
    curves=[]
    for r in s['cases']:
        z=np.load(run/f'case_{r["index"]:03d}_predictions.npz');state,goal=z['states'],z['goal_state'];angle=(state[:,4]-goal[4]+np.pi)%(2*np.pi)-np.pi
        curves.append(np.sum((state[:,2:4]-goal[2:4])**2,axis=1)+900*angle**2)
    mean=np.mean(curves,axis=0);ax.plot(np.arange(len(mean)),mean,label=f'{label} (terminal {mean[-1]:.0f})',color=color,lw=2)
ax.set_xlabel('Executed primitive steps');ax.set_ylabel('Mean block-pose cost (squared pixel units)');ax.set_xlim(0,50);ax.set_ylim(bottom=0);ax.spines[['top','right']].set_visible(False);ax.legend(frameon=False);ax.grid(alpha=.15)
fig.suptitle('Released LeWM improves closed-loop cost over simple policies',fontsize=12)
fig.text(.5,.025,'32 development scenarios • 59.4% initially successful • descriptive means, not training-seed uncertainty',ha='center',fontsize=8)
fig.tight_layout(rect=[0,.07,1,.94]);dest=root/'outputs/figures/maintrack';dest.mkdir(parents=True,exist_ok=True)
for ext in ['png','pdf']:fig.savefig(dest/f'closed_loop_reference.{ext}',dpi=170)
