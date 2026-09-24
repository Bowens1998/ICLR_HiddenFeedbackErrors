"""Task-definition sensitivity of fixed actions; optimizer seeds stay visible."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
r=json.loads(Path('outputs/maintrack/goal_interpretation_audit.json').read_text())
arms=['transformer_jepa','gru_jepa','transformer_state','gru_state'];labels=['T JEPA','GRU JEPA','T state','GRU state']
fig,axes=plt.subplots(1,3,figsize=(12,4.5))
for ax,metric,title in zip(axes,['block_cost','agent_position_cost','joint_pose_cost'],['Original block goal','Agent position only','Joint agent + block goal']):
 for i,arm in enumerate(arms):
  y=[next(c for c in r['contrasts'] if c['seed']==seed and c['arm']==arm)['cem_minus_random'][metric]['mean_difference'] for seed in [3072,3073,3074]]
  ax.scatter(i+np.array([-.12,0,.12]),y,s=30,color='#276d9c');ax.plot([i-.22,i+.22],[np.mean(y)]*2,color='#262626',lw=2)
 ax.axhline(0,color='#666666',lw=1);ax.set_xticks(range(4),labels,rotation=25,ha='right');ax.set_title(title,fontsize=11);ax.grid(axis='y',alpha=.2);ax.set_ylim(bottom=-1500)
axes[0].set_ylabel('CEM − random realized cost (positive: worse)')
fig.suptitle('The observed cost increase persists when agent position is included',fontsize=12)
fig.text(.5,.015,'Same executed actions; no replanning. Dots: 3 optimizer seeds on one training sample; bars: seed means.\n128 repeatedly examined development scenarios. Public reference uses a different training budget and is reported separately.',ha='center',fontsize=8)
fig.tight_layout(rect=[0,.11,1,.94]);out=Path('outputs/figures/maintrack');out.mkdir(parents=True,exist_ok=True)
for ext in ['png','pdf']:fig.savefig(out/f'goal_interpretation.{ext}',dpi=180)
