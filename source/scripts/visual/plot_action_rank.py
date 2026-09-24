"""Plot accepted action-choice utility; no equal-training-budget inference."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
root=Path(__file__).resolve().parents[2]
d=json.loads((root/'outputs/maintrack/visual_action_rank_summary.json').read_text())['models']
pilot=d['eight_epoch_pilot'];released=d['official_released_reference']
labels=['Native: eight-epoch pilot','Native: released reference','Oracle: released real embedding','Oracle: real pixels','Zero-action policy']
fig,axes=plt.subplots(1,2,figsize=(10,4.4),sharey=True)
for ax,task,title in zip(axes,['block_pose','agent_and_block_pose'],['Block pose','Agent + block pose']):
    a=pilot['aggregate'][task];b=released['aggregate'][task];random=a['uniform_persistence']['selected_cost']
    values=np.array([a['native_latent']['selected_cost'],b['native_latent']['selected_cost'],b['oracle_realized_latent']['selected_cost'],b['oracle_realized_pixel']['selected_cost'],pilot['zero_action_mean_cost'][task]])/random
    ax.barh(np.arange(5),values,color=['#6e99c4','#24547c','#a8afb5','#a8afb5','#baaa8a'],height=.6)
    for i,v in enumerate(values):ax.text(v+.025,i,f'{v:.2f}',va='center',fontsize=9)
    ax.axvline(1,color='#666666',ls='--',lw=1);ax.set_xlim(0,1.18);ax.set_yticks(np.arange(5),labels);ax.set_title(title);ax.set_xlabel('Mean chosen cost / mean random cost ↓')
    ax.spines[['top','right']].set_visible(False)
axes[0].invert_yaxis()
fig.suptitle('Released LeWM predictions support goal-conditioned action selection',fontsize=12)
fig.text(.5,.035,'32 shared scenarios • different training budgets • oracle scores use true future images • development only',ha='center',fontsize=8)
fig.tight_layout(rect=[0,.08,1,.94]);out=root/'outputs/figures/maintrack';out.mkdir(parents=True,exist_ok=True)
for ext in ['png','pdf']:fig.savefig(out/f'visual_action_rank.{ext}',dpi=170)
