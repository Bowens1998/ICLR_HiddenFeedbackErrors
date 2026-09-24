"""Scientific plots from the fully accepted factorial summary, without checkpoint selection."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parents[2];data=json.loads((root/'outputs/maintrack/visual_factorial_summary.json').read_text());out=root/'outputs/figures/maintrack';out.mkdir(parents=True,exist_ok=True)
arms=['transformer_jepa','gru_jepa','transformer_state','gru_state'];labels=['Transformer / JEPA','GRU / JEPA','Transformer / state labels','GRU / state labels']
fig,axes=plt.subplots(1,2,figsize=(9.8,4.4),sharey=True)
for ax,condition,title in zip(axes,['inview','stress'],['In-view candidate bank','Offscreen stress bank']):
    routes=data['conditions'][condition]['routes']
    for i,arm in enumerate(arms):
        values=[]
        for checkpoint,color,marker in [('best','#24547c','o'),('last','#9ba5af','s')]:
            m=routes[arm+'_'+checkpoint]['aggregate']['block_pose'];v=m['native']['selected_cost']/m['zero_action']['selected_cost'];values.append(v)
            ax.scatter(v,i,color=color,marker=marker,s=45,zorder=3,label=('Validation selected' if checkpoint=='best' else 'Final epoch') if i==0 else None)
        ax.plot(values,[i,i],color='#bbbbbb',lw=1,zorder=1)
    ax.axvline(1,ls='--',color='#777777',lw=1);ax.set_yticks(range(4),labels);ax.set_title(title);ax.set_xlabel('Native block cost / zero-action cost ↓');ax.spines[['top','right']].set_visible(False);ax.grid(axis='x',alpha=.15)
axes[0].invert_yaxis();fig.legend(*axes[0].get_legend_handles_labels(),loc='lower center',bbox_to_anchor=(.5,.065),ncol=2,fontsize=9)
fig.suptitle('Matched-data visual architecture and target comparison',fontsize=12)
fig.text(.5,.02,'One dataset and optimizer seed • state arms use extra labels and a physical cost • points are checkpoints',ha='center',fontsize=8)
fig.tight_layout(rect=[0,.15,1,.94])
for ext in ['png','pdf']:fig.savefig(out/f'visual_factorial_tasks.{ext}',dpi=170)
plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(9.5,4.1))
for ax,objective,title in zip(axes,['jepa','state'],['JEPA latent prediction MSE','Direct standardized state prediction MSE']):
    for arch,color in [('transformer','#24547c'),('gru','#c17e3e')]:
        epochs=data['training'][arch+'_'+objective]['epochs']
        for split,style in [('train','-'),('validation','--')]:ax.semilogy([e['epoch'] for e in epochs],[e[split]['prediction_loss'] for e in epochs],style,color=color,label=f'{arch}: {split}',lw=1.3)
    ax.set_xlabel('Epoch');ax.set_title(title,fontsize=10);ax.grid(alpha=.15);ax.spines[['top','right']].set_visible(False);ax.legend(fontsize=7,frameon=False)
fig.suptitle('Training diagnostics: target losses are not cross-objective performance scores',fontsize=11)
fig.tight_layout(rect=[0,0,1,.94])
for ext in ['png','pdf']:fig.savefig(out/f'visual_factorial_learning.{ext}',dpi=170)
