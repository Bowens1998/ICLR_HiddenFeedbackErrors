"""Display complete auxiliary-mode task results with all training pools retained."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p=argparse.ArgumentParser();p.add_argument('--effects',required=True);p.add_argument('--output-dir',required=True);a=p.parse_args()
s=json.loads(Path(a.effects).read_text());assert len(s['cells'])==24 and len(s['effects'])==36
cells={(r['arm'],r['mode'],r['checkpoint'],r['algorithm']):r for r in s['cells']};assert len(cells)==24
layout=[(arm,mode) for arm in ['transformer_jepa','gru_jepa'] for mode in ['none','inverse','inverse_goal']]
labels=['T\nNone','T\nInverse','T\nInverse + goal','GRU\nNone','GRU\nInverse','GRU\nInverse + goal']
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
fig,axes=plt.subplots(2,2,figsize=(14,8),sharex=True,sharey='row')
for col,ck in enumerate(['last','best']):
    for alg,offset,color,label in [('random',-.13,'#2166ac','Random'),('cem',.13,'#b75b19','CEM')]:
        for row,metric in enumerate(['pool_costs','pool_successes']):
            values=np.array([cells[arm,mode,ck,alg][metric] for arm,mode in layout]).T
            assert values.shape==(3,6) and np.isfinite(values).all()
            if row:values*=100
            ax=axes[row,col];xs=np.arange(6)+offset
            for rep in range(3):ax.scatter(xs,values[rep],color=color,alpha=.55,marker=['o','s','^'][rep],s=28)
            ax.scatter(xs,values.mean(0),marker='_',s=230,color=color,linewidth=2.5,label=label)
            ax.grid(axis='y',alpha=.15);ax.set_xticks(range(6),labels)
    axes[0,col].set_title(f'{ck.title()} checkpoint');axes[0,col].legend(frameon=False)
    axes[1,col].set_ylim(-2,102)
axes[0,0].set_ylabel('Mean realized block-pose cost ↓');axes[1,0].set_ylabel('Success (%) ↑')
fig.suptitle('PushT · action-supervision adaptation · three training pools')
fig.text(.5,.01,'Markers: individual pools; horizontal marks: pool means. All 128 development goals and 9,000 candidates per route.\nLast is primary; best is separate sensitivity. Auxiliary heads are unused at deployment; this is a prior-art adaptation.',ha='center',fontsize=9)
fig.tight_layout(rect=(0,.07,1,.96));out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True)
for ext in ['png','pdf']:fig.savefig(out/f'action_auxiliary_performance.{ext}',dpi=180)
plt.close(fig)
