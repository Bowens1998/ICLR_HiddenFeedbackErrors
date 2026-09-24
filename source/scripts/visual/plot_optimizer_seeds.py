import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
r=json.loads(Path('outputs/maintrack/optimizer_seed_summary.json').read_text());fig,axes=plt.subplots(1,2,figsize=(11,4.7),layout='constrained');arms=['transformer_jepa','gru_jepa','transformer_state','gru_state'];labels=['Transformer\nJEPA','GRU\nJEPA','Transformer\nstate*','GRU\nstate*'];colors=['#4c78a8','#f58518','#54a24b']
for ax,algorithm in zip(axes,['random','cem']):
 for seed,color in zip([3072,3073,3074],colors):
  y=[next(v['mean_cost'] for v in r['tables'] if v['seed']==seed and v['arm']==arm and v['algorithm']==algorithm and v['parameterization']=='full') for arm in arms]
  ax.plot(range(4),y,'o-',color=color,label=str(seed),alpha=.8)
 ax.axhline(r['zero_cost'],ls='--',color='gray',label='Zero action');ax.set_xticks(range(4),labels);ax.set_ylim(0,8000);ax.set_title('Full-action '+('random search' if algorithm=='random' else 'CEM'));ax.grid(axis='y',alpha=.2)
axes[0].set_ylabel('Mean terminal block cost (lower is better)');axes[1].legend(title='Training seed',fontsize=8)
fig.suptitle('Planner-dependent model ordering persists across three optimizer seeds\nSame training data,128 development contexts,9000 scores; *extra training state labels and six-dimensional recurrent tokens',fontsize=10)
for ext in ['png','pdf']:fig.savefig(f'outputs/figures/maintrack/optimizer_seed_replication.{ext}',dpi=180)
