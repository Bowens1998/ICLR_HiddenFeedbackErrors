"""Plot physical outcomes only: native score units differ across models."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
r=json.loads(Path('outputs/maintrack/visual_controlled_planner_summary.json').read_text())
fig,axes=plt.subplots(2,5,figsize=(15,6),sharey='row',layout='constrained')
colors=['#4c78a8','#72b7b2','#f58518','#e45756']
for index,title in enumerate(['Transformer JEPA','GRU JEPA','Transformer state*','GRU state*','Released JEPA†']):
 rows=[v for v in r['tables'] if v['model_index']==index]
 for row,key,scale in [(0,'mean_cost',1),(1,'success',100)]:
  ax=axes[row,index];ax.bar(range(4),[v[key]*scale for v in rows],color=colors);ax.set_xticks(range(4),['R-H','R-F','C-H','C-F']);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
  if row==0:ax.axhline(r['zero_mean_cost'],color='gray',ls='--');ax.set_title(title,fontsize=10)
axes[0,0].set_ylabel('Mean terminal block cost ↓');axes[1,0].set_ylabel('Success (%) ↑');axes[1,0].set_ylim(0,100)
fig.suptitle('Image-only controlled search: 9000 candidates, 128 development contexts\nR: random; C: CEM; H: held actions; F: full actions\n* Extra state labels in training. † Different training budget. One small-data training seed.',fontsize=11)
for ext in ['png','pdf']:fig.savefig(f'outputs/figures/maintrack/visual_controlled_planner.{ext}',dpi=180)
