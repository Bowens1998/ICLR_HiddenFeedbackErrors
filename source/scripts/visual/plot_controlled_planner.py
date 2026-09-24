import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
r=json.loads(Path('outputs/maintrack/controlled_planner_summary.json').read_text());fig,axes=plt.subplots(1,3,figsize=(12,4.5),sharey=True,layout='constrained');colors=['#4c78a8','#72b7b2','#f58518','#e45756']
for ax,index,title in zip(axes,[1,3,4],['Teacher forced, 3e-4','Recursive, 5e-5','Recursive, 3e-4']):
 rows=[v for v in r['tables'] if v['model_index']==index];ax.bar(range(4),[v['mean_cost'] for v in rows],color=colors,label='Realized');ax.scatter(range(4),[v['mean_predicted_cost'] for v in rows],marker='D',s=32,color='black',label='Predicted',zorder=3);ax.axhline(r['zero_mean_cost'],color='gray',ls='--',label='Zero action');ax.set_xticks(range(4),['Random\nheld','Random\nfull','CEM\nheld','CEM\nfull'],fontsize=9);ax.set_title(title,fontsize=11);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
axes[0].set_ylabel('Mean terminal block cost');axes[0].legend(fontsize=8);fig.suptitle('Controlled planning: 9000 search scores per context\n128 previously examined contexts; privileged state history; one training sample/seed',fontsize=11)
for ext in ['png','pdf']:fig.savefig(f'outputs/figures/maintrack/controlled_planner.{ext}',dpi=180)
