import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
r=json.loads(Path('outputs/maintrack/prospective_rollout_summary.json').read_text());names=['teacher_forced_expert','recursive_expert','recursive_branch','zero','uniform','sample_oracle'];labels=['Teacher forced\nboth selectors','Recursive\nexpert','Recursive\nbranch','Zero action','Uniform','Sample oracle']
fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained');colors=['#386cb0','#f28e2b','#e15759','#888888','#bbbbbb','#59a14f']
for ax,key,title in zip(axes,['mean_cost','mean_success'],['Mean block cost (lower better)','Selected-candidate success']):
 values=[r['summary'][n][key] for n in names];ax.bar(range(6),values,color=colors);ax.set_xticks(range(6),labels,fontsize=8);ax.set_title(title,fontsize=11);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
 if key=='mean_success':ax.set_ylim(0,1)
fig.suptitle('Prospective object-transport goals: 128 held-out scenarios\nPrivileged state inputs; one training sample/seed; teacher selectors chose identical weights',fontsize=11)
for ext in ['png','pdf']:fig.savefig(f'outputs/figures/maintrack/prospective_rollout.{ext}',dpi=180)
