import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
r=json.loads(Path('outputs/maintrack/action_hold_summary.json').read_text());fig,ax=plt.subplots(figsize=(8,3.5),layout='constrained');labels=['Teacher forced, 3e-4','Recursive, 5e-5','Recursive, 3e-4']
for i,v in enumerate(r['models']):
 m=v['primary_varied_minus_held'];lo,hi=v['paired_95_percentile_interval'];ax.errorbar(m,i,xerr=np.array([[m-lo],[hi-m]]),fmt='o',capsize=4,color='#386cb0')
ax.axvline(0,color='gray',linestyle='--');ax.set_yticks(range(3),labels);ax.invert_yaxis();ax.set_xlabel('Varied minus held: terminal block forecast error\nNegative favors varied actions');ax.grid(axis='x',alpha=.2);ax.set_title('Paired action timing: 64 contexts, frozen state models\nNominal 95% paired scenario intervals; one training seed',fontsize=11)
for ext in ['png','pdf']:fig.savefig(f'outputs/figures/maintrack/action_hold.{ext}',dpi=180)
