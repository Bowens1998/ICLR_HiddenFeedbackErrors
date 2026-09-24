from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,axes=plt.subplots(2,4,figsize=(10,5),layout='constrained')
for i in range(4):
 z=np.load(Path('data/pusht_prospective_v1/validation')/f'case_{i:03d}.npz')
 for row,key in [(0,'history_pixels'),(1,'goal_pixels')]:
  ax=axes[row,i];ax.imshow(z[key][-1] if row==0 else z[key]);ax.axis('off');ax.set_title(('Context' if row==0 else 'Goal')+f' {i}',fontsize=10)
fig.suptitle('First four validation scenarios in generation order; no model-based selection')
for ext in ['png','pdf']:fig.savefig(f'outputs/figures/maintrack/prospective_pusht_examples.{ext}',dpi=160)
