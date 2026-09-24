import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
r=json.loads(Path('runs/state_rollout_audit_v1/summary.json').read_text());fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
styles={'recursive':('Recursive prediction','-'),'teacher_forced':('True history restored','--'),'zero_future_action':('Zero future actions',':'),'persistence':('Persistence','-.'),'linear_extrapolation':('Linear extrapolation','--')}
for ax,metric,title in zip(axes,['block_pose_error','normalized_mse'],['Block-pose forecast error','Normalized state MSE']):
 for mode,(label,style) in styles.items():ax.plot(r['horizons_primitive_steps'],r['runs'][0]['metrics'][mode][metric]['episode_mean'],style,label=label)
 ax.set(xlabel='Future primitive steps',ylabel=title,yscale='log',xticks=r['horizons_primitive_steps']);ax.grid(alpha=.2)
axes[0].legend(fontsize=8);fig.suptitle('Privileged GRU selected by validation: expert trajectories, 64 episodes\nEpisode-balanced means; descriptive audit, no independent training-seed uncertainty',fontsize=11)
out=Path('outputs/figures/maintrack');out.mkdir(parents=True,exist_ok=True)
for ext in ['png','pdf']:fig.savefig(out/f'state_rollout_audit.{ext}',dpi=180)
