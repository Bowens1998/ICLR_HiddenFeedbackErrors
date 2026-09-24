"""Publication-size figures: native 5.5-inch width, readable 9-point labels."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from adaptation_streams import sha

ROOT=Path(__file__).resolve().parents[2];out=ROOT/'paper/iclr2027/figures';out.mkdir(parents=True,exist_ok=True)
full_path=ROOT/'runs/hpg/planner_data_adaptation_v1/readout_fiber_rollout_summary.json';matched_path=ROOT/'runs/hpg/matched_fiber_v1/summary.json';shuffled_path=ROOT/'runs/hpg/shuffled_fiber_v1/summary.json';reacher_path=ROOT/'runs/hpg/reacher_fiber_transfer_v1/fiber_rollout_summary.json'
full,matched,shuffled,reacher=[json.loads(p.read_text()) for p in (full_path,matched_path,shuffled_path,reacher_path)]
plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'legend.fontsize':8})
conditions=['unit_latent','unit_decoded_teacher','unit_physical_labels'];labels=['Latent','Coordinate teacher','Physical labels'];colors=['#555555','#8b459e','#278679']

def save(fig,name):
 fig.tight_layout(pad=.5)
 for ext in ('pdf','png'):fig.savefig(out/(name+'.'+ext),dpi=220)
 plt.close(fig)

fig,ax=plt.subplots(figsize=(5.5,2.6));xx=np.array(full['horizons'])
for condition,label,color in zip(conditions,labels,colors):
 c=next(c for c in full['contrasts'] if c['reference_stream']=='equal_four_streams' and c['condition']==condition and c['left']=='fiber')['metrics']['position_mse'];ci=np.array(c['secondary_95_percentile_interval']);ax.plot(xx,c['mean_difference'],'o-',color=color,label=label,ms=3);ax.fill_between(xx,ci[:,0],ci[:,1],color=color,alpha=.12)
ax.axhline(0,color='#444',ls=':',lw=1);ax.set_xticks(xx);ax.set_xlabel('Primitive-action horizon');ax.set_ylabel('Position-MSE change (pixel²)');ax.legend(loc='lower left');ax.grid(alpha=.15);save(fig,'single_feedback')
for name,r,paths,legend in [('displacement',None,['full','matched'],['Full correction','Matched displacement']),('guidance',shuffled,['fiber','shuffled'],['Actual observation','Other-goal observation'])]:
 fig,ax=plt.subplots(figsize=(5.5,2.6))
 for offset,path,label,color in zip([-.10,.10],paths,legend,['#8b459e','#bc783c']):
  rr={'full':full,'matched':matched}[path] if r is None else r;pp='fiber' if r is None else path
  cc=[next(c for c in rr['contrasts'] if c['reference_stream']=='equal_four_streams' and c['condition']==condition and c['left']==pp)['metrics']['position_mse'] for condition in conditions];mean=np.array([c['mean_difference'][-1] for c in cc]);ci=np.array([c['secondary_95_percentile_interval'][-1] for c in cc]);ax.errorbar(np.arange(3)+offset,mean,yerr=np.stack([mean-ci[:,0],ci[:,1]-mean]),fmt='o',capsize=3,color=color,label=label,ms=4)
 ax.set_xticks(range(3),labels);ax.set_ylabel('Endpoint position-MSE change');ax.axhline(0,color='#444',ls=':',lw=1);ax.grid(axis='y',alpha=.15);ax.legend(loc='upper center',bbox_to_anchor=(.5,1.23),ncol=2,frameon=False);save(fig,name)
# Transfer summary has the same group-path convention; inspect metric names explicitly.
fig,ax=plt.subplots(figsize=(5.5,2.6))
for path,label,color,style in [('free','Free','#777777','-'),('fiber','Readout-preserving','#8b459e','-'),('reset','Full reset','#278679','--')]:
 g=next(g for g in reacher['groups'] if g['reference_stream']=='equal_four_streams' and g['path']==path);ax.plot(reacher['horizons'],g['means']['joint_angle_mse'],'o'+style,color=color,label=label,ms=3)
ax.set_xticks(reacher['horizons']);ax.set_xlabel('Primitive-action horizon');ax.set_ylabel('Joint-angle MSE (rad²)');ax.legend(loc='upper center',bbox_to_anchor=(.5,1.22),ncol=3,frameon=False);ax.grid(alpha=.15);save(fig,'reacher_transfer')
(out/'source_manifest.json').write_text(json.dumps(dict(source_sha256=sha(__file__),inputs={str(p.relative_to(ROOT)):sha(p) for p in (full_path,matched_path,shuffled_path,reacher_path)}),indent=2)+'\n')
