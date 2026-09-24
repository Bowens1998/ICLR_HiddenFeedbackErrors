"""Quantitative supervision reversal from the frozen confirmation summary.

Figure 1 has its own editable source in figures/draw_feedback_design.py.
"""
from pathlib import Path
import json, hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'paper/iclr2027/figures'
plt.rcParams.update({'font.size':9,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
def save(fig,name):
 for ext in ('pdf','png'):fig.savefig(OUT/(name+'.'+ext),dpi=220)
 plt.close(fig)
p=ROOT/'runs/hpg/fiber_confirmation_v1/horizon_summary_protocol.json';r=json.loads(p.read_text())
cc=[next(c for c in r['contrasts'] if c['reference_stream']=='equal_four_streams' and c['kind']=='objective_difference' and c['left']=='unit_physical_labels' and c['path']==path)['metrics']['position_mse'] for path in ('teacher','free')]
means=np.array([c['mean_difference'][-1] for c in cc]);ci=np.array([c['secondary_95_percentile_interval'][-1] for c in cc])
fig,ax=plt.subplots(figsize=(5.5,2.05));ax.axvline(0,color='#666',lw=1,ls=':')
for i,(m,c,color) in enumerate(zip(means,ci,['#278679','#ab6536'])):ax.errorbar(m,i,xerr=[[m-c[0]],[c[1]-m]],fmt='o',color=color,capsize=4,ms=5)
ax.set_yticks([0,1],['Observed history','Self-generated history']);ax.invert_yaxis();ax.set_ylim(1.6,-.6);ax.set_xlim(-3500,3500);ax.set_xticks([-3000,-1500,0,1500,3000]);ax.grid(axis='x',alpha=.15);ax.set_xlabel('Physical labels minus coordinate teacher: endpoint position MSE');fig.tight_layout(pad=.6);save(fig,'reversal')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
(OUT/'story_source_manifest.json').write_text(json.dumps({'script_sha256':sha(Path(__file__)),'reversal_source':str(p.relative_to(ROOT)),'reversal_source_sha256':sha(p),'means':means.tolist(),'secondary_95_intervals':ci.tolist(),'feedback_design':'Drawn separately from figures/draw_feedback_design.py; editable SVG and vector PDF, no quantitative data.'},indent=2)+'\n')
