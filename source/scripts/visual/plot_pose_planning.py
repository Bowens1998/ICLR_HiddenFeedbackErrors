"""Scientific figure for the complete PushT development scoring comparison."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pose_planning_statistics import analyze


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
 p=argparse.ArgumentParser();p.add_argument('--summary',required=True);p.add_argument('--output',required=True);a=p.parse_args()
 r=json.loads(Path(a.summary).read_text());assert r['status']=='COMPLETE_DEVELOPMENT' and len(r['rows'])==48
 rebuilt=analyze([v['cost'] for v in r['rows']],[v['success'] for v in r['rows']]);assert rebuilt==r['analysis']
 effects=['latent_cem_minus_random','pose_cem_minus_random','interaction','pose_minus_latent_cem']
 labels=['Latent: CEM − random','Pose: CEM − random','Interaction (pose − latent)','CEM: pose − latent']
 fig,axes=plt.subplots(1,2,figsize=(10.5,4.7),sharey=True)
 colors=['#606775','#2166ac','#8056a3','#248477']
 for ax,metric,title,mult in zip(axes,['cost','success'],['Physical cost difference','Success difference (percentage points)'],[1,100]):
  for j,name in enumerate(effects):
   v=next(v for v in rebuilt['contrasts'] if v['scope']=='primary_equal_backbone_mean' and v['metric']==metric and v['contrast']==name)
   mean=v['mean']*mult;lo,hi=np.array(v['ci95'])*mult
   ax.errorbar(mean,j,xerr=[[mean-lo],[hi-mean]],fmt='o',color=colors[j],capsize=4,markersize=7,lw=1.6)
  ax.axvline(0,color='#777777',lw=1,ls='--');ax.set_title(title,fontsize=11);ax.grid(axis='x',alpha=.18)
  ax.spines[['top','right']].set_visible(False);ax.set_yticks(range(4));ax.set_yticklabels(labels,fontsize=10)
 axes[0].invert_yaxis();axes[0].set_xlabel('Lower is better for component comparisons');axes[1].set_xlabel('Higher is better for component comparisons')
 fig.suptitle('PushT: pose scoring attenuates the CEM penalty, but does not reverse it',fontsize=13,y=.96)
 fig.subplots_adjust(left=.265,right=.97,top=.82,bottom=.25,wspace=.28)
 fig.text(.03,.035,'Pose = encoded-endpoint readout with a shared encoded-goal head.\n128 previously examined development goals; equal mean over six fixed backbones.\nNominal 95% paired-goal bootstrap intervals; no training-population or fresh-goal confirmation.',fontsize=9,color='#444444')
 out=Path(a.output);out.mkdir(parents=True,exist_ok=True);files=[]
 for suffix in ['png','pdf']:
  path=out/f'pusht_pose_planning_effects.{suffix}';fig.savefig(path,dpi=180,facecolor='white');files.append(path)
 plt.close(fig)
 report=dict(summary_sha256=sha(a.summary),source_sha256=sha(__file__),files_sha256={p.name:sha(p) for p in files},status='RENDERED_REQUIRES_VISUAL_REVIEW',scope='Complete development matrix; both component effects must reverse for a reversal claim, not interaction alone.')
 (out/'pusht_pose_planning_figure.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
