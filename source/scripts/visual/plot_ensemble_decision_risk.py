"""Development outcome contrasts; no inference beyond fixed consumed banks."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from analyze_coverage_goals import sha


def main():
    p=argparse.ArgumentParser();p.add_argument('--pusht',required=True);p.add_argument('--reacher',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    files=[Path(a.reacher),Path(a.pusht)];groups=[g for f in files for g in json.loads(f.read_text())['groups']]
    labels=['Reacher / JEPA circle','Reacher / direct state','PushT / JEPA pose','PushT / direct state']
    methods=['ridge','augmented_ridge','ensemble_mean'];names=['Score-only ridge','Ridge + ensemble features','Ensemble mean'];colors=['#64748b','#b36a12','#137c8b'];records=[]
    fig,axes=plt.subplots(1,2,figsize=(10,5.7),sharey=True)
    for j,(method,name,color) in enumerate(zip(methods,names,colors)):
        c=[100*(g['costs'][method]/g['costs']['train_constant']-1) for g in groups]
        s=[100*(g['successes'][method]-g['successes']['train_constant']) for g in groups]
        y=np.arange(4)+(j-1)*.22
        for ax,values in zip(axes,[c,s]):ax.barh(y,values,height=.2,color=color,label=name)
        records.append(dict(method=method,cost_percent=c,success_percentage_points=s))
    axes[0].set_yticks(np.arange(4),labels);axes[0].invert_yaxis()
    axes[0].set_xlabel('Cost change (%) — lower is better');axes[1].set_xlabel('Success change (pp; higher is better)')
    for ax in axes:
        ax.axvline(0,color='#334155',lw=.8);ax.grid(axis='x',alpha=.2);ax.set_axisbelow(True)
        ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Cross-task development: choosing between two fixed plans',fontsize=13,y=.96)
    fig.legend(*axes[0].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.58,.88),ncol=3,frameon=False,fontsize=9)
    fig.text(.04,.035,'Reference: training-fold constant (CEM for Reacher, random for PushT). Equal means over six fixed actors.\nConsumed banks; no independent confirmation. Three-model training and extra inference are not budget-matched.\nPushT uses the declared single-action score implementation; original-search numerical differences are retained.',fontsize=8,color='#475569')
    fig.subplots_adjust(left=.23,right=.97,top=.77,bottom=.22,wspace=.18)
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(str(out)+'.png',dpi=180);fig.savefig(str(out)+'.pdf');plt.close(fig)
    Path(str(out)+'.json').write_text(json.dumps(dict(source_sha256=sha(__file__),inputs={str(f):sha(f) for f in files},labels=labels,records=records,scope='Descriptive means, no confidence intervals or significance claim; task costs normalized separately to their own constant strategy.'),indent=2)+'\n')


if __name__=='__main__':main()
