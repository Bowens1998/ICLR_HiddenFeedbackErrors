"""Complete objective/label evaluation figure; no partial-results rendering."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from adaptation_streams import sha

CONDITIONS={
 'pose_encoded':['original','clipped_latent','unit_latent','unit_decoded_teacher','unit_physical_labels'],
 'state':['original','clipped_state','unit_state'],
}
LABELS={'original':'Original','clipped_latent':'Clipped\nlatent','unit_latent':'Unit\nlatent','unit_decoded_teacher':'Image\nteacher','unit_physical_labels':'Physical\nlabels','clipped_state':'Clipped\nstate','unit_state':'Unit\nstate'}


def main():
    p=argparse.ArgumentParser()
    for key in ('evaluation','endpoints','output'):p.add_argument('--'+key,required=True)
    a=p.parse_args();e=json.loads(Path(a.evaluation).read_text());d=json.loads(Path(a.endpoints).read_text())
    assert e['status']=='COMPLETE96_FRESH_TASK_COORDINATE_ROUTES' and d['status']=='COMPLETE96_ROUTES_SELECTED_ENDPOINT_DIAGNOSTICS'
    assert len(e['rows'])==len(d['rows'])==96 and d['evaluation_sha256']==sha(a.evaluation) and d['plan_sha256']==e['plan_sha256']
    assert d['same_image_checks']==24576 and d['max_same_image_difference']==0
    primary=[c for c in e['contrasts'] if c['primary_cost']];assert len(primary)==4
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    fig,axes=plt.subplots(2,2,figsize=(15,10))
    ax=axes[0,0];ticks=[]
    for i,c in enumerate(primary):
        y=3-i;mean=c['mean_cost_difference'];lo,hi=c['cost_percentile_interval'];assert c['cost_interval_level']==.9875
        color='#007f86' if c['left'][0]=='unit_physical_labels' else '#675a9c'
        ax.hlines(y,lo,hi,color=color,lw=2);ax.plot(mean,y,'o',color=color)
        label='Labels − teacher' if c['left'][0]=='unit_physical_labels' else 'Teacher − unit latent'
        ticks.append(f"{label} / {c['left'][1].upper()}")
    ax.set_yticks([3,2,1,0],ticks);ax.axvline(0,color='#444',ls='--',lw=1);ax.set_ylim(-.7,3.7)
    ax.set_xlabel('Realized-cost difference (negative favors first condition)');ax.set_title('A  Four primary JEPA comparisons',loc='left',fontweight='bold');ax.grid(axis='x',alpha=.15)
    ax.text(.02,.015,'Nominal 98.75% shared-goal bootstrap intervals',transform=ax.transAxes,fontsize=8)
    ax=axes[0,1];entries=[(recipe,c) for recipe in CONDITIONS for c in CONDITIONS[recipe]]
    for algorithm,offset,color,marker in [('random',-.12,'#718096','o'),('cem',.12,'#b46a26','s')]:
        values=[]
        for recipe,condition in entries:
            g=next(g for g in e['groups'] if g['recipe']==recipe and g['condition']==condition and g['algorithm']==algorithm);values.append(g['mean_cost'])
        ax.scatter(values,np.arange(8)+offset,label=algorithm.upper(),color=color,marker=marker,s=35)
    ax.set_yticks(range(8),[('JEPA: ' if r=='pose_encoded' else 'State: ')+LABELS[c].replace('\n',' ') for r,c in entries]);ax.invert_yaxis();ax.set_xlim(left=0);ax.set_xlabel('Mean realized cost');ax.grid(axis='x',alpha=.15);ax.legend(frameon=False,ncol=2)
    ax.set_title('B  All original and continuation controls',loc='left',fontweight='bold')
    for ax,recipe,letter in [(axes[1,0],'pose_encoded','C'),(axes[1,1],'state','D')]:
        conditions=CONDITIONS[recipe]
        for kind,color,marker,label in [('imagined','#675a9c','o','Imagined endpoint'),('real_image','#007f86','s','Real terminal image')]:
            values=[]
            for j,c in enumerate(conditions):
                subset=[r for r in d['rows'] if r['recipe']==recipe and r['condition']==c and r['algorithm']=='cem'];assert len(subset)==6
                points=[r[kind]['position_mse'] for r in subset];values.append(np.mean(points));ax.scatter(j+np.linspace(-.055,.055,6),points,color=color,alpha=.25,s=12)
            ax.plot(range(len(conditions)),values,color=color,marker=marker,label=label)
        ax.set_xticks(range(len(conditions)),[LABELS[c] for c in conditions]);ax.set_ylabel('Block-position MSE (pixel²)');ax.grid(axis='y',alpha=.15);ax.legend(frameon=False,fontsize=9)
        ax.set_title(f"{letter}  {'JEPA' if recipe=='pose_encoded' else 'Direct state'}: CEM selected endpoints",loc='left',fontweight='bold')
    upper=max(ax.get_ylim()[1] for ax in axes[1])
    for ax in axes[1]:ax.set_ylim(0,upper)
    fig.suptitle('Fixed-perception continuation: loss coordinates and additional state labels',fontsize=14)
    fig.text(.5,.015,'128 shared fresh goals; six fixed models per recipe. Bottom panels show descriptive means and model points.\nReal-terminal images are retrospective; same-image encoding is unchanged. State and JEPA recipes also differ in architecture and normalization.',ha='center',fontsize=9)
    fig.tight_layout(rect=[0,.055,1,.96],w_pad=3,h_pad=2)
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
    for ext in ('png','pdf'):fig.savefig(out.with_suffix('.'+ext),dpi=200,bbox_inches='tight')
    out.with_suffix('.json').write_text(json.dumps(dict(evaluation_sha256=sha(a.evaluation),endpoint_sha256=sha(a.endpoints),source_sha256=sha(__file__),primary=primary,groups=e['groups'],endpoint_groups=d['groups'],scope='Complete prespecified routes only; conditional primary intervals and descriptive selected-state errors. No general JEPA advantage or novelty assertion.'),indent=2)+'\n')
    plt.close(fig)

if __name__=='__main__':main()
