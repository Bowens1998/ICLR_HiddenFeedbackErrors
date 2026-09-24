"""Complete fresh-goal adaptation evidence; primary intervals and descriptive errors."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from adaptation_streams import sha


def main():
    root=Path('runs/hpg/planner_data_adaptation_v1');ep=root/'evaluation_summary.json';dp=root/'endpoint_summary.json'
    e=json.loads(ep.read_text());d=json.loads(dp.read_text())
    assert e['status']=='COMPLETE72_FRESH_ADAPTATION_ROUTES' and d['status']=='COMPLETE72_ROUTES_SELECTED_ENDPOINT_DIAGNOSTICS' and d['evaluation_sha256']==sha(ep)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42})
    fig,axes=plt.subplots(2,2,figsize=(12.2,8.2));colors=['#7b8794','#d18a20','#007f86'];conditions=['original','expert','planner'];labels=['Original','Expert data','Planner data']
    primary=[c for c in e['contrasts'] if c['primary_cost']];assert len(primary)==4
    ax=axes[0,0]
    for i,c in enumerate(primary):
        mean=c['mean_cost_difference'];lo,hi=c['cost_percentile_interval'];color='#007f86' if c['recipe']=='state' else '#675a9c'
        ax.errorbar(mean,3-i,xerr=[[mean-lo],[hi-mean]],fmt='o',color=color,capsize=4,ms=6,lw=1.8)
    ax.axvline(0,color='#333333',ls='--',lw=1);ax.set_yticks(range(4),['State / CEM','State / random','JEPA / CEM','JEPA / random'])
    ax.set_xlim(-2600,600);ax.set_ylim(-.6,3.6);ax.set_xlabel('Planner-data minus expert-data cost (lower is better)')
    ax.set_title('A  Four prespecified primary contrasts',loc='left',fontweight='bold');ax.grid(axis='x',alpha=.15)
    ax.text(.02,.02,'Nominal 98.75% shared-goal bootstrap intervals',transform=ax.transAxes,fontsize=8)
    ax=axes[0,1];combinations=[('pose_encoded','random'),('pose_encoded','cem'),('state','random'),('state','cem')]
    for j,condition in enumerate(conditions):
        values=[next(g['mean_cost'] for g in e['groups'] if g['recipe']==recipe and g['algorithm']==algorithm and g['condition']==condition) for recipe,algorithm in combinations]
        ax.bar(np.arange(4)+(j-1)*.24,values,width=.22,color=colors[j],label=labels[j])
    ax.set_xticks(range(4),['JEPA\nrandom','JEPA\nCEM','State\nrandom','State\nCEM']);ax.set_ylabel('Mean realized cost');ax.set_ylim(0,6500);ax.legend(frameon=False,ncol=3,fontsize=8,loc='upper left');ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    ax.set_title('B  All conditions on the same fresh goals',loc='left',fontweight='bold')
    for ax,recipe,title in [(axes[1,0],'pose_encoded','C  JEPA: CEM selected endpoints'),(axes[1,1],'state','D  Direct state: CEM selected endpoints')]:
        for kind,color,marker,label in [('imagined','#675a9c','o','Imagined endpoint'),('real_image','#007f86','s','Real terminal image')]:
            values=[]
            for j,condition in enumerate(conditions):
                group=next(g for g in d['groups'] if g['recipe']==recipe and g['condition']==condition and g['algorithm']=='cem');values.append(group[kind]['position_mse'])
                actors=[r[kind]['position_mse'] for r in d['rows'] if r['recipe']==recipe and r['condition']==condition and r['algorithm']=='cem'];assert len(actors)==6
                ax.scatter(j+np.linspace(-.055,.055,6),actors,color=color,alpha=.25,s=12)
            ax.plot(range(3),values,color=color,marker=marker,lw=1.8,label=label)
        ax.set_xticks(range(3),labels);ax.set_ylabel('Block-position MSE (pixel²)');ax.set_ylim(bottom=0);ax.grid(axis='y',alpha=.15);ax.legend(frameon=False,fontsize=9);ax.set_title(title,loc='left',fontweight='bold')
    shared_upper=max(ax.get_ylim()[1] for ax in axes[1])
    for ax in axes[1]:ax.set_ylim(0,shared_upper)
    fig.suptitle('Fixed-perception adaptation: a direct-state gain, no established JEPA cost advantage',fontsize=13,y=.99)
    fig.text(.5,.018,'128 shared fresh goals; six fixed models per recipe. Bottom panels: descriptive means and model points, not confidence intervals.\nReal-terminal images are retrospective; identical-image encodings remain unchanged across adaptation conditions.',ha='center',fontsize=9)
    fig.tight_layout(rect=[0,.065,1,.96],h_pad=2.0,w_pad=2.2)
    out=Path('outputs/figures/maintrack');out.mkdir(parents=True,exist_ok=True)
    for extension in ('png','pdf'):fig.savefig(out/f'adaptation_fresh_results.{extension}',dpi=200,bbox_inches='tight')
    (out/'adaptation_fresh_results.json').write_text(json.dumps(dict(evaluation_sha256=sha(ep),endpoint_sha256=sha(dp),source_sha256=sha(__file__),primary_contrasts=primary,groups=e['groups'],endpoint_groups=d['groups'],scope='Complete frozen fresh-goal evaluation. Four primary intervals are nominal approximate marginal bootstrap intervals conditional on fixed fits; descriptive error panels are not population inference.'),indent=2)+'\n')
    plt.close(fig)

if __name__=='__main__':main()
