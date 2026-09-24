"""Publication-oriented descriptive figure; future observations labeled privileged."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from analyze_pose_selected_endpoints import sha


def main():
    terminal=Path('outputs/maintrack/pusht_terminal_encoding.json')
    substitution=Path('outputs/maintrack/pusht_pair_substitution.json')
    tr=json.loads(terminal.read_text());su=json.loads(substitution.read_text())
    assert tr['status']=='COMPLETE36_TERMINAL_ENCODING_ROUTES' and su['status']=='COMPLETE18_FIXED_PAIR_SUBSTITUTION_GROUPS'
    interfaces=['pose_encoded','pose_predicted','state'];labels=['JEPA\nimage head','JEPA\nprediction head','Direct\nstate'];records=[]
    fig,axes=plt.subplots(1,2,figsize=(11.4,5.8));colors=['#64748b','#007d8a','#c27216','#843ba1']
    for j,(method,algorithm,label) in enumerate([('imagined','random','Imagined / random'),('real_image','random','Real image / random'),('imagined','cem','Imagined / CEM'),('real_image','cem','Real image / CEM')]):
        x=np.arange(3)+(j-1.5)*.18
        for i,interface in enumerate(interfaces):
            values=np.array([100*r['metrics'][method]['precision'] for r in tr['rows'] if r['interface']==interface and r['algorithm']==algorithm]);assert len(values)==6
            axes[0].scatter(np.full(6,x[i])+np.linspace(-.025,.025,6),values,s=15,color=colors[j],alpha=.45)
            axes[0].plot([x[i]-.06,x[i]+.06],[values.mean()]*2,color=colors[j],lw=3,label=label if i==0 else None)
            records.append(dict(panel='precision',interface=interface,method=method,algorithm=algorithm,actor_values=values.tolist(),mean=float(values.mean())))
    methods=[('uncorrected','Original score'),('true_goal','True goal'),('real_image','Real endpoint image'),('true_endpoint','True endpoint')]
    for j,(method,label) in enumerate(methods):
        x=np.arange(3)+(j-1.5)*.18
        means=[]
        for interface in interfaces:
            tg=next(g for g in tr['groups'] if g['interface']==interface);sg=next(g for g in su['groups'] if g['interface']==interface)
            for name in ['uncorrected','true_endpoint']:
                np.testing.assert_allclose(tg['policies'][name]['cost'],sg['policies'][name]['cost'],atol=1e-9,rtol=0)
            cost=(tg if method=='real_image' else sg)['policies'][method]['cost'];base=sg['policies']['always_random']['cost']
            means.append(100*(cost/base-1));records.append(dict(panel='choice_cost',interface=interface,method=method,mean_cost=cost,random_mean_cost=base,change_percent=means[-1]))
        axes[1].bar(x,means,width=.16,color=colors[j],label=label)
    for ax in axes:
        ax.set_xticks(np.arange(3),labels);ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
        ax.legend(loc='upper left',fontsize=8,frameon=False)
    axes[0].set_title('(a) Same physical endpoints, different inputs',fontsize=11)
    axes[0].set_ylabel('Endpoint pose precision (%)');axes[0].set_ylim(-1,76)
    axes[1].set_title('(b) Fixed-pair scoring substitutions',fontsize=11)
    axes[1].set_ylabel('Mean cost change from random (%)');axes[1].axhline(0,color='#334155',lw=.8);axes[1].set_ylim(-25,130)
    fig.suptitle('PushT: separating image decoding from imagined-endpoint scoring',fontsize=13,y=.97)
    fig.text(.07,.035,'128 consumed goals shared by six fixed actors. Left: actor means (dots) and equal-actor mean (line); no confidence intervals.\nRight: ratio of pooled means; lower is better. True goals, real endpoint images and true endpoints are privileged diagnostics.\nBoth candidate actions are fixed. These substitutions do not establish deployable improvements or oracle replanning results.',fontsize=8,color='#475569')
    fig.subplots_adjust(left=.08,right=.97,bottom=.25,top=.86,wspace=.29)
    out=Path('outputs/figures/maintrack/pusht_error_interventions');out.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(str(out)+'.png',dpi=180);fig.savefig(str(out)+'.pdf');plt.close(fig)
    Path(str(out)+'.json').write_text(json.dumps(dict(source_sha256=sha(__file__),inputs={str(p):sha(p) for p in [terminal,substitution]},records=records,scope='Descriptive consumed-data figure; privileged substitutions explicitly labeled.'),indent=2)+'\n')


if __name__=='__main__':main()
