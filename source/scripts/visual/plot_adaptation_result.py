"""Scientific figure of the complete fixed-budget adaptation factorial."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from analyze_coverage_goals import sha


def main():
    source=Path('outputs/maintrack/pusht_adaptation_summary.json');r=json.loads(source.read_text())
    assert r['status']=='PASS_COMPLETE24_CONDITIONS'
    conditions=['trained_cls_frozen','trained_cls_adapted','initial_cls_frozen','initial_cls_adapted']
    labels=['Pretrained\nfrozen','Pretrained\nadapted','Initial\nfrozen','Initial\nadapted']
    fig,axes=plt.subplots(1,2,figsize=(10,4.7),gridspec_kw={'width_ratios':[1.25,1]})
    ax=axes[0];m=r['results']['joint_precision']['conditions']
    vals=np.array([m[k]['mean'] for k in conditions])*100
    lo=np.array([m[k]['ci95'][0] for k in conditions])*100;hi=np.array([m[k]['ci95'][1] for k in conditions])*100
    ax.bar(range(4),vals,color=['#3675a7','#73a8cc','#777777','#aaaaaa'],width=.65)
    ax.errorbar(range(4),vals,yerr=[vals-lo,hi-vals],fmt='none',color='black',capsize=4)
    ax.set_xticks(range(4),labels);ax.set_ylabel('Joint localization precision (%)');ax.set_ylim(0,10)
    ax.set_title('Same 512 downstream labels')
    for i,v in enumerate(vals):ax.text(i,hi[i]+.3,f'{v:.2f}%',ha='center',fontsize=9)
    ax=axes[1];cs=r['results']['joint_precision']['contrasts']
    keys=['pretrained_adaptation','initial_adaptation','adaptation_pretraining_interaction']
    names=['Pretrained: adapted − frozen','Initial: adapted − frozen','Adaptation × pretraining']
    for i,k in enumerate(keys):
        v=cs[k];mean=v['mean']*100;low,high=np.array(v['ci95'])*100
        ax.errorbar(mean,i,xerr=[[mean-low],[high-mean]],fmt='o',color='#3675a7',capsize=4)
    ax.axvline(0,color='#888888',ls='--',lw=1);ax.set_yticks(range(3),names);ax.invert_yaxis()
    ax.set_xlabel('Precision difference (percentage points)');ax.set_title('Paired differences and 95% intervals')
    for ax in axes:
        ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Frozen vs adapted CLS: independent static holdout',fontsize=13)
    fig.text(.5,.035,'512 shared test images • 6 fixed backbones • paired image bootstrap • unequal compute',ha='center',fontsize=9)
    fig.text(.5,.005,'Initial adapted models also fail training fit; their gap is not a clean generalization comparison.',ha='center',fontsize=9)
    fig.tight_layout(rect=[0,.09,1,.94]);out=Path('outputs/figures/maintrack');out.mkdir(parents=True,exist_ok=True)
    for ext in ['png','pdf']:fig.savefig(out/f'pusht_adaptation_factorial.{ext}',dpi=180)
    (out/'pusht_adaptation_factorial.json').write_text(json.dumps(dict(summary_sha256=sha(source),source_sha256=sha(__file__),scope='Conditional image bootstrap; no training-population or planning claim.'),indent=2)+'\n')


if __name__=='__main__':main()
