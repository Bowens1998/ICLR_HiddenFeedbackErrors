"""All matched-action conditions and reference streams, explicitly retrospective."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from adaptation_streams import sha


def main():
    p=argparse.ArgumentParser();p.add_argument('--summary',required=True);p.add_argument('--output',required=True);a=p.parse_args();r=json.loads(Path(a.summary).read_text())
    assert r['status']=='COMPLETE48_MODELS_MATCHED_ACTION_DIAGNOSTICS' and r['anchor_checks']==3072 and r['same_image_checks']==23040
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    fig=plt.figure(figsize=(14,7.2));grid=fig.add_gridspec(2,2,width_ratios=[1.2,1]);left=fig.add_subplot(grid[:,0]);axes=[fig.add_subplot(grid[0,1]),fig.add_subplot(grid[1,1])]
    groups=[g for g in r['groups'] if g['reference_stream']=='equal_four_streams'];assert len(groups)==8
    labels=['Original','Clipped\nlatent','Unit\nlatent','Image\nteacher','Physical\nlabels','Original','Clipped\nstate','Unit\nstate']
    for i,g in enumerate(groups):
        color='#675a9c' if g['recipe']=='pose_encoded' else '#007f86'
        left.bar(i,g['means']['position_mse'],color=color,alpha=.8,width=.7)
        points=[m['position_mse'] for m in g['per_model']];left.scatter(i+np.linspace(-.15,.15,6),points,color='#30343b',alpha=.55,s=18,zorder=3)
    left.set_xticks(range(8),labels,fontsize=9);left.axvline(4.5,color='#888',ls=':',lw=1);left.set_ylim(bottom=0);left.set_ylabel('Matched-action block-position MSE (pixel²)');left.grid(axis='y',alpha=.15);left.set_axisbelow(True)
    left.set_title('A  All frozen conditions on identical reference actions',loc='left',fontweight='bold');left.text(.24,.96,'JEPA',transform=left.transAxes,color='#675a9c',fontweight='bold');left.text(.8,.96,'Direct state',transform=left.transAxes,color='#007f86',fontweight='bold')
    refs=['equal_four_streams','original_jepa_random','original_jepa_cem','original_state_random','original_state_cem'];names=['Four-stream mean','JEPA random','JEPA CEM','State random','State CEM']
    for ax,first,second,title,color in [(axes[0],'unit_decoded_teacher','unit_latent','B  Image teacher − unit latent','#675a9c'),(axes[1],'unit_physical_labels','unit_decoded_teacher','C  Physical labels − image teacher','#007f86')]:
        for i,ref in enumerate(refs):
            c=next(c for c in r['contrasts'] if c['reference_stream']==ref and c['left']==first and c['right']==second)['metrics']['position_mse'];lo,hi=c['secondary_95_percentile_interval'];y=4-i
            ax.hlines(y,lo,hi,color=color,lw=2);ax.plot(c['mean_difference'],y,'o',color=color)
        ax.set_yticks(range(4,-1,-1),names);ax.axvline(0,color='#444',ls='--',lw=1);ax.set_xlim(-1000,6500);ax.set_ylim(-.6,4.6);ax.grid(axis='x',alpha=.15);ax.set_title(title,loc='left',fontweight='bold');ax.set_xlabel('Position-MSE difference (negative favors first)')
    fig.suptitle('Matched-action rollout diagnostics after objective and label interventions',fontsize=14)
    fig.text(.5,.017,'128 shared consumed goals; six fixed models; four original-policy reference streams. Bars: equal-stream means; dots: model points.\nIntervals are retrospective secondary 95% shared-goal bootstrap intervals, not new independent confirmation or simultaneous guarantees.',ha='center',fontsize=9)
    fig.tight_layout(rect=[0,.065,1,.95],w_pad=2.5,h_pad=2)
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
    for ext in ('png','pdf'):fig.savefig(out.with_suffix('.'+ext),dpi=200,bbox_inches='tight')
    out.with_suffix('.json').write_text(json.dumps(dict(summary_sha256=sha(a.summary),source_sha256=sha(__file__),groups=groups,scope='All8 conditions and all4 reference-stream effects; retrospective matched-action evidence, no new efficacy or novelty claim.'),indent=2)+'\n');plt.close(fig)

if __name__=='__main__':main()
