"""Retrospective matched-horizon mechanism figure; no prospective claims."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from adaptation_streams import sha


def main():
    p=argparse.ArgumentParser();p.add_argument('--summary',required=True);p.add_argument('--output',required=True);a=p.parse_args();r=json.loads(Path(a.summary).read_text());assert r['status']=='COMPLETE48_MODELS_FIVE_HORIZON_DIAGNOSTICS'
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    fig,axes=plt.subplots(2,2,figsize=(13,9));x=np.asarray(r['horizons'])
    def group(recipe,condition,path):return next(g for g in r['groups'] if g['reference_stream']=='equal_four_streams' and g['recipe']==recipe and g['condition']==condition and g['path']==path)
    conditions=['original','clipped_latent','unit_latent','unit_decoded_teacher','unit_physical_labels'];names=['Original','Clipped latent','Unit latent','Coordinate teacher','Physical labels'];colors=['#777777','#4685b5','#174f84','#9657a4','#cf7831']
    for ax,path,title in [(axes[0,0],'free','A  JEPA: free rollout'),(axes[0,1],'teacher','B  JEPA: actual-history feedback')]:
        for c,name,col in zip(conditions,names,colors):ax.plot(x,group('pose_encoded',c,path)['means']['position_mse'],'o-',color=col,label=name,ms=4)
        ax.plot(x,group('pose_encoded','original','observed')['means']['position_mse'],':',color='black',label='Observed-image decode',lw=1.8)
        ax.set_title(title,loc='left',fontweight='bold');ax.set_ylabel('Block-position MSE (pixel²)');ax.legend(fontsize=8,ncol=2)
    ymax=max(ax.get_ylim()[1] for ax in axes[0]);[ax.set_ylim(0,ymax) for ax in axes[0]]
    ax=axes[1,0]
    for left,right,col,label in [('unit_decoded_teacher','unit_latent','#9657a4','Coord. teacher − latent'),('unit_physical_labels','unit_decoded_teacher','#cf7831','Labels − coord. teacher')]:
        for path,ls in [('free','-'),('teacher','--')]:
            c=next(c for c in r['contrasts'] if c['reference_stream']=='equal_four_streams' and c['kind']=='objective_difference' and c['left']==left and c['right']==right and c['path']==path)['metrics']['position_mse'];ci=np.asarray(c['secondary_95_percentile_interval'])
            ax.plot(x,c['mean_difference'],ls,color=col,label=label+(' (free)' if path=='free' else ' (actual)'),lw=2);ax.fill_between(x,ci[:,0],ci[:,1],color=col,alpha=.08)
    ax.axhline(0,color='#444',ls=':',lw=1);ax.set_title('C  Objective effects depend on feedback',loc='left',fontweight='bold');ax.set_ylabel('Position-MSE difference');ax.legend(fontsize=8)
    ax=axes[1,1]
    for c,label,col in [('original','Original','#777777'),('clipped_state','Clipped state','#32888b'),('unit_state','Unit state','#164f4f')]:
        for path,ls in [('free','-'),('teacher','--')]:ax.plot(x,group('state',c,path)['means']['position_mse'],ls,color=col,label=label+(' (free)' if path=='free' else ' (actual)'),lw=2)
    ax.plot(x,group('state','original','observed')['means']['position_mse'],':',color='black',label='Observed-image decode')
    ax.set_title('D  Direct-state control',loc='left',fontweight='bold');ax.set_ylabel('Block-position MSE (pixel²)');ax.set_ylim(bottom=0);ax.legend(fontsize=8,ncol=2)
    for ax in axes.flat:ax.set_xticks(x);ax.set_xlabel('Primitive-action horizon');ax.grid(alpha=.15)
    fig.suptitle('Same trajectories, different feedback: locating the rollout error',fontsize=15)
    fig.text(.5,.018,'128 consumed shared goals × four original-policy streams; means over six fixed models. Actual-history feedback is privileged.\nPanel C bands: retrospective secondary 95% shared-goal bootstrap intervals, without simultaneous coverage. No new search or training.',ha='center',fontsize=9)
    fig.tight_layout(rect=[0,.065,1,.95],h_pad=2,w_pad=2)
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
    for ext in ('png','pdf'):fig.savefig(out.with_suffix('.'+ext),dpi=180,bbox_inches='tight')
    out.with_suffix('.json').write_text(json.dumps(dict(summary_sha256=sha(a.summary),source_sha256=sha(__file__),scope='All frozen conditions, five matched horizons; retrospective free/actual-history feedback diagnostic.'),indent=2)+'\n');plt.close(fig)

if __name__=='__main__':main()
