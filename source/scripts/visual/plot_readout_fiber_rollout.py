"""One-shot selective-feedback mechanism result."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from adaptation_streams import sha


def main():
    p=argparse.ArgumentParser();p.add_argument('--summary',required=True);p.add_argument('--output',required=True);a=p.parse_args();r=json.loads(Path(a.summary).read_text());assert r['status']=='COMPLETE18_MODELS_SINGLE_FEEDBACK_INTERVENTION'
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    fig,ax=plt.subplots(2,3,figsize=(14,8));xx=np.array(r['horizons']);bounds=np.array([c['metrics']['position_mse']['secondary_95_percentile_interval'] for c in r['contrasts'] if c['reference_stream']=='equal_four_streams']);lo=float(bounds.min());hi=float(bounds.max());pad=.05*(hi-lo);conditions=['unit_latent','unit_decoded_teacher','unit_physical_labels'];titles=['Latent objective','Coordinate-teacher objective','Physical-label objective'];colors={'free':'#777777','fiber':'#8a4b9a','reset':'#277c86'}
    for col,(condition,title) in enumerate(zip(conditions,titles)):
        for path,label in [('free','Free rollout'),('fiber','Readout-preserving correction'),('reset','Full observation reset')]:
            g=next(g for g in r['groups'] if g['reference_stream']=='equal_four_streams' and g['condition']==condition and g['path']==path)
            ax[0,col].plot(xx,g['means']['position_mse'],'o-',color=colors[path],label=label,ms=4)
        ax[0,col].set_title(title,fontweight='bold');ax[0,col].set_ylim(0,13500);ax[0,col].set_ylabel('Position MSE (pixel²)');ax[0,col].legend(fontsize=8)
        for path,label in [('fiber','Correction − free'),('reset','Full reset − free')]:
            c=next(c for c in r['contrasts'] if c['reference_stream']=='equal_four_streams' and c['condition']==condition and c['left']==path)['metrics']['position_mse'];ci=np.array(c['secondary_95_percentile_interval']);ax[1,col].plot(xx,c['mean_difference'],'o-',color=colors[path],label=label,ms=4);ax[1,col].fill_between(xx,ci[:,0],ci[:,1],color=colors[path],alpha=.12)
        ax[1,col].axhline(0,color='#444',ls=':',lw=1);ax[1,col].set_ylim(lo-pad,hi+pad);ax[1,col].set_ylabel('Position-MSE change');ax[1,col].legend(fontsize=8)
    for axis in ax.flat:axis.set_xticks(xx);axis.set_xlabel('Primitive-action horizon');axis.grid(alpha=.15)
    fig.suptitle('Preserve the current task readout, improve the predictive future',fontsize=15)
    fig.text(.5,.017,'One feedback intervention at action 5; subsequent actions and model weights fixed. Means: six models × four reference streams × 128 shared goals.\nBands: retrospective secondary 95% shared-goal intervals. Observed-token guidance is used only for this mechanism intervention.',ha='center',fontsize=9)
    fig.tight_layout(rect=[0,.075,1,.945],h_pad=2,w_pad=2)
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
    for ext in ('png','pdf'):fig.savefig(out.with_suffix('.'+ext),dpi=180,bbox_inches='tight')
    out.with_suffix('.json').write_text(json.dumps(dict(summary_sha256=sha(a.summary),source_sha256=sha(__file__)),indent=2)+'\n');plt.close(fig)

if __name__=='__main__':main()
