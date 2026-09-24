"""Matched-distance comparison of actual and other-goal observation guidance."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from adaptation_streams import sha


def main():
    p=argparse.ArgumentParser()
    for k in ('summary','output'):p.add_argument('--'+k,required=True)
    a=p.parse_args();r=json.loads(Path(a.summary).read_text());assert r['status']=='COMPLETE18_MODELS_GUIDED_VERSUS_SHUFFLED_INTERVENTION';conditions=['unit_latent','unit_decoded_teacher','unit_physical_labels'];labels=['Latent','Coordinate teacher','Physical labels']
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42});fig,ax=plt.subplots(1,2,figsize=(12,5.8))
    for offset,path,color,label in [(-.12,'fiber','#72408c','Actual observation guidance'),(.12,'shuffled','#bc783c','Other-goal observation guidance')]:
        cc=[next(c for c in r['contrasts'] if c['reference_stream']=='equal_four_streams' and c['condition']==condition and c['left']==path)['metrics']['position_mse'] for condition in conditions];mean=np.array([c['mean_difference'][-1] for c in cc]);ci=np.array([c['secondary_95_percentile_interval'][-1] for c in cc]);ax[0].errorbar(np.arange(3)+offset,mean,yerr=np.stack([mean-ci[:,0],ci[:,1]-mean]),fmt='o',capsize=4,color=color,label=label)
    cc=[next(c for c in r['guidance_contrasts'] if c['reference_stream']=='equal_four_streams' and c['condition']==condition)['metrics']['position_mse'] for condition in conditions];mean=np.array([c['mean_difference'][-1] for c in cc]);ci=np.array([c['secondary_95_percentile_interval'][-1] for c in cc]);ax[1].errorbar(range(3),mean,yerr=np.stack([mean-ci[:,0],ci[:,1]-mean]),fmt='o',capsize=4,color='#277c86')
    for i,c in enumerate(cc):ax[1].scatter(i+np.linspace(-.12,.12,6),np.array(c['per_model_differences'])[:,-1],s=18,alpha=.55,color='#277c86')
    for axis in ax:axis.set_xticks(range(3),labels);axis.axhline(0,color='#666',ls=':',lw=1);axis.grid(axis='y',alpha=.15)
    ax[0].set_title('A  Feedback source changes the intervention effect',loc='left',fontweight='bold',fontsize=11);ax[0].set_ylabel('25-action position-MSE change (pixel²)');handles,legend_labels=ax[0].get_legend_handles_labels();fig.legend(handles,legend_labels,loc='lower center',bbox_to_anchor=(.5,.12),ncol=2,frameon=False,fontsize=9)
    ax[1].set_title('B  Actual guidance wins at every model point',loc='left',fontweight='bold',fontsize=11);ax[1].set_ylabel('Actual − other-goal guidance (pixel²)')
    fig.suptitle('Same current readout and displacement, different future accuracy',fontsize=15);fig.text(.5,.035,'Six directions share a common norm for each recipient; one replacement at action 5. All 18 models and 128 goals retained.\nBars: secondary recipient-goal intervals conditional on realized donors; shared donor/recipient coupling is not resampled.',ha='center',fontsize=9);fig.tight_layout(rect=[0,.23,1,.91],w_pad=3)
    out=Path(a.output)
    for ext in ('png','pdf'):fig.savefig(out.with_suffix('.'+ext),dpi=180,bbox_inches='tight')
    out.with_suffix('.json').write_text(json.dumps(dict(summary_sha256=sha(a.summary),source_sha256=sha(__file__)),indent=2)+'\n')

if __name__=='__main__':main()
