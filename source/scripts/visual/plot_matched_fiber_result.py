"""Displacement control for the objective-dependent feedback effect."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from adaptation_streams import sha


def main():
    p=argparse.ArgumentParser()
    for k in ('full','matched','output'):p.add_argument('--'+k,required=True)
    a=p.parse_args();full=json.loads(Path(a.full).read_text());matched=json.loads(Path(a.matched).read_text());conditions=['unit_latent','unit_decoded_teacher','unit_physical_labels'];labels=['Latent','Coordinate teacher','Physical labels']
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42});fig,ax=plt.subplots(1,2,figsize=(12,5.8))
    for offset,r,color,label in [(-.12,full,'#bca4c7','Full correction'),(.12,matched,'#72408c','Matched displacement')]:
        cc=[next(c for c in r['contrasts'] if c['reference_stream']=='equal_four_streams' and c['condition']==condition and c['left']=='fiber')['metrics']['position_mse'] for condition in conditions]
        means=np.array([c['mean_difference'][-1] for c in cc]);ci=np.array([c['secondary_95_percentile_interval'][-1] for c in cc]);ax[0].errorbar(np.arange(3)+offset,means,yerr=np.stack([means-ci[:,0],ci[:,1]-means]),fmt='o',capsize=4,color=color,label=label)
    ax[0].set_xticks(range(3),labels);ax[0].set_ylabel('25-action position-MSE change (pixel²)');ax[0].set_title('A  Benefit persists at a common displacement',loc='left',fontweight='bold',fontsize=11);ax[0].legend(loc='lower left',fontsize=9)
    cc=[c for c in matched['objective_effect_interactions'] if c['reference_stream']=='equal_four_streams'];means=np.array([c['metrics']['position_mse']['mean_difference'][-1] for c in cc]);ci=np.array([c['metrics']['position_mse']['secondary_95_percentile_interval'][-1] for c in cc]);ax[1].errorbar(range(2),means,yerr=np.stack([means-ci[:,0],ci[:,1]-means]),fmt='o',color='#277c86',capsize=5)
    for i,c in enumerate(cc):ax[1].scatter(i+np.linspace(-.12,.12,6),np.array(c['metrics']['position_mse']['per_model_differences'])[:,-1],s=18,alpha=.55,color='#277c86')
    ax[1].set_xticks(range(2),['Teacher − latent','Labels − latent']);ax[1].set_xlim(-.5,1.5);ax[1].set_ylabel('Difference in matched intervention effects');ax[1].set_title('B  Coordinate objectives retain larger benefits',loc='left',fontweight='bold',fontsize=11)
    for axis in ax:axis.axhline(0,color='#666',ls=':',lw=1);axis.grid(axis='y',alpha=.15)
    fig.suptitle('Intervention magnitude explains part, but not all, of the objective-dependent effect',fontsize=14)
    fig.text(.5,.035,'One replacement at action 5; same nonlinear readout and standardized displacement across objectives.\nSix fixed model groups × four streams × 128 shared goals. Bars: secondary 95% paired-goal intervals; small dots: model points.',ha='center',fontsize=9)
    fig.tight_layout(rect=[0,.13,1,.91],w_pad=3);out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
    for ext in ('png','pdf'):fig.savefig(out.with_suffix('.'+ext),dpi=180,bbox_inches='tight')
    out.with_suffix('.json').write_text(json.dumps(dict(full_sha256=sha(a.full),matched_sha256=sha(a.matched),source_sha256=sha(__file__)),indent=2)+'\n')

if __name__=='__main__':main()
