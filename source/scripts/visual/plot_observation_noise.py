"""Plot the complete noise matrix, retaining each training pool."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    p=argparse.ArgumentParser();p.add_argument('--summary',required=True);p.add_argument('--output-dir',required=True);a=p.parse_args()
    source=Path(a.summary);s=json.loads(source.read_text());assert len(s['tables'])==144 and len(s['provenance'])==96 and s['cases']==128
    cells={(r['sigma'],r['replica'],r['architecture'],r['mode'],r['algorithm']):r for r in s['tables']};assert len(cells)==144
    modes=[('none','JEPA','#777777'),('inverse','JEPA + inverse','#2166ac'),('inverse_goal','JEPA + inverse + goal','#b75b19'),('direct_state','Direct state','#459668')]
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True);files=[]
    fig,axes=plt.subplots(2,4,figsize=(15,7),sharex=True,sharey='row')
    for col,(arch,algorithm) in enumerate((a,g) for a in ['transformer','gru'] for g in ['random','cem']):
        for mode,label,color in modes:
            for row,metric in enumerate(['mean_cost','success']):
                ys=np.array([[cells[sigma,pool,arch,mode,algorithm][metric] for sigma in [0,8,24]] for pool in range(3)])
                assert np.isfinite(ys).all()
                if row:ys*=100
                ax=axes[row,col]
                for pool in range(3):ax.plot([0,8,24],ys[pool],color=color,alpha=.3,linewidth=.7,marker=['o','s','^'][pool],markersize=3)
                ax.plot([0,8,24],ys.mean(0),color=color,linewidth=2,label=label)
                ax.grid(alpha=.15);ax.set_xticks([0,8,24])
        axes[0,col].set_title(f'{arch.title()} · {algorithm.upper()}')
        axes[1,col].set_xlabel('Noise sigma (pixel units)');axes[1,col].set_ylim(-2,102)
    axes[0,0].set_ylabel('Mean realized block-pose cost ↓');axes[1,0].set_ylabel('Success (%) ↑')
    handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,.96),ncol=4,frameon=False)
    fig.suptitle('PushT · observation noise · fixed-last checkpoints',y=.995)
    fig.text(.5,.015,'Thin lines / markers: three training pools; thick lines: pool means. Shared 128 development goals, 9,000 candidates per route.\nNoise affects history and goal images only. One fixed draw per image; no training adaptation or independent confirmation.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.08,1,.91))
    for ext in ['png','pdf']:
        path=out/f'observation_noise_performance.{ext}';fig.savefig(path,dpi=180);files.append(path)
    plt.close(fig)
    interactions=s['robustness_interactions'];assert len(interactions['rows'])==96
    fig,axes=plt.subplots(2,2,figsize=(13,8),sharex=True)
    layout=[(a,g,m) for a in ['transformer','gru'] for g in ['random','cem'] for m in ['inverse','inverse_goal']]
    labels=[f'{"T" if a=="transformer" else "GRU"}\n{g}\n{"Inv" if m=="inverse" else "Inv+goal"}' for a,g,m in layout]
    for row,reference in enumerate(['none','direct_state']):
        for col,sigma in enumerate([8,24]):
            ax=axes[row,col];ax.axhline(0,color='black',linewidth=.8)
            for x,(arch,algorithm,mode) in enumerate(layout):
                rs=sorted((r for r in interactions['rows'] if (r['sigma'],r['reference'],r['architecture'],r['algorithm'],r['mode'])==(sigma,reference,arch,algorithm,mode)),key=lambda r:r['replica']);assert [r['replica'] for r in rs]==[0,1,2]
                ys=[r['mean_cost_degradation_difference'] for r in rs]
                for pool,y in enumerate(ys):ax.scatter(x,y,marker=['o','s','^'][pool],color='#2166ac',alpha=.65,s=30)
                ax.scatter(x,np.mean(ys),marker='_',s=200,color='#111111')
            ax.set_title(f'Sigma {sigma} · reference: {reference}');ax.set_xticks(range(8),labels);ax.grid(axis='y',alpha=.15)
            ax.set_ylabel('Difference in cost degradation ↓')
    fig.suptitle('Does auxiliary supervision reduce noise-induced degradation?')
    fig.text(.5,.015,'(Auxiliary noisy − clean) − (reference noisy − clean). Negative: smaller absolute cost degradation.\nMarkers: individual pools; bars: means. This interaction is distinct from absolute noisy performance.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.08,1,.96))
    for ext in ['png','pdf']:
        path=out/f'observation_noise_interactions.{ext}';fig.savefig(path,dpi=180);files.append(path)
    plt.close(fig)
    report=dict(summary_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),files={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in files},scope='Complete scientific summary required. Rendering requires visual inspection before publication.')
    (out/'observation_noise_figure_manifest.json').write_text(json.dumps(report,indent=2)+'\n')
if __name__=='__main__':main()
