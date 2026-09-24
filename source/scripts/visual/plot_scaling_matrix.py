"""Show all training pools; fixed last and best sensitivity remain separate."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scaling_effects import analyze


p=argparse.ArgumentParser();p.add_argument('--summary',required=True);p.add_argument('--output-dir',required=True);a=p.parse_args()
summary=json.loads(Path(a.summary).read_text());assert summary['layout']=='scaling' and summary['cases']==128
analyze(summary['tables'])  # Reject partial matrices before making a figure.
rows={(r['replica'],r['arm'],r['checkpoint'],r['episodes'],r['updates'],r['algorithm']):r for r in summary['tables']}
out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True)
settings=[(256,5250),(256,21000),(1024,5250),(1024,21000)]
labels=['256 / 5,250','256 / 21,000','1,024 / 5,250','1,024 / 21,000']
colors={'jepa':'#2166ac','state':'#c76816'}
names={'jepa':'JEPA target','state':'Direct state target'}
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
for checkpoint in ['last','best']:
    fig,axes=plt.subplots(2,2,figsize=(12,8),sharex=True,sharey=True)
    for col,arch in enumerate(['transformer','gru']):
        for row,algorithm in enumerate(['random','cem']):
            ax=axes[row,col]
            for target in ['jepa','state']:
                values=np.array([[rows[(rep,arch+'_'+target,checkpoint,n,u,algorithm)]['mean_cost'] for n,u in settings] for rep in range(3)])
                for rep in range(3):ax.plot(range(4),values[rep],color=colors[target],alpha=.35,lw=.8,marker=['o','s','^'][rep],ms=4)
                ax.plot(range(4),values.mean(0),color=colors[target],lw=2.5,label=names[target])
            ax.axhline(summary['zero_cost'],color='.55',ls=':',lw=1,label='Zero-action cost')
            ax.set_title(('GRU' if arch=='gru' else 'Transformer')+' · '+('CEM' if algorithm=='cem' else 'Random'))
            ax.grid(axis='y',alpha=.15);ax.set_xticks(range(4),labels,rotation=12)
            if col==0:ax.set_ylabel('Mean realized block-pose cost ↓')
            if row==1:ax.set_xlabel('Training episodes / optimizer updates')
    axes[0,0].legend(frameon=False,fontsize=9)
    fig.suptitle(f'PushT · {checkpoint} checkpoint · identical planning budget',fontsize=14)
    fig.text(.5,.015,'Thin lines/markers: three independent training pools. Thick lines: pool means. Same 128 development goals; 9,000 candidates per case.\nTarget comparisons also change representation, supervision and parameter count. Best is a separate checkpoint-selection sensitivity.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.07,1,.96))
    for suffix in ['png','pdf']:fig.savefig(out/f'scaling_cost_{checkpoint}.{suffix}',dpi=180)
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4.8),sharey=True)
    for ax,arch in zip(axes,['transformer','gru']):
        for target in ['jepa','state']:
            delta=np.array([[rows[(rep,arch+'_'+target,checkpoint,n,u,'cem')]['mean_cost']-rows[(rep,arch+'_'+target,checkpoint,n,u,'random')]['mean_cost'] for n,u in settings] for rep in range(3)])
            for rep in range(3):ax.plot(range(4),delta[rep],color=colors[target],alpha=.35,lw=.8,marker=['o','s','^'][rep],ms=4)
            ax.plot(range(4),delta.mean(0),color=colors[target],lw=2.5,label=names[target])
        ax.axhline(0,color='.35',lw=1);ax.set_title('GRU' if arch=='gru' else 'Transformer');ax.set_xticks(range(4),labels,rotation=12);ax.grid(axis='y',alpha=.15);ax.set_xlabel('Training episodes / optimizer updates')
    axes[0].set_ylabel('CEM − random realized cost\nPositive: CEM worse');axes[0].legend(frameon=False,fontsize=9)
    fig.suptitle(f'Optimizer effect · {checkpoint} checkpoint',fontsize=14)
    fig.text(.5,.015,'Thin lines/markers: paired effects in each independent training pool. Thick lines: pool means.\nDevelopment scenarios; three pool effects are descriptive, not a population confidence interval.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.10,1,.95))
    for suffix in ['png','pdf']:fig.savefig(out/f'scaling_optimizer_effect_{checkpoint}.{suffix}',dpi=180)
    plt.close(fig)
print('WROTE four complete-matrix figures, PNG and PDF')
