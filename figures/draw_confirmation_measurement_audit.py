#!/usr/bin/env python3
"""Redraw the fixed original nine contrasts and evaluation-only core means.

Python 3, NumPy and Matplotlib only. No repository or checkpoints required.
The supplied JSON binds all displayed values to accepted evidence. This renderer
performs no resampling, hypothesis testing or across-time causal analysis.
"""
import argparse
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
import numpy as np

HERE = Path(__file__).resolve().parent
NAME = 'confirmation_measurement_audit'
INK, MUTED, GRAY = '#203247', '#596B7C', '#718096'
TEAL, LINE = '#087F83', '#D7E0E8'
TOP_SCALE, TOP_BASE = 3.23/5.0, 1.77/5.0


def setup():
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10.3,
        'axes.labelsize':10.3,'text.color':INK,'axes.labelcolor':INK,
        'xtick.color':MUTED,'ytick.color':INK,'axes.edgecolor':LINE,
        'axes.linewidth':.8,'axes.spines.top':False,'axes.spines.right':False,
        'axes.spines.left':False,'svg.fonttype':'none','svg.hashsalt':NAME,
        'pdf.fonttype':42,'savefig.facecolor':'white'})


def upper_axis(fig, rect):
    x,y,w,h=rect
    return fig.add_axes([x,TOP_BASE+y*TOP_SCALE,w,h*TOP_SCALE])


def upper_text(fig,x,y,text,**kwargs):
    return fig.text(x,TOP_BASE+y*TOP_SCALE,text,**kwargs)


def heading(fig,x,y,letter,title,subtitle=None):
    upper_text(fig,x,y,letter,fontsize=11.1,fontweight='bold',va='top',color=TEAL)
    upper_text(fig,x+.026,y,title,fontsize=11.1,fontweight='bold',va='top')
    if subtitle:
        upper_text(fig,x+.026,y-.062,subtitle,fontsize=9.6,color=MUTED,va='top')


def style(ax, zero=True):
    ax.grid(axis='x',color=LINE,linewidth=.55,alpha=.72)
    ax.set_axisbelow(True)
    ax.tick_params(axis='y',length=0,pad=7)
    ax.tick_params(axis='x',length=3,width=.6,labelsize=9.8)
    if zero:ax.axvline(0,color=GRAY,linewidth=.9,linestyle=(0,(3,3)),zorder=1)


def forest(ax,rows,labels):
    positions=np.arange(len(rows))[::-1]
    for y,row in zip(positions,rows):
        points=row['model_points']
        for j,point in enumerate(points):
            ax.scatter(point['paired_change'],y+(j-(len(points)-1)/2)*.027,
                s=15,color='#A5B3C1',edgecolors='white',linewidth=.3,zorder=2)
        value=row['estimate'];lo,hi=row['interval']
        ax.errorbar(value,y,xerr=[[value-lo],[hi-value]],fmt='o',markersize=5.5,
            capsize=2.7,linewidth=1.55,color=TEAL,zorder=3)
    ax.set_yticks(positions,labels)
    style(ax)


def draw(data,output):
    setup()
    rows=data['original_confirmation']
    assert len(rows)==9 and [len(r['model_points']) for r in rows]==[6]*5+[2]*2+[0]*2
    fig=plt.figure(figsize=(7.5,5.0))
    ax=upper_axis(fig,[.292,.20,.276,.61])
    dual=upper_axis(fig,[.748,.635,.222,.172])
    native=upper_axis(fig,[.748,.20,.222,.172])
    heading(fig,.014,.97,'A',r'Single pose readout $g_A$','6 groups · 6-direction matching family')
    heading(fig,.603,.97,'B',r'Dual readouts $g_A,g_B$','2 groups · 12-direction matching family')
    heading(fig,.603,.535,'C','Native DINO-WM: PushT','2-direction matching family')
    forest(ax,rows[:5],['Coordinate teacher:\nactual − free','Physical labels:\nactual − free',
        'Latent: actual − donor','Coordinate teacher:\nactual − donor','Physical labels:\nactual − donor'])
    ax.set_ylim(-.45,4.45);ax.axhline(2.5,color=LINE,linewidth=.65)
    ax.set_xlim(-1320,100);ax.set_xticks([-1200,-800,-400,0])
    forest(dual,rows[5:7],['Coordinate\nteacher','Physical labels'])
    dual.set_ylim(-.48,1.48);dual.set_xlim(-1320,100);dual.set_xticks([-1200,-600,0])
    dual.text(-.02,1.20,'Actual − free',transform=dual.transAxes,ha='left',color=MUTED,fontsize=9.6)
    forest(native,rows[7:9],['Actual − free','Actual − donor'])
    native.set_ylim(-.48,1.48);native.set_xlim(-5100,300);native.set_xticks([-4000,-2000,0])
    native.text(1,-.65,'Separate bank; native scale differs',transform=native.transAxes,
        ha='right',color=MUTED,fontsize=9.6)
    upper_text(fig,.10,.117,'Terminal block-position MSE change (pixels²)',ha='left',va='top',fontsize=9.6)
    upper_text(fig,.018,.014,'Negative favors actual guidance',fontsize=9.6,color=MUTED)
    handles=[Line2D([],[],color=TEAL,marker='o',markersize=5,linewidth=1.5,label='Mean + 99.444% interval'),
        Line2D([],[],color='#A5B3C1',marker='o',markersize=4,linewidth=0,label='Fixed model group')]
    fig.legend(handles=handles,loc='lower right',bbox_to_anchor=(.985,TOP_BASE-.02*TOP_SCALE),
        ncol=2,frameon=False,fontsize=9.6,handlelength=1.6,columnspacing=1.0)
    fig.add_artist(Line2D([.018,.98],[.334,.334],transform=fig.transFigure,color=LINE,linewidth=.8))
    fig.text(.018,.315,'D',fontsize=11.1,fontweight='bold',color=TEAL,va='top')
    fig.text(.046,.315,r'$g_{\mathrm{eval}}$ at insertion (action 5)',fontsize=10.6,fontweight='bold',va='top')
    fig.text(.548,.315,'E',fontsize=11.1,fontweight='bold',color=TEAL,va='top')
    fig.text(.576,.315,r'$g_{\mathrm{eval}}$ at terminal (action 25)',fontsize=10.6,fontweight='bold',va='top')
    insertion=fig.add_axes([.235,.117,.265,.148])
    terminal=fig.add_axes([.66,.117,.315,.148])
    objectives=('latent','decoded_teacher','physical_labels')
    labels=['Latent','Coordinate teacher','Physical labels']
    for axis,stage in [(insertion,'insertion'),(terminal,'terminal')]:
        for y,objective in zip([2,1,0],objectives):
            row=next(r for r in data['evaluation_only_core'] if r['stage']==stage and r['objective']==objective)
            free,actual=row['free_mse'],row['actual_mse']
            axis.plot([free,actual],[y,y],color=GRAY,linewidth=1.2,zorder=2)
            axis.scatter([free],[y],s=34,facecolors='white',edgecolors=GRAY,linewidths=1.25,zorder=4)
            axis.scatter([actual],[y],s=34,facecolors=TEAL,edgecolors='white',linewidths=.6,zorder=5)
        axis.set_ylim(-.5,2.5)
        axis.set_yticks([2,1,0],labels if stage=='insertion' else ['','',''])
        style(axis,zero=False)
        axis.xaxis.set_major_formatter(FuncFormatter(lambda x,_:f'{x/1000:g}k'))
        axis.tick_params(axis='x',labelsize=9.6,pad=3)
    insertion.set_xlim(1400,3400);insertion.set_xticks([1500,2000,2500,3000])
    terminal.set_xlim(4500,14500);terminal.set_xticks([5000,7500,10000,12500])
    fig.text(.365,.062,'Block-position MSE (pixels²)',ha='center',va='top',fontsize=9.6)
    fig.text(.815,.062,'Block-position MSE (pixels²)',ha='center',va='top',fontsize=9.6)
    branch_handles=[Line2D([],[],color=GRAY,marker='o',markerfacecolor='white',markersize=5.5,
        linewidth=0,label='Free'),Line2D([],[],color=TEAL,marker='o',markersize=5.5,linewidth=0,label='Actual')]
    fig.legend(handles=branch_handles,loc='lower left',bbox_to_anchor=(.006,-.003),ncol=2,
        frameon=False,fontsize=9.6,columnspacing=.8,handletextpad=.35)
    fig.text(.975,.014,'D–E: all six groups; descriptive means, no intervals',ha='right',fontsize=9.6,color=MUTED)
    output.mkdir(parents=True,exist_ok=True)
    fig.savefig(output/f'{NAME}.pdf',metadata={'Creator':'Scientific vector figure source',
        'Title':data['title'],'CreationDate':None,'ModDate':None})
    fig.savefig(output/f'{NAME}.svg',metadata={'Creator':'Scientific vector figure source','Title':data['title'],'Date':None})
    fig.savefig(output/f'{NAME}.png',dpi=250,metadata={'Title':data['title']})
    plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,default=HERE/f'{NAME}.json')
    p.add_argument('--output-dir',type=Path,default=HERE)
    a=p.parse_args();data=json.loads(a.data.read_text());draw(data,a.output_dir)
    print(json.dumps({'figure':NAME,'checks':data['checks'],'artifacts':{ext:hashlib.sha256((a.output_dir/f'{NAME}.{ext}').read_bytes()).hexdigest() for ext in ('pdf','svg','png')}},indent=2))

if __name__=='__main__':main()
