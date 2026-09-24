"""Five frozen primary effects with every model point, at publication width."""
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
 a=p.parse_args();r=json.loads(Path(a.summary).read_text());cc=r['primary_contrasts'];assert len(cc)==5;plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
 fig,ax=plt.subplots(figsize=(5.5,2.8));yy=np.arange(5);means=np.array([c['mean_difference'] for c in cc]);ci=np.array([c['primary_99_percentile_interval'] for c in cc])
 for i,c in enumerate(cc):ax.scatter(c['per_model_differences'],i+np.linspace(-.13,.13,6),s=13,color='#8dabb6',alpha=.8)
 ax.errorbar(means,yy,xerr=np.stack([means-ci[:,0],ci[:,1]-means]),fmt='o',capsize=3,ms=4,color='#663885');ax.axvline(0,color='#555',ls=':',lw=1);ax.set_yticks(yy,['Teacher / free','Labels / free','Latent / donor','Teacher / donor','Labels / donor']);ax.invert_yaxis();ax.set_xlabel('Actual guidance minus comparator: position-MSE change');ax.grid(axis='x',alpha=.15);fig.tight_layout(pad=.6)
 out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
 for ext in ('pdf','png'):fig.savefig(out.with_suffix('.'+ext),dpi=220)
 out.with_suffix('.json').write_text(json.dumps(dict(summary_sha256=sha(a.summary),source_sha256=sha(__file__),interval='Five prespecified 99% marginal intervals; nominal familywise 95% Bonferroni coverage.'),indent=2)+'\n')

if __name__=='__main__':main()
