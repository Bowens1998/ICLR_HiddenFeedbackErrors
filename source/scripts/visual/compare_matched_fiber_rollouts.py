"""Exact unchanged branches and full-versus-matched paired intervention effects."""
import argparse,json
from pathlib import Path
import numpy as np
from adaptation_streams import sha
from summarize_task_coordinate_horizon import contrast


def main():
    p=argparse.ArgumentParser()
    for k in ('full','matched','output'):p.add_argument('--'+k,required=True)
    a=p.parse_args();data={};bindings=[];checks=0;seeds=None
    for g in range(6):
        pair={}
        for tag,base in [('full',a.full),('matched',a.matched)]:
            d=Path(base)/f'job_{g}';r=json.loads((d/'report.json').read_text());ac=json.loads((d/'acceptance.json').read_text());assert ac['report_sha256']==sha(d/'report.json');pair[tag]=(d,r,ac)
            assert ac['source_sha256']==sha(Path(__file__).with_name('accept_'+('readout' if tag=='full' else 'matched')+'_fiber_rollout.py'))
            if seeds is None:seeds=ac['seeds']
            else:assert seeds==ac['seeds']
        for c in range(3):
            zz={}
            for tag,(d,r,ac) in pair.items():
                row=r['rows'][c];assert row['sha256']==sha(d/row['file']);zz[tag]=np.load(d/row['file']);data[g,c,tag]={k:np.array(v) for k,v in ac['rows'][c]['metrics']['fiber'].items()}
            for key in ('free_tokens','free_pose','reset_tokens','reset_pose','observed_tokens','observed_pose','true_pose','goal_tokens','seeds','reference_routes','horizons'):np.testing.assert_array_equal(zz['full'][key],zz['matched'][key]);checks+=1
        bindings.append(dict(group=g,**{tag+'_acceptance_sha256':sha(d/'acceptance.json') for tag,(d,r,ac) in pair.items()}))
    draws=np.random.default_rng(1364001).integers(0,128,(10000,128));rows=[]
    for c,name in enumerate(['unit_latent','unit_decoded_teacher','unit_physical_labels']):
        mm={}
        for metric in data[0,c,'full']:
            full=np.stack([data[g,c,'full'][metric].mean(0) for g in range(6)]);matched=np.stack([data[g,c,'matched'][metric].mean(0) for g in range(6)]);mm[metric]=contrast(matched,full,draws)
        rows.append(dict(condition=name,left='matched',right='full',metrics=mm))
    result=dict(status='PASS_ALL18_UNCHANGED_FREE_RESET_AND_REFERENCE_ARRAYS',exact_array_checks=checks,contrasts=rows,bindings=bindings,source_sha256=sha(__file__))
    with Path(a.output).open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(result['status'])

if __name__=='__main__':main()
