"""Outcome-independent convex scaling of accepted corrections to a paired norm."""
import argparse,json
from pathlib import Path
import numpy as np
from adaptation_streams import sha
from readout_fiber import readout


def main():
    p=argparse.ArgumentParser()
    for k in ('horizon','full','protocol','output'):p.add_argument('--'+k,required=True)
    p.add_argument('--index',type=int,choices=range(6),required=True);a=p.parse_args();g=a.index
    hd=Path(a.horizon)/f'job_{g}';fd=Path(a.full)/f'job_{g}';out=Path(a.output)/f'job_{g}';out.mkdir(parents=True,exist_ok=False)
    hr=json.loads((hd/'report.json').read_text());fr=json.loads((fd/'report.json').read_text());fac=json.loads((fd/'acceptance.json').read_text())
    assert fac['status']=='COMPLETE1536_FIRST_FEEDBACK_FIBER_CHECKS' and fac['constrained_valid']==1536 and fac['report_sha256']==sha(fd/'report.json')
    assert fr['source_sha256']==sha(Path(__file__).with_name('prepare_readout_fiber_rollout.py')) and fr['projection_source_sha256']==sha(Path(__file__).with_name('readout_fiber.py'))
    assert fac['source_sha256']==sha(Path(__file__).with_name('accept_readout_fiber_rollout_inputs.py'))
    entries=[];head_sha=None
    for slot in (2,3,4):
        row=hr['rows'][slot];mi=8*g+slot;b=next(b for b in fr['bindings'] if b.get('model_index')==mi)
        assert row['sha256']==b['input_sha256']==sha(hd/row['file']) and row['head_sha256']==b['head_sha256']==sha(hd/row['head_file']) and sha(fd/b['output_file'])==b['output_sha256']
        if head_sha is None:head_sha=row['head_sha256']
        else:assert head_sha==row['head_sha256']
        head=dict(np.load(hd/row['head_file']));z=dict(np.load(fd/b['output_file']));delta=z['constrained'].astype(float)-z['predicted'];norm=np.linalg.norm(delta/head['scale'],axis=-1)
        entries.append((row,b,z,delta,norm,head))
    common=np.min(np.stack([x[4] for x in entries]),axis=0);assert common.shape==(512,)
    bindings=[dict(group=g,report_sha256=sha(hd/'report.json'),acceptance_sha256=sha(hd/'acceptance.json'))];records=[]
    for row,b,z,delta,norm,head in entries:
        alpha=np.divide(common,norm,out=np.zeros_like(common),where=norm>0);assert np.all((alpha>=0)&(alpha<=1))
        correction=(z['predicted'].astype(float)+alpha[:,None]*delta).astype(np.float32)
        err=np.max(abs(readout(head,correction)-readout(head,z['predicted'])),axis=-1);actual=np.linalg.norm((correction.astype(float)-z['predicted'])/head['scale'],axis=-1)
        np.testing.assert_allclose(actual,common,rtol=1e-6,atol=1e-6);assert np.max(err)<=1e-6
        name=f"model_{row['model_index']}.npz";np.savez_compressed(out/name,predicted=z['predicted'],observed=z['observed'],constrained=correction,full=z['constrained'],alpha=alpha,target_norm=common,actual_norm=actual,full_norm=norm)
        bindings.append(dict(model_index=row['model_index'],input_sha256=row['sha256'],head_sha256=row['head_sha256'],full_file=b['output_file'],full_file_sha256=b['output_sha256'],output_file=name,output_sha256=sha(out/name)))
        records.append(dict(model_index=row['model_index'],condition=row['entry']['adaptation_condition'],max_readout_error=float(err.max()),mean_alpha=float(alpha.mean()),mean_full_norm=float(norm.mean()),mean_matched_norm=float(actual.mean())))
    result=dict(status='MATCHED1536_FIBER_INPUTS_REQUIRES_ACCEPTANCE',index=g,bindings=bindings,records=records,full_report_sha256=sha(fd/'report.json'),full_acceptance_sha256=sha(fd/'acceptance.json'),protocol_sha256=sha(a.protocol),source_sha256=sha(__file__),scope='Convex scaling of existing accepted corrections; no new projection optimization.')
    (out/'report.json').write_text(json.dumps(result,indent=2)+'\n')

if __name__=='__main__':main()
