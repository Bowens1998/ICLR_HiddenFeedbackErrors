"""Paired guided/donor projections with one six-direction common displacement."""
import argparse,json,time
from pathlib import Path
import numpy as np
from adaptation_streams import sha
from readout_fiber import project,readout


def main():
    p=argparse.ArgumentParser()
    for k in ('horizon','full','protocol','output'):p.add_argument('--'+k,required=True)
    p.add_argument('--index',type=int,choices=range(6),required=True);a=p.parse_args();g=a.index;out=Path(a.output)/f'job_{g}';out.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    hd=Path(a.horizon)/f'job_{g}';fd=Path(a.full)/f'job_{g}';hr=json.loads((hd/'report.json').read_text());fr=json.loads((fd/'report.json').read_text());fac=json.loads((fd/'acceptance.json').read_text())
    assert fac['status']=='COMPLETE1536_FIRST_FEEDBACK_FIBER_CHECKS' and fac['constrained_valid']==1536 and fac['report_sha256']==sha(fd/'report.json')
    assert fac['source_sha256']==sha(Path(__file__).with_name('accept_readout_fiber_rollout_inputs.py')) and fr['source_sha256']==sha(Path(__file__).with_name('prepare_readout_fiber_rollout.py')) and fr['projection_source_sha256']==sha(Path(__file__).with_name('readout_fiber.py'))
    rng=np.random.default_rng(1366001);permutation=rng.permutation(128)
    while np.any(permutation==np.arange(128)):permutation=rng.permutation(128)
    entries=[];norms=[];head_sha=None
    for slot in (2,3,4):
        mi=8*g+slot;row=hr['rows'][slot];b=next(b for b in fr['bindings'] if b.get('model_index')==mi)
        assert row['sha256']==b['input_sha256']==sha(hd/row['file']) and row['head_sha256']==b['head_sha256']==sha(hd/row['head_file']) and b['output_sha256']==sha(fd/b['output_file'])
        if head_sha is None:head_sha=b['head_sha256']
        else:assert head_sha==b['head_sha256']
        h=dict(np.load(hd/row['head_file']));z=dict(np.load(fd/b['output_file']));donor=z['observed'].reshape(4,128,192)[:,permutation].reshape(512,192);shuffled=[];solvers=[]
        with (out/f'solver_attempts_{mi}.jsonl').open('x') as log:
            for i,(pred,obs) in enumerate(zip(z['predicted'],donor)):
                corrected,_,solver=project(h,pred,obs);log.write(json.dumps(dict(index=i,**solver))+'\n');log.flush();assert solver['status']=='solved' and corrected is not None
                shuffled.append(corrected);solvers.append(solver)
        shuffled=np.stack(shuffled);directions=[z['constrained'].astype(float)-z['predicted'],shuffled.astype(float)-z['predicted']];nn=[np.linalg.norm(delta/h['scale'],axis=-1) for delta in directions];norms+=nn;entries.append((row,b,h,z,donor,shuffled,directions,nn,solvers));print('QP_COMPLETE',g,mi,flush=True)
    common=np.minimum.reduce(norms);bindings=[dict(group=g,report_sha256=sha(hd/'report.json'),acceptance_sha256=sha(hd/'acceptance.json'))];rows=[]
    for row,b,h,z,donor,shuffled,directions,nn,solvers in entries:
        arrays=dict(predicted=z['predicted'],observed=z['observed'],full=z['constrained'],donor=donor,shuffled_full=shuffled,target_norm=common,permutation=permutation);metrics={}
        for name,delta,norm in zip(('constrained','shuffled'),directions,nn):
            alpha=np.divide(common,norm,out=np.zeros_like(common),where=norm>0);corrected=(z['predicted'].astype(float)+alpha[:,None]*delta).astype(np.float32);actual=np.linalg.norm((corrected.astype(float)-z['predicted'])/h['scale'],axis=-1);error=float(np.max(abs(readout(h,corrected)-readout(h,z['predicted']))));assert error<=1e-6;np.testing.assert_allclose(actual,common,rtol=1e-6,atol=1e-6)
            arrays.update({name:corrected,name+'_alpha':alpha,name+'_full_norm':norm,name+'_actual_norm':actual});metrics[name]=dict(max_readout_error=error,mean_full_norm=float(norm.mean()),mean_scaled_norm=float(actual.mean()),mean_alpha=float(alpha.mean()))
        name=f"model_{row['model_index']}.npz";np.savez_compressed(out/name,**arrays);bindings.append(dict(model_index=row['model_index'],input_sha256=row['sha256'],head_sha256=row['head_sha256'],full_file=b['output_file'],full_file_sha256=b['output_sha256'],output_file=name,output_sha256=sha(out/name),solver_log_sha256=sha(out/f"solver_attempts_{row['model_index']}.jsonl")));rows.append(dict(model_index=row['model_index'],condition=row['entry']['adaptation_condition'],metrics=metrics))
    result=dict(status='SHUFFLED1536_PAIRED_FIBER_INPUTS_REQUIRES_ACCEPTANCE',index=g,bindings=bindings,rows=rows,permutation=permutation.tolist(),full_report_sha256=sha(fd/'report.json'),full_acceptance_sha256=sha(fd/'acceptance.json'),protocol_sha256=sha(a.protocol),source_sha256=sha(__file__),projection_source_sha256=sha(Path(__file__).with_name('readout_fiber.py')),elapsed_seconds=time.monotonic()-start)
    (out/'report.json').write_text(json.dumps(result,indent=2)+'\n')

if __name__=='__main__':main()
