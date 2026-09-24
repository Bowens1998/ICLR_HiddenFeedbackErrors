"""Predetermined saved-token geometry check; retain every failed attempt."""
import argparse,json,time
from pathlib import Path
import numpy as np
from adaptation_streams import sha
from readout_fiber import project,readout,geometry


def main():
    p=argparse.ArgumentParser()
    for k in ('runs','protocol','output'):p.add_argument('--'+k,required=True)
    p.add_argument('--index',type=int,choices=range(6),required=True)
    a=p.parse_args();out=Path(a.output)/f'job_{a.index}';out.mkdir(parents=True,exist_ok=False);records=[];bindings=[];started=time.monotonic()
    for g in (a.index,):
        d=Path(a.runs)/f'job_{g}';r=json.loads((d/'report.json').read_text());ac=json.loads((d/'acceptance.json').read_text());assert ac['status']=='PASS_ALL8_HORIZON_NUMPY_RECONSTRUCTIONS' and ac['report_sha256']==sha(d/'report.json')
        bindings.append(dict(group=g,report_sha256=sha(d/'report.json'),acceptance_sha256=sha(d/'acceptance.json')))
        for slot in (2,3,4):
            row=r['rows'][slot];assert sha(d/row['file'])==row['sha256'] and sha(d/row['head_file'])==row['head_sha256'];head=dict(np.load(d/row['head_file']));z=np.load(d/row['file']);saved=[]
            for ref in range(4):
                for i in range(128):
                    for h in (0,):
                        pred=z['free_tokens'][ref,i,h];obs=z['observed_tokens'][ref,i,h];baseline=readout(head,pred)
                        corrected,naive,solver=project(head,pred,obs);record=dict(group=g,model_index=row['model_index'],condition=row['entry']['adaptation_condition'],reference=int(z['reference_routes'][ref]),goal_index=i,seed=int(z['seeds'][i]),horizon=int(z['horizons'][h]),solver=solver)
                        for name,value in [('naive',naive),('constrained',corrected)]:
                            if value is None:record[name]=dict(valid=False);continue
                            err=readout(head,value)-baseline;xx=(pred.astype(float)-head['mean'])/head['scale'];oo=(obs.astype(float)-head['mean'])/head['scale'];vv=(value.astype(float)-head['mean'])/head['scale'];old=geometry(head,pred);new=geometry(head,value)
                            record[name]=dict(valid=bool(np.isfinite(value).all() and np.max(abs(err))<=1e-6 and (name=='naive' or solver['status']=='solved')),normalized_readout_max_error=float(np.max(abs(err))),physical_coordinate_errors=(abs(err)*head['target_scale']).tolist(),squared_distance_before=float(np.sum((xx-oo)**2)),squared_distance_after=float(np.sum((vv-oo)**2)),correction_norm=float(np.linalg.norm(vv-xx)),activation_changes=int(np.sum(old[3]!=new[3])+np.sum(old[4]!=new[4])))
                        records.append(record);saved.append(dict(predicted=pred,observed=obs,naive=naive,constrained=corrected if corrected is not None else np.full_like(pred,np.nan)))
            name=f"model_{row['model_index']}.npz";np.savez_compressed(out/name,**{k:np.stack([x[k] for x in saved]) for k in saved[0]})
            bindings.append(dict(model_index=row['model_index'],input_sha256=row['sha256'],head_sha256=row['head_sha256'],output_file=name,output_sha256=sha(out/name)))
            print('MODEL',row['model_index'],'valid',sum(x['constrained']['valid'] for x in records[-512:]),'/512',flush=True)
    assert len(records)==1536
    result=dict(status='FIRST_FEEDBACK_FIBERS_COMPLETE_REQUIRES_ACCEPTANCE',index=a.index,records=records,bindings=bindings,protocol_sha256=sha(a.protocol),source_sha256=sha(__file__),projection_source_sha256=sha(Path(__file__).with_name('readout_fiber.py')),elapsed_seconds=time.monotonic()-started)
    (out/'report.json').write_text(json.dumps(result,indent=2)+'\n')

if __name__=='__main__':main()
