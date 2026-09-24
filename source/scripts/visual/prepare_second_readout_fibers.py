"""Project the preregistered three second-readout configurations."""
import argparse,json,time
from pathlib import Path
import numpy as np
from adaptation_streams import sha
from readout_fiber import project,readout
from accept_readout_fiber_rollout_inputs import forward


def main():
    p=argparse.ArgumentParser()
    for k in ['horizon','donor-horizon','pilot','evaluation','protocol','output']:p.add_argument('--'+k,required=True)
    p.add_argument('--index',type=int,choices=[0,1,2],required=True);a=p.parse_args();idx=a.index;g=1 if idx==1 else 0
    out=Path(a.output)/f'job_{idx}';out.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    hd=Path(a.horizon)/f'job_{g}';dd=Path(a.donor_horizon)/f'job_{g}';pd=Path(a.pilot)/f'group_{g}'
    hr=json.loads((hd/'report.json').read_text());dr=json.loads((dd/'report.json').read_text());pr=json.loads((pd/'report.json').read_text())
    for base,report in [(hd,hr),(dd,dr),(pd,pr)]:
        ac=json.loads((base/'acceptance.json').read_text());assert ac['report_sha256']==sha(base/'report.json')
    head=dict(np.load(pd/'fitted_head.npz'));assert sha(pd/'fitted_head.npz')==pr['fitted_head_sha256']
    np.savez_compressed(out/'head.npz',**head)
    permutation=np.random.default_rng(1368001).permutation(128);entries=[];allnorms=[]
    for slot,obj in [(2,'latent'),(3,'decoded_teacher'),(4,'physical_labels')]:
        row=next(x for x in hr['rows'] if x['model_index']==8*g+slot);source=hd/row['file'];weights=row['entry']['weights_sha256'];weight_path=str(Path(row['entry']['training_path'])/'last_weights.pt')
        assert sha(source)==row['sha256']
        if idx==2 and slot!=2:
            er=json.loads((Path(a.evaluation)/'group_0/report.json').read_text());erow=next(x for x in er['rows'] if x['head']=='new' and x['objective']==obj)
            assert er['pilot_report_sha256']==sha(pd/'report.json');source=Path(a.evaluation)/'group_0'/erow['file'];assert sha(source)==erow['sha256'];weights=erow['weights_sha256'];weight_path=str(pd/obj/'last_weights.pt')
        assert sha(weight_path)==weights
        z=dict(np.load(source));drow=next(x for x in dr['rows'] if x['model_index']==8*g+slot);dp=dd/drow['file'];assert sha(dp)==drow['sha256'];dz=dict(np.load(dp));assert not set(z['seeds'])&set(dz['seeds'])
        pred=z['free_tokens'][:,:,0].reshape(512,192);obs=z['observed_tokens'][:,:,0].reshape(512,192);don=dz['observed_tokens'][:,:,0][:,permutation].reshape(512,192)
        arrays={'predicted':pred,'observed':obs,'donor':don,'free_baseline_tokens':z['free_tokens'],'observed_tokens':z['observed_tokens'],'true_pose':z['true_pose'],'seeds':z['seeds'],'reference_routes':z['reference_routes'],'permutation':permutation}
        v0,a0,b0=forward(head,pred)
        for branch,target in [('actual',obs),('donor',don)]:
            corrected=[]
            with (out/f'{obj}_{branch}_solver.jsonl').open('x') as log:
                for j,(x,y) in enumerate(zip(pred,target)):
                    value,_,solver=project(head,x,y);log.write(json.dumps({'index':j,**solver})+'\n');log.flush();assert solver['status']=='solved' and value is not None;corrected.append(value)
            corrected=np.stack(corrected);v1,a1,b1=forward(head,corrected)
            assert np.max(abs(v1-v0))<=1e-6
            assert max(0.,np.max(-np.where(a0>=0,1.,-1.)*a1),np.max(-np.where(b0>=0,1.,-1.)*b1))<=1e-6
            arrays[branch+'_full']=corrected;norm=np.linalg.norm((corrected.astype(float)-pred)/head['scale'],axis=-1);arrays[branch+'_full_norm']=norm;allnorms.append(norm)
        entries.append((obj,row,source,dp,weights,weight_path,arrays));print('PROJECTED',idx,obj,flush=True)
    common=np.minimum.reduce(allnorms);rows=[]
    for obj,row,source,dp,weights,weight_path,z in entries:
        z['target_norm']=common
        for branch in ['actual','donor']:
            norm=z[branch+'_full_norm'];alpha=np.divide(common,norm,out=np.zeros_like(common),where=norm>0)
            value=(z['predicted'].astype(float)+alpha[:,None]*(z[branch+'_full'].astype(float)-z['predicted'])).astype(np.float32)
            np.testing.assert_allclose(np.linalg.norm((value.astype(float)-z['predicted'])/head['scale'],axis=-1),common,atol=1e-6,rtol=1e-6)
            assert np.max(abs(readout(head,value)-readout(head,z['predicted'])))<=1e-6
            z[branch+'_matched']=value;z[branch+'_alpha']=alpha
        name=obj+'.npz';np.savez_compressed(out/name,**z)
        rows.append({'objective':obj,'model_index':row['model_index'],'entry':row['entry'],'weights_path':weight_path,'weights_sha256':weights,'source_file':str(source),'source_sha256':sha(source),'donor_file':str(dp),'donor_sha256':sha(dp),'file':name,'sha256':sha(out/name),'solver_sha256':{b:sha(out/f'{obj}_{b}_solver.jsonl') for b in ['actual','donor']}})
    report={'status':'SECOND_HEAD_PROJECTIONS_REQUIRE_ACCEPTANCE','index':idx,'group':g,'rows':rows,'protocol_sha256':sha(a.protocol),'source_sha256':sha(__file__),'projection_source_sha256':sha(Path(__file__).with_name('readout_fiber.py')),'head_sha256':sha(out/'head.npz'),'pilot_report_sha256':sha(pd/'report.json'),'calibration_gate_passed':pr['head_gate_passed'],'mean_target_norm':float(common.mean()),'zero_norm_cases':int((common==0).sum()),'elapsed_seconds':time.monotonic()-start}
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':main()
