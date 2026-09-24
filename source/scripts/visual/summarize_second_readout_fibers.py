"""Independent local nonlinear reconstruction and fixed paired-goal analysis."""
import argparse,json
from pathlib import Path
import numpy as np
from adaptation_streams import sha
from accept_second_readout_fibers import accept
from accept_readout_calibration_evaluation import decode


def main():
    p=argparse.ArgumentParser()
    for k in ['fibers','runs','old-runs','horizon','donor-horizon','calibrated','protocol','output']:p.add_argument('--'+k,required=True)
    a=p.parse_args();draws=np.random.default_rng(1394001).integers(0,128,(20000,128));profiles=[];packed={}
    def stats(x):
        assert x.shape==(128,5) and np.isfinite(x).all();b=np.empty((20000,5))
        for i in range(0,20000,500):b[i:i+500]=x[draws[i:i+500]].mean(1)
        lo,hi=np.quantile(b,[.025,.975],axis=0)
        return {'mean':x.mean(0).tolist(),'ci95_low':lo.tolist(),'ci95_high':hi.tolist()}
    for idx in [0,1,2]:
        g=1 if idx==1 else 0;fd=Path(a.fibers)/f'job_{idx}';rd=Path(a.runs)/f'job_{idx}';fr=json.loads((fd/'report.json').read_text());r=json.loads((rd/'report.json').read_text());fa=json.loads((fd/'acceptance.json').read_text())
        assert r['protocol_sha256']==sha(a.protocol) and r['source_sha256']==sha(Path(__file__).with_name('rollout_second_readout_fibers.py'))
        assert r['fiber_report_sha256']==sha(fd/'report.json') and r['fiber_acceptance_sha256']==sha(fd/'acceptance.json') and fa['source_files_checked']
        geometry=accept(fd,a.protocol,check_source=False);head=dict(np.load(rd/'head.npz'));fh=dict(np.load(fd/'head.npz'))
        for key in head:np.testing.assert_array_equal(head[key],fh[key])
        assert sha(rd/'head.npz')==r['head_sha256'];old=Path(a.old_runs)/f'job_{g}';old_report=json.loads((old/'report.json').read_text());rows=[];maximum=0;anchors=0
        for row in r['rows']:
            obj=row['objective'];slot={'latent':2,'decoded_teacher':3,'physical_labels':4}[obj];bound=next(x for x in fr['rows'] if x['objective']==obj)
            assert sha(rd/row['file'])==row['sha256'] and row['frozen_tensors_unchanged'] and row['weights_sha256']==bound['weights_sha256'];z=dict(np.load(rd/row['file']));f=dict(np.load(fd/bound['file']))
            source=Path(a.calibrated)/'group_0'/('new_'+obj+'.npz') if idx==2 and slot!=2 else Path(a.horizon)/f'job_{g}'/f'model_{8*g+slot}.npz'
            donor=Path(a.donor_horizon)/f'job_{g}'/f'model_{8*g+slot}.npz';assert sha(source)==bound['source_sha256'] and sha(donor)==bound['donor_sha256'];source_data=dict(np.load(source));don=dict(np.load(donor))
            np.testing.assert_array_equal(f['donor'],don['observed_tokens'][:,:,0][:,f['permutation']].reshape(512,192))
            np.testing.assert_array_equal(z['free_tokens'],source_data['free_tokens']);anchors+=512
            for key in ['true_pose','observed_tokens','seeds','reference_routes']:
                np.testing.assert_array_equal(z[key],f[key]);np.testing.assert_array_equal(z[key],source_data[key])
            assert len(np.unique(z['seeds']))==128 and not set(z['seeds'])&set(don['seeds']);errors={}
            for branch in ['free','actual','donor','reset']:
                expected={'free':'predicted','actual':'actual_matched','donor':'donor_matched','reset':'observed'}[branch]
                np.testing.assert_array_equal(z[branch+'_tokens'][:,:,0].reshape(512,192),f[expected])
                pose=decode(z[branch+'_tokens'],head);np.testing.assert_allclose(pose,z[branch+'_pose'],atol=1e-8,rtol=1e-11)
                maximum=max(maximum,float(abs(pose-z[branch+'_pose']).max()));errors[branch]=np.square(pose[...,2:4]-z['true_pose'][...,2:4]).sum(-1).mean(0)
                packed[f'config{idx}__{obj}__{branch}']=errors[branch]
            contrasts={b:stats(errors['actual']-errors[b]) for b in ['free','donor']}
            result={'objective':obj,'absolute_position_mse':{b:stats(v) for b,v in errors.items()},'actual_minus':contrasts}
            if idx!=2:
                orow=next(x for x in old_report['rows'] if x['model_index']==8*g+slot);op=old/orow['file'];assert sha(op)==orow['sha256'];oz=dict(np.load(op));np.testing.assert_array_equal(oz['free_tokens'],z['free_tokens']);np.testing.assert_array_equal(oz['true_pose'],z['true_pose'])
                olderrors={b:np.square(oz[key+'_pose'][...,2:4]-oz['true_pose'][...,2:4]).sum(-1).mean(0) for b,key in [('free','free'),('actual','fiber'),('donor','shuffled')]}
                result['original_head_actual_minus']={b:stats(olderrors['actual']-olderrors[b]) for b in ['free','donor']}
                result['new_minus_old_intervention_effect']={b:stats(errors['actual']-errors[b]-(olderrors['actual']-olderrors[b])) for b in ['free','donor']}
                for b,v in olderrors.items():packed[f'old_config{idx}__{obj}__{b}']=v
            rows.append(result)
        assert anchors==r['complete_free_anchors']==1536
        profiles.append({'index':idx,'group':g,'rows':rows,'geometry':geometry,'complete_free_anchors':anchors,'maximum_decode_discrepancy':maximum,'rollout_report_sha256':sha(rd/'report.json'),'fiber_report_sha256':sha(fd/'report.json')})
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True);np.savez_compressed(out/'paired_goal_metrics.npz',**packed)
    summary={'status':'PASS_ALL_THREE_SECOND_READOUT_CONFIGURATIONS','profiles':profiles,'horizons':[5,10,15,20,25],'goals':128,'streams':4,'bootstrap':{'draws':20000,'seed':1394001,'interval':'exploratory percentile 95%','unit':'goal, after averaging four streams'},'protocol_sha256':sha(a.protocol),'source_sha256':sha(__file__),'paired_goal_metrics_sha256':sha(out/'paired_goal_metrics.npz'),'scope':'Fixed-model readout sensitivity in one Transformer and one GRU; matched-continuation sensitivity in the same Transformer. Configurations are not independent replicates; all reused goals retained.'}
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(summary['status'])
    for p in profiles:
        for row in p['rows']:print(p['index'],row['objective'],{b:[round(v[k][-1],2) for k in ['mean','ci95_low','ci95_high']] for b,v in row['actual_minus'].items()})

if __name__=='__main__':main()
