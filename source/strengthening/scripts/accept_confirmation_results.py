"""Complete independent NumPy reconstruction before any main-effect summaries."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'strengthening/adapters'))
from contracts import sha,atomic_json
from artifact_io import atomic_npz
from evaluation_lock import add_protocol_arguments,read_protocol
from input_lock import verify_environment
from verifier import verify_token
from confirmation_results import physical_pose,block_error,CompleteCells,expected_ac_members,expected_ac_families


def main():
    p=argparse.ArgumentParser();add_protocol_arguments(p);p.add_argument('--output',required=True);a=p.parse_args()
    protocol,design=read_protocol(a.protocol_lock,a.protocol_sha256,ROOT);verify_environment(design,'fiber-projection-v1')
    out=Path(a.output)
    if out.exists():raise ValueError('Independent acceptance output already exists')
    bindings={};cells=CompleteCells();diagnostics=[];native_errors={k:np.full((256,5),np.nan) for k in ['free','actual','donor','observed','reset']}
    def read_report(path):
        row=json.loads(path.read_text())
        assert (path.parent/'DONE').exists() and row['protocol_sha256']==a.protocol_sha256
        assert row['scientific_design_sha256']==protocol['design_sha256'];bindings[str(path)]=sha(path)
        return row
    def arrays(path,digest):
        assert sha(path)==digest
        return dict(np.load(path))
    def head(path,digest):return {k:np.asarray(v,dtype=np.float64) for k,v in arrays(path,digest).items()}
    def pose_check(tokens,h,saved=None):
        pose=physical_pose(tokens,h)
        if saved is not None:np.testing.assert_allclose(pose,saved,rtol=1e-10,atol=1e-7)
        assert np.isfinite(pose).all()
        return pose
    def input_truth(path,digest):
        assert sha(path)==digest
        with np.load(path) as z:st=z['states'][[15,20,25,30,35]]
        return np.c_[st[:,:4],np.sin(st[:,4]),np.cos(st[:,4])]
    def matching_summary(m):
        return {k:m[k] for k in ['family_size','native_norms','unshrunk_common_norm',
            'effective_norm','shrink_factor','legitimate_zero_norm']}
    def check_matching(row,qarray,originals,heads,reference):
        norms=np.linalg.norm(qarray['full_standardized_directions'],axis=1)
        m=row['matching'];np.testing.assert_allclose(norms,m['native_norms'],rtol=1e-12,atol=1e-12)
        np.testing.assert_allclose(m['unshrunk_common_norm'],float(norms.min()),rtol=1e-12,atol=1e-12)
        assert m['shrink_factor'] in design['projection']['common_shrink_factors']
        assert m['effective_norm']==m['shrink_factor']*m['unshrunk_common_norm']
        assert m['legitimate_zero_norm']==(float(norms.min())==0)
        for j,(original,hs) in enumerate(zip(originals,heads,strict=True)):
            alpha=0. if norms[j]==0 else m['effective_norm']/m['native_norms'][j]
            reconstructed=(np.asarray(original,dtype=np.float64)+alpha*qarray['full_standardized_directions'][j]*reference['scale']).astype(np.float32)
            np.testing.assert_array_equal(reconstructed,qarray['corrected'][j])
            assert verify_token(original,qarray['corrected'][j],hs,reference,expected_norm=m['effective_norm'])['accepted']

    for g in range(6):
        cache=Path(protocol['outputs']['AC_cache'])/f'group_{g}';cr=read_report(cache/'report.json')
        hA=head(cache/'head_A.npz',cr['head_sha256']['head_A']);hB=head(cache/'head_B.npz',cr['head_sha256']['head_B'])
        obs=arrays(cache/'observed.npz',cr['observed_sha256']);baselines={}
        assert cr['expected_cases']==256 and cr['group']==g
        np.testing.assert_array_equal(obs['seeds'],[c['seed'] for c in protocol['input_rosters']['confirmation_A_C']['cases']])
        physics=Path(design['banks']['confirmation_A_C']['physics_output'])
        for s in range(4):
            folder=physics/f'route_{4*g+s}';report=json.loads((folder/'report.json').read_text())
            assert sha(folder/'report.json')==protocol['input_files'][str(folder/'report.json')]
            for c in report['cases']:
                np.testing.assert_array_equal(obs['truth'][s,c['index']],input_truth(folder/c['file'],c['file_sha256']))
        op=pose_check(obs['observed_tokens'],hA,obs['observed_pose'])
        for mr in cr['models']:
            m=mr['model'];z=arrays(cache/mr['file'],mr['file_sha256']);baselines[m['name']]=z
            fp=pose_check(z['free_tokens'],hA,z['free_pose']);tp=pose_check(z['observed_history_tokens'],hA,z['observed_history_pose'])
            np.testing.assert_allclose(block_error(fp,obs['truth']),z['free_block_error'],rtol=1e-10,atol=1e-6)
            np.testing.assert_allclose(block_error(tp,obs['truth']),z['observed_history_block_error'],rtol=1e-10,atol=1e-6)
            if m['kind']=='C':
                prefix='C/'+m['objective']+'/'+m['condition']
                for i in range(256):
                    for s in range(4):
                        for label,value in [('observed_history',block_error(tp[s,i],obs['truth'][s,i])),
                                            ('observed_encoding',block_error(op[s,i],obs['truth'][s,i]))]:
                            cells.put(prefix+'/'+label,i,g//2,s,value,3)
                response=mr['action_response'];assert len(response)==256*4
                assert {(r['case'],r['stream']) for r in response}=={(i,s) for i in range(256) for s in range(4)}
                for r in response:cells.put(prefix+'/action_response',r['case'],g//2,r['stream'],r['mean_square'],3)
                # Across-goal variance is descriptive. Retain all standardized latent dimensions.
                diagnostics.append(dict(kind='C_output_variance',group=g,model=m['name'],
                    free_latent_variance_per_horizon=np.var((z['free_tokens'].astype(np.float64)-hA['mean'])/hA['scale'],axis=1).mean(axis=(0,2)).tolist(),
                    observed_latent_variance_per_horizon=np.var((obs['observed_tokens'].astype(np.float64)-hA['mean'])/hA['scale'],axis=1).mean(axis=(0,2)).tolist(),
                    free_pose_variance_per_horizon_and_output=np.var(fp,axis=1).mean(axis=0).tolist(),
                    observed_pose_variance_per_horizon_and_output=np.var(op,axis=1).mean(axis=0).tolist()))
        for part in [r for r in protocol['partitions']['AC'] if r['group']==g]:
            shard=part['shard'];rp=Path(protocol['outputs']['AC_rollout'])/f'group_{g}/shard_{shard:03d}'
            qp=Path(protocol['outputs']['AC_QP'])/f'group_{g}/shard_{shard:03d}'
            rr=read_report(rp/'report.json');qr=read_report(qp/'report.json');qd=json.loads((qp/'design.json').read_text())
            assert rr['cases']==qr['cases']==part['cases'] and rr['group']==g
            assert rr['cache_report_sha256']==sha(cache/'report.json') and rr['projection_report_sha256']==sha(qp/'report.json')
            assert qr['design_sha256']==sha(qp/'design.json') and qr['qp_report_sha256']==sha(qp/'qp/report.json')
            qreport=json.loads((qp/'qp/report.json').read_text());assert (qp/'qp/DONE').exists() and not (qp/'qp/FAILURE.json').exists()
            assert qd['assignments']=={str(s):design['donor_assignments'][f'AC/{g}/{s}']['donor_indices'] for s in range(4)}
            expected=expected_ac_families(g,part['cases'])
            assert [(f['descriptor']['stream'],f['descriptor']['case'],f['descriptor']['kind']) for f in rr['families']]==expected
            assert [f['descriptor'] for f in rr['families']]==qd['families']
            assert qreport['accepted_families']==qreport['expected_families']==len(expected)
            for fi,row in enumerate(rr['families']):
                f=row['descriptor'];s=f['stream'];i=f['case'];kind=f['kind'];members=expected_ac_members(kind,g)
                assert f['member_ids']==members
                qjson=qp/f'qp/family_{fi:04d}.json';assert sha(qjson)==qreport['family_report_sha256'][fi]
                qrow=json.loads(qjson.read_text());qa=arrays(qp/f'qp/family_{fi:04d}.npz',qrow['arrays_sha256'])
                z=arrays(rp/row['file'],row['file_sha256']);assert row['matching']==qrow['matching'] and qrow['member_ids']==members
                np.testing.assert_array_equal(z['truth'],obs['truth'][s,i]);np.testing.assert_array_equal(z['corrected'][:,0],qa['corrected'])
                hs=[([hA,hB] if m.startswith('dual_A_B/') else [hA]) for m in members]
                originals=[baselines[m.split('/')[1]]['free_tokens'][s,i,0] for m in members]
                check_matching(qrow,qa,originals,hs,hA)
                diagnostics.append(dict(kind=kind,group=g,goal=i,stream=s,matching=matching_summary(row['matching'])))
                for label in ['corrected','free','reset']+(['full_feasible'] if kind.startswith('C_') else []):
                    pa=pose_check(z[label],hA,z[label+'_pose_A']);pb=pose_check(z[label],hB,z[label+'_pose_B'])
                    error=block_error(pa,z['truth']);np.testing.assert_allclose(error,z[label+'_block_error_A'],rtol=1e-10,atol=1e-6)
                    for j,member in enumerate(members):
                        constraint,model,source=member.split('/')
                        if label=='free':np.testing.assert_array_equal(z[label][j],baselines[model]['free_tokens'][s,i])
                        if label=='reset':np.testing.assert_array_equal(z[label][j,0],obs['observed_tokens'][s,i,0])
                        if kind.startswith('C_'):
                            objective=kind.removeprefix('C_');condition=model.rsplit('_',1)[1];prefix=f'C/{objective}/{condition}'
                            group=g//2;groups=3
                        else:
                            objective=model.removeprefix('A_');prefix=kind+'/'+objective;group=g;groups=6 if kind=='A_core' else 2
                        branch=source if label=='corrected' else (label+'_'+source if label=='full_feasible' else label)
                        if kind=='A_dual' and label=='corrected':branch=('dual_' if constraint=='dual_A_B' else 'single_')+source
                        cells.put(prefix+'/'+branch,i,group,s,error[j],groups)
                        cells.put(prefix+'/'+branch+'/head_B',i,group,s,block_error(pb[j],z['truth']),groups)
                        if label=='full_feasible':
                            fm=row['checks'][member]['full_feasible_matching']
                            assert verify_token(originals[j],z[label][j,0],[hA],hA,expected_norm=fm['effective_norm'])['accepted']
                            reconstructed=(originals[j].astype(np.float64)+fm['shrink_factor']*qa['full_standardized_directions'][j]*hA['scale']).astype(np.float32)
                            np.testing.assert_array_equal(z[label][j,0],reconstructed)
                            cells.put(prefix+'/full_dose_'+source,i,group,s,fm['effective_norm'],groups)
                        if kind.startswith('C_'):cells.put(prefix+'/matched_dose',i,group,s,row['matching']['effective_norm'],groups)
    cells.validate()
    bcache=Path(protocol['outputs']['B_cache']);bcr=read_report(bcache/'report.json')
    bh=head(Path(design['native_head']),bcr['head_sha256']);bmetrics=arrays(bcache/'baseline_metrics.npz',bcr['baseline_metrics_sha256'])
    bank=Path(design['banks']['confirmation_B']['output']);bm=json.loads((bank/'manifest.json').read_text())
    for part in protocol['partitions']['B']:
        shard=part['shard'];rp=Path(protocol['outputs']['B_rollout'])/f'shard_{shard:03d}';qp=Path(protocol['outputs']['B_QP'])/f'shard_{shard:03d}'
        rr=read_report(rp/'report.json');qr=read_report(qp/'report.json');qd=json.loads((qp/'design.json').read_text())
        assert rr['cases']==qr['cases']==part['cases'] and [r['case'] for r in rr['rows']]==part['cases']
        assert rr['projection_report_sha256']==sha(qp/'report.json') and rr['cache_report_sha256']==sha(bcache/'report.json')
        assert qr['design_sha256']==sha(qp/'design.json') and qr['qp_report_sha256']==sha(qp/'qp/report.json')
        qreport=json.loads((qp/'qp/report.json').read_text());assert (qp/'qp/DONE').exists() and not (qp/'qp/FAILURE.json').exists()
        assert qd['assignments']=={'0':design['donor_assignments']['B/0/0']['donor_indices']}
        assert qreport['accepted_families']==qreport['expected_families']==16
        for fi,row in enumerate(rr['rows']):
            i=row['case'];z=arrays(rp/row['file'],row['file_sha256']);br=bcr['rows'][i];base=arrays(bcache/br['file'],br['file_sha256'])
            assert i==br['index']==bm['cases'][i]['index'] and row['seed']==br['seed']==bm['cases'][i]['seed']
            np.testing.assert_array_equal(z['truth'],input_truth(bank/bm['cases'][i]['file'],bm['cases'][i]['sha256']))
            np.testing.assert_array_equal(z['truth'],base['truth']);np.testing.assert_array_equal(z['free_tokens'],base['free_tokens'][3:])
            np.testing.assert_array_equal(z['truth'],bmetrics['truth'][i])
            qjson=qp/f'qp/family_{fi:04d}.json';assert sha(qjson)==qreport['family_report_sha256'][fi]
            qrow=json.loads(qjson.read_text());qa=arrays(qp/f'qp/family_{fi:04d}.npz',qrow['arrays_sha256'])
            assert qd['families'][fi]['case']==i and qrow['member_ids']==['single_A/native/actual','single_A/native/donor']
            token=base['free_tokens'][3,:,:384].reshape(75264);check_matching(qrow,qa,[token]*2,[[bh]]*2,bh)
            for j,source in enumerate(['actual','donor']):
                tokens=z[source+'_tokens'];np.testing.assert_array_equal(tokens[0,:,:384].reshape(75264),qa['corrected'][j])
                np.testing.assert_array_equal(tokens[0,:,384:],z['free_tokens'][0,:,384:])
                np.testing.assert_array_equal(tokens[:4,:,394:],z['free_tokens'][:4,:,394:])
                pose=pose_check(tokens[...,:384].reshape(5,75264),bh,z[source+'_pose']);err=block_error(pose,z['truth'])
                np.testing.assert_allclose(err,z[source+'_block_error'],rtol=1e-10,atol=1e-6);native_errors[source][i]=err
            for label,tokens in [('free',z['free_tokens'][...,:384].reshape(5,75264)),('observed',base['observed_visual']),
                                 ('reset',base['full_visual_reset_tokens'][...,:384].reshape(5,75264))]:
                pose=pose_check(tokens,bh,bmetrics[label+'_pose'][i] if label in ['free','observed'] else None)
                native_errors[label][i]=block_error(pose,z['truth'])
            diagnostics.append(dict(kind='B_core',goal=i,matching=matching_summary(qrow['matching'])))
    assert all(np.isfinite(x).all() for x in native_errors.values())
    out.mkdir(parents=True,exist_ok=False)
    atomic_npz(out/'AC_goal_cells.npz',**cells.arrays)
    atomic_npz(out/'B_goal_cells.npz',**native_errors)
    atomic_json(out/'diagnostics.json',dict(rows=diagnostics))
    atomic_json(out/'report.json',dict(status='PASS_COMPLETE_INDEPENDENT_CONFIRMATION_RECONSTRUCTION',
        protocol_sha256=a.protocol_sha256,scientific_design_sha256=protocol['design_sha256'],
        stage_reports=bindings,AC_goal_cells_sha256=sha(out/'AC_goal_cells.npz'),B_goal_cells_sha256=sha(out/'B_goal_cells.npz'),
        diagnostics_sha256=sha(out/'diagnostics.json'),source_sha256=sha(__file__),goals_per_bank=256,
        all_primary_and_secondary_cells_complete=True,main_effects_computed=False))
    (out/'DONE').write_text('complete_independent_acceptance\n')


if __name__=='__main__':main()
