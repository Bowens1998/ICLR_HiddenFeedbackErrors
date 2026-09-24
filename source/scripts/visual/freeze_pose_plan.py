"""Freeze all PushT scoring interfaces with matched direct-state controls."""
import argparse,hashlib,json
from pathlib import Path


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser()
    for k in ['action-manifest','state-manifest','training','state-training','heads','features','score-gate','output']:p.add_argument('--'+k,required=True)
    a=p.parse_args();am=json.loads(Path(a.action_manifest).read_text());sm=json.loads(Path(a.state_manifest).read_text())
    assert am['layout']=='action_auxiliary' and sm['layout']=='scaling'
    models=[];routes=[];bindings=[]
    for index in range(6):
        rep,arch=index//2,['transformer','gru'][index%2]
        selected=[e for e in am['models'] if (e['replica'],e['arm'],e['mode'],e['checkpoint'])==(rep,arch+'_jepa','none','last')]
        assert len(selected)==1;base=dict(selected[0]);base['training_path']=str((Path(a.training)/base['training_path']).resolve())
        selected=[e for e in sm['models'] if (e['replica'],e['arm'],e['episodes'],e['updates'],e['checkpoint'])==(rep,arch+'_state',256,21000,'last')]
        assert len(selected)==1;state=dict(selected[0]);state['training_path']=str((Path(a.state_training)/state['training_path']).resolve());state['mode']='direct_state'
        reports=[]
        for entry in [base,state]:
            folder=Path(entry['training_path']);assert sha(folder/'summary.json')==entry['training_summary_sha256']
            assert sha(folder/'last_weights.pt')==entry['weights_sha256']
            reports.append(json.loads((folder/'summary.json').read_text()))
        for key in ['seed','completed_updates','expected_updates','validation_every','clip_counts','data_manifest_sha256','config_sha256','normalization','schedule']:
            assert reports[0][key]==reports[1][key],key
        for key in ['encoder','action_encoder','temporal_core']:
            assert reports[0]['initial_hashes'][key]==reports[1]['initial_hashes'][key],key
        headroot=Path(a.heads)/f'job_{index}';features=Path(a.features)/f'job_{index}'
        hr=json.loads((headroot/'report.json').read_text());ha=json.loads((headroot/'acceptance.json').read_text());fr=json.loads((features/'report.json').read_text())
        assert hr['index']==index and not hr['engineering'] and not ha['engineering'] and ha['status']=='PASS'
        assert ha['report_sha256']==sha(headroot/'report.json') and ha['verifier_sha256']==sha(Path(__file__).with_name('accept_pose_readouts.py'))
        assert hr['input_report_sha256']==sha(features/'report.json') and hr['input_acceptance_sha256']==sha(features/'acceptance.json')
        assert fr['weights_sha256']==base['weights_sha256'] and fr['manifest_sha256']==sha(a.action_manifest)
        gatepath=Path(a.score_gate)/f'job_{index}'/'acceptance.json';gate=json.loads(gatepath.read_text())
        assert gate['status']=='PASS' and gate['index']==index and len(gate['rows'])==4
        assert gate['weights_sha256']==base['weights_sha256'] and gate['head_report_sha256']==sha(headroot/'report.json')
        assert gate['head_acceptance_sha256']==sha(headroot/'acceptance.json') and gate['bank_manifest_sha256']==am['bank_manifest_sha256']
        for name,h in gate['source_sha256'].items():assert sha(Path(__file__).with_name(name))==h
        heads={}
        for row in hr['rows']:
            assert row['updates']==2000 and row['batch_size']==256
            f=headroot/row['role']/'weights.npz';assert sha(f)==row['files_sha256']['weights.npz']
            heads[row['role']]=dict(path=str(f.resolve()),sha256=sha(f))
        assert set(heads)=={'encoded','predicted'}
        for score in ['latent','pose_encoded','pose_predicted','state']:
            entry=dict(state if score=='state' else base);entry['score']=score;entry['backbone_index']=index
            entry['target_normalization']=reports[1]['target_normalization'] if score=='state' else None
            if score.startswith('pose_'):
                entry['endpoint_head']=heads[score.removeprefix('pose_')];entry['goal_head']=heads['encoded']
            mi=len(models);models.append(entry)
            for algorithm in ['random','cem']:routes.append(dict(model_index=mi,algorithm=algorithm,parameterization='full'))
        bindings.append(dict(index=index,score_gate_sha256=sha(gatepath),head_report_sha256=sha(headroot/'report.json'),head_acceptance_sha256=sha(headroot/'acceptance.json')))
    assert len(models)==24 and len(routes)==48
    result=dict(layout='pusht_nonlinear_pose',models=models,routes=routes,bindings=bindings,bank_manifest_sha256=am['bank_manifest_sha256'],
                action_manifest_sha256=sha(a.action_manifest),state_manifest_sha256=sha(a.state_manifest),freezer_sha256=sha(__file__),
                scope='Complete fixed-last 48-route development comparison, same128 previously examined goals; fresh-bank confirmation still needed. State/readout labels are additional supervision.')
    with Path(a.output).open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print('FROZEN_24_INTERFACES_48_ROUTES')


if __name__=='__main__':main()
