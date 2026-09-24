"""Bind auxiliary-method comparisons to matched accepted direct-state training pools."""
import hashlib,json
from pathlib import Path

root=Path(__file__).resolve().parents[2]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
am=root/'runs/hpg/action_planning_v1/scientific_manifest.json';sm=root/'runs/hpg/visual_updates_v1/scaling_manifest.json'
action=json.loads(am.read_text());scaling=json.loads(sm.read_text());rows=[]
for rep in range(3):
    for architecture in ['transformer','gru']:
        state_arm=architecture+'_state';sp=root/f'runs/visual_scaling_training_v1/replica_{rep}/n256/u21000/{state_arm}'
        state=json.loads((sp/'summary.json').read_text());sa=json.loads((sp/'artifact_manifest.json').read_text())
        assert sa['summary.json']==sha(sp/'summary.json')
        for ck in ['best','last']:
            e=next(x for x in scaling['models'] if (x['replica'],x['arm'],x['episodes'],x['updates'],x['checkpoint'])==(rep,state_arm,256,21000,ck))
            assert e['training_summary_sha256']==sha(sp/'summary.json') and e['weights_sha256']==sa[f'{ck}_weights.pt']
        source=sp/'source/train_visual_updates.py'
        assert 'batch_size=128' in source.read_text()
        for mode in ['none','inverse','inverse_goal']:
            ap=root/f'runs/action_auxiliary_full_v3/replica_{rep}/{architecture}_jepa/{mode}'
            r=json.loads((ap/'summary.json').read_text())
            assert r['batch_size']==128
            for key in ['seed','completed_updates','expected_updates','validation_every','clip_counts','data_manifest_sha256','config_sha256','normalization','schedule']:
                assert r[key]==state[key],key
            for key in ['encoder','action_encoder','temporal_core']:assert r['initial_hashes'][key]==state['initial_hashes'][key]
            for x in [r,state]:
                assert len(x['evaluations'])==100
                assert all(v['update_end']-v['update_start']==210 and v['train']['sequences']==210*128 for v in x['evaluations'])
            for ck in ['best','last']:
                e=next(x for x in action['models'] if (x['replica'],x['arm'],x['mode'],x['checkpoint'])==(rep,architecture+'_jepa',mode,ck))
                assert e['training_summary_sha256']==sha(ap/'summary.json')
            rows.append(dict(replica=rep,architecture=architecture,mode=mode,
                action_summary_sha256=sha(ap/'summary.json'),state_summary_sha256=sha(sp/'summary.json'),
                state_batch_source_sha256=sha(source)))
result=dict(rows=rows,action_manifest_sha256=sha(am),state_scaling_manifest_sha256=sha(sm),script_sha256=sha(Path(__file__)),
    scope='18 action/state training-pool pairings: matched data/config/action normalization/seed/updates/batch128/clip counts/shared initial modules; frozen summary and checkpoint identities checked. Model capacity, state supervision, recursive representation and objectives differ: comparison is not an isolated auxiliary-loss effect. Best checkpoints use different validation objectives; fixed-last is primary.')
out=root/'runs/hpg/action_planning_v1/state_pairing_audit.json';out.write_text(json.dumps(result,indent=2)+'\n');print('MATCHED18 ACTION/STATE TRAINING PAIRINGS')
