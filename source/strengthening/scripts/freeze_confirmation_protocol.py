"""Seal complete audited inputs and evaluator code; no model predictions are read."""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'strengthening/adapters'))
from contracts import sha
from input_lock import add_design_arguments,read_design,check_parent_design


def main():
    p=argparse.ArgumentParser();add_design_arguments(p);p.add_argument('--output',required=True);a=p.parse_args()
    d=read_design(a.design_lock,a.design_sha256,ROOT);base=Path(d['base']);release=base/'releases/feedback-strengthening-v1'
    assert json.loads((ROOT/'strengthening/manifests/development_model_roster.json').read_text())==d['model_roster']
    assert json.loads((ROOT/'strengthening/manifests/legacy_model_roster.json').read_text())==d['legacy_roster']
    assert json.loads((ROOT/'strengthening/configs/contrast_design.draft.json').read_text())==d['statistical_design']
    files={};rosters={}
    def bind(path,expected=None):
        path=Path(path);digest=sha(path)
        if expected is not None and digest!=expected:raise ValueError('Input binding mismatch: '+str(path))
        files[str(path)]=digest
        return digest
    for path,digest in d['verified_assets'].items():
        # Revalidate frozen weights/normalizers/environment inventories without loading predictions.
        if sha(path)!=digest:raise ValueError('Scientific asset changed: '+path)
    for role,spec in d['banks'].items():
        bank=Path(spec['output']);m=json.loads((bank/'manifest.json').read_text());r=json.loads((bank/'role.json').read_text())
        assert (bank/'DONE').exists() and len(m['cases'])==r['count']==256 and r['role']==role
        assert [c['index'] for c in m['cases']]==list(range(256)) and len({c['seed'] for c in m['cases']})==256
        check_parent_design(r,{'scientific_design_sha256':a.design_sha256})
        bind(bank/'manifest.json',r['parent_manifest_sha256']);bind(bank/'role.json')
        if role.endswith('A_C'):bind(bank/'acceptance.json',r['acceptance_sha256'])
        for c in m['cases']:bind(bank/c.get('file',f"case_{c['index']:03d}.npz"),c['sha256'])
        rosters[role]=dict(bank_manifest_sha256=sha(bank/'manifest.json'),
            cases=[dict(index=c['index'],seed=c['seed'],sha256=c['sha256']) for c in m['cases']])
        if not role.endswith('A_C'):continue
        for route in range(24):
            ap=Path(spec['actions_output'])/f'route_{route}';pp=Path(spec['physics_output'])/f'route_{route}'
            ar=json.loads((ap/'report.json').read_text());pr=json.loads((pp/'report.json').read_text())
            assert (ap/'DONE').exists() and (pp/'DONE').exists()
            assert ar['status']=='PASS_COMPACT_FIXED_REFERENCE_SEARCH' and pr['status']=='PASS_COMPLETE_SELECTED_PHYSICS'
            assert len(ar['cases'])==len(pr['cases'])==256
            assert ar['binding']['bank_manifest_sha256']==pr['binding']['parent_manifest_sha256']==sha(bank/'manifest.json')
            for report in [ar,pr]:check_parent_design(report['binding'],{'scientific_design_sha256':a.design_sha256})
            bind(ap/'report.json',pr['binding']['action_report_sha256']);bind(pp/'report.json')
            bind(ap/'binding.json',ar['binding_sha256']);bind(pp/'binding.json')
            for c,ac,pc in zip(m['cases'],ar['cases'],pr['cases'],strict=True):
                assert c['index']==ac['index']==pc['index'] and c['seed']==ac['seed']==pc['seed']
                assert ac['input_sha256']==pc['input_sha256']==c['sha256']
                assert pc['action_file_sha256']==ac['file_sha256'] and pc['all_two_replays_exact']
                bind(ap/ac['file'],ac['file_sha256']);bind(pp/pc['file'],pc['file_sha256'])
    audits={}
    migration=release/'artifacts/interface_migration_acceptance_v1/report.json'
    mr=json.loads(migration.read_text())
    assert mr['status']=='PASS_GENERIC_INTERFACE_DEVELOPMENT_PARITY' and (migration.parent/'DONE').exists()
    audits['interface_migration']=bind(migration)
    migration_sources={'AC_cache':'cache_AC_development.py','B_cache':'cache_B_development.py',
        'AC_rollout':'rollout_AC_development.py','B_rollout':'rollout_B_development.py',
        'B_fixed_insertion':'rollout_B_development.py'}
    for path,digest in mr['reports'].items():
        bind(path,digest)
        if 'interface_migration_' not in path:continue
        stage=next(k for k in migration_sources if 'interface_migration_'+k+'_v1' in path)
        row=json.loads(Path(path).read_text())
        assert row['source_sha256']==sha(ROOT/'strengthening/scripts'/migration_sources[stage])
    for branch,groups in [('AC',[0,1]),('B',[None])]:
        for g in groups:
            path=release/'artifacts'/f'interface_migration_{branch}_QP_v1'
            if g is not None:path=path/f'group_{g}'
            row=json.loads((path/'report.json').read_text())
            assert row['source_sha256']==sha(ROOT/'strengthening/scripts/project_development_feedback.py')
            bind(path/'report.json')
            code=json.loads((path/'qp/binding.json').read_text())['binding']['code']
            for name,digest in code.items():assert sha(ROOT/'strengthening/adapters'/name)==digest
    qa=release/'artifacts/native_QP_migration_audit_v1/report.json'
    bind(qa,mr['native_independent_solve_audit_sha256'])
    assert json.loads(qa.read_text())['source_sha256']==sha(ROOT/'strengthening/scripts/audit_native_QP_migration.py')
    assert mr['numerical_tolerances_unchanged']
    for name,status in [('AC_confirmation_input_lineage_v1','PASS_NEW_INPUT_PARENT_ISOLATION'),
                        ('B_confirmation_input_lineage_v1','PASS_NATIVE_CONFIRMATION_INPUT_ISOLATION')]:
        path=release/'artifacts'/name/'report.json';row=json.loads(path.read_text())
        assert row['status']==status and row['scientific_design_sha256']==a.design_sha256 and (path.parent/'DONE').exists()
        audits[name]=bind(path)
        if name.startswith('AC'):
            assert row['bank_manifest_sha256']['new_recipient']==rosters['confirmation_A_C']['bank_manifest_sha256']
            assert row['bank_manifest_sha256']['new_donor']==rosters['donor_bank_A_C']['bank_manifest_sha256']
        else:
            for role in ['confirmation_B','donor_bank_B']:
                assert row['bindings'][role]['bank_manifest_sha256']==rosters[role]['bank_manifest_sha256']
            bind(path.parent/'input_frames.json',row['input_frames_sha256'])
    partitions={}
    for branch,groups in [('AC',range(6)),('B',range(1))]:
        rows=[]
        for g in groups:
            for shard,start in enumerate(range(0,256,16)):
                rows.append(dict(task_index=len(rows),group=g,shard=shard,cases=list(range(start,start+16))))
        partitions[branch]=rows
    outputs={}
    for branch in ['AC','B']:
        for stage in ['cache','donor_cache','QP','rollout']:
            key=branch+'_'+stage;outputs[key]=str(release/'artifacts'/f'{branch}_confirmation_{stage}_v1')
            if Path(outputs[key]).exists():raise ValueError('New evaluation output exists before final protocol lock')
    sources=set((ROOT/'scripts/visual').glob('*.py'))
    for folder in ['adapters','scripts']:
        sources.update((ROOT/'strengthening'/folder).glob('*.py'))
    sources.update((ROOT/'strengthening/external/dino_wm').rglob('*.py'))
    sources.update(ROOT/'strengthening/manifests'/n for n in ['development_model_roster.json','legacy_model_roster.json'])
    protocol=dict(status='LOCKED_CONFIRMATION_READY',schema_version=1,created_utc=datetime.now(timezone.utc).isoformat(),
        protocol_path=str(Path(a.output).resolve()),
        design_path=str(Path(a.design_lock).resolve()),design_sha256=a.design_sha256,input_files=files,input_rosters=rosters,
        isolation_audits=audits,partitions=partitions,outputs=outputs,
        cli_assets=dict(base=str(base),assets=str(release/'assets/native-dinowm'),
            environment=str(release/'assets/native-environment/environment.json'),head_cache=str(release/'artifacts/native_observed_cache_v1')),
        evaluation_sources={str(f.relative_to(ROOT)):sha(f) for f in sorted(sources)},
        continuation_policy='Only complete identical-binding family recovery; all assigned goals retained. Numerical failure stops affected analysis. No outcome-based budget or estimand revision.',
        scientific_design='Exactly the prior immutable design; complete negative and mixed results are valid.',
        confirmation_study_evaluation_outputs_read=False,
        input_construction_note='Original reference-policy predictions are used by the fixed action-search rule only; no new study-model outcome is used to select goals, models or settings.')
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('x') as f:
        json.dump(protocol,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    print(json.dumps(dict(status=protocol['status'],sha256=sha(out),bound_input_files=len(files),partitions={k:len(v) for k,v in partitions.items()})),flush=True)


if __name__=='__main__':main()
