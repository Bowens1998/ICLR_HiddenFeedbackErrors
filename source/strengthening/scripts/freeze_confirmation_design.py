"""Verify existing assets and freeze the scientific design before new input generation."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'strengthening/adapters'))
from contracts import sha, namespace_seed
from artifact_io import array_sha


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--base', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    base = Path(a.base); release = base / 'releases/feedback-strengthening-v1'
    st = ROOT / 'strengthening'; assets = {}

    def bind(path, expected=None):
        path = Path(path); digest = sha(path)
        if expected is not None and digest != expected:
            raise ValueError('Asset mismatch: ' + str(path))
        assets[str(path)] = digest
        return digest

    roster = json.loads((st/'manifests/development_model_roster.json').read_text())
    legacy = json.loads((st/'manifests/legacy_model_roster.json').read_text())
    assert len(roster['A_groups']) == 6 and len(roster['C_models']) == 18
    normalizers = {}
    for group in roster['A_groups']:
        assert len(group['models']) == 3
        for name in ['head_A', 'head_B']:
            h = group[name]; bind(h['path'], h['sha256'])
            z = dict(np.load(h['path']))
            normalizers[f"group{group['group']}/{name}"] = {
                k: dict(shape=list(z[k].shape), dtype=str(z[k].dtype), array_sha256=array_sha(z[k]))
                for k in ['mean','scale','target_mean','target_scale']}
        for model in group['models']:
            bind(model['checkpoint'], model['checkpoint_sha256'])
            if 'report_sha256' in model:
                bind(Path(model['checkpoint']).parent/'report.json', model['report_sha256'])
        original = legacy['groups'][group['group']]['models'][0]
        bind(original['checkpoint'], original['checkpoint_sha256'])
        summary = Path(original['checkpoint']).parent/'summary.json'; bind(summary)
        info = json.loads(summary.read_text())
        normalizers[f"group{group['group']}/actions"] = info['normalization']
        bind(base/'assets/pusht-v1/models/config.json', info['config_sha256'])
    for model in roster['C_models']:
        bind(model['checkpoint'], model['checkpoint_sha256'])
        bind(Path(model['checkpoint']).parent/'report.json', model['report_sha256'])
    policies = json.loads((st/'configs/AC_reference_policies.json').read_text())
    for row in policies['rows']:
        e = row['entry']; td = Path(e['training_path'])
        bind(td/'last_weights.pt', e['weights_sha256']); bind(td/'summary.json', e['training_summary_sha256'])
        for key in ['endpoint_head','goal_head']:
            if key in e: bind(e[key]['path'], e[key]['sha256'])

    native_report = release/'artifacts/B_development_cache_v1/report.json'; bind(native_report)
    nr = json.loads(native_report.read_text()); native = nr['model_binding']
    ckpt = release/'assets/native-dinowm/outputs/outputs/pusht/checkpoints/model_latest.pth'
    bind(ckpt, native['checkpoint_sha256']); bind(ckpt.parent.parent/'hydra.yaml', native['config_sha256'])
    envpath = release/'assets/native-environment/environment.json'; bind(envpath, native['environment_sha256'])
    env = json.loads(envpath.read_text()); bind(env['backbone_weight'], native['encoder_weight_sha256'])
    commit = subprocess.check_output(['git','-C',env['backbone_source'],'rev-parse','HEAD'],text=True).strip()
    assert commit == native['encoder_source_commit']
    native_head = release/'artifacts/native_readout_v1/weights.npz'; bind(native_head, nr['head_sha256'])
    bind(release/'assets/native-simulator-environment-v2/report.json')
    bind(native_head.parent/'acceptance.json', nr['head_acceptance_sha256'])
    bind(native_head.parent/'report.json')
    nh = dict(np.load(native_head)); normalizers['native_visual_head'] = {
        k: dict(shape=list(nh[k].shape), dtype=str(nh[k].dtype), array_sha256=array_sha(nh[k]))
        for k in ['mean','scale','target_mean','target_scale']}

    inventory_path = release/'assets/evaluation-environments-v1/manifest.json'; bind(inventory_path)
    inventory = json.loads(inventory_path.read_text())
    for row in inventory['environments']:
        exe = row['metadata']['executable']; bind(exe, row['python_binary_sha256'])
        frozen = subprocess.check_output([exe,'-m','pip','freeze','--all'])
        assert hashlib.sha256(frozen).hexdigest() == row['freeze_sha256']
        bind(inventory_path.parent/row['freeze_file'], row['freeze_sha256'])

    banks_draft = json.loads((st/'configs/bank_design.draft.json').read_text())
    roles = ['confirmation_A_C','donor_bank_A_C','confirmation_B','donor_bank_B']
    banks = {r:banks_draft['banks'][r] for r in roles}
    for role, spec in banks.items():
        assert spec['count'] == 256 and spec['max_seeds'] == 8192
        spec['output'] = str(release/'artifacts'/(role+'_v1'))
        if role.endswith('A_C'):
            spec['actions_output'] = str(release/'artifacts'/(role+'_actions_v1'))
            spec['physics_output'] = str(release/'artifacts'/(role+'_physics_v1'))
        if Path(spec['output']).exists():
            raise ValueError('Confirmation inputs already exist before design freeze: ' + role)
    assignments = {}
    for g in range(6):
        for s in range(4):
            ns = f'AC_confirmation_donor/group_{g}/stream_{s}'
            assignments[f'AC/{g}/{s}'] = dict(namespace=ns,
                donor_indices=np.random.default_rng(namespace_seed(20260919,ns)).permutation(256).tolist())
    ns = 'B_confirmation_donor'
    assignments['B/0/0'] = dict(namespace=ns,
        donor_indices=np.random.default_rng(namespace_seed(20260919,ns)).permutation(256).tolist())

    sources = set((ROOT/'scripts/visual').glob('*.py'))
    sources.update(st/'scripts'/n for n in ['prepare_AC_development.py','prepare_B_development.py',
        'plan_AC_development.py','replay_AC_development.py','freeze_confirmation_design.py'])
    sources.update(st/'adapters'/n for n in ['contracts.py','input_lock.py','artifact_io.py','native_simulator.py'])
    sources.update(st/'configs'/n for n in ['bank_design.draft.json','AC_reference_policies.json','B_bank_generation.json'])
    sources.add(st/'manifests/split_lineage.json')
    sources.update((st/'external/dino_wm/env/pusht').rglob('*.py'))
    external = {}
    for folder in [base/'releases/visual-v1/stable-worldmodel/stable_worldmodel',base/'releases/visual-v1/official']:
        for file in folder.rglob('*.py'): external[str(file)] = sha(file)
    evidence = {}
    for name in ['AC_development_lineage_v1','B_development_lineage_v1','native_QP_profile_v1']:
        path = release/'artifacts'/name/'report.json'; evidence[str(path)] = bind(path)
    for g in range(6):
        for name in ['AC_development_rollout_v1','AC_development_cache_v1']:
            path = release/'artifacts'/name/f'group_{g}/report.json'; evidence[str(path)] = bind(path)
    for name in ['B_development_rollout_v1','B_development_QP_v1']:
        path = release/'artifacts'/name/'report.json'; evidence[str(path)] = bind(path)

    contract = json.loads((st/'configs/development_contract.json').read_text())
    contrasts = json.loads((st/'configs/contrast_design.draft.json').read_text())
    assert len(contrasts['primary']) == 9 and len(contrasts['secondary_C']) == 8
    lambda_lock = json.loads((st/'configs/C_lambda.lock.json').read_text())
    design = dict(status='FROZEN_SCIENTIFIC_DESIGN_INPUT_CONSTRUCTION_ONLY',schema_version=1,
        created_utc=datetime.now(timezone.utc).isoformat(), confirmation_evaluation_authorized=False,
        root_seed=20260919,base=str(base),banks=banks,model_roster=roster,legacy_roster=legacy,
        native_model=native,native_checkpoint=str(ckpt),native_head=str(native_head),
        normalizer_bindings=normalizers,reference_policies=policies,
        native_generation=json.loads((st/'configs/B_bank_generation.json').read_text()),
        donor_assignments=assignments,statistical_design=contrasts,
        projection=contract['projection'],C_lambda_lock=lambda_lock,
        groups=list(range(6)),streams=list(range(4)),dual_groups=[0,1],C_groups=[0,2,4],
        objectives=['latent','decoded_teacher','physical_labels'],C_conditions=['T0','T1','T2'],
        time_contract=dict(history_tokens=3,actions_per_token=5,insertion_action=5,endpoint_action=25,
            native_first_index=3,native_endpoint_index=7,native_visual_dimensions=75264),
        matching=dict(A_core=6,A_dual=12,C=6,B=2,C_individual_feasible='descriptive; never replaces matched C budget'),
        precision=dict(dynamics='FP32; TF32 off',projection='FP64',verification='full FP64 nonlinear head evaluated on actual FP32 inserted token'),
        failure_policy='No model, goal, horizon or sample-count selection; no failed case treated as zero. Preserve failure artifacts and stop affected analysis. Only identical-binding engineering retries. Valid zero action response or zero feasible displacement is retained and reported.',
        isolation_policy='Parent/whole-history duplication blocks analysis; individual future-pixel matches are reported, not removed. No result-dependent exclusion or reseeding.',
        source_scope='A/C prospective visible block transport conditioned by unchanged original generator; B native history-visible relative controls from one fixed model-free source; not official planning benchmark or proven OOD.',
        unresolved=['DINO author historical backbone checkout','Complete upstream native seed lineage','All foundation-model pretraining image overlap'],
        input_sources={str(f.relative_to(ROOT)):sha(f) for f in sorted(sources)},
        external_input_sources=external,verified_assets=assets,environment_inventory=inventory,development_evidence=evidence,
        next_gate='Freeze full input roster, actions, physics, donor assignments and evaluator/source hashes in a separate protocol lock before new-model evaluation.')
    out = Path(a.output); out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('x') as f:
        json.dump(design,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    print(json.dumps(dict(status=design['status'],path=str(out),sha256=sha(out),verified_assets=len(assets),
                         input_sources=len(sources),confirmation_results_exposed=False)),flush=True)


if __name__ == '__main__': main()
