"""Build audit-only manifests from read-only receipts. Does not run experiments."""
from pathlib import Path
import hashlib
import itertools
import json
import subprocess

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'strengthening'
REC = OUT / 'reports/receipts'
BASE = '/external-assets/cluster/projects/jepa-regime-study-maintrack-20260908'
RUNS = BASE + '/releases/planner-data-adaptation-v1/runs'


def read(name):
    return json.loads((REC / (name + '.json')).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save(name, data):
    (OUT / 'manifests' / name).write_text(json.dumps(data, indent=2) + '\n')


def main():
    inv, lineage, supplement = map(read, ['remote_inventory', 'remote_lineage', 'remote_supplement'])
    components = read('checkpoint_components')
    assets = {}
    for row in inv['assets']:
        if not row['exists']:
            continue  # An incorrect discovery guess is documented separately, not a required asset.
        p = row['path']
        if p in assets:
            assets[p]['roles'].append(row['role'])
        else:
            assets[p] = dict(row, roles=[row['role']], location='hpg', usage='legacy_read_only')
    for p, row in read('original_head_lineage').items():
        if row.get('exists'):
            assets[p] = dict(path=p, sha256=row['sha256'], verification='HASHED',
                             location='hpg', roles=['original_head_fit_lineage'], usage='legacy_read_only')
    for p, row in supplement.items():
        if p.endswith('/normalization.json'):
            assets[p] = dict(path=p, sha256=row['sha256'], verification='HASHED',
                             location='hpg', roles=['expert_train_action_state_normalizer'], usage='legacy_read_only')
    for role, bank in lineage['banks'].items():
        p = bank['path'] + '/manifest.json'
        assets[p] = dict(path=p, sha256=bank['manifest_sha256'], verification='HASHED',
                         location='hpg', roles=[role], usage='legacy_read_only')
    # All source entry points are hashed; this is not a declaration that all implement the new plan.
    local = list((ROOT/'scripts/visual').glob('*.py'))
    local += list((ROOT/'tests').glob('test_*.py'))
    local += [ROOT/'ICLR_feedback_strengthening_codex'/p for p in
              ['CODEX_START_HERE.md','IMPLEMENTATION_PLAN.md','source/mechanism_manuscript(9).pdf']]
    local += [ROOT/p for p in ['output/pdf/mechanism_manuscript.pdf',
             'output/latex/when_predictions_become_inputs_latex.zip',
             'output/repro/feedback_diagnostic_evidence.zip',
             'runs/hpg/storage_cleanup_20260918/final_storage_report.json']]
    local += list((ROOT/'output/repro/feedback_diagnostic_evidence').rglob('*'))
    local += list((OUT/'external/dino_wm/models').glob('*.py'))
    local += [OUT/'external/dino_wm'/p for p in ['plan.py','datasets/pusht_dset.py',
              'conf/train.yaml','conf/plan_pusht.yaml','LICENSE','README.md']]
    for p in local:
        if p.is_file():
            assets[str(p)] = dict(path=str(p), sha256=sha(p), bytes=p.stat().st_size,
                                 location='local', verification='HASHED',
                                 roles=['source_or_legacy_evidence'], usage='read_only_input')
    normalizers = []
    for p, row in inv['arrays'].items():
        if '/pose_fit/' in p and p.endswith('weights.npz'):
            normalizers.append(dict(path=p, file_sha256=assets[p]['sha256'],
                components={k:{f:v[f] for f in ['shape','dtype','sha256']}
                            for k,v in row['selected'].items()},
                input_space='192D observed encoding',
                output_order=['agent_x','agent_y','block_x','block_y','sin_theta','cos_theta'],
                fit_source='expert train endpoint features only; fit_pose_readouts.py',
                use_for_A='reuse both input and target scaling; do not refit per head'))
    roster = []
    plan = inv['metadata'][RUNS+'/fiber_confirmation_plan.json']
    for g in range(6):
        entries = []
        for slot, name in [(0,'original_21000'),(2,'latent'),(3,'decoded_teacher'),(4,'physical_labels')]:
            i = 8*g+slot
            e = plan['models'][i]
            comp = next(x for x in components['rows'] if x['model_index']==i)
            entries.append(dict(condition=name, model_index=i,
                continuation_job_index=None if slot==0 else 4*g+slot-2,
                checkpoint=e['training_path']+'/last_weights.pt',
                checkpoint_sha256=e['weights_sha256'],
                head=e['endpoint_head'], component_hashes=comp['components_sha256'],
                frozen_equal_original=comp.get('frozen_encoder_and_observation_projector_equal_original')))
        roster.append(dict(group=g, pool=g//2, architecture=['transformer_jepa','gru_jepa'][g%2],
                           models=entries, cache=RUNS+f'/formal_cache/job_{2*g}'))
    save('legacy_model_roster.json', dict(status='AUDITED_LEGACY_NOT_NEW_CONFIRMATION_ROSTER',
        groups=roster, note='Paper uses slots 2/3/4 only; do not run all 48 omnibus models.'))

    pairwise=[]
    for (an,a),(bn,b) in itertools.combinations(lineage['banks'].items(),2):
        overlaps={'seed':len({r['seed'] for r in a['rows']} & {r['seed'] for r in b['rows']})}
        for field in a['rows'][0]['fields']:
            overlaps[field]=len({r['fields'][field] for r in a['rows']} &
                                {r['fields'][field] for r in b['rows']})
        pairwise.append(dict(left=an,right=bn,exact_overlap_counts=overlaps))
    pools=[]
    episode_sets={}
    for pool in range(3):
        info={}
        for split in ['train','validation']:
            p=BASE+f'/datasets/pusht-scaling-v1/replica_{pool}/n256/{split}/episodes.npz'
            d=inv['arrays'][p]['selected']
            ids=set(d['source_episode_ids']['values'])
            episode_sets[(pool,split)]=ids
            info[split]=dict(episodes=len(ids), frames=sum(d['lengths']['values']),
                             parent_episode_ids=sorted(ids),source=p)
        info['pool']=pool
        info['train_validation_episode_overlap']=len(episode_sets[(pool,'train')] & episode_sets[(pool,'validation')])
        info['A_unique_frame_upper_bound_before_pixel_dedup']=info['train']['frames']+2048
        pools.append(info)
    cache=[]
    for g in range(6):
        root=RUNS+f'/formal_cache/job_{2*g}'
        d={s:inv['arrays'][root+'/'+s+'.npz']['selected'] for s in
           ['expert_train','expert_validation','planner_train','planner_validation']}
        ex={(int(ep),int(start)+delta) for ep,start in d['expert_train']['identities']['values'] for delta in [0,5,10,15]}
        pl={(int(route),int(i),int(seed),int(start)+delta)
            for route,i,seed,start in d['planner_train']['identities']['values'] for delta in [0,5,10,15]}
        train=set(d['expert_train']['pixel_sha256']['values'])|set(d['planner_train']['pixel_sha256']['values'])
        val=set(d['expert_validation']['pixel_sha256']['values'])|set(d['planner_validation']['pixel_sha256']['values'])
        cache.append(dict(group=g,windows={s:len(v['identities']['values']) for s,v in d.items()},
                          train_validation_four_frame_hash_overlap=len(train & val),
                          expert_train_parent_frames=len(ex),planner_train_route_frame_tuples=len(pl),
                          note='Tuple count is an upper bound on unique image payloads; shared route prefixes exist.'))
    tr={f:set() for f in ['states','actions']};va={f:set() for f in tr}
    for key,value in lineage['trajectories'].items():
        dest=tr if key.startswith('train/') else va
        for row in value['rows']:
            for f in dest:dest[f].add(row['fields'][f])
    splits=dict(status='LEGACY_LINEAGE_AUDITED_NEW_SPLITS_NOT_CREATED',banks={k:dict(path=v['path'],
        goals=len(v['rows']),manifest_sha256=v['manifest_sha256']) for k,v in lineage['banks'].items()},
        bank_pairwise=pairwise,expert_pools=pools,
        expert_cross_pool=[dict(pools=[a,b],train_overlap=len(episode_sets[a,'train']&episode_sets[b,'train']),
            validation_overlap=len(episode_sets[a,'validation']&episode_sets[b,'validation'])) for a,b in itertools.combinations(range(3),2)],
        caches=cache,planner_full_trajectory_train_validation_overlap={f:len(tr[f]&va[f]) for f in tr},
        C=dict(status='RAW_CONTINUOUS_TRAJECTORIES_EXIST_CACHE_NOT_IMPLEMENTED',
               observations_per_segment=6,action_blocks_per_segment=5,primitive_stride=5,
               starts_per_existing_35_action_trajectory=[0,5,10],
               train_segments_per_group=768,validation_segments_per_group=192),
        new_roles={k:dict(status='NOT_CREATED',manifest=None) for k in
                   ['head_train','head_validation','diagnostic_development','donor_assignment','confirmation_A_C','confirmation_B','native_basis_train']},
        unresolved=['Global raw-frame payload deduplication across expert/planner/pretraining sources',
                    'Native DINO-WM checkpoint training episode membership',
                    'Fresh root seed, immutable role manifests and enforced loader role rejection'])
    save('split_lineage.json',splits)

    decision=[]
    decision_root=ROOT/'runs/feedback_ranking_confirmation_v1'
    for g in range(6):
        for slot in [2,3,4]:
            folder=decision_root/f'scores_{g}/model_{8*g+slot}'
            paths=sorted(folder.glob('case_*.npz'))
            p=folder/'case_000.npz'
            decision.append(dict(group=g,model_index=8*g+slot,files=len(paths),
                                 bytes=sum(x.stat().st_size for x in paths),
                                 example_path=str(p),example_sha256=sha(p) if p.exists() else None,
                                 content_verification='REPRESENTATIVE_HASH_ONLY_FULL_COUNT_STAT'))
        p=decision_root/f'bank_{g}/outcomes/case_000.npz'
        if p.exists():
            assets[str(p)]=dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size,
                               verification='HASHED_REPRESENTATIVE',location='local',
                               roles=['D_realized_physical_cost_source'],usage='legacy_exploratory_only')
    save('decision_arrays.json',dict(status='REUSE_EXISTING_EXPLORATORY_NO_NEW_SIMULATION',
         score_collections=decision,remote_counts={k:v for k,v in supplement.items() if 'count' in v},
         goal_count=512,candidates_per_pool=32,
         cost_definition='block_dx**2 + block_dy**2 + 900 * wrapped_angle_error**2',
         warning='Full selected score archives counted; this audit has not rehashed every cost archive.'))
    # Verify original head fit binds the feature report actually found.
    headline=read('original_head_lineage');bindings=[]
    hbase=BASE+'/releases/pusht-nonlinear-transfer-v1/runs'
    for g in range(6):
        fit=headline[hbase+f'/pose_fit/job_{g}/report.json']
        feature=headline[hbase+f'/features/job_{g}/report.json']
        bindings.append(dict(group=g,feature_report_sha256=feature.get('sha256'),
            expected=fit.get('data',{}).get('input_report_sha256'),
            match=feature.get('sha256')==fit.get('data',{}).get('input_report_sha256'),
            counts=feature.get('data',{}).get('counts')))
    save('assets.json', dict(schema_version=1,status='AUDIT_ONLY_NOT_PROTOCOL_LOCK',
        audit_date='2026-09-19',repository=str(ROOT),
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        remote_base=BASE,legacy_assets_read_only=True,
        manuscript_match=sha(ROOT/'output/pdf/mechanism_manuscript.pdf')==sha(ROOT/'ICLR_feedback_strengthening_codex/source/mechanism_manuscript(9).pdf'),
        manuscript_sha256=sha(ROOT/'output/pdf/mechanism_manuscript.pdf'),
        assets=list(assets.values()),normalizers=normalizers,head_feature_report_bindings=bindings,
        paper_evidence_bindings={
            'Table_1_and_original_prediction_primary': {
                'input':str(ROOT/'output/repro/feedback_diagnostic_evidence/data/prediction_errors.npz'),
                'reconstruction_receipt':'reports/receipts/table1_reconstruction.json',
                'purpose':'Old endpoint absolute errors and five original prediction contrasts'},
            'Table_3_and_original_decision_primary': {
                'input':str(ROOT/'output/repro/feedback_diagnostic_evidence/data/decision_goal_metrics.npz'),
                'purpose':'Old 512-goal decision results; four original contrasts'},
            'second_readout_appendix': {
                'input':str(ROOT/'output/repro/feedback_diagnostic_evidence/data/second_readout_goal_metrics.npz'),
                'purpose':'Old three configurations, eighteen exploratory contrasts'},
            'D_exploratory_candidate_inputs': {
                'manifest':'manifests/decision_arrays.json',
                'purpose':'Existing 32-candidate predicted costs and shared physical outcomes; not new confirmation'}},
        component_receipt='reports/receipts/checkpoint_components.json',
        external=dict(dino_wm_source_commit='0a9492fa12044b852ae9e001cc74604b79c8bb0c',
            source_url='https://github.com/gaoyuezhou/dino_wm',
            catalog=read('dinowm_remote_file_catalog'),
            checkpoint_status='UNRESOLVED_NOT_DOWNLOADED',
            catalog_hash_scope='Publisher advertised archive hashes, not local verification or inner checkpoint hashes',
            legacy_official_git_commits='UNRESOLVED: release folders have no .git; individual files hashed'),
        new_inputs=dict(protocol_lock=None,root_seed=None,head_A=None,head_B=None,
                        native_checkpoint=None,native_basis=None,confirmation_A_C=None,confirmation_B=None),
        discovery_corrections=[dict(incorrect_guess=RUNS+'/hpg/feedback_ranking_v1/confirmation/contexts/manifest.json',
             verified_path=lineage['banks']['old_decision']['path']+'/manifest.json'),
             dict(note='scores_<g> uses nested model_<i>/case_*.npz; initial top-level glob returned empty, recursive inventory resolves it')],
        unresolved=['Pixel .npy files stat checked; declared whole-file hashes not recomputed',
                    'Original large HDF5 whole-file hash not recomputed',
                    'Exact current project-only allocated size: bounded du timed out',
                    'All new protocol inputs and DINO native tensor/time/normalization bindings pending']))
    print(json.dumps(dict(assets=len(assets),heads=len(normalizers),groups=len(roster),
        all_head_report_bindings_match=all(x['match'] for x in bindings),
        bank_pairwise_comparisons=len(pairwise),decision_score_files=sum(x['files'] for x in decision))))


if __name__ == '__main__':
    main()
