"""Fixed-trajectory development sensitivity with four norm-matched sources."""
import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
import torch

from adaptation_freeze import configure_dynamics_only, verify_frozen
from evaluation_precision import configure_evaluation_precision
from factorial_model import make_model
from feedback_suffix_rollout import rollout_suffixes
from image_planner_cost import ImagePlannerCost
from nonlinear_pose_cost import numpy_pose
from score_feedback_ranking import normalized_readout, region_margins, sha, state_hash


SOURCES = ('actual', 'random', 'pose_nearest', 'latent_nearest')
BRANCHES = ('free', *SOURCES, 'reset')
HORIZONS = (10, 15, 20, 25)


def match_twelve(head, projections):
    """Match all four sources and all three objectives before any rollout."""
    assert len(projections) == 3
    norms = [np.linalg.norm((z['corrections'].astype(float) - z['predicted'][:, None]) / head['scale'], axis=-1)
             for z in projections]
    common = np.min(np.stack(norms), axis=(0, 2))
    assert np.isfinite(common).all()
    results = []
    for z, full_norm in zip(projections, norms):
        delta = z['corrections'].astype(float) - z['predicted'][:, None]
        alpha = np.divide(common[:, None], full_norm, out=np.zeros_like(full_norm), where=full_norm > 0)
        corrected = (z['predicted'].astype(float)[:, None] + alpha[:, :, None] * delta).astype(np.float32)
        matched_norm = np.linalg.norm((corrected.astype(float) - z['predicted'][:, None]) / head['scale'], axis=-1)
        np.testing.assert_allclose(matched_norm, np.broadcast_to(common[:, None], matched_norm.shape), rtol=1e-6, atol=1e-6)
        error = float(np.max(abs(normalized_readout(head, corrected) - normalized_readout(head, z['predicted'])[:, None])))
        assert error <= 1e-6
        margin = min(float(np.min(region_margins(head, predicted, value)))
                     for predicted, values in zip(z['predicted'], corrected) for value in values)
        assert margin >= -1e-6
        results.append(dict(predicted=z['predicted'], full_correction_tokens=z['corrections'],
                            matched_correction_tokens=corrected, full_displacement_norms=full_norm,
                            matched_displacement_norms=matched_norm, match_alphas=alpha, common_norm=common,
                            max_readout_error=error, minimum_region_margin=margin))
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('plan', 'bank', 'development-scores', 'geometry', 'old-fibers', 'official', 'config', 'protocol', 'output'):
        parser.add_argument('--' + key, required=True)
    parser.add_argument('--group', type=int, choices=range(6), required=True)
    parser.add_argument('--cases', type=int, default=128)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    started = time.monotonic()
    bank, anchors, geometry_dir, out = [Path(getattr(args, key)) for key in ('bank', 'development_scores', 'geometry', 'output')]
    manifest = json.loads((bank / 'manifest.json').read_text())
    bank_acceptance = json.loads((bank / 'acceptance.json').read_text())
    anchor_report = json.loads((anchors / 'report.json').read_text())
    geometry_report = json.loads((geometry_dir / 'report.json').read_text())
    plan = json.loads(Path(args.plan).read_text())
    protocol_sha = sha(args.protocol)
    assert 0 < args.cases <= 128 == len(manifest['cases'])
    assert manifest['group'] == anchor_report['group'] == args.group
    assert manifest['reference_route'] == 16 * args.group
    assert manifest['bindings']['plan_sha256'] == anchor_report['plan_sha256'] == sha(args.plan)
    assert anchor_report['status'] == 'DEVELOPMENT_SCORES_REQUIRE_OUTCOME_ACCEPTANCE'
    assert anchor_report['bank_manifest_sha256'] == sha(bank / 'manifest.json')
    assert bank_acceptance['status'] == 'PASS_SHARED_PREFIX_SIMULATOR_BANK'
    assert bank_acceptance['manifest_sha256'] == sha(bank / 'manifest.json')
    assert geometry_report['status'] == 'OUTCOME_BLIND_NEAREST_DONOR_GEOMETRY_AUDIT'
    assert geometry_report['recipient_cases_per_group'] == 128
    assert geometry_report['source_sha256'] == sha(Path(__file__).with_name('audit_nearest_donor_geometry.py'))
    expected_models = [8 * args.group + slot for slot in (2, 3, 4)]
    assert [row['model_index'] for row in anchor_report['models']] == expected_models
    group_geometry = next(row for row in geometry_report['groups'] if row['group'] == args.group)
    old_dir = Path(args.old_fibers)
    if not (old_dir / 'report.json').is_file():
        old_dir = old_dir / f'job_{args.group}'
    old_report = json.loads((old_dir / 'report.json').read_text())
    assert sha(old_dir / 'report.json') == group_geometry['fiber_report_sha256']
    projections, root_rows, heads, old_arrays = [], [], [], []
    for mi, anchor_model in zip(expected_models, anchor_report['models']):
        geometry_model = next(row for row in geometry_report['models'] if row['model_index'] == mi)
        assert geometry_model['cases'] == 128 and geometry_model['valid_attempts'] == geometry_model['attempts'] == 256
        assert sha(geometry_dir / geometry_model['file']) == geometry_model['sha256']
        z = dict(np.load(geometry_dir / geometry_model['file'], allow_pickle=False))
        np.testing.assert_array_equal(z['sources'], SOURCES)
        assert z['predicted'].shape == (128, 192) and z['corrections'].shape == (128, 4, 192)
        binding = next(row for row in old_report['bindings'] if row.get('model_index') == mi)
        assert sha(old_dir / binding['output_file']) == binding['output_sha256'] == geometry_model['old_fiber_sha256']
        with np.load(old_dir / binding['output_file'], allow_pickle=False) as archive:
            old = {key: archive[key][:128].copy() for key in ('predicted', 'observed', 'full', 'shuffled_full')}
        np.testing.assert_array_equal(z['predicted'], old['predicted'])
        np.testing.assert_array_equal(z['corrections'][:, 0], old['full'])
        np.testing.assert_array_equal(z['corrections'][:, 1], old['shuffled_full'])
        hp = anchors / anchor_model['head_file']
        assert sha(hp) == anchor_model['head_sha256'] == group_geometry['head_sha256']
        heads.append(dict(np.load(hp, allow_pickle=False)))
        projections.append({key: value[:args.cases] for key, value in z.items() if key != 'sources'})
        old_arrays.append({key: value[:args.cases] for key, value in old.items()})
        root_rows.append(dict(model_index=mi, objective=anchor_model['objective'], head_sha256=sha(hp),
                              geometry_projection_sha256=geometry_model['sha256'], old_fiber_sha256=binding['output_sha256']))
    for head in heads[1:]:
        for key in heads[0]:
            np.testing.assert_array_equal(head[key], heads[0][key])
    matched = match_twelve(heads[0], projections)
    out.mkdir(parents=True, exist_ok=False)
    fiber_dir = out / 'fibers'
    fiber_dir.mkdir()
    cases_used = manifest['cases'][:args.cases]
    assert [case['index'] for case in cases_used] == list(range(args.cases))
    for row, arrays, old in zip(root_rows, matched, old_arrays):
        arrays['observed'] = old['observed']
        arrays['sources'] = np.asarray(SOURCES)
        arrays['seeds'] = np.asarray([case['seed'] for case in cases_used])
        row.update(max_readout_error=arrays.pop('max_readout_error'), minimum_region_margin=arrays.pop('minimum_region_margin'))
        name = f"model_{row['model_index']}.npz"
        np.savez_compressed(fiber_dir / name, **arrays)
        row.update(file=name, sha256=sha(fiber_dir / name))
    fiber_report = dict(status='NEAREST_DONOR_MATCHED_FIBERS_READY', group=args.group, expected_cases=args.cases,
                        models=root_rows, protocol_sha256=protocol_sha, geometry_report_sha256=sha(geometry_dir / 'report.json'),
                        old_fibers_report_sha256=sha(old_dir / 'report.json'),
                        bank_manifest_sha256=sha(bank / 'manifest.json'), source_sha256=sha(__file__),
                        scope='Four-source, three-objective displacement matching only; no future outcomes.')
    (fiber_dir / 'report.json').write_text(json.dumps(fiber_report, indent=2) + '\n')
    precision = configure_evaluation_precision()
    torch.set_num_threads(4)
    models = []
    for mi, head, fiber_row, values, anchor_model in zip(expected_models, heads, root_rows, matched, anchor_report['models']):
        entry, original = plan['models'][mi], plan['models'][8 * args.group]
        assert entry['endpoint_head'] == entry['goal_head']
        assert entry['endpoint_head']['sha256'] == fiber_row['head_sha256']
        td, od = Path(entry['training_path']), Path(original['training_path'])
        tr = json.loads((td / 'summary.json').read_text())
        assert sha(td / 'summary.json') == entry['training_summary_sha256']
        assert sha(td / 'last_weights.pt') == entry['weights_sha256'] == anchor_model['model_weights_sha256']
        assert sha(od / 'last_weights.pt') == original['weights_sha256']
        assert sha(args.config) == tr['config_sha256']
        model = make_model(args.official, args.config, entry['arm'], tr['seed'])
        model.load_state_dict(torch.load(od / 'last_weights.pt', map_location='cpu', weights_only=True))
        model = model.to(args.device)
        boundary = configure_dynamics_only(model)
        model.load_state_dict(torch.load(td / 'last_weights.pt', map_location='cpu', weights_only=True))
        verify_frozen(model, boundary)
        before = state_hash(model)
        assert before == anchor_model['state_hash']
        model_dir = out / f'model_{mi}'
        model_dir.mkdir()
        shutil.copyfile(anchors / anchor_model['head_file'], model_dir / 'head.npz')
        scored_cases = []
        anchor_cases = {case['index']: case for case in anchor_model['cases']}
        with torch.inference_mode():
            for case in cases_used:
                index = case['index']
                inp = bank / case['input_file']
                assert sha(inp) == case['input_sha256']
                z = dict(np.load(inp, allow_pickle=False))
                assert not {'terminal_states', 'goal_state', 'suffix_states', 'current_state'} & set(z)
                anchor_case = anchor_cases[index]
                assert anchor_case['input_sha256'] == case['input_sha256'] and anchor_case['seed'] == case['seed']
                ap = anchors / anchor_case['file']
                assert sha(ap) == anchor_case['sha256']
                with np.load(ap, allow_pickle=False) as archive:
                    anchor_initial = archive['initial_tokens'].copy()
                    anchor_free = archive['tokens'][0, :, 0].copy()
                    anchor_reset = archive['tokens'][3, :, 0].copy()
                np.testing.assert_array_equal(values['predicted'][index], anchor_initial[0])
                np.testing.assert_array_equal(values['observed'][index], anchor_initial[3])
                initial = np.concatenate([values['predicted'][index:index + 1], values['matched_correction_tokens'][index],
                                          values['observed'][index:index + 1]], axis=0)
                interface = ImagePlannerCost(model, z['history_pixels'], z['goal_pixels'], z['prefix'],
                                             tr['normalization'], entry['target_normalization'], 'latent')
                suffixes = torch.as_tensor(z['suffix_actions'], device=args.device)
                assert suffixes.shape == (32, 20, 2)
                past = torch.as_tensor(np.concatenate([z['prefix'][-5:], z['executed_actions']]), device=args.device)
                am, ast = [torch.tensor(tr['normalization'][key], device=args.device) for key in ('mean', 'std')]
                tokens = np.stack([rollout_suffixes(model, interface.initial,
                    torch.as_tensor(replacement, device=args.device), past, suffixes, am, ast)[:, 0].cpu().numpy()
                    for replacement in initial])
                assert tokens.shape == (6, 4, 192) and np.isfinite(tokens).all()
                np.testing.assert_array_equal(tokens[0], anchor_free)
                np.testing.assert_array_equal(tokens[5], anchor_reset)
                pose = numpy_pose(tokens, head)
                file = model_dir / f'case_{index:03d}.npz'
                np.savez_compressed(file, initial_tokens=initial, tokens=tokens, pose=pose,
                                    branches=np.array(BRANCHES), horizons=np.array(HORIZONS), common_norm=values['common_norm'][index])
                scored_cases.append(dict(index=index, seed=case['seed'], file=str(file.relative_to(out)),
                                         sha256=sha(file), input_sha256=sha(inp), anchor_score_sha256=sha(ap),
                                         free_anchor_exact=True, reset_anchor_exact=True))
        verify_frozen(model, boundary)
        assert state_hash(model) == before
        models.append(dict(model_index=mi, objective=entry['adaptation_condition'], cases=scored_cases,
                           head_file=str((model_dir / 'head.npz').relative_to(out)), head_sha256=fiber_row['head_sha256'],
                           fiber_sha256=fiber_row['sha256'], geometry_projection_sha256=fiber_row['geometry_projection_sha256'],
                           old_fiber_sha256=fiber_row['old_fiber_sha256'],
                           model_weights_sha256=entry['weights_sha256'], state_hash=before))
        print('NEAREST_DONOR_SCORED', mi, args.cases, flush=True)
        del model, boundary, interface
        if str(args.device).startswith('cuda'):
            torch.cuda.empty_cache()
    assert protocol_sha == sha(args.protocol)
    report = dict(status='NEAREST_DONOR_SCORES_REQUIRE_ACCEPTANCE', phase='development', group=args.group,
                  expected_cases=args.cases, case_count=args.cases, branches=list(BRANCHES), horizons=list(HORIZONS),
                  expected_model_indices=expected_models, models=models, protocol_sha256=protocol_sha,
                  plan_sha256=sha(args.plan), bank_manifest_sha256=sha(bank / 'manifest.json'),
                  anchor_score_report_sha256=sha(anchors / 'report.json'), geometry_report_sha256=sha(geometry_dir / 'report.json'),
                  old_fibers_report_sha256=sha(old_dir / 'report.json'),
                  fibers_report_sha256=sha(fiber_dir / 'report.json'), source_sha256=sha(__file__),
                  helper_sha256=sha(Path(__file__).with_name('feedback_suffix_rollout.py')),
                  utility_source_sha256=sha(Path(__file__).with_name('score_feedback_ranking.py')),
                  config_sha256=sha(args.config), precision=precision, elapsed_seconds=time.monotonic() - started,
                  gpu=torch.cuda.get_device_name() if str(args.device).startswith('cuda') else 'cpu',
                  scope='Readout-preserving nearest-donor sensitivity on previously consumed development recipients; selected suffix only; no action-selection outcome or independent confirmation.')
    (out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
