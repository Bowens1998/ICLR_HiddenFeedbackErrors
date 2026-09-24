"""Evaluate a frozen action-choice confirmation from fresh feedback roots."""
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
from nonlinear_pose_cost import numpy_cost, numpy_pose
from score_feedback_ranking import BRANCHES, normalized_readout, region_margins, sha, state_hash


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('plan', 'bank', 'roots', 'fibers', 'official', 'config', 'protocol', 'output'):
        parser.add_argument('--' + key, required=True)
    parser.add_argument('--group', type=int, choices=range(6), required=True)
    parser.add_argument('--expected-cases', type=int, default=512)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    started = time.monotonic()
    bank, rd, fd, out = [Path(getattr(args, key)) for key in ('bank', 'roots', 'fibers', 'output')]
    manifest = json.loads((bank / 'manifest.json').read_text())
    ba = json.loads((bank / 'acceptance.json').read_text())
    roots_report = json.loads((rd / 'report.json').read_text())
    fiber_report = json.loads((fd / 'report.json').read_text())
    plan = json.loads(Path(args.plan).read_text())
    protocol_sha = sha(args.protocol)
    assert manifest['group'] == args.group and manifest['reference_route'] == 16 * args.group
    assert len(manifest['cases']) == args.expected_cases > 0
    assert [case['index'] for case in manifest['cases']] == list(range(args.expected_cases))
    assert ba['status'] == 'PASS_SHARED_PREFIX_SIMULATOR_BANK' and ba['manifest_sha256'] == sha(bank / 'manifest.json')
    assert roots_report['status'] == 'FRESH_FEEDBACK_ROOTS_READY'
    assert fiber_report['status'] == 'FRESH_FEEDBACK_FIBERS_READY'
    assert fiber_report['roots_report_sha256'] == sha(rd / 'report.json')
    for report in (roots_report, fiber_report):
        assert report['group'] == args.group and report['case_count'] == report['expected_cases'] == args.expected_cases
        assert report['protocol_sha256'] == protocol_sha
        assert report['plan_sha256'] == sha(args.plan) == manifest['bindings']['plan_sha256']
        assert report['bank_manifest_sha256'] == sha(bank / 'manifest.json')
    assert roots_report['source_sha256'] == sha(Path(__file__).with_name('extract_fresh_feedback_roots.py'))
    assert fiber_report['source_sha256'] == sha(Path(__file__).with_name('prepare_fresh_feedback_fibers.py'))
    expected = [8 * args.group + slot for slot in (2, 3, 4)]
    assert [r['model_index'] for r in roots_report['models']] == [r['model_index'] for r in fiber_report['models']] == expected
    out.mkdir(parents=True, exist_ok=False)
    precision = configure_evaluation_precision()
    torch.set_num_threads(4)
    entries = []
    for root_row, fiber_row in zip(roots_report['models'], fiber_report['models']):
        mi = root_row['model_index']
        entry, original = plan['models'][mi], plan['models'][8 * args.group]
        assert root_row['objective'] == fiber_row['objective'] == entry['adaptation_condition']
        assert sha(rd / root_row['file']) == root_row['sha256'] == fiber_row['root_file_sha256']
        assert sha(fd / fiber_row['file']) == fiber_row['sha256']
        hp = rd / root_row['head_file']
        assert sha(hp) == root_row['head_sha256'] == fiber_row['head_sha256'] == entry['endpoint_head']['sha256']
        roots = dict(np.load(rd / root_row['file'], allow_pickle=False))
        fibers = dict(np.load(fd / fiber_row['file'], allow_pickle=False))
        for key in ('predicted', 'observed', 'donor', 'seeds', 'donor_indices', 'donor_seeds'):
            np.testing.assert_array_equal(roots[key], fibers[key])
        head = dict(np.load(hp, allow_pickle=False))
        td, od = Path(entry['training_path']), Path(original['training_path'])
        tr = json.loads((td / 'summary.json').read_text())
        assert sha(td / 'summary.json') == entry['training_summary_sha256']
        assert sha(td / 'last_weights.pt') == entry['weights_sha256'] == root_row['model_weights_sha256']
        assert sha(od / 'last_weights.pt') == original['weights_sha256']
        assert sha(args.config) == tr['config_sha256'] == roots_report['config_sha256']
        model = make_model(args.official, args.config, entry['arm'], tr['seed'])
        model.load_state_dict(torch.load(od / 'last_weights.pt', map_location='cpu', weights_only=True))
        model = model.to(args.device)
        boundary = configure_dynamics_only(model)
        model.load_state_dict(torch.load(td / 'last_weights.pt', map_location='cpu', weights_only=True))
        verify_frozen(model, boundary)
        before = state_hash(model)
        assert before == root_row['initial_state_hash']
        model_dir = out / f'model_{mi}'
        model_dir.mkdir()
        shutil.copyfile(hp, model_dir / 'head.npz')
        cases = []
        with torch.inference_mode():
            for case in manifest['cases']:
                index = case['index']
                inp = bank / case['input_file']
                assert sha(inp) == case['input_sha256']
                z = dict(np.load(inp, allow_pickle=False))
                assert not {'terminal_states', 'goal_state', 'suffix_states', 'current_state'} & set(z)
                assert int(z['seed']) == case['seed'] == int(roots['seeds'][index])
                interface = ImagePlannerCost(model, z['history_pixels'], z['goal_pixels'], z['prefix'],
                                             tr['normalization'], entry['target_normalization'], 'latent')
                np.testing.assert_array_equal(interface.initial.cpu().numpy(), roots['initial_history'][index])
                np.testing.assert_array_equal(interface.goal.cpu().numpy(), roots['goal_tokens'][index])
                population = torch.as_tensor(z['source_population_actions'], device=args.device)
                encoded = model.action_encoder(interface.normalized_actions(population))
                history = interface.initial[None].expand(300, -1, -1).clone()
                predicted = model.predict(history, encoded[:, :3])[:, -1]
                selected = int(z['source_selected_index'])
                np.testing.assert_array_equal(predicted[selected].cpu().numpy(), roots['predicted'][index])
                np.testing.assert_array_equal(population[selected, :5].cpu().numpy(), z['executed_actions'])
                initial = np.stack([fibers[key][index] for key in ('predicted', 'constrained', 'shuffled', 'observed')])
                error = float(np.max(abs(normalized_readout(head, initial[:3]) - normalized_readout(head, initial[0]))))
                assert error <= 1e-6
                norms = np.linalg.norm((initial[1:3].astype(float) - initial[0]) / head['scale'], axis=-1)
                np.testing.assert_allclose(norms, fibers['target_norm'][index], rtol=1e-6, atol=1e-6)
                for replacement in initial[1:3]:
                    assert np.min(region_margins(head, initial[0], replacement)) >= -1e-6
                suffixes = torch.as_tensor(z['suffix_actions'], device=args.device)
                past = torch.as_tensor(np.concatenate([z['prefix'][-5:], z['executed_actions']]), device=args.device)
                am, ast = [torch.tensor(tr['normalization'][key], device=args.device) for key in ('mean', 'std')]
                initial_history = torch.as_tensor(roots['initial_history'][index], device=args.device)
                tokens = np.stack([rollout_suffixes(model, initial_history,
                    torch.as_tensor(replacement, device=args.device), past, suffixes, am, ast).cpu().numpy()
                    for replacement in initial])
                assert tokens.shape == (4, 4, 32, 192) and np.isfinite(tokens).all()
                pose = numpy_pose(tokens, head)
                goal_token = roots['goal_tokens'][index]
                goal_pose = numpy_pose(goal_token, head)
                pose_cost = numpy_cost(pose[:, -1], goal_pose)
                latent_cost = np.square(tokens[:, -1].astype(float) - goal_token).sum(-1)
                output = model_dir / f'case_{index:03d}.npz'
                np.savez_compressed(output, initial_tokens=initial, tokens=tokens, pose=pose,
                                    goal_token=goal_token, goal_pose=goal_pose, pose_cost=pose_cost,
                                    latent_cost=latent_cost, branches=np.array(BRANCHES), horizons=np.array([10, 15, 20, 25]))
                cases.append(dict(index=index, seed=case['seed'], file=str(output.relative_to(out)),
                                  sha256=sha(output), input_sha256=sha(inp), max_readout_error=error,
                                  standardized_displacement_norms=norms.tolist()))
        verify_frozen(model, boundary)
        assert state_hash(model) == before
        entries.append(dict(model_index=mi, objective=entry['adaptation_condition'], cases=cases,
                            head_file=str((model_dir / 'head.npz').relative_to(out)), head_sha256=sha(hp),
                            model_weights_sha256=entry['weights_sha256'], state_hash=before,
                            roots_sha256=root_row['sha256'], fiber_sha256=fiber_row['sha256']))
        print('FRESH_SCORES_READY', mi, args.expected_cases, flush=True)
        del model, boundary, interface
        if str(args.device).startswith('cuda'):
            torch.cuda.empty_cache()
    assert protocol_sha == sha(args.protocol)
    report = dict(status='CONFIRMATION_SCORES_REQUIRE_OUTCOME_ACCEPTANCE', phase='confirmation',
                  execution_scope='frozen_confirmation' if args.expected_cases == 512 else 'engineering_smoke',
                  group=args.group, expected_cases=args.expected_cases, branches=list(BRANCHES), horizons=[10, 15, 20, 25],
                  expected_model_indices=expected, models=entries, bank_manifest_sha256=sha(bank / 'manifest.json'),
                  roots_report_sha256=sha(rd / 'report.json'), fibers_report_sha256=sha(fd / 'report.json'),
                  plan_sha256=sha(args.plan), protocol_sha256=protocol_sha, precision=precision,
                  source_sha256=sha(__file__), helper_sha256=sha(Path(__file__).with_name('feedback_suffix_rollout.py')),
                  utility_source_sha256=sha(Path(__file__).with_name('score_feedback_ranking.py')),
                  elapsed_seconds=time.monotonic() - started,
                  gpu=torch.cuda.get_device_name() if str(args.device).startswith('cuda') else 'cpu',
                  scope='Frozen one-decision suffix ranking; no candidate future outcomes or observations are read.')
    (out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
