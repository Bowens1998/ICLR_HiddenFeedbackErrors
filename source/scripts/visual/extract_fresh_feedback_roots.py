"""Encode a fresh decision-time feedback interface and fixed external donors."""
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
from image_planner_cost import ImagePlannerCost
from score_feedback_ranking import sha, state_hash


DONOR_PERMUTATION_SEED = 1368001


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('plan', 'bank', 'donor-horizon', 'official', 'config', 'protocol', 'output'):
        parser.add_argument('--' + key, required=True)
    parser.add_argument('--group', type=int, choices=range(6), required=True)
    parser.add_argument('--expected-cases', type=int, default=512)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    started = time.monotonic()
    bank, out = Path(args.bank), Path(args.output)
    manifest = json.loads((bank / 'manifest.json').read_text())
    acceptance = json.loads((bank / 'acceptance.json').read_text())
    plan = json.loads(Path(args.plan).read_text())
    assert manifest['group'] == args.group and manifest['reference_route'] == 16 * args.group
    assert manifest['bindings']['plan_sha256'] == sha(args.plan)
    assert acceptance['status'] == 'PASS_SHARED_PREFIX_SIMULATOR_BANK'
    assert acceptance['manifest_sha256'] == sha(bank / 'manifest.json')
    assert len(manifest['cases']) == acceptance['cases'] == args.expected_cases > 0
    assert [c['index'] for c in manifest['cases']] == list(range(args.expected_cases))
    assert len({c['seed'] for c in manifest['cases']}) == args.expected_cases
    protocol_sha = sha(args.protocol)
    donor_dir = Path(args.donor_horizon) / f'job_{args.group}'
    donor_report = json.loads((donor_dir / 'report.json').read_text())
    donor_acceptance = json.loads((donor_dir / 'acceptance.json').read_text())
    assert donor_acceptance['status'] == 'PASS_ALL8_HORIZON_NUMPY_RECONSTRUCTIONS'
    assert donor_acceptance['report_sha256'] == sha(donor_dir / 'report.json')
    assert donor_report['index'] == args.group
    # The previously used external donor pool is fixed, separate from both
    # development recipients and the new confirmation recipients.
    assert donor_report['plan_sha256'] == '1fd34ed8b462afc5901b6d82c9ef119ff689424be03803b92882064a5559c79b'
    permutation = np.random.default_rng(DONOR_PERMUTATION_SEED).permutation(128)
    donor_indices = np.tile(permutation, (args.expected_cases + 127) // 128)[:args.expected_cases]
    configure_evaluation_precision()
    torch.set_num_threads(4)
    out.mkdir(parents=True, exist_ok=False)
    models, shared_encodings = [], None
    cases = [dict(index=c['index'], seed=c['seed'], input_sha256=c['input_sha256']) for c in manifest['cases']]
    for slot in (2, 3, 4):
        mi = 8 * args.group + slot
        entry, original = plan['models'][mi], plan['models'][8 * args.group]
        td, od = Path(entry['training_path']), Path(original['training_path'])
        tr = json.loads((td / 'summary.json').read_text())
        assert sha(td / 'summary.json') == entry['training_summary_sha256']
        assert sha(td / 'last_weights.pt') == entry['weights_sha256']
        assert sha(od / 'last_weights.pt') == original['weights_sha256']
        assert sha(args.config) == tr['config_sha256']
        assert entry['endpoint_head'] == entry['goal_head']
        hp = Path(entry['endpoint_head']['path'])
        assert sha(hp) == entry['endpoint_head']['sha256']
        model = make_model(args.official, args.config, entry['arm'], tr['seed'])
        model.load_state_dict(torch.load(od / 'last_weights.pt', map_location='cpu', weights_only=True))
        model = model.to(args.device)
        boundary = configure_dynamics_only(model)
        model.load_state_dict(torch.load(td / 'last_weights.pt', map_location='cpu', weights_only=True))
        verify_frozen(model, boundary)
        before = state_hash(model)
        drow = next(row for row in donor_report['rows'] if row['model_index'] == mi)
        assert drow['head_sha256'] == sha(hp)
        assert drow['sha256'] == sha(donor_dir / drow['file'])
        with np.load(donor_dir / drow['file'], allow_pickle=False) as archive:
            donor = {key: archive[key].copy() for key in ('observed_tokens', 'seeds', 'reference_routes')}
        assert donor['observed_tokens'].shape == (4, 128, 5, 192)
        assert len(set(donor['seeds'].tolist())) == 128
        assert not set(donor['seeds'].tolist()) & {c['seed'] for c in cases}
        assert int(donor['reference_routes'][0]) == manifest['reference_route']
        values = {key: [] for key in ('predicted', 'observed', 'initial_history', 'goal_tokens')}
        with torch.inference_mode():
            for case in manifest['cases']:
                path = bank / case['input_file']
                assert sha(path) == case['input_sha256']
                z = dict(np.load(path, allow_pickle=False))
                assert int(z['index']) == case['index'] and int(z['seed']) == case['seed']
                assert not {'terminal_states', 'goal_state', 'suffix_states', 'current_state'} & set(z)
                interface = ImagePlannerCost(model, z['history_pixels'], z['goal_pixels'], z['prefix'],
                                             tr['normalization'], entry['target_normalization'], 'latent')
                population = torch.as_tensor(z['source_population_actions'], device=args.device)
                assert population.shape == (300, 25, 2)
                encoded = model.action_encoder(interface.normalized_actions(population))
                history = interface.initial[None].expand(300, -1, -1).clone()
                roots = model.predict(history, encoded[:, :3])[:, -1]
                repeated = model.predict(history, encoded[:, :3])[:, -1]
                np.testing.assert_array_equal(roots.cpu().numpy(), repeated.cpu().numpy())
                selected = int(z['source_selected_index'])
                np.testing.assert_array_equal(population[selected, :5].cpu().numpy(), z['executed_actions'])
                pixels = torch.as_tensor(z['current_pixels'], device=args.device).permute(2, 0, 1).float() / 255
                mean = torch.tensor([.485, .456, .406], device=args.device)[:, None, None]
                std = torch.tensor([.229, .224, .225], device=args.device)[:, None, None]
                observed = model.encode({'pixels': ((pixels - mean) / std)[None, None]})['emb'][0, 0]
                values['predicted'].append(roots[selected].cpu().numpy())
                values['observed'].append(observed.cpu().numpy())
                values['initial_history'].append(interface.initial.cpu().numpy())
                values['goal_tokens'].append(interface.goal.cpu().numpy())
        arrays = {key: np.asarray(value) for key, value in values.items()}
        arrays.update(donor=donor['observed_tokens'][0, donor_indices, 0],
                      donor_indices=donor_indices, donor_seeds=donor['seeds'][donor_indices],
                      seeds=np.asarray([case['seed'] for case in cases]))
        assert all(np.isfinite(value).all() for value in arrays.values())
        assert arrays['predicted'].shape == arrays['observed'].shape == arrays['donor'].shape == (args.expected_cases, 192)
        assert arrays['initial_history'].shape == (args.expected_cases, 3, 192)
        shared = {key: arrays[key] for key in ('observed', 'initial_history', 'goal_tokens', 'donor')}
        if shared_encodings is None:
            shared_encodings = shared
        else:
            for key in shared:
                np.testing.assert_array_equal(shared[key], shared_encodings[key])
        verify_frozen(model, boundary)
        assert state_hash(model) == before
        file, head_file = f'model_{mi}.npz', f'head_{mi}.npz'
        np.savez_compressed(out / file, **arrays)
        shutil.copyfile(hp, out / head_file)
        models.append(dict(model_index=mi, objective=entry['adaptation_condition'], file=file,
                           sha256=sha(out / file), head_file=head_file, head_sha256=sha(out / head_file),
                           model_weights_sha256=entry['weights_sha256'], initial_state_hash=before,
                           donor_file_sha256=drow['sha256'], exact_root_replay_cases=args.expected_cases))
        print('ROOTS_READY', mi, args.expected_cases, flush=True)
        del model, boundary, interface
        if str(args.device).startswith('cuda'):
            torch.cuda.empty_cache()
    assert protocol_sha == sha(args.protocol)
    report = dict(status='FRESH_FEEDBACK_ROOTS_READY', group=args.group, case_count=args.expected_cases,
                  expected_cases=args.expected_cases, models=models, cases=cases,
                  plan_sha256=sha(args.plan), protocol_sha256=protocol_sha,
                  bank_manifest_sha256=sha(bank / 'manifest.json'), bank_acceptance_sha256=sha(bank / 'acceptance.json'),
                  donor_report_sha256=sha(donor_dir / 'report.json'), donor_acceptance_sha256=sha(donor_dir / 'acceptance.json'),
                  donor_assignment=dict(permutation_seed=DONOR_PERMUTATION_SEED, size=128,
                                        rule='Repeat the fixed permutation cyclically; index shared across objectives and groups.'),
                  source_sha256=sha(__file__), sources={name: sha(Path(__file__).with_name(name)) for name in
                      ('score_feedback_ranking.py', 'image_planner_cost.py', 'factorial_model.py', 'adaptation_freeze.py', 'evaluation_precision.py')},
                  config_sha256=sha(args.config), elapsed_seconds=time.monotonic() - started,
                  gpu=torch.cuda.get_device_name() if str(args.device).startswith('cuda') else 'cpu',
                  scope='Decision-time root extraction only. No candidate future observations or outcomes are read.')
    (out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
