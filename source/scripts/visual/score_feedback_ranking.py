"""Score a fixed future-action pool after an already executed common prefix.

Reuses accepted, norm-matched confirmation corrections for the same physical
prefix. Reads scorer_inputs only: simulator outcome files are not opened here.
"""
import argparse
import hashlib
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
from nonlinear_pose_cost import numpy_pose, numpy_cost

BRANCHES = ('free', 'act', 'don', 'reset')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def state_hash(model):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def normalized_readout(head, tokens):
    value = (np.asarray(tokens, dtype=float) - head['mean']) / head['scale']
    for layer in (0, 2, 4):
        value = value @ head[f'{layer}.weight'].astype(float).T + head[f'{layer}.bias'].astype(float)
        if layer != 4:
            value = np.maximum(value, 0)
    return value


def region_margins(head, original, replacement):
    x = (original.astype(float) - head['mean']) / head['scale']
    d = (replacement.astype(float) - original) / head['scale']
    w0, w1 = [head[f'{k}.weight'].astype(float) for k in (0, 2)]
    aa = w0 @ x + head['0.bias']
    bb = w1 @ np.maximum(aa, 0) + head['2.bias']
    second = w1 @ ((aa >= 0)[:, None] * w0)
    signs = np.r_[np.where(aa >= 0, 1., -1.), np.where(bb >= 0, 1., -1.)]
    return signs * (np.r_[aa, bb] + np.r_[w0, second] @ d)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('plan', 'bank', 'fibers', 'horizon', 'official', 'config', 'protocol', 'output'):
        parser.add_argument('--' + key, required=True)
    parser.add_argument('--group', type=int, choices=range(6), required=True)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    start = time.monotonic()
    bank, out = Path(args.bank), Path(args.output)
    manifest = json.loads((bank / 'manifest.json').read_text())
    plan = json.loads(Path(args.plan).read_text())
    assert manifest['group'] == args.group
    assert manifest['reference_route'] == 16 * args.group
    assert manifest['bindings']['plan_sha256'] == sha(args.plan)
    bank_acceptance = json.loads((bank / 'acceptance.json').read_text())
    assert bank_acceptance['status'] == 'PASS_SHARED_PREFIX_SIMULATOR_BANK'
    assert bank_acceptance['manifest_sha256'] == sha(bank / 'manifest.json')
    fd, hd = Path(args.fibers) / f'job_{args.group}', Path(args.horizon) / f'job_{args.group}'
    fr, hr = [json.loads((d / 'report.json').read_text()) for d in (fd, hd)]
    fa, ha = [json.loads((d / 'acceptance.json').read_text()) for d in (fd, hd)]
    assert fa['status'] == 'PASS1536_PAIRED_CONFIRMATION_FIBER_INPUTS'
    assert ha['status'] == 'PASS_ALL8_HORIZON_NUMPY_RECONSTRUCTIONS'
    assert fa['report_sha256'] == sha(fd / 'report.json')
    assert ha['report_sha256'] == sha(hd / 'report.json')
    assert hr['plan_sha256'] == sha(args.plan)
    reference = hr['reference_bindings'][0]
    assert reference['route'] == manifest['reference_route']
    assert reference['summary_sha256'] == manifest['bindings']['source_summary_sha256']
    assert reference['acceptance_sha256'] == manifest['bindings']['source_acceptance_sha256']
    out.mkdir(parents=True, exist_ok=False)
    precision = configure_evaluation_precision()
    torch.set_num_threads(4)
    entries = []
    for slot in (2, 3, 4):
        mi = 8 * args.group + slot
        entry, original = plan['models'][mi], plan['models'][8 * args.group]
        td, od = Path(entry['training_path']), Path(original['training_path'])
        tr = json.loads((td / 'summary.json').read_text())
        assert sha(td / 'summary.json') == entry['training_summary_sha256']
        assert sha(td / 'last_weights.pt') == entry['weights_sha256']
        assert sha(od / 'last_weights.pt') == original['weights_sha256']
        assert sha(args.config) == tr['config_sha256']
        model = make_model(args.official, args.config, entry['arm'], tr['seed'])
        model.load_state_dict(torch.load(od / 'last_weights.pt', map_location='cpu', weights_only=True))
        model = model.to(args.device)
        boundary = configure_dynamics_only(model)
        model.load_state_dict(torch.load(td / 'last_weights.pt', map_location='cpu', weights_only=True))
        verify_frozen(model, boundary)
        initial_state_hash = state_hash(model)
        hrow = next(r for r in hr['rows'] if r['model_index'] == mi)
        binding = next(r for r in fr['bindings'] if r.get('model_index') == mi)
        assert binding['input_sha256'] == hrow['sha256'] == sha(hd / hrow['file'])
        assert binding['output_sha256'] == sha(fd / binding['output_file'])
        assert binding['head_sha256'] == hrow['head_sha256'] == entry['endpoint_head']['sha256']
        assert entry['endpoint_head'] == entry['goal_head']
        hp = hd / hrow['head_file']
        assert sha(hp) == hrow['head_sha256']
        head = dict(np.load(hp))
        fibers, horizons = dict(np.load(fd / binding['output_file'])), dict(np.load(hd / hrow['file']))
        assert int(horizons['reference_routes'][0]) == manifest['reference_route']
        model_dir = out / f'model_{mi}'
        model_dir.mkdir()
        shutil.copyfile(hp, model_dir / 'head.npz')
        cases = []
        with torch.inference_mode():
            for case in manifest['cases']:
                index = case['index']
                inp = bank / 'scorer_inputs' / f'case_{index:03d}.npz'
                assert sha(inp) == case['input_sha256']
                z = dict(np.load(inp))
                assert int(z['seed']) == case['seed'] == int(horizons['seeds'][index])
                np.testing.assert_array_equal(fibers['predicted'][index], horizons['free_tokens'][0, index, 0])
                np.testing.assert_array_equal(fibers['observed'][index], horizons['observed_tokens'][0, index, 0])
                interface = ImagePlannerCost(model, z['history_pixels'], z['goal_pixels'], z['prefix'],
                                             tr['normalization'], entry['target_normalization'], 'latent')
                # Keep the original 300x7 action-encoding arithmetic, without
                # evaluating any candidate future before the shared root token.
                population = torch.as_tensor(z['source_population_actions'], device=args.device)
                encoded = model.action_encoder(interface.normalized_actions(population))
                history = interface.initial[None].expand(300, -1, -1).clone()
                root = model.predict(history, encoded[:, :3])[:, -1]
                selected = int(z['source_selected_index'])
                np.testing.assert_array_equal(population[selected, :5].cpu().numpy(), z['executed_actions'])
                np.testing.assert_array_equal(root[selected].cpu().numpy(), fibers['predicted'][index])
                im = torch.as_tensor(z['current_pixels'], device=args.device).permute(2, 0, 1).float() / 255
                mean = torch.tensor([.485, .456, .406], device=args.device)[:, None, None]
                std = torch.tensor([.229, .224, .225], device=args.device)[:, None, None]
                observed = model.encode({'pixels': ((im - mean) / std)[None, None]})['emb'][0, 0]
                np.testing.assert_allclose(observed.cpu().numpy(), fibers['observed'][index], rtol=2e-5, atol=2e-5)
                np.testing.assert_array_equal(interface.goal.cpu().numpy(), horizons['goal_tokens'][index])
                initial = np.stack([fibers[k][index] for k in ('predicted', 'constrained', 'shuffled', 'observed')])
                err = float(np.max(np.abs(normalized_readout(head, initial[:3]) - normalized_readout(head, initial[0]))))
                assert err <= 1e-6
                norms = np.linalg.norm((initial[1:3].astype(float) - initial[0]) / head['scale'], axis=-1)
                np.testing.assert_allclose(norms, fibers['target_norm'][index], rtol=1e-6, atol=1e-6)
                for replacement in initial[1:3]:
                    assert np.min(region_margins(head, initial[0], replacement)) >= -1e-6
                suffixes = torch.as_tensor(z['suffix_actions'], device=args.device)
                past = torch.as_tensor(np.concatenate([z['prefix'][-5:], z['executed_actions']]), device=args.device)
                am, ast = [torch.tensor(tr['normalization'][k], device=args.device) for k in ('mean', 'std')]
                values = []
                for replacement in initial:
                    values.append(rollout_suffixes(model, interface.initial,
                        torch.as_tensor(replacement, device=args.device), past, suffixes, am, ast).cpu().numpy())
                tokens = np.stack(values)
                assert tokens.shape[:3] == (4, 4, 32) and np.isfinite(tokens).all()
                pose = numpy_pose(tokens, head)
                goal_token = interface.goal.cpu().numpy()
                goal_pose = numpy_pose(goal_token, head)
                pose_cost = numpy_cost(pose[:, -1], goal_pose)
                latent_cost = np.square(tokens[:, -1].astype(float) - goal_token).sum(-1)
                output = model_dir / f'case_{index:03d}.npz'
                np.savez_compressed(output, initial_tokens=initial, tokens=tokens, pose=pose,
                    goal_token=goal_token, goal_pose=goal_pose, pose_cost=pose_cost, latent_cost=latent_cost,
                    branches=np.array(BRANCHES), horizons=np.array([10, 15, 20, 25]))
                cases.append(dict(index=index, seed=case['seed'], file=str(output.relative_to(out)),
                    sha256=sha(output), input_sha256=sha(inp), max_readout_error=err,
                    standardized_displacement_norms=norms.tolist()))
                print('SCORED', mi, index, flush=True)
        verify_frozen(model, boundary)
        assert state_hash(model) == initial_state_hash
        entries.append(dict(model_index=mi, objective=entry['adaptation_condition'], cases=cases,
            head_file=str((model_dir / 'head.npz').relative_to(out)), head_sha256=sha(hp),
            model_weights_sha256=entry['weights_sha256'], state_hash=initial_state_hash,
            fiber_sha256=binding['output_sha256'], horizon_sha256=hrow['sha256']))
        del model, boundary, interface
        if str(args.device).startswith('cuda'):
            torch.cuda.empty_cache()
    report = dict(status='DEVELOPMENT_SCORES_REQUIRE_OUTCOME_ACCEPTANCE', group=args.group,
        branches=list(BRANCHES), horizons=[10, 15, 20, 25], phase='development',
        expected_model_indices=[8 * args.group + s for s in (2, 3, 4)],
        models=entries, bank_manifest_sha256=sha(bank / 'manifest.json'),
        plan_sha256=sha(args.plan), protocol_sha256=sha(args.protocol), precision=precision,
        elapsed_seconds=time.monotonic() - start, source_sha256=sha(__file__),
        helper_sha256=sha(Path(__file__).with_name('feedback_suffix_rollout.py')),
        gpu=torch.cuda.get_device_name() if str(args.device).startswith('cuda') else 'cpu',
        scope='One already-observed common prefix; fixed newly defined suffix pool; development only.')
    (out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
