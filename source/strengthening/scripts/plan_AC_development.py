"""Unchanged original-policy searches, with compact recoverable development outputs."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'scripts/visual'), str(ROOT/'strengthening/adapters')]
from contracts import sha, atomic_json, require_role
from artifact_io import atomic_npz, array_sha
from controlled_search import search, expand_actions
from factorial_model import make_model
from image_planner_cost import ImagePlannerCost
from nonlinear_pose_cost import NonlinearPoseCost
from evaluation_precision import configure_evaluation_precision
from score_feedback_ranking import state_hash
from input_lock import add_design_arguments,input_context,check_parent_design


def main():
    p = argparse.ArgumentParser()
    for key in ['base', 'bank', 'output']:
        p.add_argument('--'+key, required=True)
    p.add_argument('--index', type=int)
    add_design_arguments(p)
    a = p.parse_args()
    index = int(os.environ['SLURM_ARRAY_TASK_ID']) if a.index is None else a.index
    spec = ROOT/'strengthening/configs/AC_reference_policies.json'
    policies = json.loads(spec.read_text()); row = policies['rows'][index]
    e = row['entry']; route = row['route']; base = Path(a.base); bank = Path(a.bank)
    assert row['index'] == index and e['adaptation_condition'] == 'original'
    role = require_role(json.loads((bank/'role.json').read_text()),
                        ['diagnostic_development_A_C', 'donor_development_A_C','confirmation_A_C','donor_bank_A_C'])
    settings,phase_binding=input_context(a,role['role'],ROOT,__file__)
    check_parent_design(role,phase_binding);count=settings['count']
    assert (bank/'DONE').exists() and role['parent_manifest_sha256'] == sha(bank/'manifest.json')
    assert role['acceptance_sha256'] == sha(bank/'acceptance.json')
    manifest = json.loads((bank/'manifest.json').read_text())
    assert len(manifest['cases']) == role['count'] == count
    assert [c['index'] for c in manifest['cases']]==list(range(count))
    td = Path(e['training_path']); tr = json.loads((td/'summary.json').read_text())
    config = base/'assets/pusht-v1/models/config.json'
    assert sha(td/'summary.json') == e['training_summary_sha256']
    assert sha(td/'last_weights.pt') == e['weights_sha256'] and sha(config) == tr['config_sha256']
    precision = configure_evaluation_precision(); torch.set_num_threads(4)
    model = make_model(base/'releases/visual-v1/official', config, e['arm'], tr['seed'])
    model.load_state_dict(torch.load(td/'last_weights.pt', map_location='cpu', weights_only=True), strict=True)
    model = model.cuda().eval(); before = state_hash(model)
    heads = {}
    if e['score'].startswith('pose_'):
        for key in ['endpoint_head', 'goal_head']:
            assert sha(e[key]['path']) == e[key]['sha256']
            heads[key] = dict(np.load(e[key]['path']))
    out = Path(a.output)/f'route_{index}'; out.mkdir(parents=True, exist_ok=True)
    binding = dict(**phase_binding,role=role['role'], bank_manifest_sha256=sha(bank/'manifest.json'),
                   bank_role_sha256=sha(bank/'role.json'), reference_policies_sha256=sha(spec),
                   source_sha256=sha(__file__), row=row,
                   implementation_sha256={name:sha(ROOT/'scripts/visual'/name) for name in
                       ['controlled_search.py','image_planner_cost.py','nonlinear_pose_cost.py','factorial_model.py','evaluation_precision.py']})
    if (out/'binding.json').exists():
        assert json.loads((out/'binding.json').read_text()) == binding
    else:
        atomic_json(out/'binding.json', binding)
    started = time.monotonic(); rows = []
    with torch.inference_mode():
        for item in manifest['cases']:
            i = item['index']; path = bank/f'case_{i:03d}.npz'
            assert sha(path) == item['sha256']
            result = out/f'case_{i:03d}.json'; target = out/f'case_{i:03d}.npz'
            if result.exists():
                saved = json.loads(result.read_text())
                assert saved['binding_sha256'] == sha(out/'binding.json') and saved['input_sha256'] == sha(path)
                assert saved['file_sha256'] == sha(target)
                rows.append(saved); continue
            z = dict(np.load(path)); seed = int(z['seed'])
            cost = ImagePlannerCost(model, z['history_pixels'], z['goal_pixels'], z['prefix'],
                                    tr['normalization'], e['target_normalization'],
                                    'state' if e['score']=='state' else 'latent')
            if heads:
                cost = NonlinearPoseCost(cost, heads['endpoint_head'], heads['goal_head'])
            if i == 0:
                probe = torch.zeros(2,25,2,device='cuda'); probe[1] = .1
                cost.verify_native(probe)
            torch_seed = int(np.random.SeedSequence([seed,955001]).generate_state(1)[0])
            torch.cuda.synchronize(); start = time.monotonic()
            best, trace, mean = search(cost, algorithm=route['algorithm'], parameterization=route['parameterization'],
                                       device='cuda', seed=torch_seed)
            torch.cuda.synchronize(); elapsed = time.monotonic()-start
            pop = trace[best['iteration']]
            actions = expand_actions(torch.as_tensor(pop['parameters'], device='cuda'))
            pcost, tokens = cost(actions)
            np.testing.assert_array_equal(tokens.cpu().numpy(), pop['tokens'])
            np.testing.assert_array_equal(pcost.cpu().numpy(), pop['costs'])
            j = best['candidate']; selected = actions[j].cpu().numpy()
            np.testing.assert_array_equal(selected, expand_actions(best['parameters'][None])[0].cpu().numpy())
            # Keep every scalar cost and deterministic search seed, but no 9000-token prediction archive.
            cost_trace = np.stack([x['costs'] for x in trace])
            assert np.unravel_index(np.argmin(cost_trace), cost_trace.shape) == (best['iteration'], j)
            atomic_npz(target, population_actions=actions.cpu().numpy(), selected_actions=selected,
                selected_token=tokens[j].cpu().numpy(), population_costs=pcost.cpu().numpy(), cost_trace=cost_trace,
                distribution_mean=np.stack([x['mean'] for x in trace]), distribution_std=np.stack([x['std'] for x in trace]),
                final_distribution_mean=mean.cpu().numpy(), selected_index=np.asarray(j),
                selected_iteration=np.asarray(best['iteration']), seed=np.asarray(seed))
            saved = dict(index=i, seed=seed, torch_seed=torch_seed, selected_iteration=best['iteration'],
                selected_index=j, population_replay_exact=True, scored_candidates=9000,
                file=target.name, file_sha256=sha(target), input_sha256=sha(path),
                binding_sha256=sha(out/'binding.json'), planning_seconds=elapsed,
                full_trace_hashes=[{key:array_sha(x[key]) for key in ['parameters','costs','tokens','mean','std']} for x in trace])
            atomic_json(result,saved); rows.append(saved)
            print(json.dumps(dict(index=i,route=index,seconds=elapsed,accepted=True)),flush=True)
    assert len(rows) == count and state_hash(model) == before
    atomic_json(out/'report.json',dict(status='PASS_COMPACT_FIXED_REFERENCE_SEARCH',binding=binding,
        binding_sha256=sha(out/'binding.json'),cases=rows,precision=precision,gpu=torch.cuda.get_device_name(),
        elapsed_seconds=time.monotonic()-started,frozen_tensors_unchanged=True,
        scope='Input construction using original reference policies. Exact original search and selected-population arithmetic; physical replay is a separate acceptance stage. No new study model evaluated.'))
    (out/'DONE').write_text('source_actions_verified\n')


if __name__ == '__main__':
    main()
