"""Diagnose saved direct-state scores without changing search or acceptance."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from evaluation_precision import configure_evaluation_precision


def cost64(state, goal):
    angle = np.arctan2(state[..., 4], state[..., 5]) - np.arctan2(goal[4], goal[5])
    angle = (angle + np.pi) % (2 * np.pi) - np.pi
    return np.square(state[..., 2:4] - goal[2:4]).sum(-1) + 900 * angle**2


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    root = Path(a.run)
    summary = json.loads((root / 'summary.json').read_text())
    assert summary['score_space'] == 'state' and len(summary['cases']) == 128
    precision = configure_evaluation_precision()
    mean = np.array(summary['target_normalization']['mean'])
    std = np.array(summary['target_normalization']['std'])
    gm = torch.tensor(mean, device='cuda', dtype=torch.float32)
    gs = torch.tensor(std, device='cuda', dtype=torch.float32)
    artifacts = json.loads((root / 'artifact_manifest.json').read_text())
    rows = []
    for row in summary['cases']:
        index = row['index']
        name = f'case_{index:03d}_predictions.npz'
        path = root / name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == artifacts[name]
        with np.load(path) as z:
            tokens, goal, saved = z['tokens'], z['goal_tokens'], z['costs']
            full64 = cost64(tokens.astype(float) * std + mean, goal.astype(float) * std + mean)
            physical = tokens * std.astype('float32') + mean.astype('float32')
            physical_goal = goal * std.astype('float32') + mean.astype('float32')
            staged64 = cost64(physical.astype(float), physical_goal.astype(float))
            replay = []
            with torch.inference_mode():
                pg = torch.tensor(goal, device='cuda') * gs + gm
                np.testing.assert_array_equal(pg.cpu().numpy(), physical_goal)
                for t in range(30):
                    state = torch.tensor(tokens[t], device='cuda') * gs + gm
                    np.testing.assert_array_equal(state.cpu().numpy(), physical[t])
                    angle = torch.atan2(state[:, 4], state[:, 5]) - torch.atan2(pg[4], pg[5])
                    angle = torch.atan2(angle.sin(), angle.cos())
                    replay.append(((state[:, 2:4] - pg[2:4]).square().sum(1) + 900 * angle.square()).cpu().numpy())
            replay = np.stack(replay)
            mask = np.abs(full64 - saved) > .002 + 5e-5 * np.abs(saved)
            violations = []
            for t, c in np.argwhere(mask):
                violations.append(dict(iteration=int(t), candidate=int(c), saved=float(saved[t,c]), full64=float(full64[t,c]), staged64=float(staged64[t,c]), radius=float(np.linalg.norm(physical[t,c,4:6])), goal_radius=float(np.linalg.norm(physical_goal[4:6]))))
            rows.append(dict(case=index, full64_violations=violations,
                staged64_pass=bool(np.allclose(staged64,saved,rtol=5e-5,atol=.002)),
                gpu_pass=bool(np.allclose(replay,saved,rtol=5e-5,atol=.002)),
                gpu_bitwise=bool(np.array_equal(replay,saved)),
                gpu_max_difference=float(np.abs(replay-saved).max()),
                full64_argmin_same=bool(np.argmin(full64)==np.argmin(saved)),
                full64_elite_set_changed_rounds=sum(set(np.argsort(full64[t],kind='stable')[:30]) != set(np.argsort(saved[t],kind='stable')[:30]) for t in range(30))))
    output = dict(route=summary['route_index'], arm=summary['arm'], checkpoint=summary['checkpoint'],
        summary_sha256=hashlib.sha256((root/'summary.json').read_bytes()).hexdigest(),
        verifier_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), precision=precision,
        cases=rows, scope='diagnostic only; fixed saved populations, no counterfactual CEM or physical replay; original acceptance unchanged')
    Path(a.output).parent.mkdir(parents=True,exist_ok=True)
    Path(a.output).write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps({k:output[k] for k in ['route','arm','checkpoint']}))
    print('cases',len(rows),'full64 violations',sum(len(r['full64_violations']) for r in rows),'GPU bitwise',sum(r['gpu_bitwise'] for r in rows),'staged pass',sum(r['staged64_pass'] for r in rows))


if __name__ == '__main__':
    main()
