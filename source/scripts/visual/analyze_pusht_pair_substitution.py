"""Frozen privileged substitutions on fixed selected actions, never replanning."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import numpy as np
from analyze_pose_selected_endpoints import error, sha


def main():
    p = argparse.ArgumentParser(); p.add_argument('--output', required=True)
    a = p.parse_args(); out = Path(a.output); assert not out.exists()
    folder = Path('runs/pusht_pose_selected_endpoints_v1')
    summary = Path('outputs/maintrack/pusht_pose_planning_summary.json')
    validator = Path(__file__).with_name('analyze_pose_selected_endpoints.py')
    with tempfile.TemporaryDirectory() as td:
        validation = Path(td)/'validation.json'
        subprocess.run([sys.executable, str(validator), '--input', str(folder),
                        '--summary', str(summary), '--plan', 'runs/hpg/pusht_nonlinear_transfer_v1/planning_plan.json',
                        '--bank', 'data/pusht_prospective_v1/heldout', '--output', str(validation)], check=True)
        vr = json.loads(validation.read_text()); assert vr['status'] == 'COMPLETE_POSTHOC_DIAGNOSTIC'
        validation_sha = sha(validation)
    full = json.loads(summary.read_text()); actors = []; bindings = []
    for actor in range(6):
        d = folder/f'job_{actor}'; report = json.loads((d/'report.json').read_text())
        bindings.append(dict(actor=actor, report_sha256=sha(d/'report.json')))
        for offset, interface in [(2, 'pose_encoded'), (4, 'pose_predicted'), (6, 'state')]:
            routes = [actor*8+offset, actor*8+offset+1]; z = []
            for route in routes:
                entry = next(r for r in report['rows'] if r['route_index'] == route)
                assert entry['score'] == interface
                z.append(dict(np.load(d/entry['file'])))
            for key in ['seed', 'estimated_goal', 'true_goal']:
                np.testing.assert_array_equal(z[0][key], z[1][key])
            scores = {}
            for name, ep, goal in [('uncorrected', 'estimated_endpoint', 'estimated_goal'),
                                   ('true_goal', 'estimated_endpoint', 'true_goal'),
                                   ('true_endpoint', 'true_endpoint', 'estimated_goal'),
                                   ('pair_oracle', 'true_endpoint', 'true_goal')]:
                scores[name] = np.stack([error(v[ep], v[goal])[0].sum(1) for v in z], 1)
            actual = np.stack([full['rows'][r]['cost'] for r in routes], 1)
            successes = np.stack([full['rows'][r]['success'] for r in routes], 1)
            np.testing.assert_allclose(scores['pair_oracle'], actual, rtol=1e-9, atol=1e-7)
            choices = {name: value[:, 1] < value[:, 0] for name, value in scores.items()}
            choices.update(always_random=np.zeros(len(actual), bool), always_cem=np.ones(len(actual), bool))
            metrics = {}
            for name, choice in choices.items():
                cost = actual[np.arange(len(actual)), choice.astype(int)]
                if name == 'pair_oracle': np.testing.assert_allclose(cost, actual.min(1), rtol=1e-9, atol=1e-7)
                metrics[name] = dict(cost=float(cost.mean()), success=float(successes[np.arange(len(actual)), choice.astype(int)].mean()),
                                     regret=float((cost-actual.min(1)).mean()), cem_fraction=float(choice.mean()),
                                     changes_from_uncorrected=int((choice != choices['uncorrected']).sum()))
            actors.append(dict(actor=actor, interface=interface, goals=len(actual), policies=metrics))
    groups = []
    for interface in ['pose_encoded', 'pose_predicted', 'state']:
        rows = [r for r in actors if r['interface'] == interface]; assert len(rows) == 6
        policies = {name: {metric:float(np.mean([r['policies'][name][metric] for r in rows]))
                           for metric in ['cost', 'success', 'regret', 'cem_fraction']}
                    for name in rows[0]['policies']}
        groups.append(dict(interface=interface, policies=policies, per_actor=rows))
    result = dict(status='COMPLETE18_FIXED_PAIR_SUBSTITUTION_GROUPS', groups=groups, bindings=bindings,
                  summary_sha256=sha(summary), validation_sha256=validation_sha, validator_sha256=sha(validator),
                  source_sha256=sha(__file__), protocol_sha256=sha('docs/maintrack/PUSHT_PAIR_SUBSTITUTION_PROTOCOL.md'),
                  scope='Privileged scoring substitutions on fixed random/CEM actions; consumed development data. No replanning, deployable method, or causal attribution of model training failure. Original FP64 pose reconstruction, not canonical ensemble scores.')
    out.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    for g in groups: print(g['interface'], g['policies'])


if __name__ == '__main__': main()
