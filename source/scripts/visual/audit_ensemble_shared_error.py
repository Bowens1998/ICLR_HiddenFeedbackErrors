"""Frozen fixed-pair diagnostic; labels evaluate policies, never select them."""
import argparse
import json
from pathlib import Path
import numpy as np
from analyze_coverage_goals import sha


def diagnose(scores, labels, actor):
    d = scores[..., 1] - scores[..., 0]
    y = labels[:, 0]
    np.testing.assert_allclose(y, labels[:, 3] - labels[:, 2])
    mean = d.mean(1)
    mse = ((d - y[:, None]) ** 2).mean(1)
    residual = (mean - y) ** 2
    disagreement = ((d - mean[:, None]) ** 2).mean(1)
    np.testing.assert_allclose(mse, residual + disagreement, rtol=1e-10, atol=1e-12)
    votes = d < 0
    choices = dict(actor_only=votes[:, actor // 2], ensemble_mean=mean < 0,
                   other_two=d[:, [m for m in range(3) if m != actor // 2]].mean(1) < 0,
                   always_random=np.zeros(len(y), bool), always_cem=np.ones(len(y), bool))
    policies = {}
    for name, choice in choices.items():
        regret = np.where(choice, np.maximum(y, 0), np.maximum(-y, 0))
        cost = np.where(choice, labels[:, 3], labels[:, 2])
        np.testing.assert_allclose(cost - np.minimum(labels[:, 2], labels[:, 3]), regret)
        policies[name] = dict(cost=float(cost.mean()), success=float(np.where(choice, labels[:, 5], labels[:, 4]).mean()),
                              regret=float(regret.mean()), cem_fraction=float(choice.mean()),
                              changes_from_all_three=int((choice != choices['ensemble_mean']).sum()))
    masks = dict(unanimous_cem=votes.all(1), unanimous_random=(~votes).all(1), mixed=votes.any(1) & (~votes).any(1))
    regret = np.where(choices['ensemble_mean'], np.maximum(y, 0), np.maximum(-y, 0))
    strata = {name: dict(cases=int(mask.sum()), mistakes=int(((regret > 0) & mask).sum()),
                        regret_sum=float(regret[mask].sum())) for name, mask in masks.items()}
    return dict(cases=len(y), member_mse=float(mse.mean()), ensemble_residual_mse=float(residual.mean()),
                disagreement=float(disagreement.mean()), policies=policies, strata=strata)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', default='.')
    p.add_argument('--output', required=True)
    a = p.parse_args(); root = Path(a.root)
    data = root / 'outputs/maintrack/planner_risk_diagnostic_v1'
    dr = json.loads((data / 'report.json').read_text())
    assert dr['data_sha256'] == sha(data / 'paired_data.npz')
    z = np.load(data / 'paired_data.npz')
    audit_path = root / 'outputs/maintrack/pusht_ensemble_numerical_audit.json'
    audit = json.loads(audit_path.read_text())
    assert audit['status'] == 'COMPLETE12_CANONICAL_NUMERICAL_AUDITS'
    groups = []; bindings = []
    for task, score_dir, interfaces in [
        ('reacher', 'reacher_ensemble_risk_full_v1', ['circle_encoded', 'state']),
        ('pusht', 'pusht_ensemble_canonical_full_v1', ['pose_encoded', 'state'])]:
        prior_path = root / f'outputs/maintrack/{task}_ensemble_risk_crossfit_v1/report.json'
        prior = json.loads(prior_path.read_text())
        assert prior['data_report_sha256'] == sha(data / 'report.json')
        if task == 'pusht':
            assert prior['numerical_audit_sha256'] == sha(audit_path)
        for interface in interfaces:
            rows = [r for r in dr['rows'] if r['task'] == task and r['interface'] == interface]
            assert len(rows) == 6
            per_actor = []; seeds = None
            for actor, row in enumerate(rows):
                folder = root / 'runs' / score_dir / f'actor_{actor}' / interface
                r = json.loads((folder / 'report.json').read_text())
                ac = json.loads((folder / 'acceptance.json').read_text())
                key = f"{task}_{row['provenance'][0]['route']}"
                seed = z[key + '_seeds']; labels = z[key + '_labels']; inputs = z[key + '_inputs']
                if seeds is None: seeds = seed
                else: np.testing.assert_array_equal(seeds, seed)
                assert ac['status'] == 'PASS' and ac['cases'] == len(seed)
                assert r['status'] == 'EXTRACTED_REQUIRES_ACCEPTANCE'
                assert ac['actor'] == r['actor'] == actor and ac['interface'] == r['interface'] == interface
                assert ac['report_sha256'] == sha(folder / 'report.json')
                assert ac['scores_sha256'] == r['scores_sha256'] == sha(folder / 'scores.npz')
                assert r['bindings'] == row['provenance']
                np.testing.assert_array_equal([v['seed'] for v in r['cases']], seed)
                q = np.load(folder / 'scores.npz')
                s = np.stack([q[f'case_{j}_scores'] for j in range(len(seed))]).astype(np.float64)
                assert s.shape == (len(seed), 3, 2) and np.isfinite(s).all() and (s >= 0).all()
                if task == 'reacher':
                    np.testing.assert_allclose(s[:, actor // 2], inputs[:, :2], rtol=2e-5, atol=2e-5)
                else:
                    assert r['numerical_policy'] == ac['numerical_policy'] == 'canonical_single_action_v1'
                    ar = next(v for v in audit['rows'] if v['actor'] == actor and v['interface'] == interface)
                    assert ar['report_sha256'] == ac['report_sha256'] and ar['acceptance_sha256'] == sha(folder / 'acceptance.json')
                    np.testing.assert_array_equal([v['original_scores'] for v in r['cases']], inputs[:, :2])
                per_actor.append(dict(actor=actor, **diagnose(s, labels, actor)))
                bindings.append(dict(task=task, interface=interface, actor=actor, report_sha256=sha(folder/'report.json'), acceptance_sha256=sha(folder/'acceptance.json')))
            g = dict(task=task, interface=interface, goals=len(seeds), actor_goal_pairs=6*len(seeds), per_actor=per_actor)
            for name in ['member_mse', 'ensemble_residual_mse', 'disagreement']:
                g[name] = float(np.mean([v[name] for v in per_actor]))
            g['ensemble_residual_fraction'] = g['ensemble_residual_mse'] / g['member_mse'] if g['member_mse'] else None
            g['policies'] = {name: {metric: float(np.mean([v['policies'][name][metric] for v in per_actor]))
                                      for metric in ['cost', 'success', 'regret', 'cem_fraction']}
                             for name in per_actor[0]['policies']}
            g['strata'] = {name: {metric: sum(v['strata'][name][metric] for v in per_actor)
                                   for metric in ['cases', 'mistakes', 'regret_sum']}
                          for name in per_actor[0]['strata']}
            total = sum(v['regret_sum'] for v in g['strata'].values())
            g['unanimous_regret_fraction'] = (total - g['strata']['mixed']['regret_sum']) / total if total else None
            old = next(v for v in prior['groups'] if v['interface'] == interface)
            for name in ['ensemble_mean', 'always_random', 'always_cem']:
                np.testing.assert_allclose(g['policies'][name]['cost'], old['costs'][name], atol=1e-12, rtol=0)
                np.testing.assert_allclose(g['policies'][name]['success'], old['successes'][name], atol=1e-12, rtol=0)
            g['prior_report_sha256'] = sha(prior_path)
            groups.append(g)
    out = Path(a.output); assert not out.exists()
    result = dict(status='COMPLETE4_FIXED_PAIR_SHARED_ERROR_AUDITS', groups=groups, bindings=bindings,
                  data_report_sha256=sha(data/'report.json'), numerical_audit_sha256=sha(audit_path),
                  protocol_sha256=sha(root/'docs/maintrack/ENSEMBLE_SHARED_ERROR_PROTOCOL.md'), source_sha256=sha(__file__),
                  scope='Consumed development goals and fixed trained actors; deterministic error identity, not population bias/variance or causal identification. PushT canonical scores. No fit or rule selection; fixed pairs, not ensemble search; additional three-model training and scoring budget.')
    out.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    for g in groups:
        print(g['task'], g['interface'], {k:g[k] for k in ['ensemble_residual_fraction','unanimous_regret_fraction','policies','strata']})


if __name__ == '__main__':
    main()
