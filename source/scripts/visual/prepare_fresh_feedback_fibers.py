"""Construct six-way norm-matched feedback interventions for fresh recipients."""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from readout_fiber import geometry, project, readout


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def region_margin(head, original, replacement):
    _, aa, bb, m0, m1, w0, second, _ = geometry(head, original)
    displacement = (replacement.astype(float) - original.astype(float)) / head['scale']
    signs = np.r_[np.where(m0, 1., -1.), np.where(m1, 1., -1.)]
    return signs * (np.r_[aa, bb] + np.r_[w0, second] @ displacement)


def match_six_directions(entries):
    """Scale each source/objective displacement to the per-case common minimum."""
    assert len(entries) == 3
    full_norms = []
    for head, roots, actual, donor in entries:
        directions = np.stack([actual.astype(float) - roots['predicted'], donor.astype(float) - roots['predicted']], axis=1)
        full_norms.append(np.linalg.norm(directions / head['scale'], axis=-1))
    common = np.min(np.stack(full_norms), axis=(0, 2))
    output = []
    for (head, roots, actual, donor), norms in zip(entries, full_norms):
        directions = np.stack([actual.astype(float) - roots['predicted'], donor.astype(float) - roots['predicted']], axis=1)
        alpha = np.divide(common[:, None], norms, out=np.zeros_like(norms), where=norms > 0)
        replacements = (roots['predicted'].astype(float)[:, None] + alpha[:, :, None] * directions).astype(np.float32)
        realized_norms = np.linalg.norm((replacements.astype(float) - roots['predicted'][:, None]) / head['scale'], axis=-1)
        np.testing.assert_allclose(realized_norms, np.broadcast_to(common[:, None], realized_norms.shape), rtol=1e-6, atol=1e-6)
        error = float(np.max(abs(readout(head, replacements) - readout(head, roots['predicted'])[:, None])))
        assert error <= 1e-6
        margin = min(float(np.min(region_margin(head, pred, replacement)))
                     for pred, pair in zip(roots['predicted'], replacements) for replacement in pair)
        assert margin >= -1e-6
        arrays = dict(predicted=roots['predicted'], observed=roots['observed'], donor=roots['donor'],
                      constrained=replacements[:, 0], shuffled=replacements[:, 1], full=actual,
                      shuffled_full=donor, target_norm=common, displacement_norms=realized_norms,
                      full_displacement_norms=norms, alpha=alpha, seeds=roots['seeds'],
                      donor_indices=roots['donor_indices'], donor_seeds=roots['donor_seeds'])
        output.append((arrays, dict(max_readout_error=error, minimum_region_margin=margin,
                                   max_norm_error=float(np.max(abs(realized_norms - common[:, None]))))))
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('roots', 'protocol', 'output'):
        parser.add_argument('--' + key, required=True)
    parser.add_argument('--group', type=int, choices=range(6), required=True)
    parser.add_argument('--expected-cases', type=int, default=512)
    args = parser.parse_args()
    started = time.monotonic()
    root_dir, out = Path(args.roots), Path(args.output)
    report = json.loads((root_dir / 'report.json').read_text())
    protocol_sha = sha(args.protocol)
    assert report['status'] == 'FRESH_FEEDBACK_ROOTS_READY' and report['group'] == args.group
    assert report['case_count'] == report['expected_cases'] == args.expected_cases
    assert report['protocol_sha256'] == protocol_sha
    assert report['source_sha256'] == sha(Path(__file__).with_name('extract_fresh_feedback_roots.py'))
    expected = [8 * args.group + slot for slot in (2, 3, 4)]
    assert [row['model_index'] for row in report['models']] == expected
    out.mkdir(parents=True, exist_ok=False)
    entries, model_rows = [], []
    head_hashes = {row['head_sha256'] for row in report['models']}
    assert len(head_hashes) == 1
    for row in report['models']:
        mi = row['model_index']
        assert sha(root_dir / row['file']) == row['sha256']
        assert sha(root_dir / row['head_file']) == row['head_sha256']
        roots = dict(np.load(root_dir / row['file'], allow_pickle=False))
        head = dict(np.load(root_dir / row['head_file'], allow_pickle=False))
        assert roots['predicted'].shape == roots['observed'].shape == roots['donor'].shape == (args.expected_cases, 192)
        assert roots['predicted'].dtype == roots['observed'].dtype == roots['donor'].dtype == np.float32
        actual, donor = [], []
        log_name = f'solver_attempts_{mi}.jsonl'
        with (out / log_name).open('x') as log:
            for index in range(args.expected_cases):
                for source, target, saved in (('act', roots['observed'][index], actual), ('don', roots['donor'][index], donor)):
                    corrected, _, solver = project(head, roots['predicted'][index], target)
                    log.write(json.dumps(dict(index=index, model_index=mi, source=source, **solver)) + '\n')
                    log.flush()
                    assert solver['status'] == 'solved' and corrected is not None
                    assert np.isfinite(corrected).all()
                    assert np.max(abs(readout(head, corrected) - readout(head, roots['predicted'][index]))) <= 1e-6
                    assert np.min(region_margin(head, roots['predicted'][index], corrected)) >= -1e-6
                    saved.append(corrected)
        entries.append((head, roots, np.asarray(actual), np.asarray(donor)))
        model_rows.append(dict(model_index=mi, objective=row['objective'], head_sha256=row['head_sha256'],
                               root_file_sha256=row['sha256'], solver_log=log_name,
                               solver_log_sha256=sha(out / log_name), solved_projections=2 * args.expected_cases))
        print('FRESH_QP_COMPLETE', mi, 2 * args.expected_cases, flush=True)
    matched = match_six_directions(entries)
    for row, (arrays, checks) in zip(model_rows, matched):
        name = f"model_{row['model_index']}.npz"
        np.savez_compressed(out / name, **arrays)
        row.update(file=name, sha256=sha(out / name), **checks)
    assert protocol_sha == sha(args.protocol)
    result = dict(status='FRESH_FEEDBACK_FIBERS_READY', group=args.group, case_count=args.expected_cases,
                  expected_cases=args.expected_cases, models=model_rows,
                  roots_report_sha256=sha(root_dir / 'report.json'), protocol_sha256=protocol_sha,
                  plan_sha256=report['plan_sha256'], bank_manifest_sha256=report['bank_manifest_sha256'],
                  source_sha256=sha(__file__), projection_source_sha256=sha(Path(__file__).with_name('readout_fiber.py')),
                  elapsed_seconds=time.monotonic() - started, solved_projections=6 * args.expected_cases,
                  scope='Actual and external-donor fixed-region projections with six-way standardized displacement matching; every recipient retained.')
    (out / 'report.json').write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
