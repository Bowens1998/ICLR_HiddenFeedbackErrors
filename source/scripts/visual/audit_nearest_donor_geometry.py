"""Outcome-blind donor selection and fixed-readout geometric feasibility audit.

Reads only action-5 latents, frozen heads, accepted action-5 corrections and
optional already encoded observation history. No endpoint state/cost is read.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from readout_fiber import geometry, project, readout


SOURCES = ('actual', 'random', 'pose_nearest', 'latent_nearest')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def standardized(head, tokens):
    return (np.asarray(tokens, dtype=float) - head['mean']) / head['scale']


def choose_donors(head, actual, donor):
    """Choose by current-observation distance alone; exact ties use lowest index."""
    actual_x, donor_x = standardized(head, actual), standardized(head, donor)
    latent_distance = np.linalg.norm(actual_x[:, None] - donor_x[None], axis=-1)
    actual_pose, donor_pose = readout(head, actual), readout(head, donor)
    pose_distance = np.linalg.norm(actual_pose[:, None] - donor_pose[None], axis=-1)
    return dict(pose_nearest=np.argmin(pose_distance, axis=1),
                latent_nearest=np.argmin(latent_distance, axis=1)), pose_distance, latent_distance


def describe(values):
    flat = np.asarray(values, dtype=float).reshape(-1)
    valid = flat[np.isfinite(flat)]
    result = dict(valid=int(len(valid)), total=int(len(flat)))
    if len(valid):
        result.update(mean=float(valid.mean()), median=float(np.median(valid)),
                      q10=float(np.quantile(valid, .1)), q90=float(np.quantile(valid, .9)),
                      min=float(valid.min()), max=float(valid.max()))
    return result


def cosine(left, right):
    numerator = np.sum(left * right, axis=-1)
    denominator = np.linalg.norm(left, axis=-1) * np.linalg.norm(right, axis=-1)
    return np.divide(numerator, denominator, out=np.full_like(numerator, np.nan), where=denominator > 0)


def region_margin(head, predicted, replacement):
    _, aa, bb, m0, m1, w0, second, _ = geometry(head, predicted)
    signs = np.r_[np.where(m0, 1., -1.), np.where(m1, 1., -1.)]
    return signs * (np.r_[aa, bb] + np.r_[w0, second] @ ((replacement.astype(float) - predicted) / head['scale']))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('recipient-horizon', 'donor-horizon', 'old-fibers', 'output'):
        parser.add_argument('--' + key, required=True)
    parser.add_argument('--history-smoke')
    parser.add_argument('--groups', type=int, nargs='+', default=list(range(6)))
    parser.add_argument('--cases', type=int, default=128)
    parser.add_argument('--skip-projections', action='store_true')
    args = parser.parse_args()
    assert 0 < args.cases <= 128 and len(set(args.groups)) == len(args.groups)
    assert set(args.groups) <= set(range(6))
    started = time.monotonic()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    random_indices = np.random.default_rng(1368001).permutation(128)[:args.cases]
    source_stats = {name: {} for name in SOURCES}
    projection_stats = {name: {} for name in SOURCES}
    history_stats = {name: {} for name in SOURCES}
    groups, projected_models = [], []

    def collect(container, source, name, values):
        container[source].setdefault(name, []).extend(np.asarray(values, dtype=float).reshape(-1).tolist())

    for group in args.groups:
        rd = Path(args.recipient_horizon) / f'job_{group}'
        dd = Path(args.donor_horizon) / f'job_{group}'
        fd = Path(args.old_fibers) / f'job_{group}'
        rr, dr, fr = [json.loads((directory / 'report.json').read_text()) for directory in (rd, dd, fd)]
        for directory, report in ((rd, rr), (dd, dr), (fd, fr)):
            acceptance = json.loads((directory / 'acceptance.json').read_text())
            assert acceptance['report_sha256'] == sha(directory / 'report.json')
        assert dr['plan_sha256'] == '1fd34ed8b462afc5901b6d82c9ef119ff689424be03803b92882064a5559c79b'
        row = next(r for r in rr['rows'] if r['model_index'] == 8 * group + 3)
        donor_row = next(r for r in dr['rows'] if r['model_index'] == 8 * group + 3)
        head_path = rd / row['head_file']
        assert sha(head_path) == row['head_sha256'] == donor_row['head_sha256']
        assert sha(rd / row['file']) == row['sha256'] and sha(dd / donor_row['file']) == donor_row['sha256']
        head = dict(np.load(head_path, allow_pickle=False))
        with np.load(rd / row['file'], allow_pickle=False) as archive:
            actual = archive['observed_tokens'][0, :args.cases, 0].copy()
            recipient_seeds = archive['seeds'][:args.cases].copy()
        with np.load(dd / donor_row['file'], allow_pickle=False) as archive:
            donors = archive['observed_tokens'][0, :, 0].copy()
            donor_seeds = archive['seeds'].copy()
        assert actual.shape == (args.cases, 192) and donors.shape == (128, 192)
        assert not set(recipient_seeds) & set(donor_seeds)
        nearest, pose_distance, latent_distance = choose_donors(head, actual, donors)
        indices = dict(random=random_indices, **nearest)
        sources = dict(actual=actual, **{name: donors[index] for name, index in indices.items()})
        actual_x, actual_pose = standardized(head, actual), readout(head, actual)
        for source, values in sources.items():
            latent_to_actual = np.linalg.norm(standardized(head, values) - actual_x, axis=-1)
            pose_to_actual = np.linalg.norm(readout(head, values) - actual_pose, axis=-1)
            collect(source_stats, source, 'standardized_latent_distance_to_actual', latent_to_actual)
            collect(source_stats, source, 'normalized_pose_distance_to_actual', pose_to_actual)
            if source in nearest:
                collect(source_stats, source, 'latent_distance_smaller_than_random', latent_to_actual < latent_distance[np.arange(args.cases), random_indices])
                collect(source_stats, source, 'pose_distance_smaller_than_random', pose_to_actual < pose_distance[np.arange(args.cases), random_indices])
        geometry_file = f'group_{group}_selection.npz'
        np.savez_compressed(out / geometry_file, recipient_seeds=recipient_seeds, donor_seeds=donor_seeds,
                            actual=actual, donors=donors, random_indices=random_indices,
                            pose_nearest_indices=nearest['pose_nearest'], latent_nearest_indices=nearest['latent_nearest'],
                            pairwise_normalized_pose_distance=pose_distance, pairwise_standardized_latent_distance=latent_distance)
        record = dict(group=group, file=geometry_file, sha256=sha(out / geometry_file),
                      recipient_report_sha256=sha(rd / 'report.json'), donor_report_sha256=sha(dd / 'report.json'),
                      fiber_report_sha256=sha(fd / 'report.json'), head_sha256=sha(head_path),
                      unique_pose_donors=int(len(set(nearest['pose_nearest']))),
                      unique_latent_donors=int(len(set(nearest['latent_nearest']))),
                      nearest_same_index_fraction=float(np.mean(nearest['pose_nearest'] == nearest['latent_nearest'])))
        group_directions, group_models = [], []
        for slot in (2, 3, 4):
            mi = 8 * group + slot
            binding = next(r for r in fr['bindings'] if r.get('model_index') == mi)
            assert binding['head_sha256'] == sha(head_path)
            path = fd / binding['output_file']
            assert sha(path) == binding['output_sha256']
            with np.load(path, allow_pickle=False) as archive:
                old = {key: archive[key][:args.cases].copy() for key in ('predicted', 'observed', 'donor', 'full', 'shuffled_full')}
            np.testing.assert_array_equal(old['observed'], actual)
            np.testing.assert_array_equal(old['donor'], sources['random'])
            predicted, predicted_x = old['predicted'], standardized(head, old['predicted'])
            actual_root_distance = np.linalg.norm(actual_x - predicted_x, axis=-1)
            for source, values in sources.items():
                distance = np.linalg.norm(standardized(head, values) - predicted_x, axis=-1)
                collect(source_stats, source, 'standardized_latent_distance_to_predicted', distance)
                collect(source_stats, source, 'distance_to_predicted_ratio_vs_actual', np.divide(distance, actual_root_distance,
                        out=np.full_like(distance, np.nan), where=actual_root_distance > 0))
                collect(source_stats, source, 'closer_to_predicted_than_actual', distance < actual_root_distance)
            if args.skip_projections:
                continue
            corrections = dict(actual=old['full'], random=old['shuffled_full'])
            attempts = []
            for source in ('pose_nearest', 'latent_nearest'):
                corrected = []
                for index, target in enumerate(sources[source]):
                    try:
                        value, _, solver = project(head, predicted[index], target)
                        error = None if value is None else float(np.max(abs(readout(head, value) - readout(head, predicted[index]))))
                        margin = None if value is None else float(np.min(region_margin(head, predicted[index], value)))
                        valid = value is not None and solver['status'] == 'solved' and error <= 1e-6 and margin >= -1e-6
                        attempts.append(dict(source=source, index=index, valid=bool(valid), readout_error=error, minimum_region_margin=margin, **solver))
                    except Exception as error:
                        value, valid = None, False
                        attempts.append(dict(source=source, index=index, valid=False, exception=repr(error)))
                    corrected.append(value if valid else np.full_like(predicted[index], np.nan))
                corrections[source] = np.asarray(corrected)
            directions = np.stack([(corrections[source].astype(float) - predicted) / head['scale'] for source in SOURCES], axis=1)
            norms = np.linalg.norm(directions, axis=-1)
            group_directions.append(norms)
            for si, source in enumerate(SOURCES):
                collect(projection_stats, source, 'full_standardized_displacement', norms[:, si])
                collect(projection_stats, source, 'cosine_with_actual_projection', cosine(directions[:, si], directions[:, 0]))
                collect(projection_stats, source, 'correction_distance_to_actual_correction', np.linalg.norm(directions[:, si] - directions[:, 0], axis=-1))
            name = f'model_{mi}_projections.npz'
            np.savez_compressed(out / name, predicted=predicted, sources=np.array(SOURCES),
                                corrections=np.stack([corrections[source] for source in SOURCES], axis=1),
                                standardized_displacements=directions, displacement_norms=norms)
            logs = f'model_{mi}_solver_attempts.json'
            (out / logs).write_text(json.dumps(attempts, indent=2) + '\n')
            model_record = dict(model_index=mi, cases=args.cases, file=name, sha256=sha(out / name),
                                attempts=len(attempts), valid_attempts=sum(attempt['valid'] for attempt in attempts),
                                log_file=logs, log_sha256=sha(out / logs), old_fiber_sha256=binding['output_sha256'])
            projected_models.append(model_record)
            group_models.append(model_record)
            print('GEOMETRY_QP_MODEL', mi, model_record['valid_attempts'], '/', model_record['attempts'], flush=True)
        if group_directions:
            all_norms = np.stack(group_directions)  # objective, recipient, source
            common_old = np.min(all_norms[:, :, :2], axis=(0, 2))
            common_new = np.min(all_norms, axis=(0, 2))
            record['four_source_matching'] = dict(common_twelve=describe(common_new), common_six=describe(common_old),
                twelve_to_six_ratio=describe(np.divide(common_new, common_old, out=np.full_like(common_new, np.nan), where=common_old > 0)),
                reduced_fraction=float(np.mean(common_new < common_old)))
        if args.history_smoke:
            history_dir = Path(args.history_smoke) / f'roots_{group}'
            if (history_dir / 'report.json').exists():
                hr = json.loads((history_dir / 'report.json').read_text())
                history_row = next(row for row in hr['models'] if row['model_index'] == 8 * group + 3)
                assert sha(history_dir / history_row['file']) == history_row['sha256']
                with np.load(history_dir / history_row['file'], allow_pickle=False) as archive:
                    history, seeds = archive['initial_history'].copy(), archive['seeds'].copy()
                subset = [int(np.flatnonzero(recipient_seeds == seed)[0]) for seed in seeds if seed in recipient_seeds]
                history = history[[i for i, seed in enumerate(seeds) if seed in recipient_seeds]]
                x = standardized(head, history)
                old_step = x[:, 2] - x[:, 1]
                extrapolated = x[:, 2] + old_step
                actual_step = actual_x[subset] - x[:, 2]
                for source, values in sources.items():
                    sx = standardized(head, values[subset])
                    step = sx - x[:, 2]
                    denominator = np.sum(old_step ** 2, axis=-1)
                    coefficient = np.divide(np.sum(step * old_step, axis=-1), denominator,
                                            out=np.zeros_like(denominator), where=denominator > 0)
                    collect(history_stats, source, 'distance_to_last_observed_latent', np.linalg.norm(step, axis=-1))
                    collect(history_stats, source, 'distance_to_linear_history_extrapolation', np.linalg.norm(sx - extrapolated, axis=-1))
                    collect(history_stats, source, 'distance_to_history_affine_line', np.linalg.norm(step - coefficient[:, None] * old_step, axis=-1))
                    collect(history_stats, source, 'current_increment_cosine_with_actual', cosine(step, actual_step))
                record['history_smoke_cases'] = len(subset)
                record['history_report_sha256'] = sha(history_dir / 'report.json')
                record['history_file_sha256'] = history_row['sha256']
        groups.append(record)

    def summarize(container):
        return {source: {metric: describe(values) for metric, values in metrics.items()} for source, metrics in container.items()}

    result = dict(status='OUTCOME_BLIND_NEAREST_DONOR_GEOMETRY_AUDIT', groups=groups, models=projected_models,
                  recipient_cases_per_group=args.cases, sources=list(SOURCES),
                  source_geometry=summarize(source_stats), projection_geometry=summarize(projection_stats),
                  history_geometry=summarize(history_stats), source_sha256=sha(__file__),
                  projection_source_sha256=sha(Path(__file__).with_name('readout_fiber.py')),
                  elapsed_seconds=time.monotonic() - started,
                  definitions=dict(pose_distance='Euclidean distance in complete six-dimensional normalized frozen nonlinear readout output.',
                                   latent_distance='Euclidean distance after the frozen head input mean/scale standardization.',
                                   selector='Nearest to actual already-observed action-5 encoding in the fixed external donor bank. Exact ties choose lowest donor index.',
                                   random='Fixed permutation(128) from seed1368001, unchanged from the original diagnosis.',
                                   history='Feature-space proxies only: last observed latent, linear continuation from the last two observed latents, affine-line residual, and increment alignment with actual o5.',
                                   summary_units='Actual-distance source metrics: one per group/goal; root/projection metrics: one per objective/group/goal; no confidence intervals or independent-replicate claim.',
                                   matching='Twelve-way minimum is hypothetical four-source matching across three objectives, reported geometrically; no future rollout is scored.'),
                  scope='Development-only geometric and solver audit. No future endpoints, costs, rankings or selection outcomes are read. This does not establish donor-control rollout performance or identify a hidden physical variable.')
    (out / 'report.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
