"""Three fixed synthetic families through the real OSQP backend; no data files.

Run only in the audited fiber environment. This is an engineering integration
check, not S2 data generation, protocol freeze, or an experiment on real heads.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import osqp
import scipy

from projection_four import (project_t0_four_family, accept_t0_four_family,
                             OBJECTIVES, MEMBERS, PHASE, ROOT)

SEED = 23092026
SOURCE_PATHS = (
    PHASE / 'stage2/projection_four.py',
    PHASE / 'stage2/test_solver_integration.py',
    PHASE / 'scripts/s1_projection.py',
    PHASE / 'scripts/s1_common.py',
    PHASE / 'scripts/s1_readout.py',
    ROOT / 'strengthening/adapters/verifier.py',
    ROOT / 'strengthening/presubmission_v8_20260922/independent_readout/scripts/readout.py',
)


def synthetic_inputs():
    rng = np.random.default_rng(SEED)
    head = dict(mean=rng.normal(0., .1, 192), scale=np.exp(rng.uniform(-.7, .7, 192)),
        target_mean=rng.normal(size=6), target_scale=np.geomspace(.5, 100., 6),
        **{'0.weight': rng.normal(0., .02, (256, 192)), '0.bias': np.full(256, 2.),
           '2.weight': rng.normal(0., .02, (256, 256)), '2.bias': np.full(256, 2.),
           '4.weight': rng.normal(0., .02, (6, 256)), '4.bias': rng.normal(0., .1, 6)})
    cases = []
    for index in range(3):
        predicted = {objective: rng.normal(0., .2, 192).astype(np.float32) for objective in OBJECTIVES}
        guides = {name: rng.normal(0., .3, 192).astype(np.float32) for name in ('actual', 'donor')}
        if index == 2:
            # A genuine zero-objective direction, with the other three retained.
            guides['actual'] = predicted['decoded_teacher'].copy()
        cases.append((predicted, guides))
    return head, cases


def run():
    if (np.__version__, scipy.__version__, osqp.__version__) != ('1.26.4', '1.13.1', '0.6.7.post3'):
        raise ValueError('Run this missing integration gate in the audited fiber runtime only')
    head, cases = synthetic_inputs()
    assignment = np.array([2, 0, 1], dtype=np.int64)
    rows = []; start = time.monotonic()
    for goal, (predicted, guides) in enumerate(cases):
        replacements, directions, report = project_t0_four_family(predicted, guides, head,
            goal_index=goal, donor_index=int(assignment[goal]))
        accepted = accept_t0_four_family(predicted, guides, head, replacements, directions, report,
            goal_index=goal, donor_assignment=assignment)
        if (report['members'] != [list(x) for x in MEMBERS] or len(report['solvers']) != 4 or
                any(s['status'].lower() != 'solved' or s['head_count'] != 1 or s['osqp_version'] != osqp.__version__ for s in report['solvers']) or
                not all(c['accepted'] for c in accepted['checks'])):
            raise AssertionError('Incomplete real-backend four-member integration')
        if goal == 2:
            if report['legitimate_zero_norm'] is not True or accepted['effective_norm'] != 0.:
                raise AssertionError('Exact zero-target QP must retain the zero complete family')
            for oi, objective in enumerate(OBJECTIVES):
                for source in range(2): np.testing.assert_array_equal(replacements[oi, source], predicted[objective])
        rows.append(dict(goal=goal, donor_index=int(assignment[goal]), family_size=4,
            solvers=report['solvers'], native_norms=report['native_norms'], effective_norm=report['effective_norm'],
            shrink_factor=report['shrink_factor'], zero_family=report['legitimate_zero_norm'],
            independent_acceptance=accepted))
    return dict(status='PASS_SYNTHETIC_DEFAULT_OSQP_FOUR_FAMILY_INTEGRATION',
        synthetic_only=True, actual_head_or_data_files_opened=False, default_solver_not_injected=True,
        fixed_seed=SEED, families=3, directions=12, numpy=np.__version__, scipy=scipy.__version__, osqp=osqp.__version__,
        rows=rows, elapsed_seconds=time.monotonic() - start,
        source_sha256={str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in SOURCE_PATHS},
        scope='Fixed generated192→256→256→6 head and synthetic tokens only. Full native-QP/default-backend/independent-acceptance integration; not S2 scientific execution or real-data throughput.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, help='Optional exclusive JSON engineering receipt')
    args = parser.parse_args()
    if args.output and args.output.exists(): raise ValueError('Never overwrite an integration receipt')
    result = run()
    rendered = json.dumps(result, indent=2, allow_nan=False) + '\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x') as f: f.write(rendered)
    print(rendered)


if __name__ == '__main__':
    main()
