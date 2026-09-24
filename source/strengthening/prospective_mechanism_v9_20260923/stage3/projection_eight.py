"""Draft S3 A-only matching across four distinct T0/T1 model anchors.

Pure token arithmetic; no file, model, simulator, outcome or candidate loader.
The caller must authenticate one available actual image and one assigned donor
image per pool. No scientific execution authority is supplied by this module.
"""
from pathlib import Path
import sys

import numpy as np

PHASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PHASE / 'stage2'))
# Reuse numerical primitives only, never the T0-only four-member wrapper.
import projection_four as primitives

CONDITIONS = ('T0', 'T1')
OBJECTIVES = primitives.OBJECTIVES
SOURCES = primitives.SOURCES
MEMBERS = tuple((c, o, s) for c in CONDITIONS for o in OBJECTIVES for s in SOURCES)
SHAPE = (2, 2, 2, primitives.LATENT_DIM)
SHRINK_FACTORS = primitives.SHRINK_FACTORS
TOLERANCE = primitives.TOLERANCE
STATUS = 'ACCEPTED_DRAFT_S3_A_ONLY_EIGHT_MEMBER_FAMILY'
ACCEPT_STATUS = 'PASS_INDEPENDENT_DRAFT_S3_EIGHT_FAMILY_VALUE_ACCEPTANCE'
DEFAULTS = dict(shrink_factors=list(SHRINK_FACTORS), tolerance=TOLERANCE,
    eps_abs=1e-9, eps_rel=1e-9, max_iter=20000,
    receipt_atol=primitives.RECEIPT_ATOL,
    physical_receipt_atol='receipt_atol*max(1,max_target_scale)')


class EightFamilyFailure(RuntimeError):
    def __init__(self, message, detail):
        super().__init__(message)
        self.detail = detail


def _inputs(predicted, guides, head_A, pool):
    pool = primitives._index(pool, 'pool')
    if pool not in (0, 1, 2):
        raise ValueError('Exactly the three fixed Transformer pools are allowed')
    if set(predicted) != set(CONDITIONS) or set(guides) != set(SOURCES):
        raise ValueError('Both T0/T1 and only actual/donor shared guides required')
    values = dict(guides)
    for condition in CONDITIONS:
        if set(predicted[condition]) != set(OBJECTIVES):
            raise ValueError('Both coordinate objectives required for every condition')
        values.update({condition + '/' + o: predicted[condition][o] for o in OBJECTIVES})
    for name, value in values.items():
        array = np.asarray(value)
        if array.dtype != np.float32 or array.shape != (192,) or not np.isfinite(array).all():
            raise ValueError('One finite192 FP32 token required: ' + name)
    return primitives._head(head_A), pool


def _bindings(predicted, guides, head):
    values = {'anchor/' + c + '/' + o: np.asarray(predicted[c][o])
              for c in CONDITIONS for o in OBJECTIVES}
    values.update({'guide/' + s: np.asarray(guides[s]) for s in SOURCES})
    return dict(token_values_sha256=primitives._digest(values),
                head_A_values_sha256=primitives._digest(head))


def _identity(predicted, guides, head, pool, goal, donor):
    return dict(status=STATUS, draft_engineering_only=True, family_size=8,
        conditions=list(CONDITIONS), objectives=list(OBJECTIVES), sources=list(SOURCES),
        members=[list(x) for x in MEMBERS], axes=['condition', 'objective', 'source', 'latent'],
        constraint='g_A', metric='Euclidean displacement in shared g_A input-standardized coordinates',
        anchor_rule='Own model free root; only actual/donor of the same model share an anchor',
        pool=pool, group=2 * pool, goal_index=goal, donor_index=donor,
        **_bindings(predicted, guides, head),
        defaults={**DEFAULTS, 'shrink_factors': list(SHRINK_FACTORS)})


def project_matched_eight_family(predicted, guides, head_A, *, pool, goal_index,
                                 donor_index, projector=None):
    """All eight solves precede matching; every failed solve remains diagnostic.

    predicted[condition][objective] contains four independent native roots.
    guides[source] is shared across all four models, never per suffix candidate.
    Optional projector injection is reserved for synthetic engineering tests.
    """
    head, pool = _inputs(predicted, guides, head_A, pool)
    goal = primitives._index(goal_index, 'goal_index')
    donor = primitives._index(donor_index, 'donor_index')
    projector = primitives._single_direction_project if projector is None else projector
    directions, solvers, failures = [], [], []
    for condition, objective, source in MEMBERS:
        try:
            result = projector([head], predicted[condition][objective], guides[source], reference_head=head)
            delta, solver = np.asarray(result.delta), dict(result.solver)
            if (delta.dtype != np.float64 or delta.shape != (192,) or not np.isfinite(delta).all() or
                    str(solver.get('status', '')).lower() != 'solved' or solver.get('head_count') != 1):
                raise ValueError('Nonfinite, unsolved or non-A-only direction')
            directions.append(delta.copy()); solvers.append(solver)
        except Exception as exc:
            directions.append(None); solvers.append(None)
            failures.append(dict(member=[condition, objective, source], error=repr(exc),
                                 solver_detail=getattr(exc, 'detail', None)))
    if failures:
        raise EightFamilyFailure('A solver failure blocks the complete family; never a zero direction',
            dict(stage='solver', attempted_members=[list(x) for x in MEMBERS],
                 successful_directions=8 - len(failures), solvers=solvers, failures=failures))
    native = np.array([np.linalg.norm(d) for d in directions], np.float64)
    if not np.isfinite(native).all():
        raise EightFamilyFailure('Nonfinite native norm blocks the entire family', dict(stage='matching'))
    minimum = float(native.min()); attempts = []
    for shrink in SHRINK_FACTORS:
        replacements, checks = [], []
        for (condition, objective, _), direction, norm in zip(MEMBERS, directions, native, strict=True):
            anchor = predicted[condition][objective]
            fraction = 0. if norm == 0. else shrink * minimum / norm
            replacement = (anchor.astype(np.float64) + fraction * direction * head['scale']).astype(np.float32)
            check = primitives._token_verifier.verify_token(anchor, replacement, [head], head,
                expected_norm=shrink * minimum, tol=TOLERANCE)
            replacements.append(replacement); checks.append(check)
        attempts.append(dict(shrink=float(shrink), checks=checks))
        if all(c['accepted'] for c in checks):
            return (np.stack(replacements).reshape(SHAPE), np.stack(directions).reshape(SHAPE),
                dict(**_identity(predicted, guides, head, pool, goal, donor),
                    native_norms=native.tolist(), unshrunk_common_norm=minimum,
                    effective_norm=shrink * minimum, shrink_factor=float(shrink),
                    legitimate_zero_norm=minimum == 0., attempts=attempts, solvers=solvers))
    raise EightFamilyFailure('FP32 verification failed for the whole eight-member family',
        dict(stage='verification', native_norms=native.tolist(), solvers=solvers, attempts=attempts))


def accept_matched_eight_family(predicted, guides, head_A, replacements, directions, report,
                                *, pool, goal_index, donor_assignment):
    """Reconstruct the full ray/minimum/backoff and dense nonlinear checks.

    Reuses the separately implemented dense FP64 verifier from S2. It does not
    call the producing matcher or QP solve, and is not a KKT optimality proof.
    The full population caller supplies and authenticates the fixed global
    donor permutation, model/readout/normalizer hashes and available images.
    """
    head, pool = _inputs(predicted, guides, head_A, pool)
    goal = primitives._index(goal_index, 'goal_index')
    assignment = np.asarray(donor_assignment)
    if (assignment.dtype != np.int64 or assignment.ndim != 1 or len(assignment) == 0 or
            goal >= len(assignment) or
            not np.array_equal(np.sort(assignment), np.arange(len(assignment), dtype=np.int64))):
        raise ValueError('Complete global donor permutation required; no shard-local indexing')
    if (np.asarray(replacements).dtype != np.float32 or np.shape(replacements) != SHAPE or
            np.asarray(directions).dtype != np.float64 or np.shape(directions) != SHAPE or
            not np.isfinite(replacements).all() or not np.isfinite(directions).all()):
        raise ValueError('Complete FP32 replacements and FP64 eight-direction arrays required')
    expected = _identity(predicted, guides, head, pool, goal, int(assignment[goal]))
    if any(report.get(k) != v for k, v in expected.items()):
        raise ValueError('Changed identity, anchors, head, donor, order or numerical defaults')
    solvers = report.get('solvers', [])
    if len(solvers) != 8 or any(str(x.get('status', '')).lower() != 'solved' or x.get('head_count') != 1 for x in solvers):
        raise ValueError('Exactly eight solved A-only directions required')
    flat = np.asarray(directions).reshape(8, 192)
    norms = np.array([np.linalg.norm(d) for d in flat], np.float64)
    if not np.isfinite(norms).all() or len(report.get('native_norms', [])) != 8:
        raise ValueError('Invalid complete native direction norms')
    for saved, actual in zip(report['native_norms'], norms, strict=True):
        primitives._same_number(saved, float(actual), 'native norm')
    minimum = float(norms.min())
    primitives._same_number(report.get('unshrunk_common_norm'), minimum, 'common minimum')
    if report.get('legitimate_zero_norm') is not (minimum == 0.):
        raise ValueError('Changed legitimate zero-family identity')
    attempts = report.get('attempts', [])
    if not 1 <= len(attempts) <= len(SHRINK_FACTORS):
        raise ValueError('Complete first-backoff history required')
    for i, shrink in enumerate(SHRINK_FACTORS):
        values, checks = [], []
        for j, (condition, objective, _) in enumerate(MEMBERS):
            anchor = predicted[condition][objective]
            alpha = 0. if norms[j] == 0. else minimum * shrink / float(norms[j])
            candidate = (np.array(anchor, np.float64) + (flat[j] * alpha) * head['scale']).astype(np.float32)
            values.append(candidate)
            checks.append(primitives._independent_token_check(anchor, candidate, head, minimum * shrink))
        if i >= len(attempts):
            raise ValueError('Saved backoff history ends before independent acceptance')
        saved = attempts[i]
        primitives._same_number(saved.get('shrink'), shrink, 'scheduled shrink')
        if len(saved.get('checks', [])) != 8:
            raise ValueError('Missing one or more per-attempt member checks')
        for stored, actual in zip(saved['checks'], checks, strict=True):
            primitives._same_check(stored, actual, head['target_scale'])
        if all(x['accepted'] for x in checks):
            if len(attempts) != i + 1:
                raise ValueError('Not the first common accepted backoff')
            if not np.array_equal(np.stack(values).reshape(SHAPE), replacements):
                raise ValueError('Replacement differs from its own model anchor and FP32 ray')
            primitives._same_number(report.get('shrink_factor'), shrink, 'effective shrink')
            primitives._same_number(report.get('effective_norm'), minimum * shrink, 'effective norm')
            return dict(status=ACCEPT_STATUS, draft_engineering_only=True, pool=pool, group=2 * pool,
                goal_index=goal, donor_index=int(assignment[goal]), family_size=8,
                effective_norm=minimum * shrink, effective_norm_squared=(minimum * shrink) ** 2,
                shrink_factor=shrink, checks=checks,
                replacement_values_sha256=primitives._digest({'replacements': replacements}),
                direction_values_sha256=primitives._digest({'directions': directions}),
                donor_assignment_sha256=primitives._digest({'donor_assignment': assignment}),
                scope='Eight distinct-anchor A-only rays, common metric/minimum/first FP32 backoff, '
                      'dense full-function/region/norm acceptance; no QP optimality, model or image-provenance replay.')
    raise ValueError('No complete independently accepted family')
