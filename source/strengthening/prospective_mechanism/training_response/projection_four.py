"""DRAFT S2 T0-only four-direction construction and independent acceptance.

Engineering defaults reuse S1 numerical constants; this is not a scientific S2
freeze. Only g_A and two objective anchors/actual-donor guides are accepted.
No model/readout checkpoint or data loader exists here. The future role-specific
loader must authenticate T0 and observed-token ancestry; this module checks the
provided token values and complete family arithmetic, not model provenance.
"""
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys

import numpy as np

PHASE = Path(__file__).resolve().parents[1]
ROOT = PHASE.parents[1]
S1_SCRIPTS = PHASE / 'scripts'
if str(S1_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(S1_SCRIPTS))
from s1_projection import project as _single_direction_project

_spec = importlib.util.spec_from_file_location('_s2_existing_fp64_verifier', ROOT / 'strengthening/adapters/verifier.py')
_token_verifier = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_token_verifier)

OBJECTIVES = ('decoded_teacher', 'physical_labels')
SOURCES = ('actual', 'donor')
MEMBERS = tuple((o, s) for o in OBJECTIVES for s in SOURCES)
SHRINK_FACTORS = tuple(2. ** -i for i in range(9))
TOLERANCE = 1e-6
RECEIPT_ATOL = 1e-12
LATENT_DIM = 192
HEAD_KEYS = ('mean', 'scale', 'target_mean', 'target_scale',
             '0.weight', '0.bias', '2.weight', '2.bias', '4.weight', '4.bias')
STATUS = 'ACCEPTED_DRAFT_S2_T0_FOUR_MEMBER_FAMILY'


class FourFamilyFailure(RuntimeError):
    def __init__(self, message, detail):
        super().__init__(message)
        self.detail = detail


def _index(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 0:
        raise ValueError(name + ' must be a nonnegative integer')
    return int(value)


def _head(head_A):
    if not all(k in head_A for k in HEAD_KEYS):
        raise ValueError('Missing g_A serialized weights or normalization')
    h = {k: np.asarray(head_A[k], dtype=np.float64) for k in HEAD_KEYS}
    sizes = (LATENT_DIM, len(h['0.bias']), len(h['2.bias']), 6)
    for layer, nin, nout in zip((0, 2, 4), sizes[:-1], sizes[1:]):
        if h[f'{layer}.weight'].shape != (nout, nin) or h[f'{layer}.bias'].shape != (nout,):
            raise ValueError('Invalid g_A layer shape')
    if (h['mean'].shape != (LATENT_DIM,) or h['scale'].shape != (LATENT_DIM,) or
            h['target_mean'].shape != (6,) or h['target_scale'].shape != (6,) or
            not all(np.isfinite(v).all() for v in h.values()) or
            not (h['scale'] > 0).all() or not (h['target_scale'] > 0).all()):
        raise ValueError('Invalid g_A normalization or values')
    return h


def _inputs(predicted, guides, head_A):
    if set(predicted) != set(OBJECTIVES) or set(guides) != set(SOURCES):
        raise ValueError('Exactly two T0 objectives and actual/donor observed guides are required')
    for name, value in list(predicted.items()) + list(guides.items()):
        value = np.asarray(value)
        if value.dtype != np.float32 or value.shape != (LATENT_DIM,) or not np.isfinite(value).all():
            raise ValueError('Expected finite192 FP32 token: ' + name)
    return _head(head_A)


def _digest(values):
    """Value identity only; future I/O receipts must also bind actual file hashes."""
    result = hashlib.sha256()
    for name in sorted(values):
        a = np.ascontiguousarray(values[name])
        result.update(json.dumps([name, a.dtype.str, list(a.shape)], separators=(',', ':')).encode())
        result.update(a.tobytes())
    return result.hexdigest()


def _bindings(predicted, guides, head):
    values = {'anchor/' + k: np.asarray(v) for k, v in predicted.items()}
    values.update({'guide/' + k: np.asarray(v) for k, v in guides.items()})
    return dict(token_values_sha256=_digest(values), head_A_values_sha256=_digest(head))


def project_t0_four_family(predicted, guides, head_A, *, goal_index, donor_index, projector=None):
    """Construct one indivisible family; projector injection is synthetic-test only."""
    head = _inputs(predicted, guides, head_A)
    goal_index, donor_index = _index(goal_index, 'goal_index'), _index(donor_index, 'donor_index')
    projector = _single_direction_project if projector is None else projector
    directions = []; solvers = []
    for objective, source in MEMBERS:
        try:
            result = projector([head], predicted[objective], guides[source], reference_head=head)
            delta = np.asarray(result.delta)
            solver = dict(result.solver)
            if (delta.dtype != np.float64 or delta.shape != (LATENT_DIM,) or not np.isfinite(delta).all() or
                    str(solver.get('status', '')).lower() != 'solved' or solver.get('head_count') != 1):
                raise ValueError('Unsolved, nonfinite or non-A-only direction')
        except Exception as exc:
            raise FourFamilyFailure('A direction failed; the complete four-member family is blocked',
                dict(stage='solver', member=[objective, source], completed_directions=len(directions),
                     error=repr(exc))) from exc
        directions.append(delta.copy()); solvers.append(solver)
    native = np.array([np.linalg.norm(d) for d in directions], dtype=np.float64)
    if not np.isfinite(native).all():
        raise FourFamilyFailure('Nonfinite native direction norm', dict(stage='matching'))
    minimum = float(native.min()); attempts = []
    for shrink in SHRINK_FACTORS:
        corrected = []; checks = []
        for (objective, _), direction, norm in zip(MEMBERS, directions, native):
            fraction = 0. if norm == 0 else shrink * minimum / norm
            replacement = (predicted[objective].astype(np.float64) +
                           fraction * direction * head['scale']).astype(np.float32)
            check = _token_verifier.verify_token(predicted[objective], replacement, [head], head,
                                                expected_norm=shrink * minimum, tol=TOLERANCE)
            corrected.append(replacement); checks.append(check)
        attempts.append(dict(shrink=float(shrink), checks=checks))
        if all(c['accepted'] for c in checks):
            return (np.stack(corrected).reshape(2, 2, LATENT_DIM),
                    np.stack(directions).reshape(2, 2, LATENT_DIM),
                    dict(status=STATUS, draft_engineering_only=True, model_role='T0', constraint='g_A',
                         family_size=4, members=[list(x) for x in MEMBERS],
                         goal_index=goal_index, donor_index=donor_index,
                         **_bindings(predicted, guides, head), native_norms=native.tolist(),
                         unshrunk_common_norm=minimum, effective_norm=shrink * minimum,
                         shrink_factor=float(shrink), legitimate_zero_norm=minimum == 0.,
                         attempts=attempts, solvers=solvers,
                         defaults=dict(shrink_factors=list(SHRINK_FACTORS), tolerance=TOLERANCE,
                                       eps_abs=1e-9, eps_rel=1e-9, max_iter=20000,
                                       receipt_atol=RECEIPT_ATOL, physical_receipt_atol='receipt_atol*max(1,max_target_scale)')))
    raise FourFamilyFailure('FP32 numerical acceptance failed for the entire four-member family',
        dict(stage='verification', native_norms=native.tolist(), attempts=attempts))


def _independent_token_check(original, candidate, head, expected_norm):
    """Independent dense FP64 nonlinear/region/norm calculation; no QP helpers."""
    def evaluate(token):
        standardized = (np.array(token, dtype=np.float64) - head['mean']) / head['scale']
        first = np.einsum('ij,j->i', head['0.weight'], standardized) + head['0.bias']
        second = np.einsum('ij,j->i', head['2.weight'], np.clip(first, 0., None)) + head['2.bias']
        output = np.einsum('ij,j->i', head['4.weight'], np.clip(second, 0., None)) + head['4.bias']
        return output, first, second
    y0, first0, second0 = evaluate(original)
    y1, first1, second1 = evaluate(candidate)
    normalized = float(np.max(np.abs(y1 - y0)))
    signed_first = np.where(first0 >= 0., first1, -first1)
    signed_second = np.where(second0 >= 0., second1, -second1)
    region = float(max(0., -float(signed_first.min()), -float(signed_second.min())))
    displacement = (np.asarray(candidate, np.float64) - np.asarray(original, np.float64)) / head['scale']
    norm = float(np.sqrt(np.sum(displacement * displacement, dtype=np.float64)))
    physical = float(np.max(np.abs((y1 - y0) * head['target_scale'])))
    finite = bool(np.isfinite(candidate).all() and np.isfinite([normalized, region, norm, physical]).all())
    norm_ok = bool(abs(norm - expected_norm) <= TOLERANCE + TOLERANCE * abs(expected_norm))
    return dict(accepted=finite and norm_ok and normalized <= TOLERANCE and region <= TOLERANCE,
        finite=finite, norm_ok=norm_ok, actual_norm=norm, expected_norm=expected_norm,
        heads=[dict(normalized_output_deviation=normalized, region_violation=region,
                    physical_output_max_deviation=physical)])


def _same_number(actual, expected, label, *, atol=RECEIPT_ATOL):
    if isinstance(actual, bool) or not isinstance(actual, (int, float)) or not math.isfinite(actual) or not math.isclose(actual, expected, rel_tol=RECEIPT_ATOL, abs_tol=atol):
        raise ValueError('Corrupted numerical receipt: ' + label)


def _same_check(saved, actual, target_scale):
    for key in ('accepted', 'finite', 'norm_ok'):
        if saved.get(key) is not actual[key]:
            raise ValueError('Corrupted token-check boolean: ' + key)
    for key in ('actual_norm', 'expected_norm'):
        _same_number(saved.get(key), actual[key], key)
    if len(saved.get('heads', [])) != 1:
        raise ValueError('Only one g_A constraint is permitted')
    for key, value in actual['heads'][0].items():
        # Physical units amplify harmless differences between dense FP64 implementations.
        # This is receipt parity only: neither independent1e-6 feasibility gate changes.
        atol = RECEIPT_ATOL * max(1., float(np.max(target_scale))) if key == 'physical_output_max_deviation' else RECEIPT_ATOL
        _same_number(saved['heads'][0].get(key), value, key, atol=atol)


def accept_t0_four_family(predicted, guides, head_A, replacements, directions, report,
                          *, goal_index, donor_assignment):
    """Independently accept values/receipts and this goal's global donor assignment.

    The caller must authenticate observed donor-token provenance and file/model
    hashes. This routine reconstructs rays/minimum/first backoff and nonlinear
    preservation; it does not independently solve QP optimality.
    """
    head = _inputs(predicted, guides, head_A)
    goal = _index(goal_index, 'goal_index')
    assignment = np.asarray(donor_assignment)
    if (assignment.dtype != np.int64 or assignment.ndim != 1 or len(assignment) == 0 or
            goal >= len(assignment) or not np.array_equal(np.sort(assignment), np.arange(len(assignment), dtype=np.int64))):
        raise ValueError('A complete fixed global donor permutation is required')
    if (np.asarray(replacements).dtype != np.float32 or np.shape(replacements) != (2, 2, LATENT_DIM) or
            np.asarray(directions).dtype != np.float64 or np.shape(directions) != (2, 2, LATENT_DIM) or
            not np.isfinite(replacements).all() or not np.isfinite(directions).all()):
        raise ValueError('Wrong complete-family FP32/FP64 arrays')
    expected = dict(status=STATUS, draft_engineering_only=True, model_role='T0', constraint='g_A',
        family_size=4, members=[list(x) for x in MEMBERS], goal_index=goal,
        donor_index=int(assignment[goal]), **_bindings(predicted, guides, head),
        defaults=dict(shrink_factors=list(SHRINK_FACTORS), tolerance=TOLERANCE,
                      eps_abs=1e-9, eps_rel=1e-9, max_iter=20000,
                      receipt_atol=RECEIPT_ATOL, physical_receipt_atol='receipt_atol*max(1,max_target_scale)'))
    if any(report.get(k) != v for k, v in expected.items()):
        raise ValueError('Changed family identity, roster, donor, inputs or numerical defaults')
    solvers = report.get('solvers', [])
    if len(solvers) != 4 or any(str(s.get('status', '')).lower() != 'solved' or s.get('head_count') != 1 for s in solvers):
        raise ValueError('Every one of four A-only directions must have solved')
    flat = np.asarray(directions).reshape(4, LATENT_DIM)
    # Recompute the declared NumPy L2 operation independently from saved rays.
    # Its operation order also fixes subsequent FP32 ray reconstruction.
    norms = np.array([np.linalg.norm(row) for row in flat], dtype=np.float64)
    if not np.isfinite(norms).all() or len(report.get('native_norms', [])) != 4:
        raise ValueError('Invalid native direction norms')
    for saved, actual in zip(report['native_norms'], norms):
        _same_number(saved, float(actual), 'native_norm')
    minimum = float(np.min(norms))
    _same_number(report.get('unshrunk_common_norm'), minimum, 'unshrunk common norm')
    if report.get('legitimate_zero_norm') is not (minimum == 0.):
        raise ValueError('Corrupted zero-family identity')
    attempts = report.get('attempts', [])
    if not 1 <= len(attempts) <= len(SHRINK_FACTORS):
        raise ValueError('Missing complete first-backoff history')
    accepted = None
    for attempt_index, shrink in enumerate(SHRINK_FACTORS):
        candidates = []; checks = []
        for member_index, (objective, _) in enumerate(MEMBERS):
            magnitude = float(norms[member_index])
            alpha = 0. if magnitude == 0. else minimum * shrink / magnitude
            candidate = (np.array(predicted[objective], np.float64) +
                         (flat[member_index] * alpha) * head['scale']).astype(np.float32)
            candidates.append(candidate)
            checks.append(_independent_token_check(predicted[objective], candidate, head, minimum * shrink))
        if attempt_index >= len(attempts):
            raise ValueError('Backoff history ended before independent acceptance')
        saved_attempt = attempts[attempt_index]
        _same_number(saved_attempt.get('shrink'), shrink, 'scheduled shrink')
        if len(saved_attempt.get('checks', [])) != 4:
            raise ValueError('Incomplete per-attempt four-member checks')
        for saved, actual in zip(saved_attempt['checks'], checks):
            _same_check(saved, actual, head['target_scale'])
        if all(check['accepted'] for check in checks):
            if len(attempts) != attempt_index + 1:
                raise ValueError('Not the first common accepted backoff')
            reconstructed = np.stack(candidates).reshape(2, 2, LATENT_DIM)
            if not np.array_equal(reconstructed, replacements):
                raise ValueError('Saved replacements are not the accepted FP32 rays')
            _same_number(report.get('shrink_factor'), shrink, 'effective shrink')
            _same_number(report.get('effective_norm'), minimum * shrink, 'effective norm')
            accepted = (shrink, checks)
            break
    if accepted is None:
        raise ValueError('No complete independent numerical acceptance')
    return dict(status='PASS_INDEPENDENT_DRAFT_S2_FOUR_FAMILY_ACCEPTANCE',
        draft_engineering_only=True, goal_index=goal, donor_index=int(assignment[goal]),
        family_size=4, effective_norm=minimum * accepted[0], effective_norm_squared=(minimum * accepted[0]) ** 2,
        shrink_factor=accepted[0], checks=accepted[1],
        replacement_values_sha256=_digest({'replacements': replacements}),
        direction_values_sha256=_digest({'directions': directions}),
        donor_assignment_sha256=_digest({'donor_assignment': assignment}),
        scope='Complete T0 four-member value/identity/ray/first-backoff/full-function/region/norm acceptance; caller binds real data/model/source files. QP optimality and donor-image encoding not recomputed.')
