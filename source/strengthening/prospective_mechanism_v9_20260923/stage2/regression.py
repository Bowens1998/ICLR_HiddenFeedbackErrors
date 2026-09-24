"""Draft fixed S2 regression/statistics core, independent of models and inputs.

No default bootstrap seed, CI interpolation rule or numerical acceptance
tolerance is supplied: these remain explicit requirements of a future freeze.
This module does not authorize an experiment or prove external access ordering.
"""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Callable, Mapping, Sequence
import numpy as np

N_CALIBRATION = 256
N_TEST = 512
N_STRATA = 6
RIDGE_LAMBDA = 0.01
BOOTSTRAP_DRAWS = 20000
CI_PROBABILITIES = (0.025, 0.975)
RAW_COLUMNS = ('free_error_5', 'free_error_25', 'observed_history_error_25', 'common_norm_squared')
PAIRS = tuple((j, k) for j in range(4) for k in range(j + 1, 4))
BASIS_COLUMNS = (tuple(f'z{j + 1}' for j in range(4)) +
                 tuple(f'z{j + 1}^2' for j in range(4)) +
                 tuple(f'z{j + 1}*z{k + 1}' for j, k in PAIRS))
REQUIRED_BINDINGS = ('protocol_sha256', 'calibration_inputs_sha256',
                     'test_probe_inputs_sha256', 'data_isolation_sha256', 'response_contract_sha256')


def _finite(value, shape, name):
    raw = np.asarray(value)
    if raw.dtype.kind not in 'fiu' or raw.shape != shape:
        raise ValueError(f'{name}: expected numeric shape {shape}')
    a = np.array(raw, dtype=np.float64, copy=True)
    if not np.isfinite(a).all():
        raise ValueError(f'{name}: nonfinite values block the complete population')
    return a


def _ids(value, n, name):
    a = np.asarray(value)
    if a.shape != (n,) or a.dtype.kind not in 'iu' or np.any(a < 0):
        raise ValueError(f'{name}: expected {n} nonnegative integer parent IDs')
    if np.any(a > np.iinfo(np.int64).max) or len(np.unique(a)) != n:
        raise ValueError(f'{name}: repeated or unrepresentable parent IDs')
    return np.array(a, dtype=np.int64, copy=True)


def _strata(value):
    value = tuple(value)
    if len(value) != 6 or len(set(value)) != 6 or not all(isinstance(x, str) and x for x in value):
        raise ValueError('Exactly six explicit unique stratum names are required')
    return value


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def _digest(arrays, metadata):
    h = hashlib.sha256(_json(metadata).encode())
    for name in sorted(arrays):
        a = np.asarray(arrays[name])
        h.update(_json([name, a.dtype.str, list(a.shape)]).encode())
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def _readonly(arrays):
    result = {}
    for name, value in arrays.items():
        a = np.array(value, copy=True); a.setflags(write=False); result[name] = a
    return result


def _moments(x):
    """FP64 population moments along recipient axis; no epsilon floor."""
    constant = np.all(x == x[0:1], axis=0)
    mean = np.mean(x, axis=0, dtype=np.float64)
    # Exact constants (including e.g. repeated 0.1) must remain constants.
    mean = np.where(constant, x[0], mean)
    with np.errstate(over='raise', invalid='raise'):
        variance = np.mean(np.square(x - mean), axis=0, dtype=np.float64)
        scale = np.sqrt(variance)
    scale = np.where(scale == 0., 1., scale)
    if not np.isfinite(mean).all() or not np.isfinite(scale).all():
        raise ValueError('Nonfinite calibration moments; no clipping or replacement')
    return mean, scale


def quadratic_basis(z):
    """Four linear, four squared, six lexicographically ordered cross terms."""
    a = np.asarray(z, dtype=np.float64)
    if a.ndim != 3 or a.shape[1:] != (6, 4) or not np.isfinite(a).all():
        raise ValueError('Basis expects finite recipient × six strata × four features')
    with np.errstate(over='raise', invalid='raise'):
        result = np.concatenate([a, np.square(a), np.stack([a[..., j] * a[..., k] for j, k in PAIRS], axis=-1)], axis=-1)
    if not np.isfinite(result).all():
        raise ValueError('Nonfinite quadratic basis; no winsorization')
    return result


def calibration_preprocessing(ordinary, signed_g):
    """Learn both scaling stages and G moments from exactly 256 recipients."""
    x = _finite(ordinary, (256, 6, 4), 'calibration ordinary features')
    g = _finite(signed_g, (256, 6), 'calibration signed G')
    if (x < 0).any():
        raise ValueError('Ordinary losses and squared dose must be nonnegative')
    raw_mean, raw_scale = _moments(x)
    basis = quadratic_basis((x - raw_mean) / raw_scale)
    basis_mean, basis_scale = _moments(basis)
    g_mean, g_scale = _moments(g)
    moments = dict(raw_mean=raw_mean, raw_scale=raw_scale, basis_mean=basis_mean,
                   basis_scale=basis_scale, g_mean=g_mean, g_scale=g_scale)
    b = (basis - basis_mean) / basis_scale
    a = np.concatenate([b, ((g - g_mean) / g_scale)[..., None]], axis=-1)
    return _readonly(moments), b, a


def apply_preprocessing(ordinary, signed_g, moments):
    """Transform without fitting or receiving any response variable."""
    raw = np.asarray(ordinary)
    if raw.ndim != 3:
        raise ValueError('Ordinary features must have three axes')
    n = len(raw)
    x = _finite(raw, (n, 6, 4), 'ordinary features'); g = _finite(signed_g, (n, 6), 'signed G')
    if (x < 0).any():
        raise ValueError('Ordinary losses and squared dose must be nonnegative')
    shapes = dict(raw_mean=(6, 4), raw_scale=(6, 4), basis_mean=(6, 14), basis_scale=(6, 14),
                  g_mean=(6,), g_scale=(6,))
    m = {key: _finite(moments[key], shape, key) for key, shape in shapes.items()}
    if any(np.any(m[k] <= 0) for k in ['raw_scale', 'basis_scale', 'g_scale']):
        raise ValueError('Frozen feature scales must be positive')
    with np.errstate(over='raise', invalid='raise', divide='raise'):
        raw_z = (x - m['raw_mean']) / m['raw_scale']
        b = (quadratic_basis(raw_z) - m['basis_mean']) / m['basis_scale']
        a = np.concatenate([b, ((g - m['g_mean']) / m['g_scale'])[..., None]], axis=-1)
    if not np.isfinite(a).all():
        raise ValueError('Nonfinite transformed test feature')
    return b, a


def _ridge(x, u, *, verification_atol, verification_rtol):
    """Intercept elimination, checked against the full normal equations.

    Objective: ||u - D alpha - X beta||² / (6*256) + .01 ||beta||².
    Equivalently unnormalized slope equations add (6*256*.01) I.
    """
    if any(not np.isfinite(v) or v < 0 for v in [verification_atol, verification_rtol]):
        raise ValueError('Explicit nonnegative numerical tolerances required')
    n, strata, k = x.shape
    if (n, strata) != (256, 6) or k not in [14, 15]:
        raise ValueError('Wrong frozen calibration regression dimensions')
    xm = np.mean(x, axis=0, dtype=np.float64); ym, _ = _moments(u)
    xc = (x - xm).reshape(n * 6, k); yc = (u - ym).reshape(n * 6)
    rows = n * 6
    beta = np.linalg.solve(xc.T @ xc / rows + RIDGE_LAMBDA * np.eye(k), xc.T @ yc / rows)
    alpha = ym - np.einsum('sk,k->s', xm, beta)
    # Independent assembly includes six genuinely unpenalized intercepts.
    dummy = np.tile(np.eye(6), (n, 1)); design = np.concatenate([dummy, x.reshape(rows, k)], axis=1)
    penalty = np.diag(np.r_[np.zeros(6), np.full(k, RIDGE_LAMBDA)])
    gram = design.T @ design / rows + penalty; rhs = design.T @ u.reshape(rows) / rows
    full = np.linalg.solve(gram, rhs); theta = np.r_[alpha, beta]
    if not np.allclose(theta, full, atol=verification_atol, rtol=verification_rtol):
        raise ValueError('Independent normal-equation solution mismatch')
    residual = gram @ theta - rhs
    if not np.all(np.abs(residual) <= verification_atol + verification_rtol * np.abs(rhs)):
        raise ValueError('Explicit normal-equation residual exceeded frozen tolerance')
    fitted = alpha[None, :] + np.einsum('nsk,k->ns', x, beta)
    if not np.isfinite(fitted).all():
        raise ValueError('Nonfinite fitted response')
    diagnostic = dict(rows=rows, slope_columns=k, ridge_lambda=RIDGE_LAMBDA,
        unnormalized_ridge=rows * RIDGE_LAMBDA, maximum_coefficient_verification_difference=float(np.max(np.abs(theta - full))),
        maximum_normal_equation_residual=float(np.max(np.abs(residual))),
        objective=float(np.mean(np.square(u - fitted)) + RIDGE_LAMBDA * (beta @ beta)))
    return alpha, beta, fitted, diagnostic


@dataclass(frozen=True)
class CalibrationFit:
    arrays: Mapping[str, np.ndarray]
    metadata: Mapping
    sha256: str


def fit_calibration(ordinary, signed_g, response, *, recipient_ids, donor_ids,
                    stratum_ids: Sequence[str], verification_atol: float,
                    verification_rtol: float) -> CalibrationFit:
    """Fit only the two fixed calibration regressors, preserving complete inputs."""
    x = _finite(ordinary, (256, 6, 4), 'calibration ordinary features')
    g = _finite(signed_g, (256, 6), 'calibration signed G')
    y = _finite(response, (256, 6), 'calibration response')
    ids = _ids(recipient_ids, 256, 'calibration recipients'); donors = _ids(donor_ids, 256, 'calibration donors')
    if set(ids) & set(donors):
        raise ValueError('Recipient and donor parents overlap')
    strata = _strata(stratum_ids)
    moments, b, a = calibration_preprocessing(x, g)
    target_mean, _ = _moments(y)
    with np.errstate(over='raise', invalid='raise'):
        sy = float(np.sqrt(np.mean(np.square(y - target_mean), dtype=np.float64)))
    if sy == 0.:
        sy = 1.
    if not np.isfinite(sy):
        raise ValueError('Nonfinite common target scale')
    arrays = dict(moments, calibration_ordinary=x, calibration_signed_g=g, calibration_response=y,
        calibration_recipient_ids=ids, calibration_donor_ids=donors,
        calibration_basis_baseline=b, calibration_basis_augmented=a,
        target_scale=np.asarray(sy), target_stratum_mean=target_mean)
    diagnostics = {}
    for name, features in [('baseline', b), ('augmented', a)]:
        alpha, beta, fitted, diag = _ridge(features, y / sy,
            verification_atol=verification_atol, verification_rtol=verification_rtol)
        arrays.update({name + '_intercepts': alpha, name + '_slopes': beta,
                       name + '_calibration_prediction': fitted * sy,
                       name + '_calibration_residual': y - fitted * sy})
        diagnostics[name] = diag
    metadata = dict(schema='s2_fixed_quadratic_calibration_v1_draft', calibration_count=256, stratum_ids=list(strata),
        raw_columns=list(RAW_COLUMNS), ordinary_basis_columns=list(BASIS_COLUMNS), augmented_extra='signed_G_only',
        ridge_lambda=RIDGE_LAMBDA, target_scaling='single RMS around six calibration stratum means',
        moments='within-stratum FP64 population; exact constant mean=first value; zero variance scale=1',
        verification_atol=float(verification_atol), verification_rtol=float(verification_rtol),
        diagnostics=diagnostics, scientific_protocol_frozen_by_this_module=False)
    arrays = _readonly(arrays)
    return CalibrationFit(arrays, metadata, _digest(arrays, metadata))


@dataclass(frozen=True)
class TestPredictions:
    arrays: Mapping[str, np.ndarray]
    metadata: Mapping
    sha256: str


def predict_test(fit: CalibrationFit, ordinary, signed_g, *, recipient_ids, donor_ids,
                 stratum_ids: Sequence[str]) -> TestPredictions:
    """Exactly 512 held-out recipients; this API has no response argument."""
    if _digest(fit.arrays, fit.metadata) != fit.sha256:
        raise ValueError('Calibration coefficients or preprocessing changed')
    strata = _strata(stratum_ids)
    if list(strata) != fit.metadata['stratum_ids']:
        raise ValueError('Stratum order changed')
    x = _finite(ordinary, (512, 6, 4), 'test ordinary features'); g = _finite(signed_g, (512, 6), 'test signed G')
    ids = _ids(recipient_ids, 512, 'test recipients'); donors = _ids(donor_ids, 512, 'test donors')
    historical = set(fit.arrays['calibration_recipient_ids']) | set(fit.arrays['calibration_donor_ids'])
    if set(ids) & set(donors) or historical & (set(ids) | set(donors)):
        raise ValueError('Calibration/test/recipient/donor parent overlap')
    b, a = apply_preprocessing(x, g, fit.arrays); sy = float(fit.arrays['target_scale'])
    arrays = dict(test_ordinary=x, test_signed_g=g, test_recipient_ids=ids, test_donor_ids=donors,
        test_basis_baseline=b, test_basis_augmented=a)
    for name, features in [('baseline', b), ('augmented', a)]:
        value = sy * (fit.arrays[name + '_intercepts'][None, :] + np.einsum('nsk,k->ns', features, fit.arrays[name + '_slopes']))
        if not np.isfinite(value).all():
            raise ValueError('Nonfinite prediction; no recipient dropped')
        arrays[name + '_prediction'] = value
    metadata = dict(schema='s2_fixed_test_predictions_v1_draft', calibration_sha256=fit.sha256,
        count=512, stratum_ids=list(strata), response_opened_by_this_function=False)
    arrays = _readonly(arrays)
    return TestPredictions(arrays, metadata, _digest(arrays, metadata))


def _file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def _hex(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def _bootstrap_settings(seed, bit_generator, quantile_method):
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or not 0 <= int(seed) < 2**64:
        raise ValueError('An explicit frozen unsigned bootstrap seed is required')
    if bit_generator != 'PCG64' or quantile_method != 'linear':
        raise ValueError('Explicitly bind supported PCG64 and linear quantiles before use')
    return dict(bootstrap_seed=int(seed), bit_generator=bit_generator,
                quantile_method=quantile_method, draws=BOOTSTRAP_DRAWS,
                ci_probabilities=list(CI_PROBABILITIES), resampling_unit='whole_recipient')


def seal_test_predictions(output, fit: CalibrationFit, predictions: TestPredictions, *, bindings: Mapping,
                          bootstrap_seed, bit_generator: str, quantile_method: str):
    """Exclusive saved-array seal before a separate response loader is invoked.

    Caller must provide real future-frozen input/protocol hashes. The artifact
    binds predictions; it cannot prove nobody externally inspected responses.
    """
    if any(k not in bindings or not _hex(bindings[k]) for k in REQUIRED_BINDINGS):
        raise ValueError('Explicit frozen protocol/calibration/probe/isolation hashes required')
    settings = _bootstrap_settings(bootstrap_seed, bit_generator, quantile_method)
    if (_digest(fit.arrays, fit.metadata) != fit.sha256 or
            _digest(predictions.arrays, predictions.metadata) != predictions.sha256 or
            predictions.metadata['calibration_sha256'] != fit.sha256):
        raise ValueError('Modified or unrelated calibration/prediction record')
    out = Path(output); out.mkdir(parents=True, exist_ok=False)
    for name, record in [('calibration', fit), ('predictions', predictions)]:
        with (out / (name + '.npz')).open('xb') as f:
            np.savez_compressed(f, **record.arrays)
        with (out / (name + '.json')).open('x') as f:
            f.write(_json(dict(record.metadata)) + '\n')
    lock = dict(status='PREDICTIONS_SEALED_BEFORE_RESPONSE_LOADER',
        schema='s2_prediction_access_contract_v1_draft', bindings=dict(bindings),
        calibration_sha256=fit.sha256, predictions_sha256=predictions.sha256,
        files={name: _file_sha(out / name) for name in ['calibration.npz', 'calibration.json', 'predictions.npz', 'predictions.json']},
        count=512, stratum_ids=predictions.metadata['stratum_ids'],
        statistical_settings=settings,
        scope='Hash binding and API access order only; external pipeline must enforce prior nonexposure. Not a scientific protocol freeze.')
    path = out / 'PREDICTIONS.lock.json'
    with path.open('x') as f:
        f.write(_json(lock) + '\n')
    return dict(path=str(path), sha256=_file_sha(path))


def verify_prediction_seal(path, checksum, *, expected_bindings):
    path = Path(path)
    if not _hex(checksum) or _file_sha(path) != checksum:
        raise ValueError('Prediction seal changed')
    lock = json.loads(path.read_text())
    if (lock.get('status') != 'PREDICTIONS_SEALED_BEFORE_RESPONSE_LOADER' or
            lock.get('bindings') != dict(expected_bindings) or lock.get('count') != 512):
        raise ValueError('Prediction access contract identity mismatch')
    settings = lock['statistical_settings']
    if settings != _bootstrap_settings(settings['bootstrap_seed'], settings['bit_generator'], settings['quantile_method']):
        raise ValueError('Changed frozen bootstrap settings')
    expected_files = {'calibration.npz', 'calibration.json', 'predictions.npz', 'predictions.json'}
    if set(lock['files']) != expected_files:
        raise ValueError('Incomplete or unsafe prediction seal paths')
    for name, expected in lock['files'].items():
        if _file_sha(path.parent / name) != expected:
            raise ValueError('Sealed calibration or prediction payload changed')
    records = {}
    for name in ['calibration', 'predictions']:
        with np.load(path.parent / (name + '.npz'), allow_pickle=False) as z:
            arrays = {k: z[k] for k in z.files}
        meta = json.loads((path.parent / (name + '.json')).read_text())
        if _digest(arrays, meta) != lock[name + '_sha256']:
            raise ValueError('Content digest mismatch')
        records[name] = (arrays, meta)
    ca, cm = records['calibration']; pa, pm = records['predictions']
    fit = CalibrationFit(ca, cm, lock['calibration_sha256'])
    # Reconstruct frozen predictions, not just their self-consistent hashes.
    reconstructed = predict_test(fit, pa['test_ordinary'], pa['test_signed_g'],
        recipient_ids=pa['test_recipient_ids'], donor_ids=pa['test_donor_ids'], stratum_ids=pm['stratum_ids'])
    for key in ['test_basis_baseline', 'test_basis_augmented', 'baseline_prediction', 'augmented_prediction']:
        np.testing.assert_array_equal(reconstructed.arrays[key], pa[key])
    return lock, pa, pm


def paired_test_statistics(response, baseline, augmented, *, bootstrap_seed,
                           bit_generator: str, quantile_method: str):
    """One contrast: mean squared-error improvement over 512 whole recipients."""
    y = _finite(response, (512, 6), 'held-out response')
    b = _finite(baseline, (512, 6), 'baseline prediction'); a = _finite(augmented, (512, 6), 'augmented prediction')
    _bootstrap_settings(bootstrap_seed, bit_generator, quantile_method)
    with np.errstate(over='raise', invalid='raise'):
        lb, la = np.square(y - b), np.square(y - a)
        paired = np.mean(lb - la, axis=1, dtype=np.float64)
    if not np.isfinite(paired).all():
        raise ValueError('Nonfinite primary score')
    rng = np.random.Generator(np.random.PCG64(int(bootstrap_seed)))
    boot = np.empty(BOOTSTRAP_DRAWS, np.float64)
    # Generate recipient indices only. Each score already retains all six rows.
    for start in range(0, BOOTSTRAP_DRAWS, 128):
        stop = min(start + 128, BOOTSTRAP_DRAWS)
        indices = rng.integers(0, 512, size=(stop - start, 512), dtype=np.int64)
        boot[start:stop] = paired[indices].mean(axis=1, dtype=np.float64)
    ci = np.quantile(boot, CI_PROBABILITIES, method=quantile_method)
    mse_b, mse_a = float(lb.mean()), float(la.mean()); delta = float(paired.mean())
    return dict(baseline_mse=mse_b, augmented_mse=mse_a, paired_improvement=delta,
        paired_recipient_scores=paired, stratum_improvements=(lb - la).mean(axis=0),
        bootstrap_draws=BOOTSTRAP_DRAWS, bootstrap_seed=int(bootstrap_seed),
        bit_generator=bit_generator, quantile_method=quantile_method,
        bootstrap_means=boot, ci_95=ci, positive_support=bool(ci[0] > 0),
        relative_mse_reduction=delta / mse_b if mse_b > 0 else None,
        resampling_unit='whole test recipient retaining six strata and assigned donor',
        scope='Conditional on the frozen calibration sample, six model pairs, readouts, '
              'realized calibration/test donor banks and fixed assignments; recipient rows alone are resampled, with no refitting.')


def evaluate_response_after_seal(path, checksum, *, expected_bindings, response_loader: Callable,
                                 ):
    """Verify/reconstruct saved predictions before invoking the sole response callback.

    Loader returns receipt_path and receipt_sha256. The generated receipt must
    bind a predeclared response contract, this prediction-lock hash, and an NPZ
    containing response[512,6], recipient_ids and donor_ids. No unknown future
    response-array hash is required before generation. External scheduling must
    prevent earlier exposure and enforce the actual model/scorer contract.
    """
    lock, predictions, metadata = verify_prediction_seal(path, checksum, expected_bindings=expected_bindings)
    descriptor = response_loader()
    receipt_path = Path(descriptor['receipt_path'])
    if not _hex(descriptor['receipt_sha256']) or _file_sha(receipt_path) != descriptor['receipt_sha256']:
        raise ValueError('Response receipt hash mismatch')
    receipt = json.loads(receipt_path.read_text())
    expected = dict(status='ACCEPTED_S2_HELDOUT_RESPONSE', count=512,
        protocol_sha256=lock['bindings']['protocol_sha256'], prediction_lock_sha256=checksum,
        response_contract_sha256=lock['bindings']['response_contract_sha256'])
    if any(receipt.get(k) != v for k, v in expected.items()):
        raise ValueError('Response contract/prediction-lock binding mismatch')
    arrays = receipt['arrays']
    if not _hex(arrays['sha256']) or _file_sha(arrays['path']) != arrays['sha256']:
        raise ValueError('Response array hash mismatch')
    with np.load(arrays['path'], allow_pickle=False) as z:
        response = _finite(z['response'], (512, 6), 'held-out response')
        ids = _ids(z['recipient_ids'], 512, 'response recipients')
        donors = _ids(z['donor_ids'], 512, 'response donors')
    if (not np.array_equal(ids, predictions['test_recipient_ids']) or
            not np.array_equal(donors, predictions['test_donor_ids']) or
            list(_strata(receipt['stratum_ids'])) != metadata['stratum_ids']):
        raise ValueError('Response parent/stratum/source binding mismatch; no automatic reordering')
    stats = paired_test_statistics(response, predictions['baseline_prediction'],
        predictions['augmented_prediction'], bootstrap_seed=lock['statistical_settings']['bootstrap_seed'],
        bit_generator=lock['statistical_settings']['bit_generator'], quantile_method=lock['statistical_settings']['quantile_method'])
    stats.update(prediction_lock_sha256=checksum, response_arrays_sha256=arrays['sha256'],
                 response_receipt_sha256=descriptor['receipt_sha256'],
                 response_contract_sha256=lock['bindings']['response_contract_sha256'],
                 protocol_sha256=lock['bindings']['protocol_sha256'])
    return stats
