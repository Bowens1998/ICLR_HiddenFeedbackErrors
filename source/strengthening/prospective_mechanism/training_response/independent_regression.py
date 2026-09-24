"""Independent S2 FP64 saved-artifact verifier DRAFT; never loads world models.

All bindings/tolerances/seeds must be supplied externally before actual use.
No numerical or integrity function is imported from the production regression.
This verifier does not establish prior nonexposure outside its own access order.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

CAL, TEST, STRATA, DRAWS, PENALTY = 256, 512, 6, 20000, .01
RAW = ['free_error_5', 'free_error_25', 'observed_history_error_25', 'common_norm_squared']
CROSSES = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
BASIS = ['z1', 'z2', 'z3', 'z4', 'z1^2', 'z2^2', 'z3^2', 'z4^2',
         'z1*z2', 'z1*z3', 'z1*z4', 'z2*z3', 'z2*z4', 'z3*z4']
BINDINGS = {'protocol_sha256', 'calibration_inputs_sha256', 'test_probe_inputs_sha256',
            'data_isolation_sha256', 'response_contract_sha256'}
SCOPE = ('Conditional on the frozen calibration sample, six model pairs, readouts, '
         'realized calibration/test donor banks and fixed assignments; recipient rows alone are resampled, with no refitting.')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def hex_sha(x):
    return isinstance(x, str) and len(x) == 64 and all(c in '0123456789abcdef' for c in x)


def _unique_pairs(pairs):
    result = {}
    for k, v in pairs:
        require(k not in result, 'Duplicate JSON key: ' + k)
        result[k] = v
    return result


def read_json(path):
    return json.loads(Path(path).read_text(), object_pairs_hook=_unique_pairs,
                      parse_constant=lambda x: (_ for _ in ()).throw(ValueError('Nonfinite JSON: ' + x)))


def checked_file(descriptor, name):
    require(isinstance(descriptor, dict) and set(descriptor) == {'path', 'sha256'}, name + ': exact path/hash required')
    p = Path(descriptor['path'])
    require(p.is_absolute() and hex_sha(descriptor['sha256']), name + ': absolute path and SHA-256 required')
    require(file_sha(p) == descriptor['sha256'], name + ': file hash mismatch')
    return p


def load_npz(path):
    with np.load(path, allow_pickle=False) as z:
        require(len(z.files) == len(set(z.files)), 'Duplicate NPZ key')
        return {k: z[k] for k in z.files}


def content_digest(arrays, metadata):
    # Reproduce the declared serialization contract, not producer arithmetic.
    compact = lambda x: json.dumps(x, sort_keys=True, separators=(',', ':'), allow_nan=False)
    h = hashlib.sha256(compact(metadata).encode())
    for key in sorted(arrays):
        a = np.asarray(arrays[key])
        h.update(compact([key, a.dtype.str, list(a.shape)]).encode())
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def finite64(a, shape, name):
    a = np.asarray(a)
    require(a.shape == shape and a.dtype == np.dtype('float64'), name + ': exact FP64 shape required')
    require(np.isfinite(a).all(), name + ': nonfinite complete population')
    return a


def ids(a, n, name):
    a = np.asarray(a)
    require(a.shape == (n,) and a.dtype == np.dtype('int64'), name + ': exact int64 shape required')
    require(np.all(a >= 0) and len(set(a.tolist())) == n, name + ': invalid/repeated parent IDs')
    return a


def explicit_tolerances(value, name):
    require(isinstance(value, dict) and set(value) == {'atol', 'rtol'}, name + ': explicit atol/rtol required')
    for v in value.values():
        require(not isinstance(v, bool) and isinstance(v, (int, float)) and math.isfinite(v) and v >= 0,
                name + ': finite nonnegative tolerances required')
    return value


def settings(seed):
    require(not isinstance(seed, bool) and isinstance(seed, int) and 0 <= seed < 2**64,
            'Explicit unsigned 64-bit bootstrap seed required')
    return dict(bootstrap_seed=seed, bit_generator='PCG64', quantile_method='linear',
                draws=DRAWS, ci_probabilities=[.025, .975], resampling_unit='whole_recipient')


def moments(x):
    """Scalar, compensated population reductions separate from production means."""
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    mean = np.empty(x.shape[1:], np.float64)
    scale = np.empty_like(mean)
    for index in np.ndindex(mean.shape):
        values = x[(slice(None),) + index]
        if np.all(values == values[0]):
            mu, sd = float(values[0]), 1.
        else:
            mu = math.fsum(float(v) for v in values) / n
            var = math.fsum((float(v) - mu) ** 2 for v in values) / n
            sd = math.sqrt(var)
            if sd == 0.:
                sd = 1.
        mean[index], scale[index] = mu, sd
    require(np.isfinite(mean).all() and np.isfinite(scale).all(), 'Nonfinite calibration moments')
    return mean, scale


def basis(z):
    out = np.empty((len(z), STRATA, 14), np.float64)
    for j in range(4):
        out[:, :, j] = z[:, :, j]
        out[:, :, j + 4] = z[:, :, j] * z[:, :, j]
    for c, (j, k) in enumerate(CROSSES, 8):
        out[:, :, c] = z[:, :, j] * z[:, :, k]
    require(np.isfinite(out).all(), 'Nonfinite quadratic basis')
    return out


def preprocess(cal_x, cal_g, test_x, test_g):
    require(np.all(cal_x >= 0) and np.all(test_x >= 0), 'Ordinary losses/dose must be nonnegative')
    rm, rs = moments(cal_x)
    cb_raw = basis((cal_x - rm) / rs)
    bm, bs = moments(cb_raw)
    gm, gs = moments(cal_g)
    cb = (cb_raw - bm) / bs
    tb = (basis((test_x - rm) / rs) - bm) / bs
    ca = np.empty((CAL, STRATA, 15), np.float64)
    ta = np.empty((TEST, STRATA, 15), np.float64)
    ca[:, :, :14], ta[:, :, :14] = cb, tb
    ca[:, :, 14] = (cal_g - gm) / gs
    ta[:, :, 14] = (test_g - gm) / gs
    return dict(raw_mean=rm, raw_scale=rs, basis_mean=bm, basis_scale=bs, g_mean=gm, g_scale=gs), cb, ca, tb, ta


def full_ridge(features, response_scaled):
    """Full six-intercept design, stratum-major ordering, augmented SVD solve.

    The producer eliminates intercepts and uses recipient-major normal equations.
    Here all 6+K coefficients are solved jointly, penalizing only the K slopes.
    """
    k = features.shape[-1]
    require(features.shape == (CAL, STRATA, k) and k in (14, 15), 'Wrong calibration design')
    n = CAL * STRATA
    design = np.zeros((n, 6 + k), np.float64)
    response = np.empty(n, np.float64)
    for s in range(6):
        rows = slice(s * CAL, (s + 1) * CAL)
        design[rows, s] = 1.
        design[rows, 6:] = features[:, s, :]
        response[rows] = response_scaled[:, s]
    augmented = np.zeros((n + k, 6 + k), np.float64)
    augmented[:n] = design / math.sqrt(n)
    augmented[n:, 6:] = math.sqrt(PENALTY) * np.eye(k)
    rhs = np.r_[response / math.sqrt(n), np.zeros(k)]
    theta, _, rank, _ = np.linalg.lstsq(augmented, rhs, rcond=None)
    require(rank == 6 + k and np.isfinite(theta).all(), 'Rank/nonfinite failure; no model substitution')
    residual = design @ theta - response
    gradient = design.T @ residual / n
    gradient[6:] += PENALTY * theta[6:]
    objective = math.fsum(float(v) ** 2 for v in residual) / n + PENALTY * float(theta[6:] @ theta[6:])
    return theta[:6], theta[6:], gradient, objective


def predict(features, alpha, beta, sy):
    out = np.empty(features.shape[:2], np.float64)
    for s in range(6):
        out[:, s] = sy * (alpha[s] + features[:, s, :] @ beta)
    require(np.isfinite(out).all(), 'Nonfinite complete predictions')
    return out


def recompute(ca, pa):
    m, cb, ac, tb, ta = preprocess(ca['calibration_ordinary'], ca['calibration_signed_g'],
                                 pa['test_ordinary'], pa['test_signed_g'])
    y = ca['calibration_response']
    ym, _ = moments(y)
    sy = math.sqrt(math.fsum(float(v) ** 2 for v in (y - ym).flat) / (CAL * STRATA))
    if sy == 0.:
        sy = 1.
    require(math.isfinite(sy), 'Nonfinite shared target scale')
    c = dict(m, calibration_basis_baseline=cb, calibration_basis_augmented=ac,
             target_scale=np.asarray(sy), target_stratum_mean=ym)
    p = dict(test_basis_baseline=tb, test_basis_augmented=ta)
    diagnostics = {}
    for name, cx, tx in [('baseline', cb, tb), ('augmented', ac, ta)]:
        alpha, beta, grad, obj = full_ridge(cx, y / sy)
        fit = predict(cx, alpha, beta, sy)
        c.update({name + '_intercepts': alpha, name + '_slopes': beta,
                  name + '_calibration_prediction': fit, name + '_calibration_residual': y - fit})
        p[name + '_prediction'] = predict(tx, alpha, beta, sy)
        diagnostics[name] = dict(gradient=grad, objective=obj)
    return c, p, diagnostics


def compare(a, b, tolerance, name, differences):
    a, b = np.asarray(a), np.asarray(b)
    require(a.dtype.kind in 'fiu' and b.dtype.kind in 'fiu', name + ': numeric values required')
    a, b = a.astype(np.float64), b.astype(np.float64)
    require(a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all(), name + ': shape/finite mismatch')
    err = np.abs(a - b)
    require(np.all(err <= tolerance['atol'] + tolerance['rtol'] * np.abs(b)), name + ': independent numerical mismatch')
    differences[name] = float(np.max(err)) if err.size else 0.


def independent_statistics(y, baseline, augmented, seed):
    """Use the accepted sealed predictors; compensated scores and count bootstrap."""
    settings(seed)
    with np.errstate(over='raise', invalid='raise'):
        lb, la = (y - baseline) ** 2, (y - augmented) ** 2
    require(np.isfinite(lb).all() and np.isfinite(la).all(), 'Nonfinite squared loss')
    difference = lb - la
    scores = np.asarray([math.fsum(float(v) for v in difference[i]) / STRATA for i in range(TEST)])
    strata = np.asarray([math.fsum(float(v) for v in difference[:, j]) / TEST for j in range(STRATA)])
    bm = math.fsum(float(v) for v in lb.flat) / (TEST * STRATA)
    am = math.fsum(float(v) for v in la.flat) / (TEST * STRATA)
    improvement = math.fsum(float(v) for v in scores) / TEST
    rng = np.random.Generator(np.random.PCG64(seed))
    boot = np.empty(DRAWS, np.float64)
    for i in range(DRAWS):
        draw = rng.integers(TEST, size=TEST, dtype=np.int64)
        weights = np.bincount(draw, minlength=TEST)
        boot[i] = float(weights @ scores) / TEST
    ordered = np.sort(boot)
    ci = []
    for p in [.025, .975]:
        position = (DRAWS - 1) * p
        lo = math.floor(position)
        ci.append(float(ordered[lo] + (position - lo) * (ordered[lo + 1] - ordered[lo])))
    return dict(baseline_mse=bm, augmented_mse=am, paired_improvement=improvement,
                paired_recipient_scores=scores, stratum_improvements=strata,
                bootstrap_draws=DRAWS, bootstrap_seed=seed, bit_generator='PCG64', quantile_method='linear',
                bootstrap_means=boot, ci_95=np.asarray(ci), positive_support=bool(ci[0] > 0),
                relative_mse_reduction=improvement / bm if bm > 0 else None,
                resampling_unit='whole test recipient retaining six strata and assigned donor', scope=SCOPE)


def validate_records(ca, cm, pa, pm, contract, lock):
    strata = contract['expected_stratum_ids']
    require(isinstance(strata, list) and len(strata) == 6 and len(set(strata)) == 6 and
            all(isinstance(x, str) and x for x in strata), 'Six explicit ordered strata required')
    require(lock['stratum_ids'] == cm['stratum_ids'] == pm['stratum_ids'] == strata, 'Stratum order mismatch')
    require(cm.get('schema') == 's2_fixed_quadratic_calibration_v1_draft' and cm.get('calibration_count') == CAL and
            cm.get('raw_columns') == RAW and cm.get('ordinary_basis_columns') == BASIS and
            cm.get('augmented_extra') == 'signed_G_only' and cm.get('ridge_lambda') == PENALTY and
            cm.get('scientific_protocol_frozen_by_this_module') is False, 'Calibration algorithm/schema changed')
    require(cm.get('target_scaling') == 'single RMS around six calibration stratum means' and
            cm.get('moments') == 'within-stratum FP64 population; exact constant mean=first value; zero variance scale=1',
            'Calibration scaling rule changed')
    require(pm.get('schema') == 's2_fixed_test_predictions_v1_draft' and pm.get('count') == TEST and
            pm.get('response_opened_by_this_function') is False and pm.get('calibration_sha256') == lock['calibration_sha256'],
            'Prediction schema/lineage changed')
    prod_tol = explicit_tolerances(contract['production_verification_tolerances'], 'Production verification tolerances')
    require(all(cm.get('verification_' + k) == v for k, v in prod_tol.items()), 'Production tolerance drift')
    c_shapes = dict(raw_mean=(6,4), raw_scale=(6,4), basis_mean=(6,14), basis_scale=(6,14),
                    g_mean=(6,), g_scale=(6,), calibration_ordinary=(CAL,6,4), calibration_signed_g=(CAL,6),
                    calibration_response=(CAL,6), calibration_basis_baseline=(CAL,6,14),
                    calibration_basis_augmented=(CAL,6,15), target_scale=(), target_stratum_mean=(6,))
    for name, k in [('baseline',14), ('augmented',15)]:
        c_shapes.update({name + '_intercepts':(6,), name + '_slopes':(k,),
                         name + '_calibration_prediction':(CAL,6), name + '_calibration_residual':(CAL,6)})
    p_shapes = dict(test_ordinary=(TEST,6,4), test_signed_g=(TEST,6), test_basis_baseline=(TEST,6,14),
                    test_basis_augmented=(TEST,6,15), baseline_prediction=(TEST,6), augmented_prediction=(TEST,6))
    roles = {'calibration_recipient_ids':(ca,CAL), 'calibration_donor_ids':(ca,CAL),
             'test_recipient_ids':(pa,TEST), 'test_donor_ids':(pa,TEST)}
    require(set(ca) == set(c_shapes) | set(list(roles)[:2]) and set(pa) == set(p_shapes) | set(list(roles)[2:]),
            'Incomplete/unexpected sealed array schema')
    for source, shapes in [(ca,c_shapes), (pa,p_shapes)]:
        for name, shape in shapes.items():
            finite64(source[name], shape, name)
    expected_ids = contract['expected_parent_ids']
    require(set(expected_ids) == set(roles), 'Four externally bound complete parent rosters required')
    populations = []
    for name, (source, n) in roles.items():
        actual = ids(source[name], n, name)
        expected = expected_ids[name]
        require(isinstance(expected, list) and len(expected) == n and
                all(isinstance(v, int) and not isinstance(v, bool) and 0 <= v < 2**63 for v in expected),
                name + ': expected roster invalid')
        require(actual.tolist() == expected, name + ': externally bound parent ordering mismatch')
        populations.append(set(actual.tolist()))
    require(sum(map(len,populations)) == len(set.union(*populations)), 'Parent role overlap')
    require(all(np.all(ca[x] > 0) for x in ['raw_scale','basis_scale','g_scale','target_scale']), 'Nonpositive scale')


def verify_artifacts(contract):
    """Authenticate/recompute predictions before opening held-out response bytes.

    A draft verifier contract is independently hash-bound by the CLI. It is not
    a scientific freeze and cannot prove external response nonexposure.
    """
    require(contract.get('schema') == 's2_independent_regression_binding_v1_draft', 'Unknown verifier binding schema')
    tol = explicit_tolerances(contract['comparison_tolerances'], 'Independent comparison tolerances')
    expected = contract['expected_bindings']
    require(isinstance(expected, dict) and set(expected) == BINDINGS and all(hex_sha(v) for v in expected.values()),
            'Exactly five external provenance hashes required')
    frozen_settings = contract['expected_statistical_settings']
    require(frozen_settings == settings(frozen_settings['bootstrap_seed']), 'Bootstrap configuration drift')
    lock_path = checked_file(contract['prediction_lock'], 'Prediction lock')
    lock = read_json(lock_path)
    require(lock.get('status') == 'PREDICTIONS_SEALED_BEFORE_RESPONSE_LOADER' and
            lock.get('schema') == 's2_prediction_access_contract_v1_draft' and lock.get('count') == TEST and
            lock.get('bindings') == expected and lock.get('statistical_settings') == frozen_settings,
            'Prediction access contract/settings mismatch')
    names = {'calibration.npz','calibration.json','predictions.npz','predictions.json'}
    require(set(lock['files']) == names, 'Incomplete/unsafe sealed file roster')
    for name in names:
        require(hex_sha(lock['files'][name]) and file_sha(lock_path.parent / name) == lock['files'][name],
                name + ': seal payload changed')
    ca, pa = load_npz(lock_path.parent/'calibration.npz'), load_npz(lock_path.parent/'predictions.npz')
    cm, pm = read_json(lock_path.parent/'calibration.json'), read_json(lock_path.parent/'predictions.json')
    require(content_digest(ca,cm) == lock['calibration_sha256'] and content_digest(pa,pm) == lock['predictions_sha256'],
            'Sealed record content digest mismatch')
    validate_records(ca,cm,pa,pm,contract,lock)
    rc, rp, diagnostics = recompute(ca,pa)
    differences = {}
    for computed, stored, prefix in [(rc,ca,'calibration/'),(rp,pa,'predictions/')]:
        for name, value in computed.items():
            compare(value,stored[name],tol,prefix+name,differences)
    for name, k in [('baseline',14),('augmented',15)]:
        diag = cm['diagnostics'][name]
        require(diag['rows'] == CAL*STRATA and diag['slope_columns'] == k and
                diag['ridge_lambda'] == PENALTY and diag['unnormalized_ridge'] == CAL*STRATA*PENALTY,
                name + ': ridge normalization/intercept design changed')
        compare(diagnostics[name]['objective'],diag['objective'],tol,name+'/objective',differences)
        compare(diagnostics[name]['gradient'],np.zeros(6+k),tol,name+'/full_design_gradient',differences)
    # Only after all raw-feature/calibration/prediction checks may response data open.
    receipt_path = checked_file(contract['response_receipt'], 'Response receipt')
    receipt = read_json(receipt_path)
    for key, value in dict(status='ACCEPTED_S2_HELDOUT_RESPONSE',count=TEST,
                           protocol_sha256=expected['protocol_sha256'],
                           prediction_lock_sha256=contract['prediction_lock']['sha256'],
                           response_contract_sha256=expected['response_contract_sha256'],
                           stratum_ids=contract['expected_stratum_ids']).items():
        require(receipt.get(key) == value, 'Response contract/seal/order mismatch: '+key)
    response_path = checked_file(receipt['arrays'], 'Response arrays')
    response = load_npz(response_path)
    require(set(response) == {'response','recipient_ids','donor_ids'}, 'Incomplete/unexpected response arrays')
    y = finite64(response['response'],(TEST,6),'Held-out response')
    for key, saved in [('recipient_ids','test_recipient_ids'),('donor_ids','test_donor_ids')]:
        require(np.array_equal(ids(response[key],TEST,key),pa[saved]), 'Response parent ordering mismatch')
    stats = independent_statistics(y,pa['baseline_prediction'],pa['augmented_prediction'],frozen_settings['bootstrap_seed'])
    stats.update(prediction_lock_sha256=contract['prediction_lock']['sha256'],
                 response_arrays_sha256=receipt['arrays']['sha256'],
                 response_receipt_sha256=contract['response_receipt']['sha256'],
                 response_contract_sha256=expected['response_contract_sha256'],protocol_sha256=expected['protocol_sha256'])
    recorded = read_json(checked_file(contract['production_statistics'], 'Production statistics'))
    require(set(recorded) == set(stats), 'Incomplete/unexpected saved statistic fields')
    for name,value in stats.items():
        if isinstance(value,(str,bool)) or value is None or isinstance(value,int):
            require(recorded[name] == value and type(recorded[name]) is type(value), name+': statistics contract mismatch')
        else:
            compare(value,recorded[name],tol,'statistics/'+name,differences)
    report = dict(status='PASS_S2_INDEPENDENT_REGRESSION_DRAFT',scientific_protocol_frozen=False,
                  actual_provenance_accepted_only_via_external_hash_bindings=True,
                  calibration_count=CAL,test_count=TEST,stratum_count=6,bootstrap_draws=DRAWS,
                  prediction_lock_sha256=contract['prediction_lock']['sha256'],
                  response_receipt_sha256=contract['response_receipt']['sha256'],
                  production_statistics_sha256=contract['production_statistics']['sha256'],
                  expected_bindings=expected,statistical_settings=frozen_settings,comparison_tolerances=tol,
                  maximum_absolute_differences=differences,positive_support=stats['positive_support'],
                  independent_scalar_statistics={k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in stats.items()
                      if not isinstance(v,np.ndarray) or k == 'ci_95'},
                  scope=SCOPE,limitations='Saved-array arithmetic and declared provenance only; no raw model replay, '
                  'no independent admission/isolation execution, no proof of external prior nonexposure.')
    arrays = {'calibration__'+k:v for k,v in rc.items()}
    arrays.update({'predictions__'+k:v for k,v in rp.items()})
    arrays.update({'statistics__'+k:v for k,v in stats.items() if isinstance(v,np.ndarray)})
    return report, arrays


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binding',required=True)
    parser.add_argument('--binding-sha256',required=True)
    parser.add_argument('--output',required=True)
    args = parser.parse_args()
    binding_path = checked_file(dict(path=args.binding,sha256=args.binding_sha256),'Verifier binding')
    report, arrays = verify_artifacts(read_json(binding_path))
    out = Path(args.output)
    out.mkdir(parents=True,exist_ok=False)
    with (out/'recomputed.npz').open('xb') as f:
        np.savez_compressed(f,**arrays)
    report.update(binding_sha256=args.binding_sha256,verifier_source_sha256=file_sha(__file__),
                  recomputed_arrays_sha256=file_sha(out/'recomputed.npz'))
    with (out/'report.json').open('x') as f:
        json.dump(report,f,sort_keys=True,indent=2,allow_nan=False)
        f.write('\n')
    print(json.dumps(dict(status=report['status'],report=str(out/'report.json'),sha256=file_sha(out/'report.json'))))


if __name__ == '__main__':
    main()
