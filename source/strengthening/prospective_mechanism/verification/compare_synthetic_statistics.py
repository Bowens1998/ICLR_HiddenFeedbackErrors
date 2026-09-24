"""Synthetic-only production/reference cross-check; never loads experiment arrays."""
from pathlib import Path
import importlib.util
import hashlib
import json
import datetime
import numpy as np
from independent_measurement_reference import goal_vectors, count_weighted_bootstrap, NAMES

BASE = Path(__file__).resolve().parents[1]
PRODUCTION = BASE / 'scripts/summarize_s1.py'
spec = importlib.util.spec_from_file_location('s1_production_synthetic_only', PRODUCTION)
production = importlib.util.module_from_spec(spec)
spec.loader.exec_module(production)
lock = json.loads((BASE / 'protocol/DESIGN.lock.json').read_text())
rng = np.random.default_rng(73108)
p = rng.normal(size=(2, 2, 2, 256, 4, 4, 5, 6))
t = rng.normal(size=(2, 256, 4, 5, 6))
for branch in (0, 3):
    p[:, :, 1, :, :, branch] = p[:, :, 0, :, :, branch]
ix = np.random.default_rng(lock['statistics']['bootstrap_seed']).integers(0, 256, size=(20000, 256), dtype=np.int64)
reference, _ = goal_vectors(p, t)
vectors, _ = production.compute_goal_contrasts(p, t)
pv = np.column_stack([vectors[key] for key in NAMES])
np.testing.assert_allclose(reference, pv, rtol=1e-13, atol=1e-13)
ref_rows, _ = count_weighted_bootstrap(reference, ix)
prod_rows = production.bootstrap_summary(vectors, ix, lock['statistics'])
flat = prod_rows['primary'] + prod_rows['secondary']
residual = 0.0
for r, s in zip(ref_rows, flat):
    assert r['id'] == s['id']
    a = np.array([r['mean'], r['lower'], r['upper']])
    b = np.array([s['estimate']] + s['ci'])
    np.testing.assert_allclose(a, b, rtol=1e-13, atol=1e-13)
    residual = max(residual, float(np.max(np.abs(a-b))))
    assert (r['classification'] == 'positive') == s['positive_support']
sha = lambda f: hashlib.sha256(Path(f).read_bytes()).hexdigest()
report = {'status':'PASS_SYNTHETIC_PRODUCTION_REFERENCE_PARITY_ONLY',
          'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
          'fixture_seed':73108,'shape':list(p.shape),'goals':256,'draws':20000,
          'max_vector_residual':float(np.max(np.abs(reference-pv))),
          'max_mean_interval_residual':residual,
          'reference_algorithm':'per-goal count-matrix bootstrap',
          'production_algorithm':'index-gather bootstrap',
          'sources':{str(f.relative_to(BASE)):sha(f) for f in [Path(__file__),Path(__file__).with_name('independent_measurement_reference.py'),PRODUCTION,BASE/'protocol/DESIGN.lock.json']},
          'experimental_data_accessed':False}
output=BASE/'verification/SYNTHETIC_STATISTICS_PARITY.json'
output.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
