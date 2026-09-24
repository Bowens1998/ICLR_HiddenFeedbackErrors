"""Match the wide-state control to the complete fixed-data/update experiment."""
import itertools
import numpy as np
from scaling_effects import analyze as validate_scaling


def compare(scaling, wide):
    validate_scaling(scaling['tables'])
    if scaling['bank_manifest_sha256'] != wide['bank_sha256']:
        raise ValueError('Different scenario banks')
    keys=('replica','arm','checkpoint','algorithm')
    expected=set(itertools.product(range(3),['transformer_latent_state','gru_latent_state'],['best','last'],['random','cem']))
    lookup={tuple(r[k] for k in keys):r for r in wide['rows']}
    if len(wide['rows'])!=24 or set(lookup)!=expected:
        raise ValueError('Require all 24 unique wide-state routes')
    baseline={tuple(r[k] for k in keys):r for r in scaling['tables'] if r['episodes']==256 and r['updates']==21000}
    draws=np.random.default_rng(962001).integers(0,128,(10000,128))
    rows=[]
    for w in wide['rows']:
        arch=w['arm'].removesuffix('_latent_state')
        for target in ['state','jepa']:
            b=baseline[(w['replica'],arch+'_'+target,w['checkpoint'],w['algorithm'])]
            x=np.asarray(wide['per_case_costs'][str(w['route'])],dtype=float)
            y=np.asarray(scaling['per_case_costs'][str(b['route_index'])],dtype=float)
            if x.shape!=(128,) or y.shape!=(128,) or not np.isfinite(x).all() or not np.isfinite(y).all():
                raise ValueError('Invalid per-case costs')
            np.testing.assert_allclose(x.mean(),w['mean_cost'],rtol=1e-12)
            np.testing.assert_allclose(y.mean(),b['mean_cost'],rtol=1e-12)
            delta=x-y
            rows.append(dict(replica=w['replica'],architecture=arch,checkpoint=w['checkpoint'],algorithm=w['algorithm'],baseline_target=target,
                wide_route=w['route'],baseline_route=b['route_index'],mean_cost_difference=float(delta.mean()),success_difference=w['success']-b['success'],
                conditional_scenario_95_percentile_interval=np.quantile(delta[draws].mean(1),[.025,.975]).tolist()))
    aggregates=[]
    for arch,checkpoint,algorithm,target in itertools.product(['transformer','gru'],['best','last'],['random','cem'],['state','jepa']):
        selected=sorted((r for r in rows if (r['architecture'],r['checkpoint'],r['algorithm'],r['baseline_target'])==(arch,checkpoint,algorithm,target)),key=lambda r:r['replica'])
        effects=[r['mean_cost_difference'] for r in selected]
        aggregates.append(dict(architecture=arch,checkpoint=checkpoint,algorithm=algorithm,baseline_target=target,pool_cost_effects=effects,
            mean_cost_difference=float(np.mean(effects)),between_pool_cost_effect_sd=float(np.std(effects,ddof=1)),pool_success_effects=[r['success_difference'] for r in selected]))
    return dict(contrasts=rows,aggregates=aggregates,scope='Wide-state minus baseline at matched 256 episodes/21000 updates. Three training pools; best and last kept separate. Nominal scenario intervals condition on fitted models; across-pool SD descriptive. Representation, parameters, heads and objectives differ; not a pure bottleneck intervention.')
