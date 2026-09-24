"""Fixed paired adaptation factorial; all resampling uses shared image indices."""
import numpy as np

CONDITIONS=['trained_cls_frozen','trained_cls_adapted','initial_cls_frozen','initial_cls_adapted']


def contrasts(values):
    assert set(values)==set(CONDITIONS)
    pretrained=values['trained_cls_adapted']-values['trained_cls_frozen']
    initial=values['initial_cls_adapted']-values['initial_cls_frozen']
    return dict(pretrained_adaptation=pretrained,initial_adaptation=initial,
                adaptation_pretraining_interaction=pretrained-initial,
                adapted_pretrained_minus_initial=values['trained_cls_adapted']-values['initial_cls_adapted'],
                frozen_pretrained_minus_initial=values['trained_cls_frozen']-values['initial_cls_frozen'])


def summarize_matrix(values):
    assert set(values)==set(CONDITIONS)
    for v in values.values():assert v.shape==(6,512) and np.isfinite(v).all()
    # Each bootstrap statistic averages six fixed backbones, not six independent seeds.
    ix=np.random.default_rng(1256901).integers(0,512,(10000,512))
    def summary(v):
        ci=np.quantile(v[ix].mean(1),[.025,.975])
        return dict(mean=float(v.mean()),ci95=ci.tolist())
    pooled={k:v.mean(0) for k,v in values.items()}
    return dict(conditions={k:summary(v) for k,v in pooled.items()},
                contrasts={k:summary(v) for k,v in contrasts(pooled).items()},
                per_model=[dict(index=i,conditions={k:summary(v[i]) for k,v in values.items()},
                                contrasts={k:summary(v) for k,v in contrasts({k:v[i] for k,v in values.items()}).items()}) for i in range(6)])
