"""Verify every legacy latent continuation's inputs, control, and frozen tensors."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'scripts/visual'),str(ROOT/'strengthening/adapters')]
from contracts import sha, atomic_json
from run_adaptation_checkpoint_gate import tensor_digest


def main():
    p=argparse.ArgumentParser();p.add_argument('--base',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    base=Path(a.base);r=base/'releases/planner-data-adaptation-v1/runs'
    roster=json.loads((ROOT/'strengthening/manifests/legacy_model_roster.json').read_text());rows=[]
    schedule=r/'formal_stream_freeze/update_indices.npz';indices=np.load(schedule)['indices']
    assert indices.shape==(2100,128)
    for epoch in indices.reshape(210,1280):np.testing.assert_array_equal(np.sort(epoch),np.arange(1280))
    for group in roster['groups']:
        g=group['group'];entry=next(x for x in group['models'] if x['condition']=='latent')
        original=next(x for x in group['models'] if x['condition']=='original_21000')
        wrapper=Path(entry['checkpoint']);summary=wrapper.parent/'summary.json';s=json.loads(summary.read_text())
        fit=Path(s['fit_report']);f=json.loads(fit.read_text());accept=fit.parent/'acceptance.json';ac=json.loads(accept.read_text())
        assert sha(fit)==s['fit_report_sha256']==ac['report_sha256'] and sha(accept)==s['fit_acceptance_sha256']
        assert ac['status']=='PASS_TASK_COORDINATE_FORMAL_FIT_AND_CPU_PREDICTIONS'
        assert f['index']==4*g and f['objective']=='latent' and f['stream']=='planner'
        assert f['updates']==2100 and f['batch_size']==128 and f['learning_rate']==1e-5 and f['weight_decay']==1e-3
        assert f['loss_weight']==1. and f['clip_norm'] is None and f['gradient_control']=='unit_global_norm_before_adamw'
        assert f['source_sha256']==sha(ROOT/'scripts/visual/train_task_coordinate_formal.py')
        assert f['control_source_sha256']==sha(ROOT/'scripts/visual/adaptation_gradient_control.py')
        assert f['loss_source_sha256']==sha(ROOT/'scripts/visual/adaptation_task_losses.py')
        assert f['schedule_sha256']==sha(schedule)
        cache=r/f'formal_cache/job_{2*g}';cr=json.loads((cache/'report.json').read_text())
        assert f['cache_report_sha256']==sha(cache/'report.json') and f['cache_acceptance_sha256']==sha(cache/'acceptance.json')
        for item in cr['rows']:assert sha(cache/item['file'])==item['sha256']
        assert f['entry']['weights_sha256']==original['checkpoint_sha256']==sha(original['checkpoint'])
        assert str(Path(f['entry']['training_path'])/'last_weights.pt')==original['checkpoint']
        tr=json.loads((Path(original['checkpoint']).parent/'summary.json').read_text())
        assert sha(base/'assets/pusht-v1/models/config.json')==tr['config_sha256']==s['config_sha256']
        assert tr['normalization']==s['normalization']
        assert sha(wrapper)==entry['checkpoint_sha256']==f['weights_sha256']==sha(fit.parent/'last_weights.pt')
        before=torch.load(original['checkpoint'],map_location='cpu',weights_only=True)
        after=torch.load(wrapper,map_location='cpu',weights_only=True)
        assert tensor_digest(before)==f['initial_tensor_sha256']==cr['initial_tensor_sha256']
        frozen={k:v for k,v in after.items() if k not in f['trainable_names']}
        for k,v in frozen.items():assert torch.equal(before[k],v),k
        assert tensor_digest(frozen)==f['frozen_before_sha256']==f['frozen_after_sha256']==cr['frozen_tensor_sha256']
        rows.append(dict(group=g,checkpoint=str(wrapper),checkpoint_sha256=sha(wrapper),fit_report=str(fit),
            fit_report_sha256=sha(fit),acceptance_sha256=sha(accept),cache_report_sha256=sha(cache/'report.json'),
            schedule_sha256=sha(schedule),normalizers_equal=True,frozen_tensors_equal=True,source_matches=True,
            conclusion='REUSE_UNCHANGED_LATENT_CONTINUATION',new_head_affects_training=False))
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    atomic_json(out/'report.json',dict(status='PASS_SIX_LEGACY_LATENT_REUSE_BINDINGS',rows=rows,source_sha256=sha(__file__),
        scope='Complete provenance and tensor checks, existing independently accepted predictions. Optimizer trajectory not rerun.'))
    (out/'DONE').write_text('accepted\n')


if __name__=='__main__':main()
