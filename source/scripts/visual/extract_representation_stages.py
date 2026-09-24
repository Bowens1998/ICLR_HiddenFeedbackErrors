"""Training-only pre/post projector and initial CLS features; no evaluation input."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from factorial_model import make_model
from evaluation_precision import configure_evaluation_precision
from analyze_coverage_goals import sha


def main():
    p = argparse.ArgumentParser()
    for name in ['config', 'plan', 'data', 'official', 'model-config', 'output']:
        p.add_argument('--'+name, required=True)
    p.add_argument('--index', type=int, choices=range(6), required=True)
    a = p.parse_args()
    cfg = json.loads(Path(a.config).read_text())
    plan = json.loads(Path(a.plan).read_text())
    entry = plan['models'][plan['routes'][a.index*8+2]['model_index']]
    assert entry['score'] == 'pose_encoded' and entry['checkpoint'] == 'last'
    folder = Path(a.data)/f'replica_{a.index//2}'
    dr = json.loads((folder/'report.json').read_text())
    da = json.loads((folder/'acceptance.json').read_text())
    assert da['status'] == 'PASS' and not da['engineering'] and not dr['engineering']
    assert da['report_sha256'] == sha(folder/'report.json')
    assert da['verifier_sha256'] == sha(Path(__file__).with_name('accept_coverage_poses.py'))
    assert dr['replica'] == a.index//2 and dr['config_sha256'] == sha(a.config)
    row = next(v for v in dr['rows'] if v['arm'] == 'broad')
    assert row['count'] == cfg['labels_per_arm'] == 512
    for name, expected in row['files_sha256'].items():
        assert sha(folder/'broad'/name) == expected
    training = Path(entry['training_path'])
    assert sha(training/'last_weights.pt') == entry['weights_sha256']
    assert sha(training/'summary.json') == entry['training_summary_sha256']
    tr = json.loads((training/'summary.json').read_text())
    assert sha(a.model_config) == tr['config_sha256']
    torch.set_num_threads(2)
    precision = configure_evaluation_precision()
    model = make_model(a.official, a.model_config, entry['arm'], tr['seed'])
    initial = make_model(a.official, a.model_config, entry['arm'], tr['seed'])
    # Constructor determinism is checked before loading the trained checkpoint.
    for k, v in model.encoder.state_dict().items():
        torch.testing.assert_close(v, initial.encoder.state_dict()[k], rtol=0, atol=0)
    model.load_state_dict(torch.load(training/'last_weights.pt', map_location='cpu', weights_only=True), strict=True)
    model, initial = model.cuda().eval(), initial.cuda().eval()
    images = np.load(folder/'broad/pixels.npy', mmap_mode='r')
    z = np.load(folder/'broad/poses.npz')
    target, ids = z['target'], z['source_index']
    assert images.shape == (512,224,224,3) and target.shape == (512,6)
    arrays = {k: [] for k in ['trained_cls', 'projected', 'initial_cls']}
    max_native_gap = 0.
    with torch.inference_mode():
        for start in range(0,512,64):
            x = torch.tensor(np.array(images[start:start+64]), device='cuda').permute(0,3,1,2).float()/255
            x = (x-torch.tensor([.485,.456,.406],device='cuda')[None,:,None,None])/torch.tensor([.229,.224,.225],device='cuda')[None,:,None,None]
            cls = model.encoder(x, interpolate_pos_encoding=True).last_hidden_state[:,0]
            projected = model.projector(cls)
            native = model.encode({'pixels': x[:,None]})['emb'][:,0]
            torch.testing.assert_close(projected, native, rtol=2e-5, atol=2e-5)
            max_native_gap = max(max_native_gap, float((projected-native).abs().max()))
            random_cls = initial.encoder(x, interpolate_pos_encoding=True).last_hidden_state[:,0]
            for name, value in [('trained_cls',cls),('projected',projected),('initial_cls',random_cls)]:
                arrays[name].append(value.cpu().numpy())
    out = Path(a.output)/f'job_{a.index}'
    out.mkdir(parents=True, exist_ok=False)
    for name in arrays:
        arrays[name] = np.concatenate(arrays[name])
        assert arrays[name].shape == (512,192) and np.isfinite(arrays[name]).all()
    np.savez_compressed(out/'features.npz', **arrays, target=target, source_index=ids)
    r = dict(status='EXTRACTED_REQUIRES_INDEPENDENT_ACCEPTANCE', index=a.index,
             replica=a.index//2, arm=entry['arm'], initialization_seed=tr['seed'],
             features_sha256=sha(out/'features.npz'), native_projector_max_difference=max_native_gap,
             data_report_sha256=sha(folder/'report.json'), data_acceptance_sha256=sha(folder/'acceptance.json'),
             config_sha256=sha(a.config), plan_sha256=sha(a.plan), backbone_sha256=entry['weights_sha256'],
             model_config_sha256=sha(a.model_config), official_jepa_sha256=sha(Path(a.official)/'jepa.py'),
             precision=precision, gpu=torch.cuda.get_device_name(),
             sources={n:sha(Path(__file__).with_name(n)) for n in ['extract_representation_stages.py','factorial_model.py','lewm_adapter.py','evaluation_precision.py']},
             scope='Training broad512 images only. Three equal192-dimensional feature stages. No fitted probe, holdout inputs or generalization claims.')
    (out/'report.json').write_text(json.dumps(r,indent=2)+'\n')
    print('EXTRACTED_THREE_STAGES', a.index)


if __name__ == '__main__':
    main()
