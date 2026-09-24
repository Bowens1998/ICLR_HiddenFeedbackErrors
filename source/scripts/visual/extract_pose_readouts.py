"""Hash-bound PushT encoded/predicted endpoints from fixed-last JEPA models."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from factorial_model import make_model
from evaluation_precision import configure_evaluation_precision
from pose_readout_dataset import PoseReadoutClips


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    for name in ('manifest','training','data','official','config','output'):
        p.add_argument('--'+name, required=True)
    p.add_argument('--index', type=int, choices=range(6), required=True)
    p.add_argument('--engineering', action='store_true')
    a = p.parse_args()
    manifest = json.loads(Path(a.manifest).read_text())
    assert manifest['layout'] == 'action_auxiliary'
    rep, arm = a.index//2, ('transformer_jepa','gru_jepa')[a.index%2]
    entries = [e for e in manifest['models'] if
               (e['replica'],e['arm'],e['mode'],e['checkpoint']) == (rep,arm,'none','last')]
    assert len(entries) == 1
    entry = entries[0]; run = Path(a.training)/entry['training_path']
    assert sha(run/'summary.json') == entry['training_summary_sha256']
    assert sha(run/'last_weights.pt') == entry['weights_sha256']
    report = json.loads((run/'summary.json').read_text())
    assert (report['arm'],report['mode'],report['seed'],report['completed_updates']) == (arm,'none',3072,21000)
    assert sha(a.config) == report['config_sha256']
    data = Path(a.data)/f'replica_{rep}'/'n256'
    assert sha(data/'manifest.json') == entry['data_manifest_sha256'] == report['data_manifest_sha256']
    dm = json.loads((data/'manifest.json').read_text())
    assert sha(data/'normalization.json') == dm['normalization_sha256']
    assert json.loads((data/'normalization.json').read_text())['action'] == report['normalization']
    assert set(dm['splits']['train']['episode_ids']).isdisjoint(dm['splits']['validation']['episode_ids'])
    datasets = {}
    for split in ('train','validation'):
        for name in ('pixels.npy','action.npy','state.npy','episodes.npz'):
            assert sha(data/split/name) == dm['splits'][split]['files'][name], (split,name)
        datasets[split] = PoseReadoutClips(data/split)
    out = Path(a.output)/f'job_{a.index}'
    out.mkdir(parents=True, exist_ok=False)
    precision = configure_evaluation_precision(); torch.set_num_threads(2)
    model = make_model(a.official,a.config,arm,report['seed'])
    model.load_state_dict(torch.load(run/'last_weights.pt',map_location='cpu',weights_only=True),strict=True)
    model = model.cuda().eval()
    im = torch.tensor([.485,.456,.406],device='cuda')[None,None,:,None,None]
    sd = torch.tensor([.229,.224,.225],device='cuda')[None,None,:,None,None]
    am = torch.tensor(report['normalization']['mean'],device='cuda').repeat(5)
    ast = torch.tensor(report['normalization']['std'],device='cuda').repeat(5)
    counts = {}; checks = {}
    for split, dataset in datasets.items():
        chunks = {k:[] for k in ('encoded','predicted','target','identity')}
        for batch,(pixels,actions,target,identity) in enumerate(DataLoader(dataset,batch_size=64,shuffle=False,num_workers=0)):
            pixels = (pixels.cuda().float()/255-im)/sd
            actions = (actions.cuda()-am)/ast
            with torch.inference_mode():
                history = model.encode({'pixels':pixels[:,:3]})['emb']
                endpoint = model.encode({'pixels':pixels[:,3:]})['emb'][:,0]
                encoded_actions = model.action_encoder(actions)
                for step in range(2,7):
                    history = torch.cat([history,model.predict(history[:,-3:],encoded_actions[:,step-2:step+1])[:,-1:]],1)
                predicted = history[:,-1]
                if batch == 0:
                    native = model.rollout({'pixels':pixels[:,None,:3]},actions[:,None])['predicted_emb'][:,0,-1]
                    torch.testing.assert_close(native,predicted,rtol=2e-5,atol=2e-5)
                    checks[split] = dict(batch_size=len(pixels),max_abs=float((native-predicted).abs().max()))
            for name,value in [('encoded',endpoint.cpu().numpy()),('predicted',predicted.cpu().numpy()),('target',target.numpy()),('identity',identity.numpy())]:
                assert np.isfinite(value).all(); chunks[name].append(value)
            if a.engineering:
                break
        arrays = {k:np.concatenate(v) for k,v in chunks.items()}
        counts[split] = len(arrays['target'])
        assert counts[split] == (min(64,len(dataset)) if a.engineering else len(dataset))
        np.savez_compressed(out/f'{split}_features.npz',**arrays)
    source = [Path(__file__),Path(__file__).with_name('pose_readout_dataset.py'),
              Path(__file__).with_name('factorial_model.py'),Path(__file__).with_name('lewm_adapter.py'),
              Path(__file__).with_name('evaluation_precision.py'),Path(a.official)/'jepa.py',Path(a.official)/'module.py']
    result = dict(protocol='pusht_nonlinear_transfer_features_v1',engineering=a.engineering,
                  index=a.index,replica=rep,arm=arm,checkpoint='last',mode='none',
                  weights_sha256=entry['weights_sha256'],training_summary_sha256=entry['training_summary_sha256'],
                  manifest_sha256=sha(a.manifest),data_manifest_sha256=sha(data/'manifest.json'),config_sha256=sha(a.config),
                  counts=counts,native_checks=checks,precision=precision,gpu=torch.cuda.get_device_name(),
                  source_sha256={f.name:sha(f) for f in source},
                  files_sha256={f.name:sha(f) for f in out.glob('*.npz')},
                  scope='Paired endpoints only; no fitted head or planning outcome. Independent feature acceptance required.')
    (out/'report.json').write_text(json.dumps(result,indent=2)+'\n')
    (out/'COMPLETE').write_text('extraction complete; independent acceptance required\n')
    print(json.dumps(result),flush=True)


if __name__ == '__main__':
    main()
