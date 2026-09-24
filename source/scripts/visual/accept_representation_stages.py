"""Recompute all training representations with a separate batched extraction path."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from factorial_model import make_model
from evaluation_precision import configure_evaluation_precision
from analyze_coverage_goals import sha


def main():
    p=argparse.ArgumentParser()
    for k in ['run','data','plan','config','official','model-config']:
        p.add_argument('--'+k,required=True)
    a=p.parse_args();run=Path(a.run);r=json.loads((run/'report.json').read_text())
    assert r['status']=='EXTRACTED_REQUIRES_INDEPENDENT_ACCEPTANCE'
    for name,expected in r['sources'].items():assert sha(Path(__file__).with_name(name))==expected
    assert r['config_sha256']==sha(a.config) and r['plan_sha256']==sha(a.plan)
    assert r['model_config_sha256']==sha(a.model_config) and r['official_jepa_sha256']==sha(Path(a.official)/'jepa.py')
    assert sha(run/'features.npz')==r['features_sha256']
    plan=json.loads(Path(a.plan).read_text());entry=plan['models'][plan['routes'][r['index']*8+2]['model_index']]
    assert entry['weights_sha256']==r['backbone_sha256'] and entry['arm']==r['arm']
    train=Path(entry['training_path']);assert sha(train/'last_weights.pt')==entry['weights_sha256']
    assert sha(train/'summary.json')==entry['training_summary_sha256']
    tr=json.loads((train/'summary.json').read_text());assert tr['seed']==r['initialization_seed']
    data=Path(a.data)/f"replica_{r['replica']}";assert r['replica']==r['index']//2
    assert sha(data/'report.json')==r['data_report_sha256'] and sha(data/'acceptance.json')==r['data_acceptance_sha256']
    dr=json.loads((data/'report.json').read_text());da=json.loads((data/'acceptance.json').read_text())
    assert da['status']=='PASS' and not da['engineering'] and da['report_sha256']==sha(data/'report.json')
    assert da['verifier_sha256']==sha(Path(__file__).with_name('accept_coverage_poses.py'))
    row=next(v for v in dr['rows'] if v['arm']=='broad')
    for name,expected in row['files_sha256'].items():assert sha(data/'broad'/name)==expected
    features=dict(np.load(run/'features.npz'));poses=np.load(data/'broad/poses.npz')
    np.testing.assert_array_equal(features['target'],poses['target'])
    np.testing.assert_array_equal(features['source_index'],poses['source_index'])
    images=np.load(data/'broad/pixels.npy',mmap_mode='r');assert len(images)==512
    configure_evaluation_precision();torch.set_num_threads(2)
    frozen=make_model(a.official,a.model_config,entry['arm'],tr['seed'])
    frozen.load_state_dict(torch.load(train/'last_weights.pt',map_location='cpu',weights_only=True),strict=True)
    initial=make_model(a.official,a.model_config,entry['arm'],tr['seed'])
    frozen=frozen.cuda().eval();initial=initial.cuda().eval()
    maxima={k:0. for k in ['trained_cls','projected','initial_cls']}
    with torch.inference_mode():
        for begin in range(0,512,32):
            batch=np.asarray(images[begin:begin+32]).astype(np.float32).transpose(0,3,1,2)/np.float32(255)
            batch=(batch-np.array([.485,.456,.406],dtype=np.float32)[None,:,None,None])/np.array([.229,.224,.225],dtype=np.float32)[None,:,None,None]
            x=torch.from_numpy(batch.copy()).cuda()
            cls=frozen.encoder(pixel_values=x,interpolate_pos_encoding=True).last_hidden_state.select(1,0)
            random_cls=initial.encoder(pixel_values=x,interpolate_pos_encoding=True).last_hidden_state.select(1,0)
            values={'trained_cls':cls,'projected':frozen.projector(cls),'initial_cls':random_cls}
            for name,tensor in values.items():
                actual=tensor.cpu().numpy();expected=features[name][begin:begin+32]
                assert actual.shape==expected.shape==(32,192) and np.isfinite(actual).all()
                np.testing.assert_allclose(actual,expected,rtol=2e-5,atol=2e-5)
                maxima[name]=max(maxima[name],float(abs(actual-expected).max()))
    result=dict(status='PASS',index=r['index'],report_sha256=sha(run/'report.json'),features_sha256=sha(run/'features.npz'),verifier_sha256=sha(__file__),images_per_stage=512,max_abs_difference=maxima,scope='All three feature stages recomputed from bound images and frozen/initial weights using32-image batches and NumPy preprocessing; exact target and source-index checks. Shared model constructor and libraries, no claim of independent implementation of ViT.')
    with (run/'acceptance.json').open('x') as f:f.write(json.dumps(result,indent=2)+'\n')
    print('PASS_ALL1536_FEATURE_ROWS',r['index'])


if __name__=='__main__':main()
