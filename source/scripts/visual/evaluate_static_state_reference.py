"""Frozen supervised reference on consumed static holdout; no training/selection."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from factorial_model import make_model
from evaluation_precision import configure_evaluation_precision
from analyze_coverage_goals import sha


def main():
    p=argparse.ArgumentParser()
    for k in ['plan','bank','holdout-config','official','model-config','output']:p.add_argument('--'+k,required=True)
    p.add_argument('--index',type=int,choices=range(6),required=True);a=p.parse_args()
    plan=json.loads(Path(a.plan).read_text());entry=plan['models'][plan['routes'][a.index*8+6]['model_index']]
    assert entry['score']=='state' and entry['mode']=='direct_state' and entry['backbone_index']==a.index
    train=Path(entry['training_path']);assert sha(train/'last_weights.pt')==entry['weights_sha256'] and sha(train/'summary.json')==entry['training_summary_sha256']
    tr=json.loads((train/'summary.json').read_text());assert tr['config_sha256']==sha(a.model_config)
    bank=Path(a.bank);r=json.loads((bank/'report.json').read_text());acc=json.loads((bank/'acceptance.json').read_text())
    assert acc['status']=='PASS' and acc['report_sha256']==sha(bank/'report.json') and acc['verifier_sha256']==sha(Path(__file__).with_name('accept_coverage_poses.py'))
    assert r['config_sha256']==sha(a.holdout_config) and r['replica']==0 and not r['engineering']
    row=next(v for v in r['rows'] if v['arm']=='broad');assert row['count']==512
    for name,expected in row['files_sha256'].items():assert sha(bank/'broad'/name)==expected
    images=np.load(bank/'broad/pixels.npy');z=np.load(bank/'broad/poses.npz')
    precision=configure_evaluation_precision();torch.set_num_threads(2)
    model=make_model(a.official,a.model_config,entry['arm'],tr['seed']);model.load_state_dict(torch.load(train/'last_weights.pt',map_location='cpu',weights_only=True),strict=True);model=model.cuda().eval()
    encoded=[]
    with torch.inference_mode():
        for start in range(0,512,64):
            x=torch.tensor(images[start:start+64],device='cuda').permute(0,3,1,2).float()/255
            x=(x-torch.tensor([.485,.456,.406],device='cuda')[None,:,None,None])/torch.tensor([.229,.224,.225],device='cuda')[None,:,None,None]
            encoded.append(model.encode({'pixels':x[:,None]})['emb'][:,0].cpu().numpy())
    encoded=np.concatenate(encoded);assert encoded.shape==(512,6) and np.isfinite(encoded).all()
    norm=entry['target_normalization'];pred=encoded.astype(float)*np.array(norm['std'])+np.array(norm['mean'])
    out=Path(a.output)/f'job_{a.index}';out.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(out/'predictions.npz',encoded=encoded,prediction=pred,target=z['target'],source_index=z['source_index'])
    result=dict(status='COMPLETE_STATIC_STATE_REFERENCE',index=a.index,arm=entry['arm'],file_sha256=sha(out/'predictions.npz'),plan_sha256=sha(a.plan),weights_sha256=entry['weights_sha256'],bank_report_sha256=sha(bank/'report.json'),bank_acceptance_sha256=sha(bank/'acceptance.json'),source_sha256=sha(__file__),precision=precision,scope='Native frozen direct-state supervised reference. Greater state-label pretraining budget than512-label JEPA probes, not a matched-supervision comparison. Consumed static bank, diagnostic only.')
    (out/'report.json').write_text(json.dumps(result,indent=2)+'\n');print('EVALUATED_STATE_REFERENCE',a.index)


if __name__=='__main__':main()
