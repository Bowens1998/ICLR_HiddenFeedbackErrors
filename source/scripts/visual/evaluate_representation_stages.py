"""All fixed probes on the new static holdout; no fitting or model selection."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from factorial_model import make_model
from nonlinear_pose_cost import NonlinearPoseCost, numpy_pose
from evaluation_precision import configure_evaluation_precision
from analyze_coverage_goals import sha


def main():
    p=argparse.ArgumentParser()
    for k in ['config','holdout-config','plan','fits','features','bank','official','model-config','output']:
        p.add_argument('--'+k,required=True)
    p.add_argument('--index',type=int,choices=range(6),required=True);a=p.parse_args()
    plan=json.loads(Path(a.plan).read_text());entry=plan['models'][plan['routes'][a.index*8+2]['model_index']]
    fits=Path(a.fits)/f'job_{a.index}';features=Path(a.features)/f'job_{a.index}'
    fr=json.loads((fits/'report.json').read_text());fa=json.loads((fits/'acceptance.json').read_text());er=json.loads((features/'report.json').read_text())
    assert fr['index']==fa['index']==er['index']==a.index and fa['status']=='PASS'
    assert fa['report_sha256']==sha(fits/'report.json') and fa['verifier_sha256']==sha(Path(__file__).with_name('accept_representation_readouts.py'))
    assert fr['feature_report_sha256']==sha(features/'report.json') and fr['feature_acceptance_sha256']==sha(features/'acceptance.json')
    assert er['plan_sha256']==sha(a.plan) and er['backbone_sha256']==entry['weights_sha256']
    assert fr['config_sha256']==er['config_sha256']==sha(a.config)
    bank=Path(a.bank);bm=json.loads((bank/'report.json').read_text());ba=json.loads((bank/'acceptance.json').read_text());hc=json.loads(Path(a.holdout_config).read_text())
    assert hc['broad_sampling_seeds']==[1241001,1241002,1241003]
    assert ba['status']=='PASS' and not ba['engineering'] and ba['report_sha256']==sha(bank/'report.json')
    assert ba['verifier_sha256']==sha(Path(__file__).with_name('accept_coverage_poses.py'))
    assert bm['config_sha256']==sha(a.holdout_config) and bm['replica']==0 and not bm['engineering']
    row=next(v for v in bm['rows'] if v['arm']=='broad');assert row['count']==512
    for name,expected in row['files_sha256'].items():assert sha(bank/'broad'/name)==expected
    training=Path(entry['training_path']);assert sha(training/'last_weights.pt')==entry['weights_sha256'] and sha(training/'summary.json')==entry['training_summary_sha256']
    tr=json.loads((training/'summary.json').read_text());assert sha(a.model_config)==tr['config_sha256'] and er['initialization_seed']==tr['seed']
    precision=configure_evaluation_precision();torch.set_num_threads(2)
    model=make_model(a.official,a.model_config,entry['arm'],tr['seed']);model.load_state_dict(torch.load(training/'last_weights.pt',map_location='cpu',weights_only=True),strict=True);model=model.cuda().eval()
    initial=make_model(a.official,a.model_config,entry['arm'],tr['seed']).cuda().eval()
    images=np.load(bank/'broad/pixels.npy');z=np.load(bank/'broad/poses.npz');target=z['target'];ids=z['source_index']
    assert images.shape==(512,224,224,3) and target.shape==(512,6)
    tokens={k:[] for k in ['trained_cls','projected','initial_cls']}
    with torch.inference_mode():
        for start in range(0,512,64):
            x=torch.tensor(images[start:start+64],device='cuda').permute(0,3,1,2).float()/255
            x=(x-torch.tensor([.485,.456,.406],device='cuda')[None,:,None,None])/torch.tensor([.229,.224,.225],device='cuda')[None,:,None,None]
            cls=model.encoder(x,interpolate_pos_encoding=True).last_hidden_state[:,0]
            native=model.encode({'pixels':x[:,None]})['emb'][:,0]
            torch.testing.assert_close(model.projector(cls),native,rtol=2e-5,atol=2e-5)
            for k,v in [('trained_cls',cls),('projected',native),('initial_cls',initial.encoder(x,interpolate_pos_encoding=True).last_hidden_state[:,0])]:tokens[k].append(v.cpu().numpy())
    tokens={k:np.concatenate(v) for k,v in tokens.items()}
    out=Path(a.output)/f'job_{a.index}';out.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(out/'features.npz',**tokens,target=target,source_index=ids)
    rows=[]
    for stage in tokens:
        row=next(v for v in fr['rows'] if v['stage']==stage);hp=fits/stage/'weights.npz'
        assert row['labels']==512 and row['updates']==2000 and sha(hp)==row['files_sha256']['weights.npz']
        head=dict(np.load(hp));h={k:torch.tensor(v,device='cuda',dtype=torch.float64) for k,v in head.items()}
        with torch.inference_mode():pred=NonlinearPoseCost.forward(torch.tensor(tokens[stage],device='cuda'),h).cpu().numpy()
        np.testing.assert_allclose(pred,numpy_pose(tokens[stage],head),rtol=1e-10,atol=1e-9)
        np.savez_compressed(out/f'{stage}.npz',prediction=pred)
        rows.append(dict(stage=stage,head_sha256=sha(hp),file=f'{stage}.npz',file_sha256=sha(out/f'{stage}.npz')))
    result=dict(status='COMPLETE_REPRESENTATION_EVALUATION',index=a.index,rows=rows,features_sha256=sha(out/'features.npz'),fit_report_sha256=sha(fits/'report.json'),fit_acceptance_sha256=sha(fits/'acceptance.json'),bank_report_sha256=sha(bank/'report.json'),bank_acceptance_sha256=sha(bank/'acceptance.json'),plan_sha256=sha(a.plan),config_sha256=sha(a.config),holdout_config_sha256=sha(a.holdout_config),source_sha256=sha(__file__),precision=precision,scope='All512 independent static images, three fixed feature-stage probes; native projected interface and NumPy head crosschecks. Complete paired analysis required.')
    (out/'report.json').write_text(json.dumps(result,indent=2)+'\n');print('EVALUATED_THREE_STAGES',a.index)


if __name__=='__main__':main()
