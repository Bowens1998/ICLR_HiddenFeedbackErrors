"""Complete frozen/adapted × pretrained/initial evaluation, no tuning."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from factorial_model import make_model
from nonlinear_pose_cost import NonlinearPoseCost,numpy_pose
from evaluation_precision import configure_evaluation_precision
from analyze_coverage_goals import sha


def main():
    p=argparse.ArgumentParser()
    for k in ['config','holdout-config','plan','adapted','frozen','features','bank','official','model-config','output']:p.add_argument('--'+k,required=True)
    p.add_argument('--index',type=int,choices=range(6),required=True);a=p.parse_args()
    i=a.index;plan=json.loads(Path(a.plan).read_text());entry=plan['models'][plan['routes'][i*8+2]['model_index']]
    frozen=Path(a.frozen)/f'job_{i}';fr=json.loads((frozen/'report.json').read_text());fa=json.loads((frozen/'acceptance.json').read_text())
    assert fa['status']=='PASS' and fa['report_sha256']==sha(frozen/'report.json') and fr['index']==i
    assert fa['verifier_sha256']==sha(Path(__file__).with_name('accept_representation_readouts.py'))
    features=Path(a.features)/f'job_{i}';er=json.loads((features/'report.json').read_text())
    assert fr['feature_report_sha256']==sha(features/'report.json') and er['plan_sha256']==sha(a.plan) and er['backbone_sha256']==entry['weights_sha256']
    train=Path(entry['training_path']);assert sha(train/'last_weights.pt')==entry['weights_sha256'] and sha(train/'summary.json')==entry['training_summary_sha256']
    tr=json.loads((train/'summary.json').read_text());assert tr['config_sha256']==sha(a.model_config)
    bank=Path(a.bank);bm=json.loads((bank/'report.json').read_text());ba=json.loads((bank/'acceptance.json').read_text());hc=json.loads(Path(a.holdout_config).read_text())
    assert hc['broad_sampling_seeds']==[1261001,1261002,1261003]
    assert ba['status']=='PASS' and not ba['engineering'] and ba['report_sha256']==sha(bank/'report.json')
    assert ba['verifier_sha256']==sha(Path(__file__).with_name('accept_coverage_poses.py'))
    assert bm['config_sha256']==sha(a.holdout_config) and not bm['engineering'] and bm['replica']==0
    row=next(v for v in bm['rows'] if v['arm']=='broad');assert row['count']==512
    for name,expected in row['files_sha256'].items():assert sha(bank/'broad'/name)==expected
    images=np.load(bank/'broad/pixels.npy');poses=np.load(bank/'broad/poses.npz');assert images.shape==(512,224,224,3)
    precision=configure_evaluation_precision();torch.set_num_threads(2)
    out=Path(a.output)/f'job_{i}';out.mkdir(parents=True,exist_ok=False);rows=[]
    for stage in ['trained_cls','initial_cls']:
        for mode in ['frozen','adapted']:
            condition=f'{stage}_{mode}';model=make_model(a.official,a.model_config,entry['arm'],tr['seed'])
            if stage=='trained_cls':model.load_state_dict(torch.load(train/'last_weights.pt',map_location='cpu',weights_only=True),strict=True)
            norm_head=dict(np.load(frozen/stage/'weights.npz'))
            fitrow=next(v for v in fr['rows'] if v['stage']==stage)
            assert sha(frozen/stage/'weights.npz')==fitrow['files_sha256']['weights.npz']
            provenance=dict(frozen_report_sha256=sha(frozen/'report.json'),frozen_acceptance_sha256=sha(frozen/'acceptance.json'))
            if mode=='adapted':
                fit=Path(a.adapted)/f'job_{i}'/stage;ar=json.loads((fit/'report.json').read_text());aa=json.loads((fit/'acceptance.json').read_text())
                assert aa['status']=='PASS' and ar['updates']==2000 and ar['index']==aa['index']==i and ar['stage']==aa['stage']==stage
                assert ar['status']=='FIT_REQUIRES_ACCEPTANCE' and aa['report_sha256']==sha(fit/'report.json')
                assert aa['verifier_sha256']==sha(Path(__file__).with_name('accept_readout_adaptation.py'))
                assert ar['config_sha256']==sha(a.config) and ar['frozen_fit_report_sha256']==sha(frozen/'report.json') and ar['feature_report_sha256']==sha(features/'report.json')
                assert ar['weights_sha256']==sha(fit/'weights.pt')
                checkpoint=torch.load(fit/'weights.pt',map_location='cpu',weights_only=False)
                model.encoder.load_state_dict(checkpoint['encoder'],strict=True)
                head={k:v.numpy() for k,v in checkpoint['head'].items()};head.update(checkpoint['normalization'])
                for k in checkpoint['normalization']:np.testing.assert_array_equal(head[k],norm_head[k])
                provenance.update(adapted_report_sha256=sha(fit/'report.json'),adapted_acceptance_sha256=sha(fit/'acceptance.json'),weights_sha256=sha(fit/'weights.pt'))
            else:
                head=norm_head;provenance['weights_sha256']=sha(frozen/stage/'weights.npz')
            encoder=model.encoder.cuda().eval();del model;encoded=[]
            with torch.inference_mode():
                for begin in range(0,512,64):
                    x=torch.tensor(images[begin:begin+64],device='cuda').permute(0,3,1,2).float()/255
                    x=(x-torch.tensor([.485,.456,.406],device='cuda')[None,:,None,None])/torch.tensor([.229,.224,.225],device='cuda')[None,:,None,None]
                    encoded.append(encoder(x,interpolate_pos_encoding=True).last_hidden_state[:,0].cpu().numpy())
            tokens=np.concatenate(encoded);assert tokens.shape==(512,192) and np.isfinite(tokens).all()
            th={k:torch.tensor(v,device='cuda',dtype=torch.float64) for k,v in head.items()}
            with torch.inference_mode():pred=NonlinearPoseCost.forward(torch.tensor(tokens,device='cuda'),th).cpu().numpy()
            np.testing.assert_allclose(pred,numpy_pose(tokens,head),rtol=1e-10,atol=1e-9)
            np.savez_compressed(out/f'{condition}.npz',tokens=tokens,prediction=pred,target=poses['target'],source_index=poses['source_index'])
            rows.append(dict(condition=condition,file=f'{condition}.npz',file_sha256=sha(out/f'{condition}.npz'),provenance=provenance))
            del encoder,th
    result=dict(status='COMPLETE_ADAPTATION_FACTORIAL',index=i,rows=rows,plan_sha256=sha(a.plan),config_sha256=sha(a.config),holdout_config_sha256=sha(a.holdout_config),bank_report_sha256=sha(bank/'report.json'),bank_acceptance_sha256=sha(bank/'acceptance.json'),source_sha256=sha(__file__),precision=precision,scope='Four fixed conditions on512 new images; no tuning. Native CLS extraction and Torch/NumPy head agreement; independent complete paired analysis required.')
    (out/'report.json').write_text(json.dumps(result,indent=2)+'\n');print('EVALUATED_FOUR_CONDITIONS',i)


if __name__=='__main__':main()
