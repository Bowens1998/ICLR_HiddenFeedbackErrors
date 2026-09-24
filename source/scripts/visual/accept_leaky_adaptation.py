"""Verify final adaptation weights and decode all training samples; not optimizer replay."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from factorial_model import make_model
from evaluation_precision import configure_evaluation_precision
from leaky_pose_readout import numpy_leaky_pose as numpy_pose
from analyze_coverage_goals import sha,metrics


def main():
    p=argparse.ArgumentParser()
    for k in ['run','config','plan','data','features','frozen-fits','official','model-config']:p.add_argument('--'+k,required=True)
    a=p.parse_args();run=Path(a.run);r=json.loads((run/'report.json').read_text());cfg=json.loads(Path(a.config).read_text())
    assert r['activation']=='leaky_relu' and r['negative_slope']==.01
    assert r['status']=='FIT_REQUIRES_ACCEPTANCE' and r['updates']==cfg['updates']==2000
    assert r['config_sha256']==sha(a.config) and r['source_sha256']==sha(Path(__file__).with_name('fit_leaky_adaptation.py'))
    assert r['weights_sha256']==sha(run/'weights.pt')
    i=r['index'];stage=r['stage'];assert stage in ['trained_cls','initial_cls']
    plan=json.loads(Path(a.plan).read_text());entry=plan['models'][plan['routes'][i*8+2]['model_index']]
    f=Path(a.features)/f'job_{i}';fr=json.loads((f/'report.json').read_text())
    assert r['feature_report_sha256']==sha(f/'report.json') and fr['plan_sha256']==sha(a.plan)
    frozen=Path(a.frozen_fits)/f'job_{i}';assert r['frozen_fit_report_sha256']==sha(frozen/'report.json')
    ar=json.loads((frozen/'acceptance.json').read_text());assert ar['status']=='PASS' and ar['report_sha256']==sha(frozen/'report.json')
    source=Path(a.data)/f'replica_{i//2}';assert sha(source/'report.json')==fr['data_report_sha256']
    dr=json.loads((source/'report.json').read_text());row=next(v for v in dr['rows'] if v['arm']=='broad')
    for name,expected in row['files_sha256'].items():assert sha(source/'broad'/name)==expected
    training=Path(entry['training_path']);assert sha(training/'last_weights.pt')==entry['weights_sha256'] and sha(training/'summary.json')==entry['training_summary_sha256']
    tr=json.loads((training/'summary.json').read_text());assert sha(a.model_config)==tr['config_sha256']
    base=make_model(a.official,a.model_config,entry['arm'],tr['seed'])
    if stage=='trained_cls':base.load_state_dict(torch.load(training/'last_weights.pt',map_location='cpu',weights_only=True),strict=True)
    # Trusted locally generated checkpoint includes NumPy normalization arrays.
    checkpoint=torch.load(run/'weights.pt',map_location='cpu',weights_only=False)
    assert set(checkpoint)=={'encoder','head','normalization'}
    initial=base.encoder.state_dict();d=max(float((checkpoint['encoder'][k]-v).abs().max()) for k,v in initial.items())
    np.testing.assert_allclose(d,r['encoder_parameter_delta'],rtol=0,atol=0);assert d>0
    h0=dict(np.load(frozen/stage/'initial.npz'));hd=max(float((v-torch.tensor(h0[k])).abs().max()) for k,v in checkpoint['head'].items())
    np.testing.assert_allclose(hd,r['head_parameter_delta'],rtol=0,atol=0);assert hd>0
    h=dict(np.load(frozen/stage/'weights.npz'))
    for k,v in checkpoint['normalization'].items():np.testing.assert_array_equal(v,h[k])
    assert all(torch.isfinite(v).all() for group in ['encoder','head'] for v in checkpoint[group].values())
    base.encoder.load_state_dict(checkpoint['encoder'],strict=True);encoder=base.encoder.cuda().eval();del base
    head=nn.Sequential(nn.Linear(192,256),nn.LeakyReLU(.01),nn.Linear(256,256),nn.LeakyReLU(.01),nn.Linear(256,6)).double().cuda()
    head.load_state_dict(checkpoint['head'],strict=True);head.eval()
    configure_evaluation_precision();torch.set_num_threads(2)
    images=np.load(source/'broad/pixels.npy');poses=np.load(source/'broad/poses.npz');assert len(images)==512
    tokens=[];predictions=[]
    norm=checkpoint['normalization'];mean=torch.tensor(norm['mean'],device='cuda',dtype=torch.float64);scale=torch.tensor(norm['scale'],device='cuda',dtype=torch.float64)
    with torch.inference_mode():
        for begin in range(0,512,32):
            x=np.array(images[begin:begin+32],dtype=np.float32).transpose(0,3,1,2)/255
            x=(x-np.array([.485,.456,.406],dtype=np.float32)[None,:,None,None])/np.array([.229,.224,.225],dtype=np.float32)[None,:,None,None]
            emb=encoder(torch.tensor(x,device='cuda'),interpolate_pos_encoding=True).last_hidden_state[:,0]
            prediction=head((emb.double()-mean)/scale).cpu().numpy()*norm['target_scale']+norm['target_mean']
            tokens.append(emb.cpu().numpy());predictions.append(prediction)
    tokens=np.concatenate(tokens);predictions=np.concatenate(predictions)
    nh={k:v.numpy() for k,v in checkpoint['head'].items()};nh.update(norm)
    np.testing.assert_allclose(predictions,numpy_pose(tokens,nh),rtol=1e-10,atol=1e-9)
    np.savez_compressed(run/'accepted_train_predictions.npz',tokens=tokens,prediction=predictions,target=poses['target'],source_index=poses['source_index'])
    x=(tokens.astype(float)-norm['mean'])/norm['scale'];layers=[]
    for layer in (0,2):
        pre=x@nh[f'{layer}.weight'].T+nh[f'{layer}.bias'];x=np.where(pre>=0,pre,.01*pre)
        layers.append(dict(layer=layer,negative_fraction=float((pre<0).mean()),nonpositive_all_rows_units=int((pre<=0).all(0).sum()),mean_activation_variance=float(x.var(0).mean())))
    result=dict(activation='leaky_relu',negative_slope=.01,layers=layers,status='PASS',index=i,stage=stage,report_sha256=sha(run/'report.json'),verifier_sha256=sha(__file__),train_predictions_sha256=sha(run/'accepted_train_predictions.npz'),metrics={k:float(v.mean()) for k,v in metrics(predictions,poses['target']).items()},scope='Final weights/source/data bindings, exact unchanged normalizers, encoder/head deltas independently checked against initialization; all512 training images decoded with Torch float64 head and independent NumPy agreement. Does not replay2000 optimizer updates, independently implement encoder, or establish test performance.')
    with (run/'acceptance.json').open('x') as out:out.write(json.dumps(result,indent=2)+'\n')
    print('PASS_ADAPTED_MODEL',i,stage)


if __name__=='__main__':main()
