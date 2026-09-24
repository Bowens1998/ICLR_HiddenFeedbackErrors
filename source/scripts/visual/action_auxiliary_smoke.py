"""Actual visual minibatch integration and device timing; engineering only."""
import argparse,hashlib,json,time,os
from pathlib import Path
import numpy as np
import torch
from factorial_model import make_model
from action_auxiliary import ActionAuxiliary
from train_visual_updates import Clips,tensor_hash


def main():
    p=argparse.ArgumentParser()
    for key in ['data','official','config','output']:p.add_argument('--'+key,required=True)
    a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    clips=Clips(Path(a.data)/'train');batch=[clips[i] for i in range(16)]
    pixels=torch.stack([v[0] for v in batch]).cuda().float()/255
    pixels=(pixels-torch.tensor([.485,.456,.406],device='cuda')[None,None,:,None,None])/torch.tensor([.229,.224,.225],device='cuda')[None,None,:,None,None]
    actions=torch.stack([v[1] for v in batch]).cuda().float()[:,:3]
    norm=json.loads((Path(a.data)/'normalization.json').read_text())['action']
    actions=(actions-torch.tensor(norm['mean'],device='cuda').repeat(5))/torch.tensor(norm['std'],device='cuda').repeat(5)
    rows=[]
    for arm in ['transformer_jepa','gru_jepa']:
        initial=None
        for mode in ['none','inverse','inverse_goal']:
            torch.manual_seed(3072);model=make_model(a.official,a.config,arm,3072)
            hashes={k:tensor_hash(getattr(model,k)) for k in ['encoder','action_encoder','predictor']}
            if initial is None:initial=hashes
            assert initial==hashes
            model=model.cuda().train();heads=ActionAuxiliary().cuda().train()
            from module import SIGReg
            reg=SIGReg().cuda();parameters=list(model.parameters())+list(heads.parameters());opt=torch.optim.AdamW(parameters,lr=5e-5,weight_decay=1e-3)
            torch.cuda.reset_peak_memory_stats();timings=[];losses=[]
            for step in range(8):
                torch.cuda.synchronize();start=time.perf_counter();opt.zero_grad(set_to_none=True)
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    z=model.encode({'pixels':pixels,'action':actions});pred=model.predict(z['emb'][:,:3],z['act_emb'])
                    wm=(pred-z['emb'][:,1:]).square().mean()+.09*reg(z['emb'].transpose(0,1))
                    extra,parts=heads.objective(z['emb'],actions,mode);loss=wm+extra
                assert torch.isfinite(loss);loss.backward()
                assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.encoder.parameters())
                assert all((p.grad is not None)==(mode!='none') for p in heads.inverse.parameters())
                assert all((p.grad is not None)==(mode=='inverse_goal') for p in heads.goal.parameters())
                torch.nn.utils.clip_grad_norm_(parameters,1.);opt.step();torch.cuda.synchronize()
                timings.append(time.perf_counter()-start);losses.append(float(loss.detach()))
            # Inference does not need auxiliary modules; base model serializes independently.
            weights=out/f'{arm}_{mode}.pt';torch.save(model.state_dict(),weights)
            loaded=make_model(a.official,a.config,arm,3072);loaded.load_state_dict(torch.load(weights,map_location='cpu',weights_only=True),strict=True)
            rows.append(dict(arm=arm,mode=mode,initial_hashes=hashes,steps=8,batch=16,losses=losses,seconds_per_step=timings,warm_median_seconds=float(np.median(timings[2:])),peak_allocated_bytes=torch.cuda.max_memory_allocated(),weights_sha256=hashlib.sha256(weights.read_bytes()).hexdigest()))
            del opt,model,heads,loaded,parameters,reg,z,pred,wm,extra,parts,loss;torch.cuda.empty_cache()
    files=['action_auxiliary_smoke.py','action_auxiliary.py','factorial_model.py','lewm_adapter.py','train_visual_updates.py','update_budget.py']
    report=dict(rows=rows,gpu=torch.cuda.get_device_name(),torch=torch.__version__,job=os.environ.get('SLURM_JOB_ID'),data_manifest_sha256=hashlib.sha256((Path(a.data)/'manifest.json').read_bytes()).hexdigest(),source_sha256={f:hashlib.sha256(Path(__file__).with_name(f).read_bytes()).hexdigest() for f in files},scope='engineering repeated minibatch only; warm timing at batch16 is not full-training throughput or scientific performance')
    (out/'summary.json').write_text(json.dumps(report,indent=2)+'\n');(out/'COMPLETE').write_text('six arm/mode integrations passed\n')

if __name__=='__main__':main()
