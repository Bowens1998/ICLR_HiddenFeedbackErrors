"""Same-device first-step audit: original repeat versus inactive auxiliary."""
import argparse,json,hashlib
from pathlib import Path
import torch
from factorial_model import make_model
from action_auxiliary import ActionAuxiliary
from train_visual_updates import Clips,tensor_hash

p=argparse.ArgumentParser()
for k in ['data','official','config','output']:p.add_argument('--'+k,required=True)
a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4)
def rng():return {'cpu':hashlib.sha256(torch.random.get_rng_state().numpy().tobytes()).hexdigest(),'cuda':hashlib.sha256(torch.cuda.get_rng_state().cpu().numpy().tobytes()).hexdigest()}
# Initialize CUDA before every paired path so lazy startup is not a confound.
torch.cuda.init();d=Clips(Path(a.data)/'train');batch=[d[i] for i in range(128)]
pixels=torch.stack([v[0] for v in batch]).cuda().float()/255
pixels=(pixels-torch.tensor([.485,.456,.406],device='cuda')[None,None,:,None,None])/torch.tensor([.229,.224,.225],device='cuda')[None,None,:,None,None]
norm=json.loads((Path(a.data)/'normalization.json').read_text())['action'];actions=torch.stack([v[1] for v in batch]).cuda().float()
actions=(actions-torch.tensor(norm['mean'],device='cuda').repeat(5))/torch.tensor(norm['std'],device='cuda').repeat(5)
report=[]
for arm in ['transformer_jepa','gru_jepa']:
    references={}
    for label in ['original','original_repeat','auxiliary_none']:
        torch.manual_seed(3072);model=make_model(a.official,a.config,arm,3072);trace={'model_constructed':rng()}
        heads=ActionAuxiliary() if label=='auxiliary_none' else None;trace['heads_constructed']=rng()
        model=model.cuda().train()
        if heads is not None:heads=heads.cuda().train()
        from module import SIGReg
        reg=SIGReg().cuda();trace['before_forward']=rng()
        params=list(model.parameters())+([] if heads is None else list(heads.parameters()))
        opt=torch.optim.AdamW(params,lr=5e-5,weight_decay=1e-3);opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            z=model.encode({'pixels':pixels,'action':actions});pred=model.predict(z['emb'][:,:3],z['act_emb'][:,:3]);trace['after_predict']=rng()
            prediction=(pred-z['emb'][:,1:]).square().mean();regularizer=reg(z['emb'].transpose(0,1));loss=prediction+.09*regularizer
            if heads is not None:extra,_=heads.objective(z['emb'],actions[:,:3],'none');loss=loss+extra
        trace['after_loss']=rng();loss.backward();trace['after_backward']=rng()
        grads={k:v.grad.detach().cpu().clone() for k,v in model.named_parameters() if v.grad is not None}
        gn=torch.nn.utils.clip_grad_norm_(params,1.);opt.step()
        weights={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        differences={}
        if label=='original':references={'grads':grads,'weights':weights,'trace':trace}
        else:
            for name,current in [('grads',grads),('weights',weights)]:
                changed=[k for k,v in current.items() if not torch.equal(v,references[name][k])]
                differences[name]={'changed_tensors':len(changed),'max_abs':max([float((current[k]-references[name][k]).abs().max()) for k in changed] or [0.]),'first_keys':changed[:5]}
            differences['rng_matches']={k:trace[k]==references['trace'][k] for k in trace}
        report.append(dict(arm=arm,label=label,prediction=float(prediction.detach()),regularizer=float(regularizer.detach()),loss=float(loss.detach()),gradient_norm=float(gn),trace=trace,differences=differences))
        del model,heads,opt,params,reg,z,pred,loss,grads,weights;torch.cuda.empty_cache()
(out/'report.json').write_text(json.dumps(dict(rows=report,gpu=torch.cuda.get_device_name(),scope='single fixed minibatch, same GPU, initialized CUDA; does not establish213-step equivalence'),indent=2)+'\n')
