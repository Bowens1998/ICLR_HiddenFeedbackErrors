"""Budget-matched random shooting and clipped diagonal CEM with explicit traces."""
import torch

def expand_actions(parameters):
    if parameters.shape[-1]==2:return parameters[:,:,None,:].expand(-1,-1,5,-1).reshape(-1,25,2)
    assert parameters.shape[-1]==10
    return parameters.reshape(-1,25,2)

def search(cost_function,*,algorithm,parameterization,device,seed,steps=30,samples=300,elites=30):
    assert algorithm in ['random','cem'] and parameterization in ['held','full'] and 1<=elites<=samples
    generator=torch.Generator(device=device).manual_seed(seed);dim=2 if parameterization=='held' else 10
    mean=torch.zeros(5,dim,device=device);std=torch.full_like(mean,.2);trace=[];best=None
    for iteration in range(steps):
        parameters=(mean+std*torch.randn(samples,5,dim,device=device,generator=generator)).clamp(-.35,.35)
        parameters[0]=mean # includes zero initially; current distribution mean in CEM
        costs,tokens=cost_function(expand_actions(parameters));assert costs.shape==(samples,) and torch.isfinite(costs).all()
        order=torch.argsort(costs,stable=True);idx=int(order[0]);value=float(costs[idx])
        if best is None or value<best['cost']:best={'cost':value,'iteration':iteration,'candidate':idx,'parameters':parameters[idx].clone(),'tokens':tokens[idx].clone()}
        trace.append({'mean':mean.cpu().numpy(),'std':std.cpu().numpy(),'parameters':parameters.cpu().numpy(),'costs':costs.cpu().numpy(),'tokens':tokens.cpu().numpy()})
        if algorithm=='cem':
            selected=parameters[order[:elites]];mean=selected.mean(0);std=selected.std(0,correction=0).clamp_min(1e-3)
    # Primary policy returns the best scored candidate in both algorithms, not an unscored final mean.
    return best,trace,mean
