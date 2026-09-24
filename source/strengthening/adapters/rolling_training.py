"""C training graphs: rolling observed history, recursive BPTT, latent anchor."""
import sys
from pathlib import Path
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts/visual'))
from adaptation_task_losses import coordinates
from factorial_model import state_features


def rolling_predictions(model, observed, actions, condition):
    if condition not in ['T0','T1','T2']:
        raise ValueError('Unknown rolling condition')
    if observed.ndim!=3 or observed.shape[1]!=6 or actions.shape[:2]!=(observed.shape[0],5):
        raise ValueError('Expected six continuous observations and five aligned action blocks')
    encoded=model.action_encoder(actions)
    history=observed[:,:3];predictions=[]
    for step in range(3):
        context=observed[:,step:step+3] if condition=='T0' else history[:,-3:]
        token=model.predict(context,encoded[:,step:step+3])[:,-1:]
        predictions.append(token[:,0])
        if condition!='T0':history=torch.cat([history,token],dim=1)  # Deliberately no detach: full three-step BPTT.
    return torch.stack(predictions,dim=1)


def rolling_loss(model, observed, actions, raw_states, head, objective, condition, anchor_lambda=0.):
    if any(value.requires_grad for value in head.values()):
        raise ValueError('Head parameters and normalizers must be frozen')
    if objective not in ['decoded_teacher','physical_labels']:
        raise ValueError('C has only the two coordinate objectives')
    prediction=rolling_predictions(model,observed,actions,condition)
    target=(coordinates(observed[:,3:],head) if objective=='decoded_teacher' else
            (state_features(raw_states[:,3:].double())-head['target_mean'])/head['target_scale']).detach()
    coordinate_loss=(coordinates(prediction,head)-target).square().mean()
    anchor=((prediction.double()-observed[:,3:].detach().double())/head['scale']).square().mean()
    total=coordinate_loss+anchor_lambda*anchor if condition=='T2' else coordinate_loss
    return total,dict(coordinate=coordinate_loss,anchor=anchor,predictions=prediction)
