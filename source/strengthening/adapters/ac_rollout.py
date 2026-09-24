"""Preserve the original 300-candidate arithmetic and three-token rolling context."""
import torch


def rollout_population(model, initial, encoded_actions, selected, replacement=None, observed=None):
    if initial.shape!=(3,192) or encoded_actions.shape[:2]!=(300,7):
        raise ValueError('A/C fixes a three-token history and the original selected population of 300')
    history=initial[None].expand(300,-1,-1).clone();outputs=[]
    for step in range(2,7):
        nxt=model.predict(history[:,-3:],encoded_actions[:,step-2:step+1])[:,-1:]
        if step==2 and replacement is not None:
            nxt=nxt.clone();nxt[selected,0]=replacement
        outputs.append(nxt[selected,0])
        appended=nxt if observed is None else observed[step-2][None,None].expand(300,1,-1)
        history=torch.cat([history,appended],1)
    torch.testing.assert_close(history[:,:3],initial[None].expand(300,-1,-1),rtol=0,atol=0)
    return torch.stack(outputs)
