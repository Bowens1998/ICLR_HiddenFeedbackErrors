"""Native control units and explicit visual-only feedback, with no observed-proprio injection."""
import torch
from native_model import replace_visual


def prepare_native_inputs(archive, image_size):
    from datasets.img_transforms import default_transform
    from datasets.pusht_dset import ACTION_MEAN, ACTION_STD, PROPRIO_MEAN, PROPRIO_STD
    images=torch.as_tensor(archive['pixels']).permute(0,3,1,2).float()/255.0
    images=default_transform(image_size)(images)
    # The bank already stores relative controls divided by the native scale 100.
    # Dividing by 100 again would silently change all executed controls.
    actions=(torch.as_tensor(archive['actions'],dtype=torch.float32)-ACTION_MEAN)/ACTION_STD
    proprio=(torch.as_tensor(archive['proprio'][[0,5,10]],dtype=torch.float32)-PROPRIO_MEAN)/PROPRIO_STD
    history=dict(visual=images[:3][None].cuda(),proprio=proprio[None].cuda())
    blocks=actions.reshape(1,7,10).cuda()
    return history,blocks,images


def native_feedback_rollout(model,history,blocks,replacement=None):
    if blocks.shape!=(1,7,10):raise ValueError('Native diagnostic fixes batch1 and seven control blocks')
    z=model.encode(history,blocks[:,:3])
    earlier=z.clone()
    for t in range(3,7):
        nxt=model.predict(z[:,-model.num_hist:])[:,-1:]
        nxt=model.replace_actions_from_z(nxt,blocks[:,t:t+1])
        if t==3 and replacement is not None:
            nxt=replace_visual(nxt,replacement.reshape(1,1,75264))
        z=torch.cat([z,nxt],dim=1)
    z=torch.cat([z,model.predict(z[:,-model.num_hist:])[:,-1:]],dim=1)
    torch.testing.assert_close(z[:,:3],earlier,rtol=0,atol=0)
    return z
