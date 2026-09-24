"""Fixed readout task coordinates; no trainable head or new-label ambiguity."""
import torch
from nonlinear_pose_cost import NonlinearPoseCost
from factorial_model import state_features

OBJECTIVES=('latent','decoded_teacher','physical_labels')


def coordinates(tokens,head):
    return (NonlinearPoseCost.forward(tokens,head)-head['target_mean'])/head['target_scale']


def target_for(objective,observed,states,head):
    if objective=='latent':return observed[:,1:].detach()
    if objective=='decoded_teacher':return coordinates(observed[:,1:],head).detach()
    if objective=='physical_labels':
        return ((state_features(states.double())-head['target_mean'])/head['target_scale'])[:,1:].detach()
    raise ValueError(objective)


def loss_for(objective,prediction,target,head):
    if objective not in OBJECTIVES:raise ValueError(objective)
    value=prediction if objective=='latent' else coordinates(prediction,head)
    return (value-target).square().mean()
