"""Deployed candidate scores: only model tokens and training normalization are inputs."""
import numpy as np

def token_scores(predicted,goal,arm,normalization):
    if arm.endswith('jepa'):
        distance=np.square(predicted-goal).sum(1)
        return {'block_pose':distance,'agent_and_block_pose':distance.copy()}
    assert predicted.shape[-1]==goal.shape[-1]==6
    mean,std=np.array(normalization['mean']),np.array(normalization['std'])
    state=predicted*std+mean;target=goal*std+mean
    angle=np.arctan2(state[:,4],state[:,5])-np.arctan2(target[4],target[5])
    angle=np.arctan2(np.sin(angle),np.cos(angle))
    block=np.square(state[:,2:4]-target[2:4]).sum(1)+900*np.square(angle)
    return {'block_pose':block,'agent_and_block_pose':block+np.square(state[:,:2]-target[:2]).sum(1)}
