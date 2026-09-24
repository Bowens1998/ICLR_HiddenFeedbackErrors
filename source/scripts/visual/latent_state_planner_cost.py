"""Image-only planning through a state head; full latent tokens remain recurrent."""
import torch
from image_planner_cost import ImagePlannerCost

class LatentStatePlannerCost(ImagePlannerCost):
    def __init__(self,model,history_images,goal_image,prefix_actions,action_normalization,target_normalization):
        super().__init__(model,history_images,goal_image,prefix_actions,action_normalization,None,'latent')
        self.mean=torch.tensor(target_normalization['mean'],device=self.device)
        self.std=torch.tensor(target_normalization['std'],device=self.device)
        self.goal_state=model.state_head(self.goal)*self.std+self.mean
    def __call__(self,actions):
        _,token=super().__call__(actions)
        state=self.model.state_head(token)*self.std+self.mean
        angle=torch.atan2(state[:,4],state[:,5])-torch.atan2(self.goal_state[4],self.goal_state[5]);angle=torch.atan2(angle.sin(),angle.cos())
        cost=(state[:,2:4]-self.goal_state[2:4]).square().sum(1)+900*angle.square()
        return cost,token
