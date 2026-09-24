"""Image-only cost interface: no simulator state or future outcome inputs."""
import numpy as np
import torch

class ImagePlannerCost:
    def __init__(self,model,history_images,goal_image,prefix_actions,action_normalization,target_normalization,score_space):
        self.model=model;self.device=next(model.parameters()).device;self.score_space=score_space
        im=torch.tensor([.485,.456,.406],device=self.device)[None,:,None,None];isd=torch.tensor([.229,.224,.225],device=self.device)[None,:,None,None]
        def pixels(images):return (torch.as_tensor(np.array(images),device=self.device).permute(0,3,1,2).float()/255-im)/isd
        self.pixels=pixels(history_images)
        self.initial=model.encode({'pixels':self.pixels[:,None]})['emb'][:,0]
        self.goal=model.encode({'pixels':pixels(goal_image[None])[:,None]})['emb'][0,0]
        self.prefix=torch.as_tensor(prefix_actions,device=self.device).reshape(1,2,10)
        self.am=torch.tensor(action_normalization['mean'],device=self.device).repeat(5);self.asd=torch.tensor(action_normalization['std'],device=self.device).repeat(5)
        if score_space=='state':
            self.mean=torch.tensor(target_normalization['mean'],device=self.device);self.std=torch.tensor(target_normalization['std'],device=self.device);self.goal_state=self.goal*self.std+self.mean
        else:assert score_space=='latent'
    def normalized_actions(self,actions):return (torch.cat([self.prefix.expand(len(actions),-1,-1),actions.reshape(-1,5,10)],1)-self.am)/self.asd
    def __call__(self,actions):
        encoded=self.model.action_encoder(self.normalized_actions(actions));history=self.initial[None].expand(len(actions),-1,-1).clone()
        for k in range(2,7):history=torch.cat([history,self.model.predict(history[:,-3:],encoded[:,k-2:k+1])[:,-1:]],1)
        token=history[:,-1]
        if self.score_space=='latent':cost=(token-self.goal).square().sum(1)
        else:
            state=token*self.std+self.mean;angle=torch.atan2(state[:,4],state[:,5])-torch.atan2(self.goal_state[4],self.goal_state[5]);angle=torch.atan2(angle.sin(),angle.cos());cost=(state[:,2:4]-self.goal_state[2:4]).square().sum(1)+900*angle.square()
        return cost,token
    def verify_native(self,actions):
        native=self.model.rollout({'pixels':self.pixels[None,None].expand(1,len(actions),-1,-1,-1,-1)},self.normalized_actions(actions)[None])['predicted_emb'][0,:,-1]
        _,manual=self(actions);torch.testing.assert_close(native,manual,rtol=2e-5,atol=2e-5)
