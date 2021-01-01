import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class RolloutPolicy:
    def __init__(self, env):
        self.env = env
    def act(self, obs, render=False):
        return self.env.action_space.sample()
    def rollout(self, obs, model, render=False):
        d = False
        rsum = 0
        while not d:
            obs, r, d, _ = model.step(obs, self.act(obs))
            rsum += r
        return rsum

class Model:
    def __init__(self, env):
        self.env = env
    def step(self, obs, action):
        self._set_env(obs)
        start_mark = self.env.start_mark
        obs, r, done, _ = self.env.step(action)
        if self.env.mark == start_mark:
            r *= -1
        return obs, r, done, _
    def available_actions(self, obs):
        self._set_env(obs)
        return self.env.available_actions()
    def _set_env(self, obs):
        self.env.done = False
        self.env.board = list(obs[0])
        self.env.mark  = obs[1] 
    def get_num_actions(self):
        return self.env.action_space.n

class TreePolicy:
    def __init__(self):
        pass
    def act(self, obs, available_actions):
        import random
        return random.choice(available_actions)
    def get_action_probs(self, obs, available_actions):
        return [1/len(available_actions) for i in range(len(available_actions))]



class Backbone(nn.Module):
    def __init__(self, net_arch, middle_activation=F.relu, last_activation=F.relu):
        super().__init__()
        self.middle_activation = middle_activation
        self.last_activation = last_activation
        self.layers = nn.ModuleList([nn.Linear(a, b) for a, b in zip(net_arch[:-1], net_arch[1:])])
    def forward(self, h):
        h = h.view(h.shape[0], -1)*1.0
        for lay in self.layers[:-1]:
            h = self.middle_activation(lay(h))
        h = self.layers[-1](h)
        h = self.last_activation(h)
        return h

class ValueFunction(nn.Module):
    def __init__(self, net_arch, backbone):
        super().__init__()
        self.backbone = backbone
        self.layers = nn.ModuleList([nn.Linear(a, b) for a, b in zip(net_arch[:-1], net_arch[1:])])
    def forward(self, h):
        h = torch.tensor(h, dtype=torch.float)
        h = h.view(h.shape[0], -1)
        h = self.backbone(h)
        for lay in self.layers[:-1]:
            h = F.relu(lay(h))
        h = self.layers[-1](h)
        return h
    def get(self, obs):
        obs = self.obs2testorobs(obs)
        value = self.forward(obs)
        return value.detach().cpu().item()
    def obs2testorobs(self, obs):
        l2 = [1] if obs[1]=='O' else [-1]
        obs = torch.tensor([list(obs[0])+l2])
        obs[obs==2] = -1
        return obs


class NNTreePolicy(nn.Module):
    def __init__(self, net_arch, backbone, temperature=1):
        super().__init__()
        self.backbone = backbone
        self.temperature = temperature
        self.layers = nn.ModuleList([nn.Linear(a, b) for a, b in zip(net_arch[:-1], net_arch[1:])])
    def forward(self, x, available_actions=None):
        h = x
        h = self.backbone(h)
        for lay in self.layers[:-1]:
            h = F.tanh(lay(h))
        h = self.layers[-1](h)/self.temperature
        h = torch.softmax(h, dim=1)
        return h
    def act(self, obs, available_actions):
        obs = self.obs2testorobs(obs)
        action_probs = self.forward(obs)[0].detach().cpu().numpy()
        weights = []
        for action in available_actions:
            weights.append(action_probs[action])
        weights = np.array(weights)
        weights /= weights.sum()
        action = np.random.choice(available_actions, p=weights)
        return action     
    def act_greedy(self, obs, available_actions):
        obs = self.obs2testorobs(obs)
        action_probs = self.forward(obs)[0].detach().cpu().numpy()
        max_prob, best_a = -1, -1
        for action in available_actions:
            if max_prob < action_probs[0][action].item():
                max_prob = action_probs[0][action].item()
                best_a = action
        return best_a
    def get_action_probs(self, obs, available_actions=None):
        obs = self.obs2testorobs(obs)
        h = self.forward(obs)
        return h.tolist()[0]
    def obs2testorobs(self, obs):
        l2 = [1] if obs[1]=='O' else [-1]
        obs = torch.tensor([list(obs[0])+l2])
        obs[obs==2] = -1
        return obs

class CNNTreePolicy(NNTreePolicy):
    def __init__(self, net_arch, backbone, temperature=1):
        pass