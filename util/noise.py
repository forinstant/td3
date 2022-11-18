import torch
import gym
import numpy as np

class GaussianExploration(object):
    def __init__(self, action_space, max_sigma=1.0, min_sigma=1.0, decay_period=400):
        self.low = action_space.low
        self.high = action_space.high
        self.max_sigma = max_sigma
        self.min_sigma = min_sigma
        self.decay_period = decay_period

    def get_action(self, action, t=0):
        action=action.flatten()
        sigma = self.max_sigma - (self.max_sigma ) * min(1.0, t / self.decay_period)
        action = action + np.random.normal(size=action.shape[0]) * sigma
        return np.clip(action, self.low, self.high)