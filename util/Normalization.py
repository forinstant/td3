import gym
import numpy as np
def Normalized_action(action,action_space):
    low_bound = action_space.low
    upper_bound =action_space.high

    action = low_bound + (action + 1.0) * 0.5 * (upper_bound - low_bound)
    # 将经过tanh输出的值重新映射回环境的真实值内
    action = np.clip(action, low_bound, upper_bound)

    return action
