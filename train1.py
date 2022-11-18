import gym
from util.noise import GaussianExploration
from td3_net import TD3
from util.Normalization import Normalized_action
import numpy as  np
def main():
    env = gym.make('Pendulum-v1')#环境加载
    noise=GaussianExploration(action_space=env.action_space)
    state_dim = env.observation_space.shape[0]#状态
    action_dim = env.action_space.shape[0]#动作
    hidden_dim = 256#隐藏层
    TD = TD3(action_dim, state_dim, hidden_dim)#输入神经网络
    max_frames = 10000
    max_steps = 500
    frame_idx = 0

    while frame_idx < max_frames:
        state = env.reset()
        # 重置环境
        episode_reward = 0
        # 回合奖励重置
        for step in range(max_steps):
            action = TD.actor_net.get_action(state)#获取动作 Pendulum环境 action范围-1~1
            action = noise.get_action(action, step)#添加高斯噪声 action范围[-1~1]
            action[0] = Normalized_action(action[0], action_space=env.action_space)#action映射到真实值
            next_state, reward, done, _ = env.step(action[0])
            TD.learn(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward
            if done:
                break
        frame_idx += 1
        print("第%d轮的reward=%d" % (frame_idx, episode_reward))
        #TD.save(eposide=frame_idx,reward=episode_reward)

if __name__ == '__main__':
    main()



