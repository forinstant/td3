import gym
from util.noise import GaussianExploration
from td3_net import TD3
from util.Normalization import Normalized_action
import numpy as  np
def main():
    env = gym.make('Pendulum-v1')#环境加载
    noise=GaussianExploration(action_space=env.action_space)
    #噪声动作输出

    state_dim = env.observation_space.shape[0]#状态
    action_dim = env.action_space.shape[0]#动作
    hidden_dim = 256#隐藏层

    TD = TD3(action_dim, state_dim, hidden_dim)#输入神经网络
    max_frames = 10000

    max_steps = 500

    frame_idx = 0
    #回合从0开始
    rewards = []
    batch_size = 128
    train_step=0
# 当在回合内，训练继续
    while True:
        state = env.reset()


        episode_reward = 0

        for step in range(max_steps):
            if train_step>=200:
                env.render()
            # step<500
            action = TD.actor_net.get_action(state)
            action = noise.get_action(action, step)

            action[0] = Normalized_action(action[0], action_space=env.action_space)
            # 获取动作
            # 添加噪声
            next_state, reward, done, _ = env.step(action[0])
            # 赋值给next_state,reward,done

            TD.learn(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward
            if done:
                break
        train_step = train_step + 1
        print("第%d轮的reward=%d" % (train_step, episode_reward))

    env.close()
if __name__ == '__main__':
    main()
