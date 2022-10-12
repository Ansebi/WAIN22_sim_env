
import os
import numpy as np
import pandas as pd
import torch
import stable_baselines3 as sb3
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style='darkgrid', palette='muted', font='monospace', font_scale=1.2)


class RLAgents:
    '''setup 3 RL agents (A2C, DQN, PPO) to learn a given environment'''

    def __init__(self, env, random_seed):
        self.env = env
        self.random_seed = random_seed
        self.setup()
        os.makedirs('./images/', exist_ok=True)

    def fix_random_seed(self):
        '''set random seed for reproducibility'''
        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        self.env.fix_random_seed()

    def setup(self):
        '''configure the policy, all updates are once per batch_size'''
        self.fix_random_seed()
        self.agents = {}
        # Advantage Actor Critic
        self.agents['A2C'] = sb3.A2C('MlpPolicy', self.env,
                                     seed=self.random_seed, learning_rate=1e-4)
        # Deep Q Network
        self.agents['DQN'] = sb3.DQN('MlpPolicy', self.env, learning_starts=0,
                                     seed=self.random_seed, learning_rate=1e-4)
        # Proximal Policy Optization
        self.agents['PPO'] = sb3.PPO('MlpPolicy', self.env,
                                     seed=self.random_seed, learning_rate=1e-4)

    def train(self, num_timesteps):
        '''train agents on the given environment'''
        self.rewards = {}
        for agent_name, agent_policy in self.agents.items():
            print(f'training {agent_name}-agent on {self.env.name}-environment...')
            self.fix_random_seed()
            callback = Callback()
            agent_policy.learn(num_timesteps, callback=callback)
            self.rewards[agent_name] = callback.rewards

    def plot_rewards(self):
        '''plot training rewards'''
        fig, ax = plt.subplots(figsize=(8,4))
        for agent_name, agent_rewards in self.rewards.items():
            rewards_smooth = pd.Series(agent_rewards).rolling(1000).mean().to_list()
            plt.plot(rewards_smooth, linewidth=3, label=agent_name)
        plt.xlabel('number of agent-environment interactions')
        plt.ylabel('average user rating')
        plt.legend(loc='lower right')
        plt.tight_layout()
        plt.savefig(f'./images/{self.env.name}_rewards.png', format='png', dpi=300)
        plt.show()


class Callback(sb3.common.callbacks.BaseCallback):
    '''callback that evaluates agent's policy on given timesteps'''

    def __init__(self):
        super().__init__()
        self.rewards = []

    def _on_step(self):
        try:
            self.rewards.append(self.locals['rewards'].item())
        except:
            self.rewards.append(self.locals['infos'][0]['episode']['r'])
        return True

