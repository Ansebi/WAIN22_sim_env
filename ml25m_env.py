
import gym
import numpy as np
import pandas as pd


class ML25MEnv(gym.Env):
    '''generate a contextual bandit environment based on MovieLens 25M Dataset'''

    def __init__(self, random_seed=0):
        super().__init__()
        self.name = 'ml25m'
        self.random_seed = random_seed
        self.setup()

    def setup(self):
        '''setup the environment'''
        self.fix_random_seed()
        str_to_np = lambda x: np.fromstring(x[1:-1], sep=' ')

        # setup state space
        self.state_data = pd.read_csv('./data/MovieLens25M/users_active.csv', converters={'encoding': str_to_np})
        self.states = np.array(self.state_data.encoding.to_list())
        self.observation_space = gym.spaces.Box(low=0., high=100.,
                                                shape=(self.states.shape[1],), dtype=np.float32)

        # setup action space
        self.action_data = pd.read_csv('./data/MovieLens25M/movies_popular.csv', converters={'encoding': str_to_np})
        self.actions = np.array(self.action_data.encoding.to_list())
        self.action_space = gym.spaces.Discrete(len(self.action_data))

        # setup reward signal
        cossim = lambda s,a: np.dot(s, a.T) / (np.linalg.norm(s) * np.linalg.norm(a))
        self.reward = lambda s,a: min(5., np.round(13 * cossim(s,a)**.8) / 2)

    def fix_random_seed(self):
        '''fix random seed for reproducibility'''
        self.seed(self.random_seed)
        self.rng = np.random.default_rng(seed=self.random_seed)

    def reset(self):
        '''observe a new state -- pick a random user'''
        state_ind = self.rng.integers(len(self.state_data))
        self.state = self.states[state_ind]
        return self.state

    def step(self, action_ind):
        '''given an observed state take an action and receive reward'''
        s = self.state
        a = self.actions[action_ind]
        r = self.reward(s,a)
        done = True
        info = {}
        return s, r, done, info


if __name__ == '__main__':

    env = ML25MEnv(random_seed=0)

