
import gym
import numpy as np
import pandas as pd


class IMDbEnv(gym.Env):
    '''generate a contextual bandit environment based on IMDb Dataset'''

    def __init__(self, random_seed=0):
        super().__init__()
        self.name = 'imdb'
        self.random_seed = random_seed
        self.num_feedback = 50
        self.setup()

    def setup(self):
        '''setup the environment'''
        self.fix_random_seed()
        str_to_np = lambda x: np.fromstring(x[1:-1], sep=' ')

        # setup state space
        self.movies = pd.read_csv('./data/IMDb/movies.csv', converters={'encoding': str_to_np})
        self.movies_enc = np.array(self.movies.encoding.to_list())
        self.observation_space = gym.spaces.Box(low=-self.num_feedback, high=self.num_feedback,
                                                shape=(self.movies_enc.shape[1],), dtype=np.float32)

        # setup action space
        self.action_data = pd.read_csv('./data/IMDb/movies_popular.csv', converters={'encoding': str_to_np})
        self.actions = np.array(self.action_data.encoding.to_list())
        self.action_space = gym.spaces.Discrete(len(self.action_data))

        # setup reward signal
        cossim = lambda s,a: np.dot(s, a.T) / (np.linalg.norm(s) * np.linalg.norm(a))
        self.reward = lambda s,a: np.clip(np.round(9 * (.5 + cossim(s,a)/2)**.5), 1, 10)

    def fix_random_seed(self):
        '''fix random seed for reproducibility'''
        self.seed(self.random_seed)
        self.rng = np.random.default_rng(seed=self.random_seed)

    def reset(self):
        '''observe a new state -- generate a user from random normalized feedback'''
        feedback = (self.rng.integers(1, 11, size=self.num_feedback) - 5.5) / 4.5
        inds = self.rng.integers(len(self.movies), size=self.num_feedback)
        self.state = np.matmul(feedback, self.movies_enc[inds])
        return self.state

    def step(self, action_ind):
        '''given an observed state take an action and receive reward'''
        s = self.state
        a = self.actions[action_ind]
        r = self.reward(s,a)
        done = True
        info = {}
        return s, r, done, info


    def test_params(self):
        import matplotlib.pyplot as plt
        import seaborn as sns
        sns.set_theme(style='darkgrid', palette='muted', font='monospace', font_scale=2)
        cossim = lambda s,a: np.dot(s, a.T) / (np.linalg.norm(s) * np.linalg.norm(a))
        S = [env.reset() for _ in range(1000)]

        for p1 in np.linspace(.5,.6,3):
            for p2 in np.linspace(9,10,3):
                self.reward = lambda s,a: np.clip(np.ceil(p2 * (.5 + cossim(s,a)/2)**p1), 1, 10)

                fig, ax = plt.subplots(figsize=(8,6))
                rr = []
                for s in S:
                    for a in env.movies_enc:
                        rr.append(env.reward(s,a))
                plt.hist(rr, bins=np.linspace(1,10,10), density=True)
                plt.xlim(1,10)
                plt.savefig(f'./params_c/{int(10*p1)}_{int(10*p2)}.png', dpi=300, format='png')
                plt.close()



if __name__ == '__main__':

    env = IMDbEnv(random_seed=0)

