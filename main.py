
import argparse

from ml25m_env import ML25MEnv
from imdb_env import IMDbEnv
from rl_agents import RLAgents


def parse_arguments():
    '''parse input arguments'''
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--seed', default=2022, help='value of random seed, default is 2022')
    parser.add_argument('-e', '--environment', default='ml25m', help='which environment to use: ml25m, imdb')
    args = parser.parse_args()
    return int(args.seed), args.environment


if __name__ == '__main__':

    # parse input arguments
    seed, env_name = parse_arguments()

    # create contextual bandit environment
    if env_name == 'ml25m':
        env = ML25MEnv(random_seed=seed)
    elif env_name == 'imdb':
        env = IMDbEnv(random_seed=seed)
    else:
        raise SystemExit(f'\nenvironment {env_name} is not implemented...')

    # train rl agents
    rl = RLAgents(env, random_seed=seed)
    rl.train(num_timesteps=100000)
    rl.plot_rewards()

