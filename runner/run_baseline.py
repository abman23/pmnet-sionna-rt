import os
import time
from datetime import datetime

import yaml
from torch.distributions import Distribution

from agent.ppo import PPOAgent
from agent.random import RandomAgent
from agent.brute_force import BruteForceAgent
from multi_agent.async_random import AsyncRandom
from multi_agent.async_ppo import AsyncPPO
from env.utils_v1 import ROOT_DIR, plot_rewards

VERSION = 'v30'

if __name__ == "__main__":
    # prevent error caused by simplex check failure
    Distribution.set_default_validate_args(False)

    start = time.time()

    # random
    config_random = yaml.safe_load(open(os.path.join(ROOT_DIR, f'config/mdp_{VERSION}.yaml'), 'r'))
    random_agent = AsyncRandom(config=config_random, log_file=os.path.join(ROOT_DIR, f'log/runner_random_{VERSION}.log'),
                               version=VERSION)

    # # brute-force, pick one pixel in a 8x8 block
    # config_bf = yaml.safe_load(open(os.path.join(ROOT_DIR, f'config/mdp_{VERSION}.yaml')))
    # bf_sparse = BruteForceAgent(config=config_bf, log_file=os.path.join(ROOT_DIR, f'log/runner_bf_{VERSION}.log'),
    #                             version=VERSION)

    # train these baseline agents and evaluate every some steps
    # timestamp = datetime.now().strftime('%m%d_%H%M')
    timestamp = '0423_0127'
    # random_agent.train_and_eval(log=True, timestamp=timestamp)
    # filename_rand = f"{VERSION}_{random_agent.algo_name}_{timestamp}.json"
    # bf_sparse.train_and_eval(log=True, timestamp=timestamp)
    # filename_bf_sparse = f"{VERSION}_{bf_sparse.algo_name}_{timestamp}.json"
    #
    # evaluate random agent
    random_agent.test(timestamp, duration=20, log=True)

    # # plot the reward curves
    # plot_rewards(output_name="rand_ppo", algo_names=["random", "ppo"],
    #              data_filenames=[filename_rand, 'v30_ppo_0423_0127.json'],
    #              # data_filenames=['random_0321_1812.json', 'brute-force_0321_1817.json', 'brute-force_0321_1839.json', 'ppo_0321_2245.json'],
    #              version=VERSION, evaluation=True, log=True, n_epi=config_random['stop']['training_iteration'], timestamp=timestamp)

    end = time.time()
    print(f"total runtime: {end - start}s")
