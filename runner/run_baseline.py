import os
import time
from datetime import datetime

import yaml
from torch.distributions import Distribution

from agent.ppo import PPOAgent
from agent.random import RandomAgent
from agent.brute_force import BruteForceAgent
from env.utils_v1 import ROOT_DIR, plot_rewards

VERSION = 'v16'

if __name__ == "__main__":
    # prevent error caused by simplex check failure
    Distribution.set_default_validate_args(False)

    start = time.time()
    # config_ppo = yaml.safe_load(open(os.path.join(ROOT_DIR, 'config/ppo_v1.yaml'), 'r'))
    # ppo = PPOAgent(config=config_ppo, log_file=os.path.join(ROOT_DIR, 'log/runner_ppo.log'), version="v13")
    # random
    config_random = yaml.safe_load(open(os.path.join(ROOT_DIR, 'config/mdp_v16.yaml'), 'r'))
    random_agent = RandomAgent(config=config_random, log_file=os.path.join(ROOT_DIR, 'log/runner_random.log'),
                               version=VERSION)
    # # brute-force, pick one pixel in a 4x4 block
    # config_bf_dense = yaml.safe_load(open(os.path.join(ROOT_DIR, 'config/ppo_v14.yaml')))
    # bf_dense = BruteForceAgent(config=config_bf_dense, log_file=os.path.join(ROOT_DIR, 'log/runner_bf_dense.log'),
    #                            version="v14")
    # brute-force, pick one pixel in a 8x8 block
    config_bf_sparse = yaml.safe_load(open(os.path.join(ROOT_DIR, 'config/mdp_v16.yaml')))
    # config_bf_sparse['env']['action_space_size'] = config_bf_sparse['env']['map_size'] // 8
    bf_sparse = BruteForceAgent(config=config_bf_sparse, log_file=os.path.join(ROOT_DIR, 'log/runner_bf_sparse.log'),
                                version=VERSION)

    # train these baseline agents and evaluate every some steps
    timestamp = datetime.now().strftime('%m%d_%H%M')
    random_agent.train_and_eval(log=True, timestamp=timestamp)
    filename_rand = "random_" + timestamp + ".json"
    bf_sparse.train_and_eval(log=True, timestamp=timestamp)
    filename_bf_sparse = "brute-force_" + timestamp + ".json"

    # evaluate random agent
    random_agent.test(timestamp, duration=50, steps_per_map=1, log=True)

    # plot the reward curves
    plot_rewards(output_name="rand_bf_ppo", algo_names=["random", "bf_sparse", "ppo"],
                 data_filenames=[filename_rand, filename_bf_sparse, 'ppo_0331_2247.json'],
                 # data_filenames=['random_0321_1812.json', 'brute-force_0321_1817.json', 'brute-force_0321_1839.json', 'ppo_0321_2245.json'],
                 version=VERSION, evaluation=True, log=True, n_epi=500)

    end = time.time()
    print(f"total runtime: {end - start}s")
