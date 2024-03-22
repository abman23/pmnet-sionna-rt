import os
import time
import yaml
from torch.distributions import Distribution

from agent.ppo import PPOAgent
from agent.random import RandomAgent
from agent.brute_force import BruteForceAgent
from env.utils_v1 import ROOT_DIR, plot_rewards

if __name__ == "__main__":
    # prevent error caused by simplex check failure
    Distribution.set_default_validate_args(False)

    start = time.time()
    # config_ppo = yaml.safe_load(open(os.path.join(ROOT_DIR, 'config/ppo_v1.yaml'), 'r'))
    # ppo = PPOAgent(config=config_ppo, log_file=os.path.join(ROOT_DIR, 'log/runner_ppo.log'), version="v13")
    # random
    config_random = yaml.safe_load(open(os.path.join(ROOT_DIR, 'config/ppo_v14.yaml'), 'r'))
    random_agent = RandomAgent(config=config_random, log_file=os.path.join(ROOT_DIR, 'log/runner_random.log'),
                               version="v14")
    # brute-force, pick one pixel in a 4x4 block
    config_bf_dense = yaml.safe_load(open(os.path.join(ROOT_DIR, 'config/ppo_v14.yaml')))
    bf_dense = BruteForceAgent(config=config_bf_dense, log_file=os.path.join(ROOT_DIR, 'log/runner_bf_dense.log'),
                               version="v14")
    # brute-force, pick one pixel in a 8x8 block
    config_bf_sparse = yaml.safe_load(open(os.path.join(ROOT_DIR, 'config/ppo_v14.yaml')))
    config_bf_sparse['env']['action_space_size'] = config_bf_sparse['env']['map_size'] // 8
    bf_sparse = BruteForceAgent(config=config_bf_sparse, log_file=os.path.join(ROOT_DIR, 'log/runner_bf_sparse.log'),
                                version="v14")

    # train these baseline agents and evaluate every some steps
    filename_rand = random_agent.train_and_eval(log=True)
    filename_bf_dense = bf_dense.train_and_eval(log=True)
    filename_bf_sparse = bf_sparse.train_and_eval(log=True)

    # plot the reward curves
    plot_rewards(output_name="rand_bf_ppo", algo_names=["random", "bf_dense", "bf_sparse", "ppo"],
                 data_filenames=[filename_rand, filename_bf_dense, filename_bf_sparse, 'ppo_0321_2245.json'],
                 # data_filenames=['random_0321_1812.json', 'brute-force_0321_1817.json', 'brute-force_0321_1839.json', 'ppo_0321_2245.json'],
                 version='v14', evaluation=True, log=True)

    end = time.time()
    print(f"total runtime: {end - start}s")
