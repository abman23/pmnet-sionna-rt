import json
import os
import time
from tqdm import tqdm

import numpy as np

from env.utils_v1 import ROOT_DIR, dict_update
from multi_agent.async_agent import Agent


class MABruteForce(Agent):
    def __init__(self, config: dict, log_file: str, version: str) -> None:
        super().__init__(config, log_file, version)

        self.algo_name = 'brute-force'

    def train_and_eval(self, log: bool = True, **kwargs):
        num_episode = self.config["stop"].get("training_iteration", 10)
        num_steps_per_episode = self.config['report'].get('min_sample_timesteps_per_iteration', 2000)
        eval_interval = self.config["eval"].get("evaluation_interval", 10)
        num_maps_per_eval = self.config["eval"].get("num_maps_per_eval", 1)
        data_saving_interval = self.config["agent"].get("data_saving_interval", 10)

        run_on_train_set = kwargs.get('run_on_train_set', False)
        env_config_train = self.config['env']
        env_train = self.env_class(config=env_config_train)
        env_config_eval = dict_update(self.config['env'], self.config['eval']['evaluation_config']['env_config'])
        env_eval = self.env_class(config=env_config_eval)

        # evaluation data
        ep_eval = np.arange(0, num_episode, eval_interval) + eval_interval
        ep_reward_mean = np.empty(num_episode // eval_interval, dtype=float)
        ep_reward_std = np.empty(num_episode // eval_interval, dtype=float)
        # training data
        ep_train = np.arange(num_episode) + 1
        ep_reward_mean_train = np.empty(num_episode, dtype=float)
        ep_reward_std_train = np.zeros(num_episode, dtype=float)

        timestamp = kwargs["timestamp"]
        start_info = f"==========={self.algo_name.upper()} train and eval started at {timestamp}==========="
        if log:
            self.logger.info(start_info)
        print(start_info)

        time_train_start = time.time()
        for i in range(num_episode):
            reward_train = []
            reward_eval = []

            if run_on_train_set:
                # training
                num_maps_per_train = num_steps_per_episode // env_train.n_steps_truncate
                for j in range(num_maps_per_train):
                    env_train.reset()
                    _, reward = env_train.calc_optimal_locations()
                    reward_train.append(reward)
                ep_reward_mean_train[i] = np.mean(reward_train)
                ep_reward_std_train[i] = np.std(reward_train)
            time_total_s = time.time() - time_train_start
            print("\n")
            print(f"================TRAINING # {i + 1}================")
            print(f"time_total_s: {time_total_s}")

            if (i + 1) % eval_interval == 0:
                # evaluation
                for j in tqdm(range(num_maps_per_eval)):
                    env_eval.reset()
                    _, reward = env_eval.calc_optimal_locations()
                    reward_eval.append(reward)
                idx = (i + 1) // eval_interval - 1
                ep_reward_mean[idx] = np.mean(reward_eval)
                ep_reward_std[idx] = np.std(reward_eval)
                time_total_s = time.time() - time_train_start
                print(f"================EVALUATION AT # {i + 1}================")
                print(f"time_total_s: {time_total_s}")

            if i == num_episode - 1:
                if log:
                    self.logger.debug(self.config)
                    self.logger.info("=============TRAINING ENDED=============")
                else:
                    print("=============TRAINING ENDED=============")
                    print(self.config)

            if log and ((i + 1) % data_saving_interval == 0 or i == num_episode - 1):
                # save the training and evaluation data periodically
                data = {
                    "ep_train": ep_train.tolist(),
                    "ep_reward_mean_train": ep_reward_mean_train.tolist(),
                    "ep_reward_std_train": ep_reward_std_train.tolist(),
                    "ep_eval": ep_eval.tolist(),
                    "ep_reward_std": ep_reward_std.tolist(),
                    "ep_reward_mean": ep_reward_mean.tolist(),
                }
                json.dump(data, open(os.path.join(ROOT_DIR, f"data/{self.version}_{self.algo_name}_{timestamp}.json"), 'w'))

        if log:
            time_total_s = time.time() - time_train_start
            self.logger.info(f"train and eval total time: {time_total_s}s")

        return timestamp
