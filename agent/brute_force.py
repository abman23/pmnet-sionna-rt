import json
import os
from datetime import datetime
import time

import numpy as np
from matplotlib import pyplot as plt

from agent.agent import Agent
from env.utils_v1 import dict_update, ROOT_DIR, calc_optimal_locations


class BruteForceAgent(Agent):
    def __init__(self, config: dict, log_file: str, version: str) -> None:
        super().__init__(config, log_file, version)

        self.algo_name = 'brute-force'

    def train_and_eval(self, log: bool = True):
        num_episode = self.config["stop"].get("training_iteration", 10)
        eval_interval = self.config["eval"].get("evaluation_interval", 5)
        eval_duration = self.config["eval"].get("evaluation_duration", 1)
        data_saving_interval = self.config["agent"].get("data_saving_interval", 10)

        env_config_train = self.config['env']
        env_train = self.env_class(config=env_config_train)
        env_config_eval = dict_update(self.config['env'], self.config['eval']['evaluation_config']['env_config'])

        # evaluation data
        if eval_interval is None:
            ep_eval = np.arange(0)
            ep_reward_mean = np.empty(0, dtype=float)
            ep_reward_std = np.empty(0, dtype=float)
        else:
            ep_eval = np.arange(0, num_episode, eval_interval) + eval_interval
            ep_reward_mean = np.empty(num_episode // eval_interval, dtype=float)
            ep_reward_std = np.empty(num_episode // eval_interval, dtype=float)
        # training data
        ep_train = np.arange(num_episode)
        ep_reward_mean_train = np.empty(num_episode, dtype=float)

        timestamp = datetime.now().strftime('%m%d_%H%M')
        start_info = f"==========={self.algo_name} train and eval started at {timestamp}==========="
        if log:
            self.logger.info(start_info)
        print(start_info)

        time_train_start = time.time()
        for i in range(num_episode):
            _, info_dict = env_train.reset()
            reward_eval = []

            # training
            action = calc_optimal_locations(env_train.dataset_dir, env_train.map_suffix, info_dict["map_index"],
                                            env_train.coverage_threshold, env_train.upsampling_factor)
            obs, reward, terminated, truncated, info = env_train.step(action)
            ep_reward_mean_train[i] = info['r_c']  # not normalized coverage reward
            time_total_s = time.time() - time_train_start
            print("\n")
            print(f"================TRAINING # {i + 1}================")
            print(f"time_total_s: {time_total_s}")

            if eval_interval is not None and (i + 1) % eval_interval == 0:
                # test
                for _ in range(eval_duration):
                    env_eval = self.env_class(config=env_config_eval)
                    _, info_dict = env_eval.reset()
                    action = calc_optimal_locations(env_eval.dataset_dir, env_eval.map_suffix, info_dict["map_index"],
                                                    env_eval.coverage_threshold, env_eval.upsampling_factor)
                    obs, reward, terminated, truncated, info = env_eval.step(action)
                    reward_eval.append(info['r_c'])
                ep_r_mean, ep_r_std = np.mean(reward_eval), np.std(reward_eval)
                idx = (i + 1) // eval_interval - 1
                ep_reward_mean[idx] = ep_r_mean
                ep_reward_std[idx] = ep_r_std
                time_total_s = time.time() - time_train_start
                print(f"================EVALUATION AT # {i + 1}================")
                print(f"time_total_s: {time_total_s}")

            if i == num_episode - 1:
                if log:
                    self.logger.debug(self.config)
                    self.logger.info("=============TRAINING ENDED=============")
                else:
                    print(self.config)
                    print("=============TRAINING ENDED=============")

            if log and ((i + 1) % data_saving_interval == 0 or i == num_episode - 1):
                # save the training and evaluation data periodically
                data = {
                    "ep_train": ep_train.tolist(),
                    "ep_reward_mean_train": ep_reward_mean_train.tolist(),
                    "ep_eval": ep_eval.tolist(),
                    "ep_reward_std": ep_reward_std.tolist(),
                    "ep_reward_mean": ep_reward_mean.tolist(),
                }
                json.dump(data, open(os.path.join(ROOT_DIR, f"data/{self.algo_name}_{timestamp}.json"), 'w'))

        if log:
            time_total_s = time.time() - time_train_start
            self.logger.info(f"train and eval total time: {time_total_s}s")

        return f"{self.algo_name}_{timestamp}.json"

        # if eval_interval is not None:
        #     # plot the mean reward in evaluation
        #     fig, ax = plt.subplots()
        #     fig.set_size_inches(10, 6)
        #     ax.plot(ep_eval, ep_reward_mean, color="blue")
        #     sup = list(map(lambda x, y: x + y, ep_reward_mean, ep_reward_std))
        #     inf = list(map(lambda x, y: x - y, ep_reward_mean, ep_reward_std))
        #     ax.fill_between(ep_eval, inf, sup, color="blue", alpha=0.2)
        #     ax.set(xlabel="training_step", ylabel="mean reward per step",
        #            title=f"{self.algo_name.upper()} Evaluation Results")
        #     ax.grid()
        #     if log:
        #         fig.savefig(f"./figures/{self.version}_{self.algo_name}_{timestamp}_eval.png")
        #
        # # plot mean reward in training
        # fig, ax = plt.subplots()
        # fig.set_size_inches(10, 6)
        # ax.plot(ep_train, ep_reward_mean_train, color='red')
        # ax.set(xlabel="training_step", ylabel="mean reward per step",
        #        title=f"{self.algo_name.upper()} Training Results")
        # ax.grid()
        # if log:
        #     fig.savefig(f"./figures/{self.version}_{self.algo_name}_{timestamp}_train.png")
        #
        # plt.show()
