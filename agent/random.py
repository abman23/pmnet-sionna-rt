import json
import os
from datetime import datetime
import time

import numpy as np
from matplotlib import pyplot as plt

from agent.agent import Agent
from env.utils_v1 import dict_update, ROOT_DIR


class RandomAgent(Agent):
    def __init__(self, config: dict, log_file: str, version: str) -> None:
        super().__init__(config, log_file, version)

        self.algo_name = 'random'

    def train_and_eval(self, log: bool = True):
        num_episode = self.config["stop"].get("training_iteration", 10)
        num_steps_per_episode = self.config['report'].get('min_sample_timesteps_per_iteration', 2000)
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
        start_info = f"===========train and eval started at {timestamp}==========="
        if log:
            self.logger.info(start_info)
        print(start_info)

        time_train_start = time.time()
        terminated, truncated = False, False
        env_train.reset()
        for i in range(num_episode):
            reward_train_mean = 0.
            reward_eval = []

            # training
            for j in range(num_steps_per_episode):
                if terminated or truncated:
                    env_train.reset()
                action = env_train.np_random.choice(np.where(env_train.mask == 1)[0])
                obs, reward, terminated, truncated, info = env_train.step(action)
                reward_train_mean += info['reward'] / num_steps_per_episode
            ep_reward_mean_train[i] = reward_train_mean
            time_total_s = time.time() - time_train_start
            print("\n")
            print(f"================TRAINING # {i + 1}================")
            print(f"time_total_s: {time_total_s}")

            if eval_interval is not None and (i + 1) % eval_interval == 0:
                # evaluation
                for _ in range(eval_duration):
                    env_eval = self.env_class(config=env_config_eval)
                    env_eval.reset()
                    term, trunc = False, False
                    reward_per_ep = 0.
                    num_steps = 0
                    while not term and not trunc:
                        action = env_eval.np_random.choice(np.where(env_eval.mask == 1)[0])
                        obs, reward, term, trunc, info = env_eval.step(action)
                        reward_per_ep += info['reward']
                        num_steps += 1
                    reward_eval.append(reward_per_ep / num_steps)
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
                    "ep_reward_mean": ep_reward_mean_train.tolist(),
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
