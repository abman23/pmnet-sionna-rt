import json
import logging
from datetime import datetime
from logging import Logger

import numpy as np
from matplotlib import pyplot as plt
from ray.rllib.algorithms import Algorithm
from ray.tune.logger import pretty_print

from env_v0 import BaseEnvironment
from utils import save_map


class Agent(object):
    """Abstract base class for drl agent

    """
    agent: Algorithm
    algo_name: str

    def __init__(self, config: dict, log_file: str) -> None:
        self.config: dict = config
        # set a logger
        self.logger: Logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.DEBUG)
        handler = logging.FileHandler(log_file, encoding='utf-8', mode='a')
        formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

    def train_and_eval(self):
        """Train and evaluate the agent.
         Plot the average training/evaluation reward per environment step vs. training step.

        """
        num_training_step = self.config["stop"].get("training_iteration", 10)
        eval_interval = self.config["eval"].get("evaluation_interval", 5)

        # evaluation data
        ep_eval = np.arange(0, num_training_step, eval_interval) + eval_interval
        print(f"ep_eval: {ep_eval}")
        ep_reward_mean = np.empty(num_training_step // eval_interval, dtype=float)
        ep_reward_std = np.empty(num_training_step // eval_interval, dtype=float)
        # training data
        ep_train = np.arange(num_training_step)
        ep_reward_mean_train = np.empty(num_training_step, dtype=float)

        for i in range(num_training_step):
            if (i + 1) % eval_interval == 0:
                self.logger.info(f"================EVALUATION AT # {i + 1}================")

            # one training step (may include multiple environment episodes)
            result = self.agent.train()

            print("\n")
            print(f"================training # {i}================")
            print(f"timesteps_total: {result['timesteps_total']}")
            print(f"time_total_s: {result['time_total_s']}")

            if i == num_training_step - 1:
                # save the result and checkpoint
                self.logger.info("=============A WHOLE TRAINING PERIOD ENDED=============")
                self.logger.info(pretty_print(result))
                self.logger.debug(self.config)
                checkpoint_dir = self.agent.save(
                    f"./checkpoint/{self.algo_name}_{datetime.now().strftime('%m%d_%H%M')}").checkpoint.path
                print(f"Checkpoint saved in directory {checkpoint_dir}")

            # calculate the training mean reward per step
            episodes_this_iter = result["sampler_results"]["episodes_this_iter"]
            ep_len_train = np.array(result["sampler_results"]["hist_stats"]["episode_lengths"][-episodes_this_iter:])
            ep_reward_train = np.array(result["sampler_results"]["hist_stats"]["episode_reward"][-episodes_this_iter:])
            ep_r_per_step = ep_reward_train / ep_len_train
            ep_reward_mean_train[i] = np.mean(ep_r_per_step)

            if (i + 1) % eval_interval == 0:
                # calculate the evaluation mean reward per step
                ep_len = np.array(result["evaluation"]["hist_stats"]["episode_lengths"])
                ep_r_sum = np.array(result["evaluation"]["hist_stats"]["episode_reward"])
                # print(f"ep_len: {ep_len}")
                # print(f"ep_r_sum: {ep_r_sum}")
                ep_r_per_step = ep_r_sum / ep_len
                ep_r_mean, ep_r_std = np.mean(ep_r_per_step), np.std(ep_r_per_step)
                idx = (i + 1) // eval_interval - 1
                ep_reward_mean[idx] = ep_r_mean
                ep_reward_std[idx] = ep_r_std

        # plot the mean reward in evaluation
        fig, ax = plt.subplots()
        ax.plot(ep_eval, ep_reward_mean, color="blue")
        sup = list(map(lambda x, y: x + y, ep_reward_mean, ep_reward_std))
        inf = list(map(lambda x, y: x - y, ep_reward_mean, ep_reward_std))
        ax.fill_between(ep_eval, inf, sup, color="blue", alpha=0.2)
        ax.set(xlabel="training_step", ylabel="mean reward per step", title=f"{self.algo_name.upper()} Evaluation Results")
        ax.grid()
        fig.savefig(f"./figures/{self.algo_name}_eval_{datetime.now().strftime('%m%d_%H%M')}.png")

        # plot mean reward in training
        fig, ax = plt.subplots()
        ax.plot(ep_train, ep_reward_mean_train, color='red')
        ax.set(xlabel="training_step", ylabel="mean reward per step", title=f"{self.algo_name.upper()} Training Results")
        ax.grid()
        fig.savefig(f"./figures/{self.algo_name}_train_{datetime.now().strftime('%m%d_%H%M')}.png")

    def eval_plot(self, duration: int = 3):
        """Plot the agent action (TX location) and the optimal TX location in a same map, given a new env.

        """
        env_config: dict = json.loads(json.dumps(self.config.get("env")))
        env_config["evaluation"] = True
        env_config["preset_map_path"] = None
        env_config["eval_plot"] = True
        env_config["n_maps"] = duration
        env_eval = BaseEnvironment(config=env_config)

        for i in range(duration):
            term, trunc = False, False
            obs, _ = env_eval.reset()
            while not (term or trunc):
                action = self.agent.compute_single_action(obs)
                obs, reward, term, trunc, info = env_eval.step(action)
