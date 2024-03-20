import json
import logging
import os.path
from datetime import datetime
from logging import Logger

import numpy as np
from matplotlib import pyplot as plt
from ray import train, tune
from ray.rllib.algorithms import Algorithm, AlgorithmConfig
from ray.tune.logger import pretty_print

from env.utils_v1 import ROOT_DIR, dict_update


class Agent(object):
    """Abstract base class for DRL agent

    """
    agent_config: AlgorithmConfig  # initialized by specific agent
    agent: Algorithm
    algo_name: str

    def __init__(self, config: dict, log_file: str, version: str = "v11") -> None:
        self.config: dict = config
        # set a logger
        self.logger: Logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.DEBUG)
        handler = logging.FileHandler(log_file, encoding='utf-8', mode='a')
        formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        # set an environment class
        if version == "v11":
            from env.env_v11 import BaseEnvironment
        elif version == "v12":
            from env.env_v12 import BaseEnvironment
        elif version == "v13":
            from env.env_v13 import BaseEnvironment
        else:
            from env.env_v04 import BaseEnvironment
        self.env_class = BaseEnvironment
        self.version: str = version

    def train_and_eval(self, log: bool = True):
        """Train and evaluate the agent.
         Plot the average training/evaluation reward per environment step vs. training step.

        """
        self.agent = self.agent_config.build()
        num_episode = self.config["stop"].get("training_iteration", 10)
        eval_interval = self.config["eval"].get("evaluation_interval", 5)
        data_saving_interval = self.config["agent"].get("data_saving_interval", 10)

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
        start_info = f"==========={self.algo_name.upper()} train and eval started at {timestamp}==========="
        if log:
            self.logger.info(start_info)
        print(start_info)

        for i in range(num_episode):
            # one training step (may include multiple environment episodes)
            result = self.agent.train()

            print("\n")
            print(f"================TRAINING # {i + 1}================")
            print(f"timesteps_total: {result['timesteps_total']}")
            print(f"time_total_s: {result['time_total_s']}")
            if eval_interval is not None and (i + 1) % eval_interval == 0:
                print(f"================EVALUATION AT # {i + 1}================")
            # if not log:
            #     # print for debug ONLY
            #     print(pretty_print(result))

            if i == num_episode - 1:
                if log:
                    # save the result and checkpoint
                    self.logger.info(pretty_print(result))
                    self.logger.debug(self.config)
                    self.logger.info("=============TRAINING ENDED=============")
                    checkpoint_dir = self.agent.save(
                        f"./checkpoint/{self.algo_name}_{timestamp}").checkpoint.path
                    print(f"Checkpoint saved in directory {checkpoint_dir}")
                else:
                    print("=============TRAINING ENDED=============")
                    print(self.config)

            # calculate the training mean reward per step
            episodes_this_iter = result["sampler_results"]["episodes_this_iter"]
            ep_len_train = np.array(result["sampler_results"]["hist_stats"]["episode_lengths"][-episodes_this_iter:])
            ep_reward_train = np.array(result["sampler_results"]["hist_stats"]["episode_reward"][-episodes_this_iter:])
            ep_r_per_step = ep_reward_train / ep_len_train
            ep_reward_mean_train[i] = np.mean(ep_r_per_step)

            if eval_interval is not None and (i + 1) % eval_interval == 0:
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

        if eval_interval is not None:
            # plot the mean reward in evaluation
            fig, ax = plt.subplots()
            fig.set_size_inches(10, 6)
            ax.plot(ep_eval, ep_reward_mean, color="blue")
            sup = list(map(lambda x, y: x + y, ep_reward_mean, ep_reward_std))
            inf = list(map(lambda x, y: x - y, ep_reward_mean, ep_reward_std))
            ax.fill_between(ep_eval, inf, sup, color="blue", alpha=0.2)
            ax.set(xlabel="training_step", ylabel="mean reward per step",
                   title=f"{self.algo_name.upper()} Evaluation Results")
            ax.grid()
            if log:
                fig.savefig(f"./figures/{self.version}_{self.algo_name}_{timestamp}_eval.png")

        # plot mean reward in training
        fig, ax = plt.subplots()
        fig.set_size_inches(10, 6)
        ax.plot(ep_train, ep_reward_mean_train, color='red')
        ax.set(xlabel="training_step", ylabel="mean reward per step",
               title=f"{self.algo_name.upper()} Training Results")
        ax.grid()
        if log:
            fig.savefig(f"./figures/{self.version}_{self.algo_name}_{timestamp}_train.png")

        plt.show()

    def test(self, duration: int = 3, log: bool = True):
        """Test the trained agent on both the training maps and test maps.

        """
        env_config: dict = json.loads(json.dumps(self.config.get("env")))

        # test on training maps
        env_config["test_algo"] = self.algo_name + "_used"
        env_eval = self.env_class(config=env_config)

        coverage_reward_mean_overall = 0.0
        reward_opt_mean = 0.0
        for i in range(duration):
            coverage_reward_mean = 0.0
            cnt = 0
            term, trunc = False, False
            obs, _ = env_eval.reset()
            reward_opt = env_eval.coverage_opt  # np.sum(env_eval.coverage_map_opt)
            while not (term or trunc):
                action = self.agent.compute_single_action(obs)
                row, col = env_eval.calc_upsampling_loc(action)
                coverage_reward = env_eval.calc_coverage(row, col)
                coverage_reward_mean += coverage_reward
                cnt += 1
                obs, reward, term, trunc, info = env_eval.step(action)

            coverage_reward_mean /= cnt
            coverage_reward_mean_overall += coverage_reward_mean / duration
            reward_opt_mean += reward_opt / duration
            info = f"average coverage reward for trained map {i}: {coverage_reward_mean}, optimal reward: {reward_opt}, ratio: {coverage_reward_mean / reward_opt}"
            print(info)

        # test on new maps
        env_config = dict_update(env_config, self.config['eval']['evaluation_config']['env_config'])
        env_config["test_algo"] = self.algo_name + "_new"
        env_eval = self.env_class(config=env_config)

        coverage_reward_mean_overall_new = 0.0
        reward_opt_mean_new = 0.0
        for i in range(duration):
            coverage_reward_mean = 0.0
            cnt = 0
            term, trunc = False, False
            obs, _ = env_eval.reset()
            reward_opt = env_eval.coverage_opt
            while not (term or trunc):
                action = self.agent.compute_single_action(obs)
                row, col = env_eval.calc_upsampling_loc(action)
                coverage_reward = env_eval.calc_coverage(row, col)
                coverage_reward_mean += coverage_reward
                cnt += 1
                obs, reward, term, trunc, info = env_eval.step(action)

            coverage_reward_mean /= cnt
            coverage_reward_mean_overall_new += coverage_reward_mean / duration
            reward_opt_mean_new += reward_opt / duration
            info = f"average coverage reward for new map {i}: {coverage_reward_mean}, optimal reward: {reward_opt}, ratio: {coverage_reward_mean / reward_opt}"
            print(info)

        info1 = (f"overall average coverage reward for trained maps: {coverage_reward_mean_overall},"
                 f" average optimal reward: {reward_opt_mean},"
                 f" ratio: {coverage_reward_mean_overall / reward_opt_mean}")
        info2 = (f"overall average coverage reward for new maps: {coverage_reward_mean_overall_new}"
                 f", average optimal reward: {reward_opt_mean_new},"
                 f" ratio: {coverage_reward_mean_overall_new / reward_opt_mean_new}")
        if log:
            self.logger.info(info1)
            self.logger.info(info2)
        else:
            print(info1)
            print(info2)

    def param_tuning(self, lr_schedule=None, bs_schedule=None, gamma_schedule=None, training_iteration: int = 100):
        if bs_schedule is None:
            bs_schedule = [32]
        if lr_schedule is None:
            lr_schedule = [1e-5]
        if gamma_schedule is None:
            gamma_schedule = [0.9]

        config = self.agent_config.training(lr=tune.grid_search(lr_schedule),
                                            train_batch_size=tune.grid_search(bs_schedule),
                                            gamma=tune.grid_search(gamma_schedule))
        tuner = tune.Tuner(
            self.algo_name.upper(),
            run_config=train.RunConfig(
                stop={"training_iteration": training_iteration},
            ),
            param_space=config.to_dict(),
        )

        results = tuner.fit()

        # Get the best result based on a particular metric.
        best_result = results.get_best_result(metric="episode_reward_mean", mode="max", scope="last-5-avg")
        print(best_result)
        # # Get the best checkpoint corresponding to the best result.
        # best_checkpoint = best_result.checkpoint

