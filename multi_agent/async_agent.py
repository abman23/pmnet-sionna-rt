import json
import logging
import os.path
import time
from logging import Logger

import numpy as np
from matplotlib import pyplot as plt
from ray import train, tune
from ray.rllib.algorithms import Algorithm, AlgorithmConfig
from ray.tune.logger import pretty_print

from env.utils_v2 import save_map_with_tx
from env.utils_v1 import ROOT_DIR, dict_update


class Agent(object):
    """Base class for DRL agent under the asynchronous MDP setting.

    """
    agent_config: AlgorithmConfig  # initialized by specific agent
    agent: Algorithm
    algo_name: str

    def __init__(self, config: dict, log_file: str, version: str = "v21") -> None:
        self.config: dict = config
        # set a logger
        self.logger: Logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.DEBUG)
        handler = logging.FileHandler(log_file, encoding='utf-8', mode='a')
        formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        # set an environment class
        if version == "v21":
            from env.env_v21 import BaseEnvironment
        else:
            from env.env_v21 import BaseEnvironment
        self.env_class = BaseEnvironment
        self.version: str = version

    def train_and_eval(self, log: bool = True, **kwargs):
        """Train and evaluate the agent.
         Plot the average training/evaluation reward per environment step vs. training step.

        """
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
        ep_train = np.arange(num_episode) + 1
        ep_reward_mean_train = np.empty(num_episode, dtype=float)
        ep_reward_std_train = np.empty(num_episode, dtype=float)

        timestamp = kwargs["timestamp"]
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
                        f"./checkpoint/{self.version}_{self.algo_name}_{timestamp}").checkpoint.path
                    print(f"Checkpoint saved in directory {checkpoint_dir}")
                else:
                    print("=============TRAINING ENDED=============")
                    print(self.config)

            # calculate the training mean reward per step
            reward_per_round_mean = result["custom_metrics"]["reward_per_round_mean"]
            reward_per_round_std = result["custom_metrics"]["reward_per_round_std"]
            ep_reward_mean_train[i] = reward_per_round_mean
            ep_reward_std_train[i] = reward_per_round_std

            if eval_interval is not None and (i + 1) % eval_interval == 0:
                # calculate the evaluation mean reward per step
                ep_r_mean = result['evaluation']["custom_metrics"]["reward_per_round_mean"]
                ep_r_std = result['evaluation']["custom_metrics"]["reward_per_round_std"]
                idx = (i + 1) // eval_interval - 1
                ep_reward_mean[idx] = ep_r_mean
                ep_reward_std[idx] = ep_r_std

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
                json.dump(data,
                          open(os.path.join(ROOT_DIR, f"data/{self.version}_{self.algo_name}_{timestamp}.json"), 'w'))
                # save the model
                self.agent.save(os.path.join(ROOT_DIR, f"checkpoint/{self.version}_{self.algo_name}_{timestamp}"))

        if eval_interval is not None:
            # plot the mean reward in evaluation
            fig, ax = plt.subplots()
            fig.set_size_inches(10, 6)
            ax.plot(ep_eval, ep_reward_mean, color="blue")
            sup = list(map(lambda x, y: x + y, ep_reward_mean, ep_reward_std))
            inf = list(map(lambda x, y: x - y, ep_reward_mean, ep_reward_std))
            ax.fill_between(ep_eval, inf, sup, color="blue", alpha=0.5)
            ax.set(xlabel="training_step", ylabel="mean reward per step",
                   title=f"{self.algo_name.upper()} Evaluation Results")
            ax.grid()
            if log:
                fig.savefig(os.path.join(ROOT_DIR, f"figures/{self.version}_{self.algo_name}_{timestamp}_eval.png"))

        # plot mean reward in training
        fig, ax = plt.subplots()
        fig.set_size_inches(10, 6)
        ax.plot(ep_train, ep_reward_mean_train, color='red')
        sup = list(map(lambda x, y: x + y, ep_reward_mean_train, ep_reward_std_train))
        inf = list(map(lambda x, y: x - y, ep_reward_mean_train, ep_reward_std_train))
        ax.fill_between(ep_train, inf, sup, color="red", alpha=0.5)
        ax.set(xlabel="training_step", ylabel="mean reward per step",
               title=f"{self.algo_name.upper()} Training Results")
        ax.grid()
        if log:
            fig.savefig(os.path.join(ROOT_DIR, f"figures/{self.version}_{self.algo_name}_{timestamp}_train.png"))

        plt.show()

    def continue_train(self, start_episode: int, data_path: str, model_path: str, **kwargs) -> None:
        """Continue training the agent from the given checkpoint and training data.

        """
        # load the training data
        training_data = json.load(open(data_path))
        num_episode = self.config["stop"].get("training_iteration", 10)
        training_data['ep_train'] = list(range(1, num_episode + 1))
        eval_interval = self.config["eval"].get("evaluation_interval", 5)
        training_data['ep_eval'] = list(range(eval_interval, num_episode + 1, eval_interval))
        data_saving_interval = self.config["agent"].get("data_saving_interval", 10)

        # reload the model
        self.agent.restore(model_path)

        timestamp = kwargs["timestamp"]
        log = kwargs['log']
        start_info = f"==========={self.algo_name.upper()} Training at {timestamp} Continue==========="
        if log:
            self.logger.info(start_info)
        print(start_info)

        for i in range(start_episode, num_episode):
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
                        os.path.join(ROOT_DIR,
                                     f"checkpoint/{self.version}_{self.algo_name}_{timestamp}")).checkpoint.path
                    print(f"Checkpoint saved in directory {checkpoint_dir}")
                else:
                    print("=============TRAINING ENDED=============")
                    print(self.config)

                    # calculate the training mean reward per step
                    reward_per_round_mean = result["custom_metrics"]["reward_per_round_mean"]
                    reward_per_round_std = result["custom_metrics"]["reward_per_round_var"]
                    if i >= len(training_data['ep_reward_mean_train']):
                        training_data['ep_reward_mean_train'].append(reward_per_round_mean)
                        training_data['ep_reward_std_train'].append(reward_per_round_std)
                    else:
                        training_data['ep_reward_mean_train'][i] = reward_per_round_mean
                        training_data['ep_reward_std_train'][i] = reward_per_round_std

                    if eval_interval is not None and (i + 1) % eval_interval == 0:
                        # calculate the evaluation mean reward per step
                        ep_r_mean = result['evaluation']["custom_metrics"]["reward_per_round_mean"]
                        ep_r_std = result['evaluation']["custom_metrics"]["reward_per_round_var"]
                        idx = (i + 1) // eval_interval - 1
                        if idx >= len(training_data['ep_reward_mean']):
                            training_data['ep_reward_mean'].append(ep_r_mean)
                            training_data['ep_reward_std'].append(ep_r_std)
                        else:
                            training_data['ep_reward_mean'][idx] = ep_r_mean
                            training_data['ep_reward_std'][idx] = ep_r_std

            if log and ((i + 1) % data_saving_interval == 0 or i == num_episode - 1):
                # save the training and evaluation data periodically
                json.dump(training_data,
                          open(os.path.join(ROOT_DIR, f"data/{self.version}_{self.algo_name}_{timestamp}.json"), 'w'))
                # save the model
                self.agent.save(os.path.join(ROOT_DIR, f"checkpoint/{self.version}_{self.algo_name}_{timestamp}"))

        if eval_interval is not None:
            # plot the mean reward in evaluation
            fig, ax = plt.subplots()
            fig.set_size_inches(10, 6)
            ax.plot(training_data['ep_eval'], training_data['ep_reward_mean'], color="blue")
            sup = list(map(lambda x, y: x + y, training_data['ep_reward_mean'], training_data['ep_reward_std']))
            inf = list(map(lambda x, y: x - y, training_data['ep_reward_mean'], training_data['ep_reward_std']))
            ax.fill_between(training_data['ep_eval'], inf, sup, color="blue", alpha=0.2)
            ax.set(xlabel="training_step", ylabel="mean reward",
                   title=f"{self.algo_name.upper()} Evaluation Results")
            ax.grid()
            if log:
                fig.savefig(os.path.join(ROOT_DIR, f"figures/{self.version}_{self.algo_name}_{timestamp}_eval.png"))

        # plot mean reward in training
        fig, ax = plt.subplots()
        fig.set_size_inches(10, 6)
        ax.plot(training_data['ep_train'], training_data['ep_reward_mean_train'], color='red')
        sup = list(map(lambda x, y: x + y, training_data['ep_reward_mean_train'], training_data['ep_reward_std_train']))
        inf = list(map(lambda x, y: x - y, training_data['ep_reward_mean_train'], training_data['ep_reward_std_train']))
        ax.fill_between(training_data['ep_train'], inf, sup, color="red", alpha=0.2)
        ax.set(xlabel="training_step", ylabel="mean reward",
               title=f"{self.algo_name.upper()} Training Results")
        ax.grid()
        if log:
            fig.savefig(os.path.join(ROOT_DIR, f"figures/{self.version}_{self.algo_name}_{timestamp}_train.png"))

        plt.show()

    def test(self, timestamp: str, duration: int = 25, log: bool = True, suffix: str = 'after'):
        """Test the trained agent on both the training maps and test maps.

        """
        msg = f"\n=============Test for {self.algo_name.upper()} {suffix.upper()} Training============="
        print(msg)
        if log:
            self.logger.info(msg)

        env_config: dict = json.loads(json.dumps(self.config.get("env")))

        # test on training maps
        env_config["algo_name"] = self.algo_name + "_used"
        env_config["n_episodes_per_map"] = 1
        env_eval = self.env_class(config=env_config)
        env_eval.evaluation = True  # select map in sequence at each reset

        coverage_reward_mean_overall = 0.0
        reward_opt_mean = 0.0
        num_roi_mean = 0

        for i in range(duration):
            before_reset = time.time()
            obs, info_dict = env_eval.reset()
            after_reset = time.time()
            # number of RoI pixels - black
            num_roi = int(env_eval.map_size ** 2 - env_eval.pixel_map.sum())
            num_roi_mean += num_roi / duration
            locs_opt, reward_opt = env_eval.calc_optimal_locations()
            after_calc_opt = time.time()
            locs = []

            before_action = time.time()
            for _ in range(env_eval.n_bs):
                action = self.agent.compute_single_action(obs)
                row, col = env_eval.calc_upsampling_loc(action)
                locs.append((row, col))
                obs, _, _, _, info_dict = env_eval.step(action)
            after_action = time.time()

            accumulated_reward = info_dict['accumulated_reward']
            coverage_reward_mean_overall += accumulated_reward / duration
            reward_opt_mean += reward_opt / duration
            info = (f"coverage reward for trained map {i} with index {env_eval.map_idx}: {accumulated_reward}, "
                    f"optimal reward: {reward_opt}, "
                    f"ratio: {accumulated_reward / reward_opt}, num_roi: {num_roi}")
            time_info = (
                f"one map total {time.time() - before_reset:.4f}s, env reset {after_reset - before_reset:.4f}s, "
                f"calc opt action {after_calc_opt - after_reset:.4f}s, inference {after_action - before_action:.4f}s")

            if i % 10 == 0:
                print(time_info)
                print(info)
                if log:
                    self.logger.info(time_info)
                    self.logger.info(info)
                    if i == 0:
                        # plot the optimal TX location and location corresponding the best action in STEP_PER_MAP steps
                        test_map_path = os.path.join(ROOT_DIR, 'figures/test_maps',
                                                     self.version + '_' + timestamp + '_' + self.algo_name + '_train_' +
                                                     str(i) + '_' + suffix + '.png')
                        save_map_with_tx(filepath=test_map_path, pixel_map=env_eval.pixel_map, mark_size=5,
                                         target_locs=locs_opt, curr_locs=locs)

        # test on new maps
        env_config = dict_update(env_config, self.config['eval']['evaluation_config']['env_config'])
        env_config["algo_name"] = self.algo_name + "_new"
        env_eval = self.env_class(config=env_config)

        coverage_reward_mean_overall_new = 0.0
        reward_opt_mean_new = 0.0
        num_roi_mean_new = 0

        start_time = time.time()
        for i in range(duration):
            obs, info_dict = env_eval.reset()
            # number of RoI pixels - black
            num_roi = int(env_eval.map_size ** 2 - env_eval.pixel_map.sum())
            num_roi_mean_new += num_roi / duration
            locs_opt, reward_opt = env_eval.calc_optimal_locations()
            locs = []
            for _ in range(env_eval.n_bs):
                action = self.agent.compute_single_action(obs)
                row, col = env_eval.calc_upsampling_loc(action)
                locs.append((row, col))
                obs, _, _, _, info_dict = env_eval.step(action)

            accumulated_reward = info_dict['accumulated_reward']
            coverage_reward_mean_overall_new += accumulated_reward / duration
            reward_opt_mean_new += reward_opt / duration
            info = (f"coverage reward for test map {i} with index {env_eval.map_idx}: {accumulated_reward}, "
                    f"optimal reward: {reward_opt}, "
                    f"ratio: {accumulated_reward / reward_opt}, num_roi: {num_roi}")

            if i % 10 == 0:
                print(f"{time.time() - start_time:.4f}s so far")
                print(info)
                if log:
                    self.logger.info(info)
                    if i == 0:
                        # plot the optimal TX location and location corresponding the best action in STEP_PER_MAP steps
                        test_map_path = os.path.join(ROOT_DIR, 'figures/test_maps',
                                                     f"{self.version}_{timestamp}_{self.algo_name}_test_{i}_{suffix}.png")
                        save_map_with_tx(filepath=test_map_path, pixel_map=env_eval.pixel_map, mark_size=5,
                                         target_locs=locs_opt, curr_locs=locs)

        info1 = (f"overall average coverage reward for trained maps: {coverage_reward_mean_overall},"
                 f" average optimal reward: {reward_opt_mean},"
                 f" ratio: {coverage_reward_mean_overall / reward_opt_mean},"
                 f" average number of RoI pixels: {num_roi_mean},"
                 f" percentage of coverage: {coverage_reward_mean_overall / num_roi_mean * 100}")
        info2 = (f"overall average coverage reward for test maps: {coverage_reward_mean_overall_new}"
                 f", average optimal reward: {reward_opt_mean_new},"
                 f" ratio: {coverage_reward_mean_overall_new / reward_opt_mean_new},"
                 f" average number of RoI pixels: {num_roi_mean_new},"
                 f" percentage of coverage: {coverage_reward_mean_overall_new / num_roi_mean_new * 100}")
        if log:
            self.logger.info(info1)
            self.logger.info(info2)
        print(info1)
        print(info2)
