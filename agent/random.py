import json
import os
from datetime import datetime
import time

import numpy as np
from matplotlib import pyplot as plt

from agent.agent import Agent
from env.utils import save_map
from env.utils_v1 import dict_update, ROOT_DIR, calc_optimal_locations


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
        env_eval = self.env_class(config=env_config_eval)

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
                reward_train_mean += info['r_c'] / num_steps_per_episode
            ep_reward_mean_train[i] = reward_train_mean
            time_total_s = time.time() - time_train_start
            print("\n")
            print(f"================TRAINING # {i + 1}================")
            print(f"time_total_s: {time_total_s}")

            env_eval.reset()
            if eval_interval is not None and (i + 1) % eval_interval == 0:
                # evaluation
                # now it only supports evaluating for one episode
                # because we want to use the same map as that in agent training
                term, trunc = False, False
                reward_per_ep = 0.
                num_steps = 0
                while not term and not trunc:
                    action = env_eval.np_random.choice(np.where(env_eval.mask == 1)[0])
                    obs, reward, term, trunc, info = env_eval.step(action)
                    reward_per_ep += info['r_c']
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
                    "ep_reward_mean": ep_reward_mean.tolist(),
                }
                json.dump(data, open(os.path.join(ROOT_DIR, f"data/{self.algo_name}_{timestamp}.json"), 'w'))

        if log:
            time_total_s = time.time() - time_train_start
            self.logger.info(f"train and eval total time: {time_total_s}s")

        return timestamp

    def test(self, timestamp: str, duration: int = 3, steps_per_map: int = 10, log: bool = True):
        """Test the agent on both the training maps and test maps.

        """
        env_config: dict = json.loads(json.dumps(self.config.get("env")))

        # test on training maps
        env_config["test_algo"] = self.algo_name + "_used"
        env_config["n_episodes_per_map"] = 1
        env_eval = self.env_class(config=env_config)
        env_eval.evaluation = True  # randomly select map at each reset

        coverage_reward_mean_overall = 0.0
        reward_opt_mean = 0.0
        for i in range(duration):
            coverage_reward_mean = 0.0
            cnt = 0
            term, trunc = False, False
            obs, _ = env_eval.reset()
            action_opt, reward_opt = calc_optimal_locations(env_eval.dataset_dir, env_eval.map_suffix, env_eval.map_idx,
                                                            env_eval.coverage_threshold, env_eval.upsampling_factor)
            loc_opt = env_eval.calc_upsampling_loc(action_opt)
            reward_highest, loc_highest = 0, (-1, -1)
            while not (term or trunc) and cnt < steps_per_map:
                action = env_eval.np_random.choice(np.where(env_eval.mask == 1)[0])
                row, col = env_eval.calc_upsampling_loc(action)
                coverage_reward = env_eval.calc_coverage(row, col)
                coverage_reward_mean += coverage_reward
                if coverage_reward > reward_highest:
                    reward_highest = coverage_reward
                    loc_highest = (row, col)
                cnt += 1
                obs, reward, term, trunc, info = env_eval.step(action)

            coverage_reward_mean /= cnt
            coverage_reward_mean_overall += coverage_reward_mean / duration
            reward_opt_mean += reward_opt / duration
            info = f"average coverage reward for trained map {i} in {cnt} steps: {coverage_reward_mean}, optimal reward: {reward_opt}, ratio: {coverage_reward_mean / reward_opt}"
            if log:
                self.logger.info(info)
            print(info)
            if i == 0 or i == duration - 1:
                # plot the optimal TX location and location corresponding the best action in STEP_PER_MAP steps
                test_map_path = os.path.join(ROOT_DIR, 'figures/test_maps',
                                             self.version + '_' + timestamp + '_' + self.algo_name + '_train_' + str(
                                                 i) + '.png')
                save_map(filepath=test_map_path, pixel_map=env_eval.pixel_map, reverse_color=False,
                         mark_size=5, mark_loc=loc_opt, mark_locs=[loc_highest])

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
            action_opt, reward_opt = calc_optimal_locations(env_eval.dataset_dir, env_eval.map_suffix, env_eval.map_idx,
                                                            env_eval.coverage_threshold, env_eval.upsampling_factor)
            loc_opt = env_eval.calc_upsampling_loc(action_opt)
            reward_highest, loc_highest = 0, (-1, -1)
            while not (term or trunc) and cnt < steps_per_map:
                action = env_eval.np_random.choice(np.where(env_eval.mask == 1)[0])
                row, col = env_eval.calc_upsampling_loc(action)
                coverage_reward = env_eval.calc_coverage(row, col)
                coverage_reward_mean += coverage_reward
                if coverage_reward > reward_highest:
                    reward_highest = coverage_reward
                    loc_highest = (row, col)
                cnt += 1
                obs, reward, term, trunc, info = env_eval.step(action)

            coverage_reward_mean /= cnt
            coverage_reward_mean_overall_new += coverage_reward_mean / duration
            reward_opt_mean_new += reward_opt / duration
            info = f"average coverage reward for test map {i} in {cnt} steps: {coverage_reward_mean}, optimal reward: {reward_opt}, ratio: {coverage_reward_mean / reward_opt}"
            if log:
                self.logger.info(info)
            print(info)
            if i == 0 or i == duration - 1:
                # plot the optimal TX location and location corresponding the best action in STEP_PER_MAP steps
                test_map_path = os.path.join(ROOT_DIR, 'figures/test_maps',
                                             self.version + '_' + timestamp + '_' + self.algo_name + '_test_' + str(
                                                 i) + '.png')
                save_map(filepath=test_map_path, pixel_map=env_eval.pixel_map, reverse_color=False,
                         mark_size=5, mark_loc=loc_opt, mark_locs=[loc_highest])

        info1 = (f"overall average coverage reward for trained maps: {coverage_reward_mean_overall},"
                 f" average optimal reward: {reward_opt_mean},"
                 f" ratio: {coverage_reward_mean_overall / reward_opt_mean}")
        info2 = (f"overall average coverage reward for test maps: {coverage_reward_mean_overall_new}"
                 f", average optimal reward: {reward_opt_mean_new},"
                 f" ratio: {coverage_reward_mean_overall_new / reward_opt_mean_new}")
        if log:
            self.logger.info(info1)
            self.logger.info(info2)
        print(info1)
        print(info2)

