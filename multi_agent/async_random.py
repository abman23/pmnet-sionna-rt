import json
import os
import time

import numpy as np

from env.utils_v2 import draw_map_with_tx, plot_coverage
from env.utils_v1 import ROOT_DIR, dict_update
from multi_agent.async_agent import Agent


class AsyncRandom(Agent):
    def __init__(self, config: dict, log_file: str, version: str) -> None:
        super().__init__(config, log_file, version)

        self.algo_name = 'random'

    def train_and_eval(self, log: bool = True, **kwargs):
        num_episode = self.config["stop"].get("training_iteration", 10)
        num_steps_per_episode = self.config['report'].get('min_sample_timesteps_per_iteration', 2000)
        eval_interval = self.config["eval"].get("evaluation_interval", 5)
        num_maps_per_eval = self.config["eval"].get("num_maps_per_eval", 1)
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
        ep_train = np.arange(num_episode) + 1
        ep_reward_mean_train = np.empty(num_episode, dtype=float)
        ep_reward_std_train = np.empty(num_episode, dtype=float)

        timestamp = kwargs["timestamp"]
        start_info = f"==========={self.algo_name.upper()} train and eval started at {timestamp}==========="
        if log:
            self.logger.info(start_info)
        print(start_info)

        time_train_start = time.time()
        for i in range(num_episode):
            reward_train = []
            reward_eval = []

            terminated, truncated = False, False
            before_reset = time.time()
            env_train.reset()
            after_reset = time.time()
            # training
            for j in range(num_steps_per_episode):
                action = env_train.np_random.choice(np.where(env_train.mask == 1)[0])
                obs, reward, terminated, truncated, info = env_train.step(action)
                if terminated or truncated:
                    steps, n_bs = info['steps'], info['n_bs']
                    n_rounds = np.ceil(steps / n_bs)
                    reward_train.append(info['accumulated_reward'] / n_rounds)
                    env_train.reset()
            after_step = time.time()
            ep_reward_mean_train[i] = np.mean(reward_train)
            ep_reward_std_train[i] = np.std(reward_train)
            time_total_s = time.time() - time_train_start
            print("\n")
            print(f"================TRAINING # {i + 1}================")
            print(f"time_total_s: {time_total_s}")
            print(f"reset time: {after_reset - before_reset}s, step time: {after_step - after_reset}s")

            if eval_interval is not None and (i + 1) % eval_interval == 0:
                # evaluation
                for j in range(num_maps_per_eval):
                    env_eval.reset()
                    term, trunc = False, False

                    while not term and not trunc:
                        action = env_eval.np_random.choice(np.where(env_eval.mask == 1)[0])
                        obs, reward, term, trunc, info = env_eval.step(action)
                        if terminated or truncated:
                            steps, n_bs = info['steps'], info['n_bs']
                            n_rounds = np.ceil(steps / n_bs)
                            reward_eval.append(info['accumulated_reward'] / n_rounds)
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
                json.dump(data,
                          open(os.path.join(ROOT_DIR, f"data/{self.version}_{self.algo_name}_{timestamp}.json"), 'w'))

        if log:
            time_total_s = time.time() - time_train_start
            self.logger.info(f"train and eval total time: {time_total_s}s")

    def test(self, timestamp: str, duration: int = 3, log: bool = True, suffix: str = "after",
             test_on_trained: bool = False):
        """Test the agent on both the training maps and test maps.

        """
        msg = f"\n=============Test for {self.algo_name.upper()} {suffix.upper()}============="
        print(msg)
        if log:
            self.logger.info(msg)

        env_config_train: dict = json.loads(json.dumps(self.config.get("env")))
        env_config_test = dict_update(env_config_train, self.config['eval']['evaluation_config']['env_config'])
        if test_on_trained:
            env_configs = {'training': env_config_train, 'test': env_config_test}
        else:
            env_configs = {'test': env_config_test}

        start_time = time.time()
        for env_type, env_config in env_configs.items():
            env_eval = self.env_class(config=env_config)
            env_eval.evaluation = True  # select map in sequence at each reset

            reward_mean_overall = 0.0
            reward_opt_mean = 0.0
            num_roi_mean = 0

            for i in range(duration):
                obs, info_dict = env_eval.reset()
                # number of RoI pixels
                num_roi = np.sum(env_eval.pixel_map == env_eval.non_building_pixel)
                num_roi_mean += num_roi / duration
                locs_opt, reward_opt = env_eval.calc_optimal_locations()
                locs = []

                for _ in range(env_eval.n_bs):
                    action = env_eval.np_random.choice(np.where(env_eval.mask == 1)[0])
                    row, col = env_eval.calc_upsampling_loc(action)
                    locs.append((row, col))
                    obs, _, _, _, info_dict = env_eval.step(action)

                accumulated_reward = info_dict['accumulated_reward']
                reward_mean_overall += accumulated_reward / duration
                reward_opt_mean += reward_opt / duration
                info = (
                    f"reward for {env_type} map {i} with index {env_eval.map_idx}: {accumulated_reward}, "
                    f"optimal reward: {reward_opt}, "
                    f"ratio: {accumulated_reward / reward_opt}, num_roi: {num_roi}"
                )

                if i % 10 == 0:
                    print(f"{time.time() - start_time:.4f}s so far")
                    print(info)
                    if log:
                        self.logger.info(info)

                if log and (i == 0 or i == duration - 1):
                    if env_eval.reward_type == 'coverage':
                        # plot coverage area of deployed TXs and optimal TXs
                        coverage_map, _ = env_eval.calc_coverage(locs)
                        coverage_map_opt, _ = env_eval.calc_coverage(locs_opt)
                        overall_rewards = env_eval.calc_rewards_for_all_locations()
                        coverage_map_dir = os.path.join(ROOT_DIR, 'figures/coverage_map')
                        os.makedirs(coverage_map_dir, exist_ok=True)
                        coverage_map_path = os.path.join(coverage_map_dir,
                                                         f'{self.version}_{timestamp}_{self.algo_name}_{env_eval.map_idx}_{suffix}.png')
                        plot_coverage(filepath=coverage_map_path, pixel_map=env_eval.pixel_map,
                                      coverage_curr=coverage_map,
                                      coverage_opt=coverage_map_opt, tx_locs=locs, opt_tx_locs=locs_opt, rewards=overall_rewards, save=True)
                    elif env_eval.reward_type == 'capacity':
                        # plot capacity map (overlap of power maps corresponding to multiple TX locations)
                        capacity_map, _ = env_eval.calc_capacity(locs)
                        capacity_map_opt, _ = env_eval.calc_capacity(locs_opt)
                        overall_rewards = env_eval.calc_rewards_for_all_locations()
                        capacity_map_dir = os.path.join(ROOT_DIR, 'figures/capacity_map')
                        os.makedirs(capacity_map_dir, exist_ok=True)
                        capacity_map_path = os.path.join(capacity_map_dir,
                                                         f'{self.version}_{timestamp}_{self.algo_name}_{env_eval.map_idx}.png')
                        plot_coverage(filepath=capacity_map_path, pixel_map=env_eval.pixel_map,
                                      coverage_curr=capacity_map,
                                      coverage_opt=capacity_map_opt, tx_locs=locs, opt_tx_locs=locs_opt, rewards=overall_rewards, save=True)

            info1 = (
                f"overall average reward for {env_type} maps: {reward_mean_overall},"
                f" average optimal reward: {reward_opt_mean},"
                f" ratio: {reward_mean_overall / reward_opt_mean},"
                f" average number of RoI pixels: {num_roi_mean},"
            )
            if env_eval.reward_type == 'coverage':
                info1 += f" percentage of coverage: {reward_mean_overall / num_roi_mean * 100}"
            else:
                info1 += f" overall average capacity: {reward_mean_overall}"
            if log:
                self.logger.info(info1)
            print(info1)
