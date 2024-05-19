import itertools
import json
import os
from typing import Any, SupportsFloat

import gymnasium as gym
import numpy as np
from gymnasium.core import ActType, ObsType
from gymnasium.spaces import Discrete, Dict, Box
from gymnasium.utils import seeding

from env.utils_v1 import ROOT_DIR, load_map_normalized

RANDOM_SEED: int | None = None  # manually set random seed


class BaseEnvironment(gym.Env):
    """MDP environment of single-BS, version 1.9.
    Old dataset, capacity reward.

    """
    pixel_map: np.ndarray  # current building map
    version: str = "v19"
    steps: int
    map_idx: int  # index of the current building map
    n_deployed_bs: int  # number of deployed BS
    accumulated_reward: int  # accumulated reward in one episode
    tx_locs: list  # Deployed TX locations
    r_prev: float  # reward before deploying the current TX

    def __init__(self, config: dict) -> None:
        """Initialize the base MDP environment.

        """
        # directory contains building maps, training and test power maps
        self.dataset_dir = config.get("dataset_dir", "resource/new_usc")
        evaluation_mode = config.get("evaluation", False)
        self.evaluation = evaluation_mode
        # training or test env
        self.map_suffix = "test" if evaluation_mode else "train"
        # indices of maps used for training or test
        if not evaluation_mode:
            self.map_indices = np.arange(1, 2412, 2)[:964]
        else:
            self.map_indices = np.arange(1, 2412, 2)[964:]
        self.coverage_threshold: float = 240. / 255
        self.non_building_pixel: float = config["non_building_pixel"]
        self.reward_type: str = config.get('reward_type', 'coverage')

        self.n_bs: int = config.get("n_bs", 2)
        self.n_maps: int = config.get("n_maps", 1)
        self.n_steps_truncate: int = config.get("n_steps_truncate", 10)
        # count the number of used cropped maps
        self.n_trained_maps: int = 0

        map_size = config.get("map_size", 256)
        # number of pixels in one row or column of the cropped map
        self.map_size: int = map_size
        action_space_size = config.get("action_space_size", 64)
        assert map_size % action_space_size == 0, f"map_size {map_size} must be divisible by action_space_size {action_space_size}"
        # used for calculating the location corresponding to action in the reduced action space
        self.upsampling_factor = map_size // action_space_size
        self.action_space_size: int = action_space_size
        # action mask has the same shape as the action
        self.mask: np.ndarray = np.empty(action_space_size ** 2, dtype=np.int8)
        self.no_masking: bool = config.get("no_masking", False)

        self.action_space: Discrete = Discrete(action_space_size ** 2)
        if self.no_masking:
            self.observation_space: Box = Box(low=0., high=1., shape=(map_size ** 2,), dtype=np.float32)
        else:
            self.observation_space: Dict = Dict(
                {
                    "observations": Box(low=0., high=1., shape=(map_size ** 2,), dtype=np.float32),
                    "action_mask": Box(low=0., high=1., shape=(self.action_space.n,), dtype=np.int8)
                }
            )

        # fix a random seed
        self._np_random, seed = seeding.np_random(RANDOM_SEED)

    def reset(
            self, *, seed: int | None = None, options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        super().reset(seed=seed, options=options)

        self.steps = 0
        self.n_deployed_bs = 0
        self.accumulated_reward = 0
        # switch the building map
        # map_idx uniquely determines a map in the dataset
        # choose the map in sequence in evaluation while randomly choose in training
        if not self.evaluation and self.n_trained_maps % self.n_maps == 0:
            # shuffle the training map indices after all maps are trained
            self.map_indices = np.random.permutation(self.map_indices)
        self.map_idx = int(self.map_indices[self.n_trained_maps % self.n_maps])
        self.n_trained_maps += 1
        # update map and action mask
        map_path = os.path.join(self.dataset_dir, 'map', str(self.map_idx) + '.png')
        self.pixel_map = load_map_normalized(map_path)
        # 1 - building, 0 - free space
        self.mask = self._calc_action_mask()

        # # initial state - no deployed TX
        self.tx_locs = []
        self.r_prev = 0.
        # obs = np.concatenate([self.pixel_map, self.coverage_map], axis=None)
        obs = self.pixel_map.reshape(-1)

        if self.no_masking:
            observation = obs
        else:
            observation = {
                "observations": obs,
                "action_mask": self.mask
            }

        info_dict = {
            "n_trained_maps": self.n_trained_maps,
            "map_suffix": self.map_suffix,
            "map_index": self.map_idx,
            "accumulated_reward": self.accumulated_reward,
        }

        return observation, info_dict

    def step(
            self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        row, col = self.calc_upsampling_loc(action)

        # calculate reward
        self.tx_locs.append((row, col))
        _, r_new = self.calc_capacity(self.tx_locs)
        r = r_new - self.r_prev
        self.accumulated_reward += r
        self.r_prev = r_new

        self.steps += 1
        trunc = self.steps >= self.n_steps_truncate  # truncate if reach the step limit

        # reset TX locations and reward if all TXs are deployed
        if not trunc and len(self.tx_locs) == self.n_bs:
            self.tx_locs = []
            self.r_prev = 0.

        info_dict = {
            "steps": self.steps,
            "loc": [row, col],
            "reward": r,
            "n_bs": self.n_bs,
            "accumulated_reward": self.accumulated_reward,
        }

        # update observation
        self.n_deployed_bs = (self.n_deployed_bs + 1) % self.n_bs
        # if self.n_deployed_bs == 0:
        #     # reset coverage map
        #     self.coverage_map = np.zeros_like(self.pixel_map)
        # obs = np.concatenate((self.pixel_map, self.coverage_map), axis=None)
        obs = self.pixel_map.reshape(-1)

        if self.no_masking:
            observation = obs
        else:
            observation = {
                "observations": obs,
                "action_mask": self.mask
            }

        return observation, r, False, trunc, info_dict
    
    def calc_rewards_for_all_locations(self) -> dict:
        """Calculate the reward for all possible TX locations.
    
        Returns:
            A dictionary containing rewards for all possible locations.
        """
        rewards = {}
        
        data_dir = os.path.join(self.dataset_dir, f'overall_{self.reward_type}_{self.n_bs}')
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        filename = os.path.join(data_dir, f"overall_{self.reward_type}_{self.n_bs}_{self.map_idx}.json")
        if not os.path.exists(filename):
            all_actions = itertools.combinations_with_replacement(range(self.action_space_size ** 2), 1)
    
            for actions in all_actions:
                tx_locs = []
                flag = False
                for action in actions:
                    row, col = self.calc_upsampling_loc(action)
                    if self.pixel_map[row, col] == self.non_building_pixel:
                        # skip non-building pixel
                        flag = True
                        break
                    tx_locs.append((row, col))
                if flag:
                    continue
    
                if self.reward_type == 'coverage':
                    _, reward = self.calc_coverage(tx_locs)
                else:  # capacity
                    _, reward = self.calc_capacity(tx_locs)
                
                rewards[f'{row},{col}'] = reward

            # save result to avoid repeated computation
            json.dump(rewards, open(filename, 'w'))
        else:
            rewards = json.load(open(filename))
    
        return rewards

    def calc_optimal_locations(self) -> tuple[list, int]:
        """Calculate the optimal TX locations that maximize the reward.

            Returns:
                (actions, optimal coverage reward).

        """
        locs_opt, reward_opt = [(-1, -1)], -1

        data_dir = os.path.join(self.dataset_dir, f'optimal_{self.reward_type}_{self.n_bs}')
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        filename = os.path.join(data_dir, f"optimal_{self.reward_type}_{self.n_bs}_{self.map_idx}.json")
        if not os.path.exists(filename):
            # print(f"No existing optimal reward {filename}")
            all_actions = itertools.combinations_with_replacement(range(self.action_space_size ** 2), self.n_bs)

            for actions in all_actions:
                tx_locs = []
                flag = False
                for action in actions:
                    row, col = self.calc_upsampling_loc(action)
                    if self.pixel_map[row, col] == self.non_building_pixel: 
                        # skip non-building pixel
                        flag = True
                        break
                    tx_locs.append((row, col))
                if flag:
                    continue

                if self.reward_type == 'coverage':
                    _, reward = self.calc_coverage(tx_locs)
                else:  # capacity
                    _, reward = self.calc_capacity(tx_locs)
                if reward > reward_opt:
                    reward_opt = reward
                    locs_opt = tx_locs

            # save result to avoid repeatedly computation
            result = {"locs_opt": locs_opt, "reward_opt": reward_opt}
            json.dump(result, open(filename, 'w'))
        else:
            result = json.load(open(filename))
            locs_opt, reward_opt = result["locs_opt"], result["reward_opt"]

        return locs_opt, reward_opt

    def calc_coverage(self, tx_locs: list) -> tuple[np.ndarray, int]:
        """Calculate the overall coverage reward of multiple TXs, given their locations.

        Args:
            tx_locs: tx location tuples

        Returns:
            (Coverage map, number of covered pixels in the map)

        """
        overall_coverage = np.zeros_like(self.pixel_map, dtype=int)
        for tx_loc in tx_locs:
            row, col = tx_loc[0], tx_loc[1]
            loc_idx = row * self.map_size + col
            pmap_filename = f'pmap_{self.map_idx}_{loc_idx}.png'
            # pmap_dir = os.path.join(self.dataset_dir, f'pmap_{self.map_suffix}')
            pmap_dir = os.path.join(self.dataset_dir, 'power_map', f'{self.map_idx}')
            pmap_path = os.path.join(pmap_dir, pmap_filename)
            power_map = load_map_normalized(pmap_path)
            # compute pixels covered by one TX from power map
            covered = np.where(power_map >= self.coverage_threshold, 1, 0)
            # update overall covered pixels
            overall_coverage = np.maximum(overall_coverage, covered)

        return overall_coverage, int(overall_coverage.sum())

    def calc_capacity(self, tx_locs: list[tuple]) -> tuple[np.ndarray, float]:
        """Calculate the capacity reward of TXs (between 0 and 1), given their locations.

        Args:
            tx_locs: [(row_1, col_1), (row_2, col_2) ...].

        Returns:
            (Capacity map, average max grayscale value of each RoI pixel given different TX locations)

        """
        # only consider RoI while RoI are black pixels (value=0) in the power map
        num_roi = np.sum(self.pixel_map == self.non_building_pixel)

        capacity_map = np.zeros_like(self.pixel_map, dtype=float)
        for row, col in tx_locs:
            loc_idx = row * self.map_size + col
            pmap_filename = f'pmap_{self.map_idx}_{loc_idx}.png'
            pmap_dir = os.path.join(self.dataset_dir, 'power_map', f'{self.map_idx}')
            pmap_path = os.path.join(pmap_dir, pmap_filename)
            power_map = load_map_normalized(pmap_path)
            capacity_map = np.maximum(capacity_map, power_map)

        avg_capacity = capacity_map.sum() / num_roi
        # threshold = 0.714076
        # transformed_avg_capacity = ((avg_capacity - threshold) if avg_capacity > threshold else 0)/(1-threshold)
        
        transformed_avg_capacity = np.exp(avg_capacity * 10) / np.exp(10)
        return capacity_map, float(transformed_avg_capacity)

    def calc_upsampling_loc(self, action: int) -> tuple:
        """Calculate the location corresponding to a 'space-reduced' action by upsampling.

        Args:
            action: action in the reduced action space.

        Returns:
            Coordinate of the location - (row, col).

        """
        row_r, col_r = divmod(action, self.action_space_size)
        row = row_r * self.upsampling_factor + (self.upsampling_factor - 1) // 2
        col = col_r * self.upsampling_factor + (self.upsampling_factor - 1) // 2
        return row, col

    def _calc_action_mask(self) -> np.ndarray:
        """Calculate the action mask in the reduced action space.

        Returns:
            A 0-1 flatten array of the action mask.

        """
        idx = np.arange((self.upsampling_factor - 1) // 2, self.map_size, self.upsampling_factor)
        # filter out non-building pixel
        action_pixels = np.where(self.pixel_map[idx][:, idx] != self.non_building_pixel, 1, 0)
        return action_pixels.reshape(-1).astype(np.int8)
