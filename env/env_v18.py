import json
import logging
import os
from typing import Any, SupportsFloat

import gymnasium as gym
import numpy as np
from gymnasium.core import ActType, ObsType
from gymnasium.spaces import Discrete, MultiBinary, Dict, Box
from gymnasium.utils import seeding
from retrying import retry

from env.utils_v1 import ROOT_DIR, load_map_normalized

RANDOM_SEED: int | None = None  # manually set random seed


class BaseEnvironment(gym.Env):
    """MDP environment of single-agent autoBS, version 1.8.
    Adapted based on the new dataset, directly using reward matrix instead of power maps.

    """
    pixel_map: np.ndarray  # current building map
    version: str = "v18"
    steps: int
    map_idx: int  # index of the current building map
    reward_matrix: np.ndarray

    def __init__(self, config: dict) -> None:
        """Initialize the base MDP environment.

        """
        # directory contains building maps, training and test power maps
        self.dataset_dir = config.get("dataset_dir", "resource/usc_old")
        evaluation_mode = config.get("evaluation", False)
        self.evaluation = evaluation_mode
        self.test_algo = config.get("algo_name", None)
        # training or test env
        self.map_suffix = "test" if evaluation_mode else "train"
        # indices of maps used for training or test
        self.map_indices: np.ndarray = np.arange(1001, 1101) if evaluation_mode else np.arange(1, 1001)

        self.n_maps: int = config.get("n_maps", 100)
        # number of continuous steps using one cropped map
        self.n_episodes_per_map: int = config.get("n_episodes_per_map", 1)
        self.n_steps_truncate: int = config.get("n_steps_truncate", 10)
        self.n_episodes: int = 0
        # coefficients of reward items
        self.coefficient_dict = config.get("coefficient_dict", {})
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
                    "action_mask": MultiBinary(action_space_size ** 2)
                }
            )

        # fix a random seed
        self._np_random, seed = seeding.np_random(RANDOM_SEED)
        # if not evaluation_mode:
        #     logger.info(f"=============NEW ENV {self.version.upper()} INITIALIZED=============")

    # retry if action mask contains no 1 (no valid action in the reduced action space)
    @retry(stop_max_attempt_number=3)
    def reset(
            self, *, seed: int | None = None, options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        super().reset(seed=seed, options=options)

        self.steps = 0
        if self.n_episodes % self.n_episodes_per_map == 0:
            # switch the building map
            # map_idx uniquely determines a map in the dataset
            # choose the map in sequence in evaluation while randomly choose in training
            if self.evaluation:
                self.map_idx = int(self.map_indices[self.n_trained_maps % self.n_maps])
            else:
                self.map_idx = self.np_random.choice(self.map_indices)
            # print(f"Map {self.map_idx} selected")
            self.n_trained_maps += 1
            # update map and action mask
            map_path = os.path.join(ROOT_DIR, self.dataset_dir, 'map', str(self.map_idx) + '.png')
            self.pixel_map = load_map_normalized(map_path)
            # 1 - building, 0 - free space
            self.mask = self._calc_action_mask()
            if self.mask.sum() < 1:
                print("retry")
                raise Exception("mask sum < 1, no available action")

            # load reward matrix
            reward_matrix_path = os.path.join(ROOT_DIR, self.dataset_dir, 'reward_matrix',
                                              f'reward_{self.map_idx}.json')
            self.reward_matrix = np.array(json.load(open(reward_matrix_path)))
            # print(self.reward_matrix)
        self.n_episodes += 1

        if self.no_masking:
            observation = self.pixel_map.reshape(-1).astype(np.float32)
        else:
            observation = {
                "observations": self.pixel_map.reshape(-1).astype(np.float32),
                "action_mask": self.mask
            }

        info_dict = {
            "n_episodes": self.n_episodes,
            "n_trained_maps": self.n_trained_maps,
            "map_suffix": self.map_suffix,
            "map_index": self.map_idx,
        }
        # if self.n_episodes % 5 == 0:
        #     logger.info(info_dict)

        return observation, info_dict

    def step(
            self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        assert 0 <= action < self.action_space.n, f"action {action} must be in the action space [0, {self.action_space.n}]"
        # print(f"action: {action}")
        row, col = self.calc_upsampling_loc(action)

        # calculate reward
        r_c = self.calc_coverage(action)  # coverage reward
        # print(r_c)
        r = r_c

        self.steps += 1
        trunc = self.steps >= self.n_steps_truncate  # truncate if reach the step limit

        if self.no_masking:
            observation = self.pixel_map.reshape(-1).astype(np.float32)
        else:
            observation = {
                "observations": self.pixel_map.reshape(-1).astype(np.float32),
                "action_mask": self.mask
            }

        info_dict = {
            "steps": self.steps,
            "loc": (row, col),
            "reward": r,
            "r_c": r_c,
            "algo_name": self.test_algo,
        }

        return observation, r, False, trunc, info_dict

    def calc_coverage(self, action: int):
        """Calculate the coverage reward of a TX, given its location.

        Args:
            action: index in the action space.

        Returns:
            Number of covered pixels in the map.

        """
        y, x = divmod(action, self.action_space_size)
        return self.reward_matrix[y, x]

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
        # filter out white pixel - non-building
        action_pixels = np.where(self.pixel_map[idx][:, idx] != 1, 1, 0)
        return action_pixels.reshape(-1).astype(np.int8)


if __name__ == "__main__":
    reward_matrix_path = os.path.join(ROOT_DIR, 'resource/new_usc', 'reward_matrix',
                                      f'reward_{1}.json')
    reward_matrix = np.array(json.load(open(reward_matrix_path)))
    print(reward_matrix.shape)
    print(reward_matrix[5:10, 5:10])
