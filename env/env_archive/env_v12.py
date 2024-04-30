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

from env.utils_v1 import ROOT_DIR, calc_coverages

RANDOM_SEED: int | None = None  # manually set random seed

# set a logger
logger = logging.getLogger("env_v12")
logger.setLevel(logging.INFO)
log_path = os.path.join(ROOT_DIR, "log/env_v12.log")
handler = logging.FileHandler(log_path, encoding='utf-8', mode='a')
formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


class BaseEnvironment(gym.Env):
    """MDP environment of autoBS, version 1.2.
    Reward function : coverage reward (normalized)

    """
    pixel_map: np.ndarray  # current building map
    power_map: np.ndarray  # flatten power map corresponding to pixel map
    loc_tx_opt: tuple  # the optimal location of TX
    coverage_opt: int  # the coverage reward corresponding to the optimal TX location
    coverage_rewards: dict[int, int]  # coverage matrices corresponding to each valid TX location
    power_maps: dict[int, np.ndarray]  # power maps corresponding to each TX location
    max_dis_opt: float  # the maximum distance from the optimal TX location to any pixel on the map
    version: str = "v12"
    steps: int

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
        self.map_indices: np.ndarray = np.arange(2, 2 + 32 * 50, 32) if evaluation_mode else np.arange(1, 1 + 32 * 100,
                                                                                                       32)
        # the threshold of luminance (larger power value, brighter pixel) that we consider a pixel as 'covered' by TX
        self.coverage_threshold: float = 220. / 255  # todo: may need to be changed for the new dataset

        self.n_maps: int = config.get("n_maps", 1)
        # number of continuous steps using one cropped map
        self.n_steps_per_map: int = config.get("n_steps_per_map", 10)
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
            self.observation_space: Box = Box(low=0., high=1., shape=(map_size ** 2,), dtype=np.float64)
        else:
            self.observation_space: Dict = Dict(
                {
                    "observations": Box(low=0., high=1., shape=(map_size ** 2,), dtype=np.float64),
                    "action_mask": MultiBinary(action_space_size ** 2)
                }
            )

        # fix a random seed
        self._np_random, seed = seeding.np_random(RANDOM_SEED)
        # logger.info("=============NEW ENV INITIALIZED=============")

    # retry if action mask contains no 1 (no valid action in the reduced action space)
    @retry(stop_max_attempt_number=3)
    def reset(
            self, *, seed: int | None = None, options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        super().reset(seed=seed, options=options)

        self.steps = 0
        # map_idx uniquely determines a map in the dataset
        # choose the map in sequence in training while randomly choose in evaluation
        if self.evaluation:
            map_idx = self.np_random.choice(self.map_indices)
        else:
            map_idx = self.map_indices[self.n_trained_maps % self.n_maps]
        self.n_trained_maps += 1

        # get map array, calculate the coverage-related information by the map index
        info_tuple = calc_coverages(self.dataset_dir, self.map_suffix, map_idx, self.coverage_threshold,
                                    self.upsampling_factor)
        self.pixel_map = info_tuple[0]
        self.loc_tx_opt = info_tuple[1]
        self.coverage_opt = info_tuple[2]
        self.coverage_rewards = info_tuple[3]
        self.power_maps = info_tuple[4]
        # info_filename = os.path.join(ROOT_DIR, self.dataset_dir, f"dataset_{map_idx}.json")
        # info_dict: dict = json.load(open(info_filename))
        # self.pixel_map = np.array(info_dict['map'], dtype=np.int8)
        # self.loc_tx_opt = info_dict['loc_opt']
        # self.coverage_opt = info_dict['coverage_opt']
        # self.coverage_rewards = info_dict['coverage_rewards']
        # self.power_maps = info_dict['pmaps']

        # calculate the maximum distance from a pixel to the optimal location
        row_opt, col_opt = self.loc_tx_opt[0], self.loc_tx_opt[1]
        row_dis = max(self.map_size - 1 - row_opt, row_opt)
        col_dis = max(self.map_size - 1 - col_opt, col_opt)
        self.max_dis_opt = np.sqrt(row_dis ** 2 + col_dis ** 2)

        # 1 - building, 0 - free space
        self.mask = self._calc_action_mask()
        if self.mask.sum() < 1:
            print("retry")
            raise Exception("mask sum < 1, no available action")
        # if self.evaluation:
        #     print(f"sum of mask in evaluation: {self.mask.sum()}")
        # else:
        #     print(f"sum of mask in training: {self.mask.sum()}")

        # choose a random initial action
        init_action = self.np_random.choice(np.where(self.mask == 1)[0])
        row, col = self.calc_upsampling_loc(init_action)
        self.power_map = self._get_power_map(row, col)

        if self.no_masking:
            observation = self.power_map
        else:
            observation = {
                "observations": self.power_map,
                "action_mask": self.mask
            }
        info_dict = {
            # "action_mask": self.mask,
            # "cropped_map_shape": self.pixel_map.shape,
            "map_index": map_idx,
            "loc_tx_opt": self.loc_tx_opt,
            "max_dis_opt": self.max_dis_opt,
            "coverage_opt": self.coverage_opt,
            "init_action": (row, col),
        }
        # if self.evaluation:
        #     logger.info(info_dict)
        return observation, info_dict

    def step(
            self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        assert 0 <= action < self.action_space.n, f"action {action} must be in the action space [0, {self.action_space.n}]"
        # print(f"action: {action}")
        row, col = self.calc_upsampling_loc(action)

        # calculate reward
        r_c = self.calc_coverage(row, col)  # coverage reward
        r_e = self.coverage_opt  # optimal coverage reward
        # coverage reward only
        # The reward value should be in the range [0,1]
        r = r_c / r_e if r_e > 0 else 0.

        term = True if r == 1. else False  # terminate if the location (action) is optimal
        self.steps += 1
        trunc = self.steps >= self.n_steps_per_map  # truncate if reach the step limit

        self.power_map = self._get_power_map(row, col)
        if self.no_masking:
            observation = self.power_map
        else:
            observation = {
                "observations": self.power_map,
                "action_mask": self.mask
            }

        info_dict = {
            "steps": self.steps,
            "loc": [row, col],
            "loc_opt": self.loc_tx_opt,
            "reward": r,
            "r_c": r_c,
            # "detailed_rewards": f"r_c = {r_c}, r_e = {r_e}",
        }
        # logger.info(info_dict)
        if self.test_algo and (self.steps % (np.ceil(self.n_steps_per_map / 4)) == 0 or term or trunc):
            logger.info(info_dict)

        # # plot the current and optimal TX locations
        # if self.algo_name and (term or trunc):
        #     save_map(
        #         f"./figures/test_map_archive/{datetime.now().strftime('%m%d_%H%M')}_{self.algo_name}_{self.n_trained_maps}.png",
        #         self.pixel_map,
        #         True,
        #         self.loc_tx_opt,
        #         mark_locs=[[row, col]],
        #     )

        return observation, r, term, trunc, info_dict

    def calc_coverage(self, row: int, col: int) -> int:
        """Calculate the coverage reward of a TX, given its location.

        Args:
            row: The row coordinate of the location.
            col: The column coordinate of the location.

        Returns:
            Number of covered pixels in the map.

        """
        tx_idx = row * self.map_size + col
        if tx_idx in self.coverage_rewards.keys():
            return self.coverage_rewards[tx_idx]
        else:
            # invalid TX location gets a huge penalty
            return -int(1e5)

    def _get_power_map(self, row: int, col: int) -> np.ndarray:
        """Get the power map given a TX location.

        Args:
            row: The row coordinate of the location.
            col: The column coordinate of the location.

        Returns:
            A flatten power map.

        """
        tx_idx = row * self.map_size + col
        if tx_idx in self.power_maps.keys():
            power_map = self.power_maps[tx_idx]
        else:
            # return building map for invalid TX location
            power_map = self.pixel_map.astype(np.float64)
        return power_map.reshape(-1)

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
        return self.pixel_map[idx][:, idx].reshape(-1)
