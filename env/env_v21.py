import itertools
import json
import logging
import math
import os
from typing import Any, SupportsFloat

import gymnasium as gym
import numpy as np
from gymnasium.core import ActType, ObsType
from gymnasium.spaces import Discrete, Dict, Box
from gymnasium.utils import seeding
from retrying import retry

from env.utils import calc_fspl
from env.utils_v1 import ROOT_DIR, load_map_normalized

RANDOM_SEED: int | None = None  # manually set random seed

# # set a logger
# logger = logging.getLogger("env_v21")
# logger.setLevel(logging.INFO)
# log_path = os.path.join(ROOT_DIR, "log/env_v21.log")
# handler = logging.FileHandler(log_path, encoding='utf-8', mode='a')
# formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
# handler.setFormatter(formatter)
# logger.addHandler(handler)


class BaseEnvironment(gym.Env):
    """MDP environment of FSPL-based multi-BS, where agents take actions asynchronously (implemented as single-agent).
    Reward function : coverage reward (not normalized)
    State : building map + coverage map

    """
    pixel_map: np.ndarray  # current building map
    version: str = "v21"
    steps: int
    map_idx: int  # index of the current building map
    n_deployed_bs: int  # number of deployed BS
    accumulated_reward: int  # accumulated reward in one episode
    coverage_map: np.ndarray  # coverage of previous deployed TXs

    def __init__(self, config: dict) -> None:
        """Initialize the base MDP environment.

        """
        # directory contains building maps, training and test power maps
        self.dataset_dir = config.get("dataset_dir", "resource/usc_old")
        evaluation_mode = config.get("evaluation", False)
        self.evaluation = evaluation_mode
        self.algo_name = config.get("algo_name", None)
        # training or test env
        self.map_suffix = "test" if evaluation_mode else "train"
        # indices of maps used for training or test
        self.map_indices: np.ndarray = np.arange(2, 2 + 32 * 100, 32) if evaluation_mode else np.arange(1, 1 + 16 * 1000,
                                                                                                        16)
        # the threshold of path loss value that we consider a pixel as 'covered' by TX
        radius_thr = 0.1 * 880  # meters
        # self.coverage_threshold: float = calc_fspl(radius_thr)
        self.coverage_threshold: float = 0.2 * 256  # radius_thr

        self.n_bs: int = config.get("n_bs", 2)
        self.n_maps: int = config.get("n_maps", 1)
        # number of continuous steps using one cropped map
        self.n_episodes_per_map: int = config.get("n_episodes_per_map", 10)
        self.n_steps_truncate: int = config.get("n_steps_truncate", 10)
        self.n_episodes: int = 0
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
            self.observation_space: Box = Box(low=0., high=1., shape=(map_size ** 2 * 2,), dtype=np.float32)
        else:
            self.observation_space: Dict = Dict(
                {
                    "observations": Box(low=0., high=1., shape=(map_size ** 2 * 2,), dtype=np.float32),
                    "action_mask": Box(low=0., high=1., shape=(self.action_space.n,), dtype=np.int8)
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
        self.n_deployed_bs = 0
        self.accumulated_reward = 0
        if self.n_episodes % self.n_episodes_per_map == 0:
            # switch the building map
            # map_idx uniquely determines a map in the dataset
            # choose the map in sequence in evaluation while randomly choose in training
            if self.evaluation:
                self.map_idx = int(self.map_indices[self.n_trained_maps % self.n_maps])
            else:
                self.map_idx = self.np_random.choice(self.map_indices)
            self.n_trained_maps += 1
            # update map and action mask
            map_path = os.path.join(ROOT_DIR, self.dataset_dir, 'map', str(self.map_idx) + '.png')
            self.pixel_map = load_map_normalized(map_path)
            # 1 - building, 0 - free space
            self.mask = self._calc_action_mask()
            if self.mask.sum() < 1:
                print("retry")
                raise Exception("mask sum < 1, no available action")
        self.n_episodes += 1

        # initial state - empty coverage map
        self.coverage_map = np.zeros_like(self.pixel_map)
        obs = np.concatenate([self.pixel_map, self.coverage_map], axis=None)

        if self.no_masking:
            observation = obs
        else:
            observation = {
                "observations": obs,
                "action_mask": self.mask
            }

        info_dict = {
            "n_episodes": self.n_episodes,
            "n_trained_maps": self.n_trained_maps,
            "map_suffix": self.map_suffix,
            "map_index": self.map_idx,
            "accumulated_reward": self.accumulated_reward,
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
        covered_new, _ = self.calc_coverage([(row, col)])
        r_prev = self.coverage_map.sum()
        self.coverage_map = np.maximum(self.coverage_map, covered_new)
        r = self.coverage_map.sum() - r_prev
        self.accumulated_reward += r

        self.steps += 1
        trunc = self.steps >= self.n_steps_truncate  # truncate if reach the step limit

        # update observation
        self.n_deployed_bs = (self.n_deployed_bs + 1) % self.n_bs
        if self.n_deployed_bs == 0:
            # reset coverage map
            self.coverage_map = np.zeros_like(self.pixel_map)
        obs = np.concatenate((self.pixel_map, self.coverage_map), axis=None)

        if self.no_masking:
            observation = obs
        else:
            observation = {
                "observations": obs,
                "action_mask": self.mask
            }

        info_dict = {
            "steps": self.steps,
            "loc": [row, col],
            "reward": r,
            "algo_name": self.algo_name,
            "n_deployed_bs": self.n_deployed_bs,
            "n_bs": self.n_bs,
            "accumulated_reward": self.accumulated_reward,
        }
        # logger.info(info_dict)
        # if self.algo_name and (self.steps % (np.ceil(self.n_steps_per_map / 4)) == 0 or trunc):
        #     logger.info(info_dict)

        return observation, r, False, trunc, info_dict

    def calc_optimal_locations(self) -> tuple[list, int]:
        """Calculate the optimal TX locations that maximize the coverage reward.

            Returns:
                (actions, optimal coverage reward).

        """
        locs_opt, coverage_opt = [(-1, -1)], -1

        data_dir = os.path.join(ROOT_DIR, self.dataset_dir, 'optimal')
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        filename = os.path.join(data_dir, f"optimal_{self.map_idx}.json")
        if not os.path.exists(filename):
            # print(f"No existing optimal reward {filename}")
            all_actions = itertools.combinations_with_replacement(range(self.action_space_size**2), self.n_bs)

            for actions in all_actions:
                tx_locs = []
                flag = False
                for action in actions:
                    row, col = self.calc_upsampling_loc(action)
                    if self.pixel_map[row, col] == 0.:
                        # skip illegal action
                        flag = True
                        break
                    tx_locs.append((row, col))
                if flag:
                    continue
                _, coverage = self.calc_coverage(tx_locs)
                if coverage > coverage_opt:
                    coverage_opt = coverage
                    locs_opt = tx_locs

            # save result to avoid repeatedly computation
            result = {"locs_opt": locs_opt, "coverage_opt": coverage_opt}
            json.dump(result, open(filename, 'w'))
        else:
            result = json.load(open(filename))
            locs_opt, coverage_opt = result["locs_opt"], result["coverage_opt"]

        return locs_opt, coverage_opt

    def calc_coverage(self, tx_locs: list) -> tuple[np.ndarray, int]:
        """Calculate the overall coverage reward of multiple TXs, given their locations.

        Args:
            tx_locs: tx location tuples

        Returns:
            (Coverage map, number of covered pixels in the map)

        """
        # calculate path loss in each pixel point
        values_pl = np.zeros_like(self.pixel_map, dtype=int)
        for tx_loc in tx_locs:
            row, col = tx_loc[0], tx_loc[1]
            top = max(0, math.ceil(row - self.coverage_threshold))
            bottom = min(self.map_size, math.floor(row + self.coverage_threshold) + 1)
            left = max(0, math.ceil(col - self.coverage_threshold))
            right = min(self.map_size, math.floor(col + self.coverage_threshold) + 1)
            for i in range(top, bottom):
                for j in range(left, right):
                    # we ignore non-ROI area (buildings pixel) when calculating the coverage
                    if self.pixel_map[i, j] == 1. or values_pl[i, j] == 1:
                        continue

                    dis = np.linalg.norm([row - i, col - j])

                    # convert path loss value to a 0-1 indicator of coverage
                    covered = 1 if dis < self.coverage_threshold else 0
                    values_pl[i, j] = covered

        return values_pl, int(values_pl.sum())

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
        return self.pixel_map[idx][:, idx].reshape(-1).astype(np.int8)
