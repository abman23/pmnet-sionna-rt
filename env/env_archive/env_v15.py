import logging
import os
from typing import Any, SupportsFloat

import gymnasium as gym
import numpy as np
from gymnasium.core import ActType, ObsType
from gymnasium.spaces import Discrete, MultiBinary, Dict, Box
from gymnasium.utils import seeding
from retrying import retry

from env.utils_v1 import ROOT_DIR, calc_coverages, load_map_normalized, calc_optimal_locations
from dataset_builder.generate_pmap import generate_pmaps

RANDOM_SEED: int | None = None  # manually set random seed

# set a logger
logger = logging.getLogger("env_v15")
logger.setLevel(logging.INFO)
log_path = os.path.join(ROOT_DIR, "log/env_v15.log")
handler = logging.FileHandler(log_path, encoding='utf-8', mode='a')
formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


class BaseEnvironment(gym.Env):
    """MDP environment of autoBS, version 1.5.
    Reward function : coverage reward (not normalized)
    State: building map + power map

    """
    pixel_map: np.ndarray  # current building map
    version: str = "v15"
    steps: int
    map_idx: int  # index of the current building map
    power_maps: dict[int, np.ndarray]
    obs: np.ndarray  # observation, flatten building map concatenated by flatten power maps with random TX locations

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
        self.map_indices: np.ndarray = np.arange(2, 2 + 32 * 100, 32) if evaluation_mode else np.arange(1, 1 + 32 * 500,
                                                                                                        32)
        # the threshold of luminance (larger power value, brighter pixel) that we consider a pixel as 'covered' by TX
        self.coverage_threshold: float = 220. / 255
        # number of power maps stacked to the observation
        self.n_power_maps_obs: int = 3

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
            self.observation_space: Box = Box(low=0., high=1.,
                                              shape=(map_size ** 2 * (self.n_power_maps_obs + 1),), dtype=np.float32)
        else:
            self.observation_space: Dict = Dict(
                {
                    "observations": Box(low=0., high=1.,
                                        shape=(map_size ** 2 * (self.n_power_maps_obs + 1),), dtype=np.float32),
                    "action_mask": MultiBinary(action_space_size ** 2)
                }
            )

        # fix a random seed
        self._np_random, seed = seeding.np_random(RANDOM_SEED)
        if not evaluation_mode:
            logger.info(f"=============NEW ENV {self.version.upper()} INITIALIZED=============")

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
            # choose the map in sequence in training while randomly choose in evaluation
            # if self.evaluation:
            #     self.map_idx = self.np_random.choice(self.map_indices)
            # else:
            #     self.map_idx = int(self.map_indices[self.n_trained_maps % self.n_maps])
            self.map_idx = int(self.map_indices[self.n_trained_maps % self.n_maps])
            self.n_trained_maps += 1
            # update map and action mask
            map_path = os.path.join(ROOT_DIR, self.dataset_dir, 'map', str(self.map_idx) + '.png')
            self.pixel_map = load_map_normalized(map_path)
            # 1 - building, 0 - free space
            self.mask = self._calc_action_mask()
            if self.mask.sum() < 1:
                print("retry")
                raise Exception("mask sum < 1, no available action")
            # generate power maps
            # self.power_maps = generate_pmaps(self.map_idx, self.upsampling_factor, True, save=False)
            self.load_pmaps()
        self.n_episodes += 1

        # choose a random initial action
        init_action = self.np_random.choice(np.where(self.mask == 1)[0])
        row, col = self.calc_upsampling_loc(init_action)
        # choose some random initial TX locations
        init_tx_locs = self.np_random.choice(list(self.power_maps.keys()), self.n_power_maps_obs)
        init_power_maps = [self.power_maps[tx_idx].reshape(-1).astype(np.float32) for tx_idx in init_tx_locs]
        obs = [self.pixel_map.reshape(-1).astype(np.float32)]
        obs.extend(init_power_maps)
        self.obs = np.concatenate(obs)

        if self.no_masking:
            observation = self.obs
        else:
            observation = {
                "observations": self.obs,
                "action_mask": self.mask
            }

        info_dict = {
            "n_episodes": self.n_episodes,
            "n_trained_maps": self.n_trained_maps,
            "map_suffix": self.map_suffix,
            "map_index": self.map_idx,
            "init_action": (row, col),
        }
        if self.n_episodes % 5 == 0:
            logger.info(info_dict)

        return observation, info_dict

    def step(
            self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        assert 0 <= action < self.action_space.n, f"action {action} must be in the action space [0, {self.action_space.n}]"
        # print(f"action: {action}")
        row, col = self.calc_upsampling_loc(action)

        # calculate reward
        r_c = self.calc_coverage(row, col)  # coverage reward
        # coverage reward only
        r = r_c

        self.steps += 1
        trunc = self.steps >= self.n_steps_truncate  # truncate if reach the step limit

        if self.no_masking:
            observation = self.obs
        else:
            observation = {
                "observations": self.obs,
                "action_mask": self.mask
            }

        info_dict = {
            "steps": self.steps,
            "loc": [row, col],
            # "loc_opt": self.loc_tx_opt,
            "reward": r,
            "r_c": r_c,
            "algo_name": self.test_algo,
            # "detailed_rewards": f"r_c = {r_c}, r_e = {r_e}",
        }
        # logger.info(info_dict)
        # if self.algo_name and (self.steps % (np.ceil(self.n_steps_per_map / 4)) == 0 or trunc):
        #     logger.info(info_dict)

        return observation, r, False, trunc, info_dict

    def calc_coverage(self, row: int, col: int) -> int:
        """Calculate the coverage reward of a TX, given its location.

        Args:
            row: The row coordinate of the location.
            col: The column coordinate of the location.

        Returns:
            Number of covered pixels in the map.

        """
        tx_idx = row * self.map_size + col
        try:
            pmap_arr = self.power_maps[tx_idx]
            coverage_matrix = np.where(pmap_arr >= self.coverage_threshold, 1, 0)
            coverage = int(coverage_matrix.sum())
            return coverage
        except Exception:
            # invalid TX location gets a huge penalty
            return -int(1e5)

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

    def load_pmaps(self):
        """Load power maps corresponding to the current building map from images.

        """
        self.power_maps = {}
        for action in range(self.action_space_size ** 2):
            row, col = self.calc_upsampling_loc(action)
            if self.pixel_map[row, col] == 1.:
                tx_idx = row * self.map_size + col
                pmap_path = os.path.join(ROOT_DIR, self.dataset_dir, 'pmap_' + self.map_suffix,
                                         'pmap_' + str(self.map_idx) + '_' + str(tx_idx) + '.png')
                pmap_arr = load_map_normalized(pmap_path)
                self.power_maps[tx_idx] = pmap_arr
