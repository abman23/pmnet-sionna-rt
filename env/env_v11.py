import logging
import os.path
import time
from datetime import datetime
from typing import Any, SupportsFloat

import gymnasium as gym
import numpy as np
import yaml
from gymnasium.core import ActType, ObsType
from gymnasium.spaces import Discrete, MultiBinary, Dict, Box
from gymnasium.utils import seeding
from matplotlib import pyplot as plt
from retrying import retry

from env.utils_v1 import ROOT_DIR, calc_coverages

RANDOM_SEED: int | None = None  # manually set random seed

# set a logger
logger = logging.getLogger("env_v11")
logger.setLevel(logging.INFO)
log_path = os.path.join(ROOT_DIR, "log/env_v11.log")
handler = logging.FileHandler(log_path, encoding='utf-8', mode='a')
formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


class BaseEnvironment(gym.Env):
    """MDP environment of autoBS, version 1.1.
    Reward function : coverage reward + distance penalty (normalized)

    """
    pixel_map: np.ndarray  # current building map
    power_map: np.ndarray  # flatten power map corresponding to pixel map
    loc_tx_opt: tuple  # the optimal location of TX
    coverage_opt: int  # the coverage reward corresponding to the optimal TX location
    coverage_matrices: dict[int, np.ndarray]  # coverage matrices corresponding to each valid TX location
    power_maps: dict[int, np.ndarray]  # power maps corresponding to each TX location
    max_dis_opt: float  # the maximum distance from the optimal TX location to any pixel on the map
    version: str = "v11"
    steps: int

    def __init__(self, config: dict) -> None:
        """Initialize the base MDP environment.

        """
        # directory contains building maps, training and test power maps
        self.dataset_dir = config.get("dataset_dir", "resource/usc_old")
        evaluation_mode = config.get("evaluation", False)
        self.evaluation = evaluation_mode
        self.test_algo = config.get("test_algo", None)
        # training or test env
        self.map_suffix = "test" if evaluation_mode else "train"
        # indices of maps used for training or test
        self.map_indices: np.ndarray = np.arange(2, 2 + 32 * 100, 32) if evaluation_mode else np.arange(1, 1 + 32 * 100,
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

        self.action_space: Discrete = Discrete(action_space_size ** 2)
        self.observation_space: Dict = Dict(
            {
                "observations": Box(low=-np.inf, high=np.inf, shape=(map_size ** 2,)),
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
        # choose the map in sequence in evaluation
        if self.evaluation:
            map_idx = self.map_indices[self.n_trained_maps % self.n_maps]
        else:
            map_idx = self.np_random.choice(self.map_indices)
        self.n_trained_maps += 1

        # get map array, calculate the coverage-related information by the map index
        info_tuple = calc_coverages(self.dataset_dir, self.map_suffix, map_idx, self.coverage_threshold,
                                    self.upsampling_factor)
        self.pixel_map = info_tuple[0]
        self.loc_tx_opt = info_tuple[1]
        self.coverage_opt = info_tuple[2]
        self.coverage_matrices = info_tuple[3]
        self.power_maps = info_tuple[4]

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
        if self.evaluation:
            print(f"sum of mask in evaluation: {self.mask.sum()}")
        else:
            print(f"sum of mask in training: {self.mask.sum()}")

        # choose a random initial action
        init_action = self.np_random.choice(np.where(self.mask == 1)[0])
        row, col = self.calc_upsampling_loc(init_action)
        self.power_map = self._get_power_map(row, col)

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
            "init_action": (row, col),
        }
        # if self.evaluation:
        #     logger.info(info_dict)
        # print("restart")
        # print(info_dict)
        return observation, info_dict

    def step(
            self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        assert 0 <= action < self.action_space.n, f"action {action} must be in the action space [0, {self.action_space.n}]"
        # print(f"action: {action}")
        row, col = self.calc_upsampling_loc(action)
        try:
            # calculate reward
            coverage_matrix = self.calc_coverage(row, col)
        except Exception as e:
            # print(f"action mask value: {self.mask[action]}")
            # print(f"loc_tx_opt: {self.loc_tx_opt}, action: {action}, (row, col): {(row, col)}")
            raise e
        r_c = np.sum(coverage_matrix)  # coverage reward
        r_e = self.coverage_opt  # optimal coverage reward

        p_d = -np.linalg.norm(np.array([row, col]) - self.loc_tx_opt)  # distance penalty
        k = self.max_dis_opt
        # calculate reward
        a = self.coefficient_dict.get("r_c", 1.)
        b = self.coefficient_dict.get("p_d", 1.)
        # combine coverage reward and distance penalty together to form the final reward
        # The reward value should be in the range [-1,1]
        r = a * r_c/r_e + b * p_d/k if r_e > 0 else p_d/k

        term = True if r == 1. else False  # terminate if the location (action) is optimal
        self.steps += 1
        trunc = self.steps >= self.n_steps_per_map  # truncate if reach the step limit
        # # change a different map and count the number of used resource
        # if self.steps % self.n_steps_per_map == 0:
        #     self.n_trained_maps += 1
        #     self.pixel_map = self.cropped_maps[self.n_trained_maps % self.n_maps]

        observation = {
            "observations": self.power_map,
            "action_mask": self.mask
        }

        info_dict = {
            "steps": self.steps,
            "loc": [row, col],
            "loc_opt": self.loc_tx_opt,
            "reward": r,
            "detailed_rewards": f"r_c = {r_c}, r_e = {r_e}, p_d = {p_d}, k = {k}",
        }
        # logger.info(info_dict)
        if self.test_algo and (self.steps % (np.ceil(self.n_steps_per_map / 4)) == 0 or term or trunc):
            logger.info(info_dict)

        # # plot the current and optimal TX locations
        # if self.test_algo and (term or trunc):
        #     save_map(
        #         f"./figures/test_maps/{datetime.now().strftime('%m%d_%H%M')}_{self.test_algo}_{self.n_trained_maps}.png",
        #         self.pixel_map,
        #         True,
        #         self.loc_tx_opt,
        #         mark_locs=[[row, col]],
        #     )

        return observation, r, term, trunc, info_dict

    def calc_coverage(self, row: int, col: int) -> np.ndarray:
        """Calculate the coverage of a TX, given its location.

        Args:
            row: The row coordinate of the location.
            col: The column coordinate of the location.

        Returns:
            A coverage map where 0 = uncovered, 1 = covered.

        """
        tx_idx = row * self.map_size + col
        return self.coverage_matrices[tx_idx]

    def _get_power_map(self, row: int, col: int) -> np.ndarray:
        """Get the power map given a TX location.

        Args:
            row: The row coordinate of the location.
            col: The column coordinate of the location.

        Returns:
            A flatten power map.

        """
        tx_idx = row * self.map_size + col
        power_map = self.power_maps[tx_idx]
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


if __name__ == "__main__":
    start = time.time()

    config = yaml.safe_load(open('../config/ppo_v1.yaml', 'r'))
    env_config = config['env']
    env = BaseEnvironment(config=env_config)
    # env.reset()

    fig, ax = plt.subplots()

    for i in range(3):
        env.reset()
        terminated, truncated = False, False
        r_ep = []
        n = 0
        while not terminated and not truncated:
            action = env.np_random.choice(np.where(env.mask == 1)[0])
            obs, reward, terminated, truncated, info = env.step(action)
            print(info)
            n += 1
            r_ep.append(reward)

        ax.plot(list(range(n)), r_ep, label=f"episode # {i}")

    ax.set(xlabel="step", ylabel="reward", title="Random Sample")
    ax.legend()
    ax.grid()
    # fig.savefig(f"./figures/random_{datetime.now().strftime('%m%d_%H%M')}.png")
    plt.show()

    end = time.time()
    print(f"total runtime: {end - start}s")
