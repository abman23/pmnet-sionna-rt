import random
import time
from datetime import datetime
import logging
from typing import Any, SupportsFloat
import matplotlib.pyplot as plt

import gymnasium as gym
import numpy as np
from gymnasium.core import ActType, ObsType
from gymnasium.spaces import Discrete, MultiBinary, Dict
from gymnasium.utils import seeding

from utils import generate_map, crop_map, calc_coverage, find_opt_loc, calc_pl_threshold, load_map, save_map
from config import config_run_test, config_run_train, config_run3

RANDOM_SEED: int | None = None  # manually set random seed
# RATIO_BUILDINGS: float = .5  # the ratio of buildings for a randomly generated pixel map
# ORIGINAL_MAP_SIZE: int = 1000
# # a random generated usc map
# usc_map_rand: np.ndarray = generate_map(ORIGINAL_MAP_SIZE, ORIGINAL_MAP_SIZE,
#                                         ratio_buildings=RATIO_BUILDINGS, seed=RANDOM_SEED)

# set a logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.FileHandler(f"/Users/ylu/Documents/USC/WiDeS/BS_Deployment/log/{__name__}.log", encoding='utf-8', mode='a')
formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


class BaseEnvironment(gym.Env):
    """MDP environment for the BS deployment problem, aiming to maximize the total path loss in an area.

    """

    def __init__(self, config: dict) -> None:
        """Initialize the base MDP environment.

        """
        self.pixel_map: np.ndarray | None = None  # the pixel map used for interacting with the agent
        self.loc_tx_opt: np.ndarray | None = None  # the optimal location of TX
        self.coverage_map_opt: np.ndarray | None = None  # the coverage map corresponding to the optimal TX location

        # a pixel map where 1 = building, 0 = free space
        original_map_path = config.get("original_map_path", "resource/usc.png")
        original_map: np.ndarray = load_map(original_map_path)
        # print(f"original map with shape {original_map.shape}")
        # print(original_map)
        original_map_scale: float = config.get("original_map_scale", 880 / 256)
        ratio_coverage = config.get("ratio_coverage", .01)
        self.original_map: np.ndarray = original_map
        self.original_map_scale: float = original_map_scale
        # the threshold of path loss such that we consider a pixel as 'covered' by the TX
        self.thr_pl: float = calc_pl_threshold(original_map, original_map_scale, ratio_coverage)

        map_size = config.get("cropped_map_size", 256)
        # number of pixels in one row or column of the cropped map
        self.cropped_map_size: int = map_size
        self.n_maps: int = config.get("n_maps", 1)  # todo: not used now
        # number of continuous steps using one cropped map
        self.n_steps_per_map: int = config.get("n_steps_per_map", 10)
        # # since the cropped map will be scaled to the same size as the original map, its scale will be smaller
        # self.cropped_map_scale: float = original_map_scale * map_size / original_map.shape[0]

        # print(f"threshold: {self.thr_pl}, cropped_map_size: {map_size}, ratio_coverage: {ratio_coverage}")
        self.coefficient_dict = config.get("coefficient_dict", {})

        self.max_steps: int = config.get("max_steps", 100)  # todo: not used now
        self.steps: int = 0
        # count the number of used cropped resource
        self.n_trained_maps: int = 0

        self.action_space: Discrete = Discrete(map_size * map_size)
        self.observation_space: Dict = Dict(
            {
                "observations": MultiBinary([map_size, map_size]),
                "action_mask": MultiBinary(map_size * map_size)
            }
        )
        # action mask has the same shape as the action
        self.mask: np.ndarray = np.empty(map_size * map_size, dtype=np.int8)
        self.no_masking: bool = config.get("no_masking", True)

        # fix a random seed
        self._np_random, seed = seeding.np_random(RANDOM_SEED)

        # crop some resource from the original one and calculate the optimal TX location and coverage for each map
        self.cropped_maps = crop_map(self.original_map, self.cropped_map_size,
                                     self.n_maps, self.np_random)
        self.locs_opt: list[np.ndarray] = []
        self.coverages_opt: list[np.ndarray] = []
        s = time.time()
        for cropped_map in self.cropped_maps:
            loc_opt, coverage_opt = find_opt_loc(cropped_map, self.original_map_scale, self.thr_pl)
            self.locs_opt.append(loc_opt)
            self.coverages_opt.append(coverage_opt)
        e = time.time()
        logger.info("=============NEW ENV INITIALIZED=============")
        logger.info(f"exhaustive search time for {self.n_maps} cropped resource: {e - s} s")

    def reset(
            self, *, seed: int | None = None, options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        super().reset(seed=seed, options=options)

        self.steps = 0
        # 'randomly' (controlled by the seed) crop a smaller map from the original one
        # todo: scale cropped map to original one's size (64x64 -> 256x256)
        map_idx = self.np_random.choice(self.n_maps)
        self.pixel_map = self.cropped_maps[map_idx]
        self.loc_tx_opt = self.locs_opt[map_idx]
        self.coverage_map_opt = self.coverages_opt[self.n_trained_maps % self.n_maps]
        # self.pixel_map = crop_map(self.original_map, self.cropped_map_size, n=1, rng=self.np_random)[0]
        self.n_trained_maps += 1
        # 1 - building, 0 - free space
        self.mask = self.pixel_map.reshape(-1)
        # # calculate the optimal TX location
        # s = time.time()
        # self.loc_tx_opt, self.coverage_map_opt = find_opt_loc(self.pixel_map, self.original_map_scale, self.thr_pl)
        # e = time.time()

        obs_dict = {
            "observations": self.pixel_map,
            "action_mask": self.mask
        }
        info_dict = {
            # "action_mask": self.mask,
            # "cropped_map_shape": self.pixel_map.shape,
            "map_index": map_idx,
            "loc_tx_opt": self.loc_tx_opt,
            # "time_exhaustive_search": e - s
        }
        logger.info(info_dict)
        # print(info_dict)
        # print(f"optimal coverage reward: {np.sum(self.coverage_map_opt)}")
        # print(f"RoI area: {np.sum(self.pixel_map == 0)}")
        # save_map(f"log/cropped_map_{datetime.now().strftime('%m%d_%H%M')}.png",
        #          self.pixel_map, mark_loc=self.loc_tx_opt)
        # np.savetxt('log/cropped_map.txt', self.pixel_map, delimiter=',', fmt='%d')
        # save_map(f"log/coverage_{datetime.now().strftime('%m%d_%H%M')}.png",
        #          self.coverage_map_opt, mark_loc=self.loc_tx_opt)
        # np.savetxt('log/coverage_map_opt.txt', self.coverage_map_opt, delimiter=',', fmt='%d')
        return obs_dict, info_dict

    def step(
            self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        assert 0 <= action < self.action_space.n
        # print(f"action: {action}")
        row, col = divmod(action, self.cropped_map_size)
        # calculate reward
        coverage_matrix = self._calc_coverage(row, col)
        r_c = np.sum(coverage_matrix)  # coverage reward
        r_e = np.sum(self.coverage_map_opt)  # optimal coverage reward
        p_d = -np.linalg.norm(np.array([row, col]) - self.loc_tx_opt)  # distance penalty
        p_b = -100 if (self.pixel_map[row, col] != 1 and self.no_masking) else 0  # RoI penalty
        a = self.coefficient_dict.get("r_c", .1)
        b = self.coefficient_dict.get("p_d", 1.)
        c = self.coefficient_dict.get("p_b", 1.)
        r = a * (r_c - r_e) + b * p_d + c * p_b

        term = True if r == 0 else False  # terminate if the location (action) is optimal
        self.steps += 1
        trunc = self.steps >= self.n_steps_per_map  # truncate if reach the step limit
        # # change a different map and count the number of used resource
        # if self.steps % self.n_steps_per_map == 0:
        #     self.n_trained_maps += 1
        #     self.pixel_map = self.cropped_maps[self.n_trained_maps % self.n_maps]
        obs_dict = {
            "observations": self.pixel_map,
            "action_mask": self.mask
        }

        info_dict = {
            "steps": self.steps,
            "detailed_rewards": f"r_c = {r_c}, r_e = {r_e}, p_d = {p_d}, p_b = {p_b}",
        }
        # if self.steps % 100 == 0 or term or trunc:
        #     print(info_dict)
        #     print(f"action: {[row, col]}, loc_opt: {self.loc_tx_opt}")

        return obs_dict, r, term, trunc, info_dict

    def _calc_coverage(self, x_tx: int, y_tx: int) -> np.ndarray:
        """Calculate the coverage of a TX, given its location and a threshold.

        Args:
            x_tx: The X coordinate of the TX.
            y_tx: The Y coordinate of the TX.

        Returns:
            A coverage map where 0 = uncovered, 1 = covered.

        """
        return calc_coverage(x_tx, y_tx, self.pixel_map, map_scale=self.original_map_scale, threshold=self.thr_pl)


if __name__ == "__main__":
    env = BaseEnvironment(config=config_run_test["env"])
    # env.reset()

    fig, ax = plt.subplots()

    for i in range(3):
        env.reset()
        terminated, truncated = False, False
        r_ep = []
        n = 0
        while not terminated and not truncated:
            obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
            n += 1
            r_ep.append(reward)

        ax.plot(list(range(n)), r_ep, label=f"episode # {i}")

    ax.set(xlabel="step", ylabel="reward", title="Random Sample")
    ax.legend()
    ax.grid()
    fig.savefig(f"./figures/random_{datetime.now().strftime('%m%d_%H%M')}.png")
