import json
import time
from datetime import datetime
import logging
from typing import Any, SupportsFloat
import matplotlib.pyplot as plt

import gymnasium as gym
import numpy as np
import yaml
from gymnasium.core import ActType, ObsType
from gymnasium.spaces import Discrete, MultiBinary, Dict, Box
from gymnasium.utils import seeding
from retrying import retry

from env.utils import crop_map, calc_coverage, find_opt_loc, calc_pl_threshold, load_map, save_map

RANDOM_SEED: int | None = None  # manually set random seed

# set a logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.FileHandler(f"./log/{__name__}.log", encoding='utf-8', mode='a')
formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


class BaseEnvironment(gym.Env):
    """MDP environment for the BS deployment problem, aiming to maximize the total path loss in an area.

    """
    pixel_map: np.ndarray  # current pixelated cropped map
    power_map: np.ndarray  # flatten power map corresponding to pixel map
    loc_tx_opt: tuple  # the optimal location of TX
    coverage_map_opt: np.ndarray  # the coverage map corresponding to the optimal TX location
    max_dis_opt: float  # the maximum distance from the optimal TX location to any pixel on the map
    version: str = "v04"

    def __init__(self, config: dict) -> None:
        """Initialize the base MDP environment.

        """
        # a pixel map where 1 = building, 0 = free space
        original_map_path = config.get("original_map_path", "resource/usc.png")
        original_map: np.ndarray = load_map(original_map_path)
        original_map_scale: float = config.get("original_map_scale", 880 / 256)
        ratio_coverage = config.get("ratio_coverage", .01)
        self.original_map: np.ndarray = original_map
        self.original_map_scale: float = original_map_scale

        preset_config: dict = {}
        # if not None, use a given set of cropped maps and the corresponding configurations
        preset_map_path: str | None = config.get("preset_map_path", None)
        if preset_map_path:
            with open(preset_map_path, "r", encoding="utf-8") as preset_file:
                preset_config = json.load(preset_file)

        # the threshold of path loss such that we consider a pixel as 'covered' by the TX
        if preset_map_path:
            self.thr_pl: float = preset_config.get("thr_pl")
        else:
            self.thr_pl: float = calc_pl_threshold(original_map, original_map_scale, ratio_coverage)

        map_size = config.get("cropped_map_size", 64)
        # number of pixels in one row or column of the cropped map
        self.cropped_map_size: int = map_size
        self.n_maps: int = config.get("n_maps", 1)
        # number of continuous steps using one cropped map
        self.n_steps_per_map: int = config.get("n_steps_per_map", 10)
        # # since the cropped map will be scaled to the same size as the original map, its scale will be smaller
        # self.cropped_map_scale: float = original_map_scale * map_size / original_map.shape[0]

        # print(f"threshold: {self.thr_pl}, cropped_map_size: {map_size}, ratio_coverage: {ratio_coverage}")
        self.coefficient_dict = config.get("coefficient_dict", {})

        self.max_steps: int = config.get("max_steps", 100)
        self.steps: int = 0
        # count the number of used cropped resource
        self.n_trained_maps: int = 0

        self.evaluation = config.get("evaluation", False)
        self.test_algo = config.get("test_algo", None)

        action_space_size = config.get("action_space_size", 32)
        assert map_size % action_space_size == 0, f"map_size {map_size} must be divisible by action_space_size {action_space_size}"
        self.upsampling_factor = map_size // action_space_size
        self.action_space_size: int = action_space_size
        # action mask has the same shape as the action
        self.mask: np.ndarray = np.empty(action_space_size**2, dtype=np.int8)
        self.no_masking: bool = config.get("no_masking", True)
        self.action_space: Discrete = Discrete(action_space_size**2)
        if self.no_masking:
            self.observation_space = Box(low=-np.inf, high=np.inf, shape=(map_size**2,))
        else:
            self.observation_space: Dict = Dict(
                {
                    "observations": Box(low=-np.inf, high=np.inf, shape=(map_size**2,)),
                    "action_mask": MultiBinary(action_space_size**2)
                }
            )

        # fix a random seed
        self._np_random, seed = seeding.np_random(RANDOM_SEED)

        # crop some resource from the original one and calculate the optimal TX location and coverage for each map
        if preset_map_path:
            self.cropped_maps = np.array(preset_config.get("cropped_maps"), dtype=np.int8)
            self.locs_opt: list[tuple] = preset_config.get("locs_opt")
            self.coverages_opt = np.array(preset_config.get("coverages_opt"), dtype=np.int8)
            logger.info(f"loaded {self.n_maps} maps from {preset_map_path}")
        else:
            self.cropped_maps = crop_map(self.original_map, self.cropped_map_size,
                                         self.n_maps, self.np_random)
            self.locs_opt: list[tuple] = []
            self.coverages_opt: list[np.ndarray] = []
            s = time.time()
            for cropped_map in self.cropped_maps:
                loc_opt, coverage_opt = find_opt_loc(cropped_map, self.original_map_scale, self.thr_pl)
                self.locs_opt.append(loc_opt)
                self.coverages_opt.append(coverage_opt)
            e = time.time()
            logger.info(f"exhaustive search time for {self.n_maps} cropped maps: {e - s} s")
        logger.info("=============NEW ENV INITIALIZED=============")

    # retry if action mask contains no 1 (no valid action in the reduced action space)
    @retry(stop_max_attempt_number=3)
    def reset(
            self, *, seed: int | None = None, options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        super().reset(seed=seed, options=options)

        self.steps = 0
        # choose the map in sequence in evaluation
        if self.evaluation:
            map_idx = self.n_trained_maps % self.n_maps
        else:
            map_idx = self.np_random.choice(self.n_maps)
        self.pixel_map = self.cropped_maps[map_idx]
        self.loc_tx_opt = self.locs_opt[map_idx]
        self.coverage_map_opt = self.coverages_opt[map_idx]
        self.n_trained_maps += 1

        # calculate the maximum distance from a pixel to the optimal location
        row_opt, col_opt = self.loc_tx_opt[0], self.loc_tx_opt[1]
        row_dis = max(self.cropped_map_size - row_opt, row_opt)
        col_dis = max(self.cropped_map_size - col_opt, col_opt)
        self.max_dis_opt = np.sqrt(row_dis**2 + col_dis**2)

        # 1 - building, 0 - free space
        self.mask = self._calc_action_mask()

        # choose a random initial action
        init_action = self.np_random.choice(np.where(self.mask == 1)[0])
        row, col = self.calc_upsampling_loc(init_action)
        self.power_map = self._calc_power_map(row, col)

        if self.no_masking:
            observation = self.power_map
        else:
            observation = {
                "observations": self.power_map,
                "action_mask": self.mask
            }
        info_dict = {
            # "action_mask": self.mask,
            "cropped_map_shape": self.pixel_map.shape,
            "map_index": map_idx,
            "loc_tx_opt": self.loc_tx_opt,
            "max_dis_opt": self.max_dis_opt,
            "init_action": init_action,
            "row, col": (row, col),
        }
        if self.evaluation:
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
        return observation, info_dict

    def step(
            self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        assert 0 <= action < self.action_space.n, f"action {action} must be in the action space [0, {self.action_space.n}]"
        # print(f"action: {action}")
        row, col = self.calc_upsampling_loc(action)

        # calculate reward
        coverage_matrix = self.calc_coverage(row, col)
        r_c = np.sum(coverage_matrix)  # coverage reward
        r_e = np.sum(self.coverage_map_opt)  # optimal coverage reward

        p_d = -np.linalg.norm(np.array([row, col]) - self.loc_tx_opt)  # distance penalty
        # calculate reward
        a = self.coefficient_dict.get("r_c", 1.)
        b = self.coefficient_dict.get("p_d", 1.)
        # normalization
        r = a * (r_c / r_e) + b * (p_d / self.max_dis_opt)

        term = True if r == 0 else False  # terminate if the location (action) is optimal
        self.steps += 1
        trunc = self.steps >= self.n_steps_per_map  # truncate if reach the step limit
        # # change a different map and count the number of used resource
        # if self.steps % self.n_steps_per_map == 0:
        #     self.n_trained_maps += 1
        #     self.pixel_map = self.cropped_maps[self.n_trained_maps % self.n_maps]
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
            "detailed_rewards": f"r_c = {r_c}, r_e = {r_e}, p_d = {p_d}, dis_max = {self.max_dis_opt}",
            "coverage_matrix": coverage_matrix.tolist
        }
        # logger.info(info_dict)
        if self.test_algo and (self.steps % (np.ceil(self.n_steps_per_map/4)) == 0 or term or trunc):
            logger.info(info_dict)

        # plot the current and optimal TX locations
        if self.test_algo and (term or trunc):
            save_map(
                f"./figures/test_maps/{self.version}_{datetime.now().strftime('%m%d_%H%M')}_{self.test_algo}_{self.n_trained_maps}.png",
                self.pixel_map,
                True,
                self.loc_tx_opt,
                mark_locs=[[row, col]],
            )

        return observation, r, term, trunc, info_dict

    def calc_coverage(self, row: int, col: int) -> np.ndarray:
        """Calculate the coverage of a TX, given its location and a threshold.

        Args:
            row: The row coordinate of the location.
            col: The column coordinate of the location.

        Returns:
            A coverage map where 0 = uncovered, 1 = covered.

        """
        return calc_coverage(row, col, self.pixel_map, map_scale=self.original_map_scale, threshold=self.thr_pl)

    def _calc_power_map(self, row: int, col: int) -> np.ndarray:
        """Calculate the power map given a TX location.

        Args:
            row: The row coordinate of the location.
            col: The column coordinate of the location.

        Returns:
            A flatten power map.

        """
        power_map = calc_coverage(row, col, self.pixel_map, map_scale=self.original_map_scale, threshold=None)
        flatten_power_map = power_map.reshape(-1)
        return flatten_power_map
        # # min max normalization
        # return (flatten_power_map - flatten_power_map.min()) / (flatten_power_map.max() - flatten_power_map.min())

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
        idx = np.arange((self.upsampling_factor-1)//2, self.cropped_map_size, self.upsampling_factor)
        return self.pixel_map[idx][:, idx].reshape(-1)


if __name__ == "__main__":
    start = time.time()

    config = yaml.safe_load(open('config/dqn_test.yaml', 'r'))
    env_config = config['env']
    env = BaseEnvironment(config=env_config)
    # env.reset()

    fig, ax = plt.subplots()

    for i in range(3):
        obs_dict, _ = env.reset()
        mask = obs_dict["action_mask"]
        terminated, truncated = False, False
        r_ep = []
        n = 0
        while not terminated and not truncated:
            action = env.action_space.sample(mask=mask)
            print(f"step: {n}, action: {action}")
            obs, reward, terminated, truncated, info = env.step(action)
            print(info)
            n += 1
            r_ep.append(reward)

        ax.plot(list(range(n)), r_ep, label=f"episode # {i}")

    ax.set(xlabel="step", ylabel="reward", title="Random Sample")
    ax.legend()
    ax.grid()
    # fig.savefig(f"./figures/random_{datetime.now().strftime('%m%d_%H%M')}.png")

    end = time.time()
    print(f"total runtime: {end - start}s")

    fig.show()
