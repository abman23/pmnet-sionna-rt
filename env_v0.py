import time
from datetime import datetime
from typing import Any, SupportsFloat, Tuple
import matplotlib.pyplot as plt

import gymnasium as gym
import numpy as np
from gymnasium.core import ActType, ObsType
from gymnasium.spaces import Discrete, MultiBinary, Dict
from gymnasium.utils import seeding

from utils import generate_map, crop_map, calc_path_loss, find_opt_loc, calc_pl_threshold

RATIO_BUILDINGS: float = .5  # the ratio of buildings for a randomly generated pixel map
RANDOM_SEED: int | None = 2024  # manually set random seed
ORIGINAL_MAP_SIZE: int = 1000
# a random generated usc map (todo: replace it by the real map)
usc_map_rand: np.ndarray = generate_map(ORIGINAL_MAP_SIZE, ORIGINAL_MAP_SIZE,
                                        ratio_buildings=RATIO_BUILDINGS, seed=RANDOM_SEED)


class BaseEnvironment(gym.Env):
    """MDP environment for the BS deployment problem, aiming to maximize the total path loss in an area.

    """

    def __init__(self, config: dict) -> None:
        """Initialize the base MDP environment.

        """
        self.cropped_maps: list[np.ndarray] | None = None  # the list of cropped maps
        self.pixel_map: np.ndarray | None = None  # the pixel map used for interacting with the agent
        self.loc_tx_opt: np.ndarray | None = None  # the optimal location of TX
        self.coverage_map_opt: np.ndarray | None = None  # the coverage map corresponding to the optimal TX location

        map_size = config.get("map_size", 64)
        self.map_size: int = map_size
        self.n_maps: int = config.get("n_cropped", 100)
        # a pixel map where 1 = building, 0 = free space
        original_map = config.get("original_map", usc_map_rand)
        print(f"random original map with size {original_map.shape[0]} x {original_map.shape[1]}:")
        print(original_map)
        ratio_coverage = config.get("ratio_coverage", .01)
        self.original_map: np.ndarray = original_map
        # the threshold of path loss such that we consider a pixel as 'covered' by the TX
        self.thr_pl: float = calc_pl_threshold(original_map, ratio_coverage=ratio_coverage)
        print(f"threshold: {self.thr_pl}, map_size: {map_size}, ratio_coverage: {ratio_coverage}")
        self.coefficient_dict = config.get("coefficient_dict", {})

        self.max_steps: int = config.get("max_steps", 100)
        self.steps: int = 0
        self.n_trained_maps: int = 0

        self.action_space: Discrete = Discrete(map_size * map_size)
        self.observation_space: Dict = Dict(
            {
                "observations": MultiBinary([map_size, map_size]),
                "action_mask": MultiBinary(map_size * map_size)
            }
        )

        self.mask: np.ndarray = np.empty(map_size * map_size, dtype=np.int8)
        self.no_masking: bool = config.get("no_masking", True)

        # fix a random seed
        self._np_random, seed = seeding.np_random(RANDOM_SEED)

    def step(
            self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        assert 0 <= action < self.action_space.n
        # print(f"action: {action}")
        row, col = divmod(action, self.map_size)

        coverage_matrix = self._calc_coverage(row, col)
        r_c = np.sum(coverage_matrix)  # coverage reward
        r_e = np.sum(self.coverage_map_opt)  # optimal coverage reward
        p_d = -np.linalg.norm(action - self.loc_tx_opt)  # distance penalty
        p_b = -100 if self.pixel_map[row, col] != 1 else 0  # RoI penalty
        a = self.coefficient_dict.get("r_c", .1)
        b = self.coefficient_dict.get("p_d", 1.)
        c = self.coefficient_dict.get("p_b", 1.)
        r = a * (r_c - r_e) + b * p_d + c * p_b

        term = True if r_c == r_e else False
        self.steps += 1
        trunc = self.steps >= self.max_steps
        info = {"steps": self.steps}
        if self.steps % 10 == 0 or term or trunc:
            info["detailed_rewards"] = f"r_c = {r_c}, r_e = {r_e}, p_d = {p_d}, p_b = {p_b}"
            print(info)
        obs_dict = {
            "observations": self.pixel_map,
            "action_mask": self.mask
        }

        return obs_dict, r, term, trunc, info

    def reset(
            self, *, seed: int | None = None, options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        super().reset(seed=seed, options=options)

        if self.cropped_maps is None:
            self.cropped_maps = crop_map(self.original_map, self.map_size, self.n_maps, self.np_random)
        self.steps = 0
        # generate a new cropped map and calculate the optimal TX location
        self.pixel_map = self.cropped_maps[self.n_trained_maps % self.n_maps]
        # 1 - building, 0 - free space
        self.mask = self.pixel_map.reshape(-1)
        self.n_trained_maps += 1
        s = time.time()
        self.loc_tx_opt, self.coverage_map_opt = find_opt_loc(self.pixel_map, self.thr_pl)
        e = time.time()

        obs_dict = {
            "observations": self.pixel_map,
            "action_mask": self.mask
        }
        info_dict = {
            "n_trained_maps": self.n_trained_maps,
            # "pixel_map": self.pixel_map,
            # "action_mask": self.mask,
            "cropped_map_shape": self.pixel_map.shape,
            "loc_tx_opt": self.loc_tx_opt,
            # "coverage_map_opt": self.coverage_map_opt,
            "time_exhaustive_search": e - s
        }
        print(info_dict)
        return obs_dict, info_dict

    def _calc_coverage(self, x_tx: int, y_tx: int) -> np.ndarray:
        """Calculate the coverage of a TX, given its location and a threshold.

        Args:
            x_tx: The X coordinate of the TX.
            y_tx: The Y coordinate of the TX.

        Returns:
            A coverage map where 0 = uncovered, 1 = covered.

        """
        return calc_path_loss(x_tx, y_tx, self.pixel_map, threshold=self.thr_pl)


if __name__ == "__main__":
    env = BaseEnvironment(config={})
    # _, info = env.reset(seed=2024)
    # print(info)

    fig, ax = plt.subplots()

    for i in range(3):
        env.reset()
        terminated, truncated = False, False
        r_ep = []
        n = 0
        while not terminated and not truncated:
            state, reward, terminated, truncated, info = env.step(env.action_space.sample())
            n += 1
            r_ep.append(reward)

            # if info.get("detailed_rewards", None) is not None:
            #     print(info)

        ax.plot(list(range(n)), r_ep, label=f"map # {i}")

    ax.set(xlabel="step", ylabel="reward", title="Random Sample")
    ax.legend()
    ax.grid()
    fig.savefig(f"./figures/random_{datetime.now().strftime('%m%d_%H%M')}.png")
