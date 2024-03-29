import itertools
import logging
import os
from typing import Any, SupportsFloat

import gymnasium as gym
import numpy as np
from gymnasium.core import ActType, ObsType
from gymnasium.spaces import Discrete, MultiBinary, Dict, Box
from gymnasium.utils import seeding
from ray.rllib import MultiAgentEnv
from ray.rllib.utils.typing import MultiAgentDict
from retrying import retry

from env.utils import calc_fspl
from env.utils_v1 import ROOT_DIR, calc_coverages, load_map_normalized, calc_optimal_locations
from dataset_builder.generate_pmap import generate_pmaps

RANDOM_SEED: int | None = None  # manually set random seed

# set a logger
logger = logging.getLogger("env_v20")
logger.setLevel(logging.INFO)
log_path = os.path.join(ROOT_DIR, "log/env_v20.log")
handler = logging.FileHandler(log_path, encoding='utf-8', mode='a')
formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


class BaseEnvironment(MultiAgentEnv):
    """MDP environment for multi-BS scenario, version 2.0.
    Reward function : global coverage reward (not normalized)
    State: building map + power map

    """
    pixel_map: np.ndarray  # current building map
    version: str = "v20"
    steps: int
    map_idx: int  # index of the current building map
    power_maps: dict[int, np.ndarray]

    def __init__(self, config: dict) -> None:
        """Initialize the base MDP environment.

        """
        super().__init__()

        # directory contains building maps, training and test power maps
        self.dataset_dir = config.get("dataset_dir", "resource/usc_old")
        evaluation_mode = config.get("evaluation", False)
        self.evaluation = evaluation_mode
        self.algo_name = config.get("algo_name", None)
        # training or test env
        self.map_suffix = "test" if evaluation_mode else "train"
        # indices of maps used for training or test
        self.map_indices: np.ndarray = np.arange(2, 2 + 4 * 100, 4) if evaluation_mode else np.arange(1, 1 + 4 * 1000,
                                                                                                      4)
        # the threshold of path loss value that we consider a pixel as 'covered' by TX
        radius_thr = 0.1 * 880 / 256
        self.coverage_threshold: float = calc_fspl(radius_thr)

        self.n_bs: int = config.get("n_bs", 2)
        self.n_maps: int = config.get("n_maps", 1)
        # number of continuous steps using one cropped map
        self.n_episodes_per_map: int = config.get("n_episodes_per_map", 10)
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

        # MultiAgentEnv related setting
        self._agent_ids = set(str(idx) for idx in range(self.n_bs))
        self.action_space: Dict = Dict(
            {
                idx: Discrete(action_space_size ** 2)
                for idx in self._agent_ids
            }
        )

        if self.no_masking:
            self.observation_space: Dict = Dict(
                {
                    idx: Box(low=0., high=1., shape=(map_size ** 2,), dtype=np.float32)
                    for idx in self._agent_ids
                }
            )
        else:
            self.observation_space: Dict = Dict(
                {
                    idx: Dict(
                        {
                            "observations": Box(low=0., high=1.,
                                                shape=(map_size ** 2,), dtype=np.float32),
                            "action_mask": MultiBinary(action_space_size ** 2)
                        }
                    )
                    for idx in self._agent_ids
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
    ) -> tuple[MultiAgentDict, MultiAgentDict]:
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
            self.map_idx = int(self.map_indices[self.n_trained_maps % len(self.map_indices)])
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
            # self.load_pmaps()
        self.n_episodes += 1

        # choose a random initial action
        init_actions = self.np_random.choice(np.where(self.mask == 1)[0], size=self.n_bs)
        init_locs = []
        for action in init_actions:
            init_locs.append(self.calc_upsampling_loc(action))
        obs = self._mark_tx_locs(init_locs)

        if self.no_masking:
            observation = {
                idx: obs
                for idx in self._agent_ids
            }
        else:
            observation = {
                idx:
                    {"observations": obs,
                     "action_mask": self.mask}
                for idx in self._agent_ids
            }

        info_dict = {
            "n_episodes": self.n_episodes,
            "n_trained_maps": self.n_trained_maps,
            "map_suffix": self.map_suffix,
            "map_index": self.map_idx,
            "init_locs": init_locs,
        }
        if self.n_episodes % 5 == 0:
            logger.info(info_dict)

        return observation, {idx: info_dict for idx in self._agent_ids}

    def step(
            self, action_dict: MultiAgentDict
    ) -> tuple[
        MultiAgentDict, MultiAgentDict, MultiAgentDict, MultiAgentDict, MultiAgentDict
    ]:
        tx_locs = {}
        for idx, action in action_dict.items():
            tx_locs[idx] = self.calc_upsampling_loc(action)

        # calculate reward
        r_c = self.calc_coverage(list(tx_locs.values()))  # coverage reward
        reward_dict = {idx: r_c for idx in self._agent_ids}

        self.steps += 1
        trunc = self.steps >= self.n_steps_truncate  # truncate if reach the step limit
        trunc_dict = {idx: trunc for idx in self._agent_ids}
        trunc_dict["__all__"] = trunc

        term_dict = {idx: False for idx in self._agent_ids}
        term_dict["__all__"] = False

        obs = self._mark_tx_locs(list(tx_locs.values()))
        if self.no_masking:
            observation = {
                idx: obs
                for idx in self._agent_ids
            }
        else:
            observation = {
                idx:
                    {"observations": obs,
                     "action_mask": self.mask}
                for idx in self._agent_ids
            }

        info_dict = {
            "steps": self.steps,
            "tx_locs": tx_locs,
            # "loc_opt": self.loc_tx_opt,
            "r_c": r_c,
            "algo_name": self.algo_name,
            # "detailed_rewards": f"r_c = {r_c}, r_e = {r_e}",
        }
        # logger.info(info_dict)
        # if self.algo_name and (self.steps % (np.ceil(self.n_steps_per_map / 4)) == 0 or trunc):
        #     logger.info(info_dict)

        return observation, reward_dict, term_dict, trunc_dict, {idx: info_dict for idx in self._agent_ids}

    def calc_coverage(self, tx_locs: list) -> int:
        """Calculate the overall coverage reward of multiple TXs, given their locations.

        Args:
            tx_locs: tx location tuples

        Returns:
            Number of covered pixels in the map.

        """
        # calculate path loss in each pixel point
        values_pl = []
        for i in range(self.map_size):
            pl_row = []
            for j in range(self.map_size):
                # we ignore non-ROI area (buildings pixel) when calculating the coverage
                if self.pixel_map[i, j] != 0:
                    pl_row.append(0)
                    continue

                pl = -np.inf
                for loc in tx_locs:
                    # choose the TX with maximum path loss value
                    row, col = loc[0], loc[1]
                    dis = np.linalg.norm([row - i, col - j]) * 880 / 256
                    pl = max(pl, calc_fspl(dis))

                # convert path loss value to a 0-1 indicator of coverage
                covered = 1 if pl > self.coverage_threshold else 0
                pl_row.append(covered)
            values_pl.append(pl_row)

        return np.array(values_pl).sum()

    def calc_optimal_locations(self) -> tuple[list, int]:
        """Calculate the optimal TX locations that maximize the coverage reward.

            Returns:
                (actions, optimal coverage reward).

        """
        locs_opt, coverage_opt = [(-1, -1)], -1
        all_actions = list(itertools.combinations(range(self.action_space_size), 2))

        for actions in all_actions:
            tx_locs = []
            for action in actions:
                tx_locs.append(self.calc_upsampling_loc(action))
            coverage = self.calc_coverage(tx_locs)
            if coverage > coverage_opt:
                coverage_opt = coverage
                locs_opt = tx_locs

        return locs_opt, coverage_opt

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

    def _mark_tx_locs(self, locs: list[tuple], marker_size: int = 11) -> np.ndarray:
        """Mark TX locations on the current building map.

        Returns:
            The current building map with TX locations marked as 2.

        """
        bld_map = self.pixel_map.copy()
        for loc in locs:
            y, x = loc[0], loc[1]
            y_top, y_bottom = max(0, y - (marker_size - 1) // 2), min(self.map_size, y + marker_size // 2 + 1),
            x_left, x_right = max(0, x - (marker_size - 1) // 2), min(self.map_size, x + marker_size // 2 + 1)
            bld_map[y_top: y_bottom, x_left: x_right] = 2.  # mark tx location

        return bld_map.reshape(-1)


