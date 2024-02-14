import math
from typing import Tuple

import numpy as np


def generate_map(n_rows: int, n_cols: int, ratio_buildings: float = .5, seed: int | None = None) -> np.ndarray:
    """Randomly generate a pixel map with buildings and free spaces with a given ratio of buildings.

    Args:
        n_rows: The number of rows in the pixel map.
        n_cols: The number of columns in the pixel map.
        ratio_buildings: The ratio of buildings in the map.
        seed: A random seed set for numpy.

    Returns:
        A map represented by a 2d numpy array.

    """
    rng = np.random.default_rng(seed)
    map_flatten = np.zeros(n_rows * n_cols, dtype=np.int8)
    if 0 < ratio_buildings <= 1:
        num_buildings = int(ratio_buildings * n_rows * n_cols)
        loc_buildings = rng.choice(n_rows * n_cols, size=num_buildings, replace=False)
        map_flatten[loc_buildings] = 1

    return map_flatten.reshape((n_rows, n_cols))


def crop_map(original_map: np.ndarray, map_size: int, n: int, rng: np.random.Generator) -> list[np.ndarray]:
    """Crop N MAP_SIZE x MAP_SIZE pixel square which must contain a building pixel from the ORIGINAL_MAP.

    Args:
        original_map: The original pixel map to crop.
        map_size: The size of cropped map.
        n: The number of cropped maps.
        rng: A random number generator.

    Returns:
        A cropped square pixel map.

    """
    assert np.sum(original_map) > 0

    maps = []
    for _ in range(n):
        s_map = np.zeros((map_size, map_size), dtype=np.int8)
        while np.sum(s_map) == 0:
            x_start = rng.choice(original_map.shape[0] - map_size)
            y_start = rng.choice(original_map.shape[1] - map_size)
            s_map = original_map[x_start:x_start + map_size, y_start:y_start + map_size]
        maps.append(s_map)
    return maps


def calc_path_loss(x: int, y: int, map: np.ndarray, threshold: float | None = None, option: str = "FSPL") -> np.ndarray:
    """Calculate path loss for every pixel in the map and return the path loss map, given a TX location.

    Args:
        x: The X coordinate of TX.
        y: The Y coordinate of TX.
        map: The pixel map to calculate FSPL for.
        threshold: A threshold of FSPL for counting each pixel as covered by the TX or not.
        option: The type of path loss.

    Returns:
        The path loss map if THRESHOLD is None, otherwise a cover map.

    """
    n_row, n_col = map.shape
    assert 0 <= x < n_row and 0 <= y < n_col

    values_pl = []
    for i in range(n_row):
        pl_row = []
        for j in range(n_col):
            dis = np.linalg.norm([x - i, y - i])
            if option == "FSPL":
                pl = calc_fspl(dis)
            else:
                pl = 0.
            if threshold is not None:
                pl = 1 if pl > threshold else 0
            pl_row.append(pl)
        values_pl.append(pl_row)

    return np.array(values_pl, dtype=np.float32) if threshold is None else np.array(values_pl, dtype=np.int8)


def calc_pl_threshold(original_map: np.ndarray, ratio_coverage: float, option: str = "FSPL") -> float:
    """Calculate the threshold of path loss value given a PIXEL_MAP and an expected coverage rate.

    Args:
        original_map: A pixel map with buildings and free spaces.
        ratio_coverage: The ratio of pixels with higher average path loss value than the threshold.

    Returns:
        The threshold we find.

    """
    assert 0 < ratio_coverage < 1

    radius = math.sqrt(ratio_coverage ** 2 * original_map.shape[0] * original_map.shape[1] / math.pi)
    if option == "FSPL":
        return calc_fspl(radius)
    else:
        return -np.inf


def find_opt_loc(pixel_map: np.ndarray, pl_threshold: float | None = None) -> Tuple[np.ndarray, np.ndarray]:
    """Find the optimal location of TX which maximizes the overall path loss.

    Args:
        pixel_map: A pixel map to find the optimal TX location for.
        pl_threshold: : The threshold of path loss.

    Returns:
        The optimal TX location and the corresponding coverage map.

    """
    loc_opt = np.ones(2, dtype=np.int32) * -1
    pl_sum_opt = -np.inf
    coverage_map_opt = np.empty((pixel_map.shape[0], pixel_map.shape[1]), dtype=np.int8)

    for x in range(pixel_map.shape[0]):
        for y in range(pixel_map.shape[1]):
            if pixel_map[x][y] == 1:
                coverage_map = calc_path_loss(x, y, pixel_map, threshold=pl_threshold)
                pl_sum = np.sum(coverage_map)
                if pl_sum > pl_sum_opt:
                    pl_sum_opt = pl_sum
                    loc_opt[0], loc_opt[1] = x, y
                    coverage_map_opt = coverage_map

    return loc_opt, coverage_map_opt

def calc_fspl(dis: float) -> float:
    """Calculate the free-space path loss (FSPL) at a point given a distance.

    Args:
        dis: Distance between the TX and the target point.

    Returns:
        The FSPL value.

    """
    if dis == 0:
        return 0.
    beta = 3
    lg_param = -1.0203
    return 10 * beta * (lg_param - np.log(dis) / np.log(10))
