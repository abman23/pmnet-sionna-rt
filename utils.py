import math
from typing import Tuple
from PIL import Image, ImageDraw
import numpy as np
import json
from gymnasium.utils import seeding


def load_map(filepath: str) -> np.ndarray:
    """Convert a building map image (black - building, white - free space) to a numpy array.

    Args:
        filepath: Path of the map.

    Returns:
        The corresponding 0-1 numpy array (1 - building, 0 - free space).

    """
    image = Image.open(filepath)
    # convert the image to grayscale
    image_gray = image.convert("L")
    image_array = np.array(image_gray)
    threshold = 128  # threshold of the grayscale value to map black to 1 and white to 0

    return (image_array < threshold).astype(np.int8)


def save_map(filepath: str, pixel_map: np.ndarray, reverse_color: bool = True, mark_loc: np.ndarray | None = None,
             **kwargs) -> None:
    """Save building map array as a black-white image.

    Args:
        filepath: Path of the image.
        pixel_map: Building map array.
        reverse_color: Whether to reverse the 0 1 value of each pixel or not.
        mark_loc (optional): Add a marker to the image given its location.

    Returns:

    """
    if reverse_color:
        pixel_map = (np.ones_like(pixel_map, dtype=np.uint8) - pixel_map)
    # convert the binary array to an image
    image_from_array = Image.fromarray((255 * pixel_map).astype(np.uint8), mode='L')

    if mark_loc is not None:
        image_from_array = image_from_array.convert('RGB')
        # Calculate the pixel coordinates of the top-left corner of the circle
        xy = (mark_loc[1], mark_loc[0])
        # Create an ImageDraw object to draw on the image
        draw = ImageDraw.Draw(image_from_array)
        # Draw the red point
        draw.point(xy, fill='red')

    if "mark_locs" in kwargs.keys():
        locs = kwargs["mark_locs"]
        draw = ImageDraw.Draw(image_from_array)
        for loc in locs:
            xy = (loc[1], loc[0])
            draw.point(xy, fill='blue')

    image_from_array.save(filepath)


def save_cropped_maps(original_map: np.ndarray, map_size: int, n: int, map_scale: float, ratio_coverage: float,
                      rng: np.random.Generator, filepath: str = './resource/setup.json') -> None:
    """Crop N pixel maps, calculate the corresponding optimal TX locations, coverages and save them in a file.

    Args:
        original_map: The original pixel map to crop.
        map_size: The size of cropped map.
        map_scale: The ratio between physical length and pixel (meter/pixel).
        ratio_coverage: The ratio of pixels with higher average path loss value than the threshold.
        n: The number of cropped resource.
        rng: A random number generator.
        filepath: Path of the output file.

    """
    threshold = calc_pl_threshold(original_map, map_scale, ratio_coverage)
    cropped_maps = crop_map(original_map, map_size, n, rng)
    locs_opt = []
    coverages_opt = []

    for i, cropped_map in enumerate(cropped_maps):
        loc_opt, coverage_opt = find_opt_loc(cropped_map, map_scale, threshold)
        cropped_maps[i] = cropped_map.tolist()
        locs_opt.append(loc_opt)
        coverages_opt.append(coverage_opt.tolist())

    res_dict = {
        "thr_pl": threshold,
        "cropped_maps": cropped_maps,
        "locs_opt": locs_opt,
        "coverages_opt": coverages_opt,
    }
    with open(filepath, "w", encoding="utf-8") as output_file:
        json.dump(res_dict, output_file)


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
    """Crop N pixel maps which must contain a building pixel from the ORIGINAL_MAP.

    Args:
        original_map: The original pixel map to crop.
        map_size: The size of cropped map.
        n: The number of cropped resource.
        rng: A random number generator.

    Returns:
        A cropped square pixel map with the same size as the original map.

    """
    assert np.sum(original_map) > 0  # the original map must contain building areas
    row_scale = original_map.shape[0] // map_size
    col_scale = original_map.shape[1] // map_size

    maps = []
    for _ in range(n):
        s_map = np.zeros((map_size, map_size), dtype=np.int8)
        while np.sum(s_map) == 0:
            x_start = rng.choice(original_map.shape[0] - map_size)
            y_start = rng.choice(original_map.shape[1] - map_size)
            s_map = original_map[x_start:x_start + map_size, y_start:y_start + map_size]
        # s_map = np.repeat(np.repeat(s_map, col_scale, axis=1), row_scale, axis=0)
        maps.append(s_map)
    return maps


def calc_coverage(x: int, y: int, map: np.ndarray, map_scale: float,
                  threshold: float | None = None, option: str = "FSPL") -> np.ndarray:
    """Calculate the coverage of a TX given its location and a building map.

    Args:
        x: The X coordinate of TX.
        y: The Y coordinate of TX.
        map: The pixel map to calculate FSPL for.
        map_scale: The ratio between physical length and pixel (meter/pixel).
        threshold: A threshold of FSPL for counting each pixel as covered by the TX or not.
        option: The type of path loss.

    Returns:
        The path loss map if THRESHOLD is None, otherwise a TX coverage map.

    """
    n_row, n_col = map.shape
    assert 0 <= x < n_row and 0 <= y < n_col

    values_pl = []
    for i in range(n_row):
        pl_row = []
        for j in range(n_col):
            # we ignore non-ROI area (buildings pixel) when calculating the coverage
            if map[i][j] == 1 and threshold is not None:
                pl_row.append(0)
                continue

            dis = np.linalg.norm([x - i, y - j]) * map_scale
            if option == "FSPL":
                pl = calc_fspl(dis)
            else:
                pl = 0.
            if threshold is not None:
                # convert path loss value to a 0-1 indicator of coverage if a threshold is given
                covered = 1 if pl > threshold else 0
                pl_row.append(covered)
            else:
                pl_row.append(pl)
        values_pl.append(pl_row)

    return np.array(values_pl, dtype=np.int8) if threshold is not None else np.array(values_pl, dtype=np.float32)


def calc_pl_threshold(original_map: np.ndarray, map_scale: float, ratio_coverage: float, option: str = "FSPL") -> float:
    """Calculate the threshold of path loss value given a PIXEL_MAP and an expected coverage rate.

    Args:
        original_map: A pixel map with buildings and free spaces.
        map_scale: The ratio between physical length and pixel (meter/pixel).
        ratio_coverage: The ratio of pixels with higher average path loss value than the threshold.

    Returns:
        The threshold we find.

    """
    assert 0 < ratio_coverage < 1

    radius = math.sqrt(ratio_coverage * map_scale ** 2 * original_map.shape[0] * original_map.shape[1] / math.pi)
    if option == "FSPL":
        return calc_fspl(radius)
    else:
        return -np.inf


def find_opt_loc(pixel_map: np.ndarray, map_scale: float, pl_threshold: float | None = None) -> tuple[tuple, np.ndarray]:
    """Find the optimal location of TX which maximizes the overall coverage.

    Args:
        pixel_map: A pixel map to find the optimal TX location for.
        map_scale: The ratio between physical length and pixel (meter/pixel).
        pl_threshold: : The threshold of path loss.

    Returns:
        The optimal TX location and the corresponding coverage map.

    """
    loc_opt = (-1, -1)
    coverage_sum_opt = -np.inf
    coverage_map_opt = np.empty((pixel_map.shape[0], pixel_map.shape[1]), dtype=np.int8)
    loc_center = np.array([pixel_map.shape[0] // 2, pixel_map.shape[1] // 2], dtype=np.int32)
    dis_center_loc_opt = np.inf

    for x in range(pixel_map.shape[0]):
        for y in range(pixel_map.shape[1]):
            if pixel_map[x][y] == 1:
                coverage_map = calc_coverage(x, y, pixel_map, map_scale, threshold=pl_threshold)
                coverage_sum = np.sum(coverage_map)
                loc = (x, y)
                dis_center_loc = np.linalg.norm(loc - loc_center)
                # update the optimal location if the sum of coverage is greater or the location is closer to the center
                if (coverage_sum > coverage_sum_opt or
                        (coverage_sum == coverage_sum_opt and dis_center_loc < dis_center_loc_opt)):
                    coverage_sum_opt = coverage_sum
                    loc_opt = loc
                    coverage_map_opt = coverage_map
                    dis_center_loc_opt = dis_center_loc

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


if __name__ == '__main__':
    # use this script to generate cropped maps and save them as json
    original_map = load_map('./resource/usc.png')
    map_size = 64
    n = 400
    map_scale = 880 / 256
    ratio_coverage = .2 / 16
    rng, seed = seeding.np_random(2024)
    filepath = './resource/setup_400.json'
    save_cropped_maps(original_map, map_size, n, map_scale, ratio_coverage, rng, filepath)
