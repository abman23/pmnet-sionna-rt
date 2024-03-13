import json
import os

import numpy as np
from PIL import Image
from tqdm import tqdm

# project root directory
ROOT_DIR: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_map_normalized(filepath: str) -> np.ndarray:
    """Convert map image to array (pixel value normalized to the range [0,1]).

    """
    image = Image.open(filepath).convert('L')
    image_arr = np.array(image, dtype=np.float32) / 255

    return image_arr


def calc_coverages(dataset_dir: str, map_suffix: str, map_idx: int,
                   coverage_threshold: float, upsampling_factor: int) -> tuple[np.ndarray, tuple, int, dict, dict]:
    """Calculate coverage matrix, optimal coverage reward and optimal TX location given a map index.

    Returns:
        (map, optimal TX location, optimal coverage reward, coverage matrices, power maps)

    """
    map_dir = os.path.join(ROOT_DIR, dataset_dir, 'map')
    pmap_dir = os.path.join(ROOT_DIR, dataset_dir, 'pmap_' + map_suffix)

    pmaps = {}
    coverage_matrices = {}
    loc_opt, coverage_opt = (-1, -1), -1
    dis_center_loc_opt = np.inf
    map_path = os.path.join(map_dir, str(map_idx) + '.png')
    map_arr = load_map_normalized(map_path)
    map_size = map_arr.shape[0]
    # we only consider TX location corresponding to reduced action
    n_steps = map_size // upsampling_factor
    for row in range(n_steps):
        for col in range(n_steps):
            # upsampled TX location
            y, x = row * upsampling_factor + (upsampling_factor - 1) // 2, col * upsampling_factor + (
                    upsampling_factor - 1) // 2
            if map_arr[y, x] == 1.:  # white pixel - building
                loc_idx = map_size * y + x  # 1d index of TX location
                pmap_path = os.path.join(pmap_dir, 'pmap_' + str(map_idx) + '_' + str(loc_idx) + '.png')
                pmap_arr = load_map_normalized(pmap_path)
                pmaps[loc_idx] = pmap_arr
                coverage_matrix = np.where(pmap_arr >= coverage_threshold, 1, 0)
                coverage_matrices[loc_idx] = coverage_matrix
                coverage = int(coverage_matrix.sum())
                dis_center_loc = (y - map_size // 2) ** 2 + (x - map_size // 2) ** 2
                # exhaustively search the optimal TX location
                # break the tie using distance between optimal location and map center
                if coverage > coverage_opt or (coverage == coverage_opt and dis_center_loc < dis_center_loc_opt):
                    coverage_opt = coverage
                    loc_opt = (y, x)
                    dis_center_loc_opt = dis_center_loc

    return map_arr.astype(np.int8), loc_opt, coverage_opt, coverage_matrices, pmaps


def calc_coverages_and_save(dataset_dir: str, output_dir: str, map_indices: np.ndarray, map_suffix: str,
                            coverage_threshold: float, upsampling_factor: int = 4) -> None:
    """Calculate coverage matrices, optimal coverage value and optimal TX location given some map indices and save them
    to a JSON file.

    """
    maps = {}
    locs_opt = {}
    coverages_opt = {}
    coverage_matrices = {}  # 2-layer dict, coverage_matrices[map_idx][loc_idx] = [0,0,1,..]

    for i in tqdm(range(len(map_indices))):
        map_idx = int(map_indices[i])
        map, loc_opt, coverage_opt, coverage_matrices, pmaps = calc_coverages(dataset_dir, map_suffix, map_idx,
                                                                              coverage_threshold, upsampling_factor)
        # convert numpy array to list
        map = map.tolist()
        for loc_idx in coverage_matrices.keys():
            coverage_matrices[loc_idx] = coverage_matrices[loc_idx].tolist()
            pmaps[loc_idx] = pmaps[loc_idx].tolist()

        res_dict = {'map': map, 'loc_opt': loc_opt, 'coverage_opt': coverage_opt,
                    'coverage_matrices': coverage_matrices, 'pmaps': pmaps}

        output_filepath = os.path.join(ROOT_DIR, output_dir, "dataset_" + str(map_idx) + '.json')
        with open(output_filepath, "w", encoding="utf-8") as output_file:
            json.dump(res_dict, output_file)

    print(f"Found optimal locations for all {map_suffix} maps")

# if __name__ == '__main__':
# calc_coverages_and_save(dataset_dir='../resource/usc_old', output_dir='../resource/usc_old_json',
#                          map_suffix='train',
#                          coverage_threshold=220. / 255, map_indices=np.arange(1, 1 + 32 * 1, 32, dtype=int))
# calc_coverages_and_save(dataset_dir='../resource/usc_old', output_dir='../resource/usc_old_json',
#                          map_suffix='test',
#                          coverage_threshold=220. / 255, map_indices=np.arange(2, 2 + 32 * 50, 32, dtype=int))
