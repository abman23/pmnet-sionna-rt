from datetime import datetime
import json
import os

import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
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
        (building map, optimal TX location, optimal coverage reward, coverage matrices, power maps)

    """
    map_dir = os.path.join(ROOT_DIR, dataset_dir, 'map')
    pmap_dir = os.path.join(ROOT_DIR, dataset_dir, 'pmap_' + map_suffix)

    pmaps = {}
    coverage_rewards = {}
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
                coverage = int(coverage_matrix.sum())
                coverage_rewards[loc_idx] = coverage
                dis_center_loc = (y - map_size // 2) ** 2 + (x - map_size // 2) ** 2
                # exhaustively search the optimal TX location
                # break the tie using distance between optimal location and map center
                if coverage > coverage_opt or (coverage == coverage_opt and dis_center_loc < dis_center_loc_opt):
                    coverage_opt = coverage
                    loc_opt = (y, x)
                    dis_center_loc_opt = dis_center_loc

    return map_arr.astype(np.int8), loc_opt, coverage_opt, coverage_rewards, pmaps


def calc_coverages_and_save(dataset_dir: str, output_dir: str, map_indices: np.ndarray, map_suffix: str,
                            coverage_threshold: float, upsampling_factor: int = 4) -> None:
    """Calculate coverage matrices, optimal coverage value and optimal TX location given some buildings_map indices and save them
    to a JSON file.

    """
    for i in tqdm(range(len(map_indices))):
        map_idx = int(map_indices[i])
        buildings_map, loc_opt, coverage_opt, coverage_matrices, pmaps = calc_coverages(dataset_dir, map_suffix,
                                                                                        map_idx,
                                                                                        coverage_threshold,
                                                                                        upsampling_factor)
        # convert numpy array to list
        buildings_map = buildings_map.tolist()
        for loc_idx in coverage_matrices.keys():
            coverage_matrices[loc_idx] = coverage_matrices[loc_idx].tolist()
            pmaps[loc_idx] = pmaps[loc_idx].tolist()

        res_dict = {'buildings_map': buildings_map, 'loc_opt': loc_opt, 'coverage_opt': coverage_opt,
                    'coverage_rewards': coverage_matrices, 'pmaps': pmaps}

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


def dict_update(old_dict: dict, new_dict: dict) -> dict:
    """Updates the old dictionary with new key-value pairs in the new dictionary.

    Returns:
        The updated dictionary (not in-placed).
    """
    returned_dict = json.loads(json.dumps(old_dict))
    for key, value in new_dict.items():
        returned_dict[key] = value

    return returned_dict


# todo: this method can be moved into env class
def calc_optimal_locations(dataset_dir: str, map_suffix: str, map_idx: int,
                           coverage_threshold: float, upsampling_factor: int) -> tuple[int, int]:
    """Calculate the optimal TX location given a map index.

        Returns:
            (action index, optimal coverage reward).

    """
    map_dir = os.path.join(ROOT_DIR, dataset_dir, 'map')
    pmap_dir = os.path.join(ROOT_DIR, dataset_dir, 'pmap_' + map_suffix)

    loc_opt, coverage_opt = (-1, -1), -1
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
                coverage_matrix = np.where(pmap_arr >= coverage_threshold, 1, 0)
                coverage = int(coverage_matrix.sum())
                # exhaustively search the optimal TX location
                if coverage > coverage_opt:
                    coverage_opt = coverage
                    loc_opt = (row, col)

    return loc_opt[0] * n_steps + loc_opt[1], coverage_opt


def plot_rewards(output_name: str, algo_names: list[str], data_filenames: list[str], version: str,
                 evaluation: bool = True, log: bool = False, n_epi: int = 10):
    """Plot rewards curve of multiple algorithms.

    """
    if evaluation:
        fig, axes = plt.subplots(2, 1)
        fig.set_size_inches(10, 12)
    else:
        fig, axes = plt.subplots()
        fig.set_size_inches(10, 6)

    for algo_name, filename in zip(algo_names, data_filenames):
        data_path = os.path.join(ROOT_DIR, 'data', filename)
        algo_data = json.load(open(data_path))

        if evaluation:
            # plot reward in evaluation
            ep_eval = algo_data['ep_eval'][:n_epi//5]
            ep_reward_mean = algo_data['ep_reward_mean'][:n_epi//5]
            ep_reward_std = algo_data['ep_reward_std'][:n_epi//5]
            # print(algo_name)
            # print(len(ep_eval), len(ep_reward_mean))
            ax = axes[1]
            ax.plot(ep_eval, ep_reward_mean, label=algo_name.upper())
            sup = list(map(lambda x, y: x + y, ep_reward_mean, ep_reward_std))
            inf = list(map(lambda x, y: x - y, ep_reward_mean, ep_reward_std))
            ax.fill_between(ep_eval, inf, sup, alpha=0.2)

        # plot reward in training
        ep_train = algo_data['ep_train'][:n_epi]
        ep_reward_mean_train = algo_data['ep_reward_mean_train'][:n_epi]
        ax = axes[0] if evaluation else axes
        ax.plot(ep_train, ep_reward_mean_train, label=algo_name.upper())

    if evaluation:
        ax = axes[1]
        ax.set(xlabel="training_step", ylabel="mean reward per step",
               title=f"Evaluation Results")
        ax.grid()
        ax.legend()
    ax = axes[0] if evaluation else axes
    ax.set(xlabel="training_step", ylabel="mean reward per step",
           title=f"Training Results")
    ax.grid()
    ax.legend()

    timestamp = datetime.now().strftime('%m%d_%H%M')
    if log:
        fig.savefig(os.path.join(ROOT_DIR, f"figures/compare/{version}_{output_name}_{timestamp}.png"))
    plt.show()


if __name__ == "__main__":
    plot_rewards(output_name="ppo", algo_names=['ppo_v16'],
                 data_filenames=['ppo_0331_1940.json'],
                 version='v16', evaluation=True, log=False, n_epi=350)
