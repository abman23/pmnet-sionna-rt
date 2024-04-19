import json
import os

import numpy as np
import torch

from dataset_builder.generate_pmap import create_dataset, inference
from dataset_builder.pmnet_v3 import PMNet
from env.utils_v1 import ROOT_DIR


def generate_reward(map_idx: int, dir_dataset: str, map_size: int, upsampling_factor: int, coverage_thr: float,
                    checkpoint: str = 'model_0.00136.pt', batch_size: int = 32, save: bool = False):
    """Generate coverage reward matrix, where each value is the coverage reward of deploying TX
    at the corresponding location.

    coverage_thr: Percentage of RoI pixels to be considered as 'covered'.
    e.g., 20 means the top 20% RoI area with the strongest signal.

    """
    # Load PMNet pre-trained model
    pretrained_model = os.path.join(ROOT_DIR, f'dataset_builder/checkpoints/{checkpoint}')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = PMNet(n_blocks=[3, 3, 27, 3],
                  atrous_rates=[6, 12, 18],
                  multi_grids=[1, 2, 4],
                  output_stride=8, )
    model.load_state_dict(torch.load(pretrained_model, map_location=device))
    model = model.to(device)

    # Generate path-loss maps using PMNet
    idx, tensors, tx_layers = create_dataset(input_dir_base=dir_dataset, index=map_idx, tx_size=3,
                                             upsampling_factor=upsampling_factor, non_building_pixel_value=1.,
                                             device=device)
    power_maps = inference(model=model, idx=idx, tensors=tensors, mark_tx=False, tx_layers=tx_layers,
                           batch_size=batch_size, save=True, dir_base=dir_dataset, dir_img=f'power_map')
    # print(power_maps.keys())

    # Compute the coverage threshold,
    # which averages out the threshold of all power maps corresponding to the building map.
    thr = []
    for power_map in power_maps.values():
        mask = power_map > 0.
        thr.append(np.percentile(power_map[mask], 100 - coverage_thr))
    thr_avg = np.mean(thr)

    # Calculate reward based on power maps and fill up the reward matrix
    matrix_dim = map_size // upsampling_factor
    rewards = []
    for i in range(matrix_dim):
        rewards_row = []
        for j in range(matrix_dim):
            row = i * upsampling_factor + (upsampling_factor - 1) // 2
            col = j * upsampling_factor + (upsampling_factor - 1) // 2
            idx = row * map_size + col
            if idx not in power_maps.keys():
                # skip non-building pixel
                # print(f'miss {idx}')
                rewards_row.append(0)
            else:
                pmap_arr = power_maps[idx]
                coverage_matrix = np.where(pmap_arr >= thr_avg, 1, 0)
                coverage = int(coverage_matrix.sum())
                rewards_row.append(coverage)
        rewards.append(rewards_row)

    # print(len(rewards))
    if save:
        reward_dir = os.path.join(ROOT_DIR, dir_dataset, 'reward_matrix')
        os.makedirs(reward_dir, exist_ok=True)
        reward_matrix = os.path.join(reward_dir, f"reward_{map_idx}.json")
        json.dump(rewards, open(os.path.join(reward_matrix), 'w'))

    return np.array(rewards)


if __name__ == '__main__':
    for map_idx in range(1, 1501):
        reward_matrix = generate_reward(map_idx=map_idx, dir_dataset='resource/usc_new', map_size=256,
                                        upsampling_factor=8, coverage_thr=20, checkpoint='model_0.00136.pt',
                                        batch_size=32, save=True)
    # print(reward_matrix[reward_matrix != 0])
    # print((reward_matrix != 0).sum())
    # print(reward_matrix.shape)
