import json
import os

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from PIL import Image

from env.utils_v1 import ROOT_DIR, load_map_normalized


class RewardDataset(Dataset):
    def __init__(self, dir_dataset: str, data_type: str, map_indices: np.ndarray, map_dim: int, matrix_dim: int,
                 reward_thr: float = 240. / 255):
        self.dir_maps: str = os.path.join(ROOT_DIR, 'resource', dir_dataset, 'map')
        self.dir_pmaps: str = os.path.join(ROOT_DIR, 'resource', dir_dataset, f'pmap_{data_type}')
        self.dir_rewards: str = os.path.join(ROOT_DIR, 'resource', dir_dataset, 'reward_matrix')
        self.map_indices: np.ndarray = map_indices
        self.map_dim: int = map_dim
        self.matrix_dim: int = matrix_dim
        self.upsampling_factor: int = map_dim // matrix_dim
        self.reward_thr: float = reward_thr

    def __len__(self):
        return self.map_indices.size

    def __getitem__(self, idx):
        # building map
        map_idx = self.map_indices[idx % self.map_indices.size]
        map_path = os.path.join(self.dir_maps, f"{map_idx}.png")
        map_arr = Image.open(map_path).convert('L')
        map_tensor = ToTensor()(map_arr)  # 3D tensor, 1 x MAP_DIM x MAP_DIM

        # reward matrix, flatten 1d tensor
        reward_path = os.path.join(self.dir_rewards, f'reward_{map_idx}.json')
        rewards = json.load(open(reward_path))
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32)
        # rewards = torch.empty(self.matrix_dim**2, dtype=torch.float32)
        # for i in range(self.matrix_dim):
        #     for j in range(self.matrix_dim):
        #         row = i * self.upsampling_factor + (self.upsampling_factor - 1) // 2
        #         col = j * self.upsampling_factor + (self.upsampling_factor - 1) // 2
        #         idx = i * self.matrix_dim + j
        #         if map_tensor[0, row, col] == 0.:
        #             rewards[idx] = 0
        #         else:
        #             tx_idx = row * self.map_dim + col
        #             pmap_path = os.path.join(self.dir_pmaps, f'pmap_{map_idx}_{tx_idx}.png')
        #             pmap_arr = load_map_normalized(pmap_path)
        #             coverage_matrix = np.where(pmap_arr >= self.reward_thr, 1, 0)
        #             coverage = int(coverage_matrix.sum())
        #             rewards[idx] = coverage

        return map_tensor, rewards_tensor
