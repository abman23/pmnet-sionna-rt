#!/usr/bin/env python
# coding: utf-8

# Power Map Generation
import os
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

from dataset_builder.pmnet_v3 import PMNet
from env.utils_v1 import ROOT_DIR, calc_coverages_and_save


# Dataset preparation
def load_maps(dir_base: str = "usc", indices: np.ndarray = np.arange(100, dtype=int)) -> dict[int, np.ndarray]:
    """Load pixel maps as np.ndarray from images.
    
    """
    arr_maps = {}
    dir_maps = os.path.join(ROOT_DIR, dir_base, "map")
    for idx in indices:
        filename = os.path.join(dir_maps, str(idx)) + ".png"
        arr_map = Image.open(filename).convert('L')
        arr_maps[idx] = (np.array(arr_map,
                                  dtype=np.float32) - 0) / 255  # 256 x 256 matrix with value in [0,1] (grayscale)
        # arr_maps[idx] = np.asarray(io.imread(filename))

    return arr_maps


def generate_tx_layer(arr_map: np.ndarray, tx_size: int = 1, upsampling_factor: int = 4) -> dict[int, np.ndarray]:
    """Generate TX layers (same shape as the map) corresponding to all valid TX locations on a map.

    """
    tx_layers = {}
    map_size = arr_map.shape[0]
    n_steps = map_size // upsampling_factor
    for row in range(n_steps):
        for col in range(n_steps):
            # only generate upsampled TX location corresponding to the action in auto BS
            y, x = row * upsampling_factor + (upsampling_factor - 1) // 2, col * upsampling_factor + (
                    upsampling_factor - 1) // 2
            if arr_map[y, x] == 1.:  # white pixel - building
                arr_tx = np.zeros_like(arr_map, dtype=np.uint8)  # black background
                y_top, y_bottom = max(0, y - (tx_size - 1) // 2), min(map_size, y + tx_size // 2 + 1),
                x_left, x_right = max(0, x - (tx_size - 1) // 2), min(map_size, x + tx_size // 2 + 1)
                arr_tx[y_top: y_bottom, x_left: x_right] = 1  # white tx location

                idx = map_size * y + x  # 1d index of TX location
                tx_layers[idx] = arr_tx

    return tx_layers


def create_dataset(input_dir_base: str = "usc", index: int = 1, tx_size: int = 1, upsampling_factor: int = 4,
                   device: str = "cpu") -> tuple[list[str], torch.Tensor, dict[int, np.ndarray]]:
    """Create dataset for PMNet (cropped maps + TX locations).

    """
    indices = np.array(index, dtype=int).reshape(-1)
    arr_maps = load_maps(input_dir_base, indices)
    arr_map = arr_maps[index]
    idx_map_tx, tensors = [], []  # index (map index + tx index), tensor ([map, tx], ch=2)
    tx_layers = generate_tx_layer(arr_map, tx_size, upsampling_factor)
    for idx_tx, tx_layer in tx_layers.items():
        idx_data = str(index) + '_' + str(idx_tx)
        idx_map_tx.append(idx_data)
        # # save tx location as a separate image
        # tx_layer_grayscale = tx_layer * 255
        # img_tx = Image.fromarray(tx_layer_grayscale, mode='L')
        # img_tx.save(os.path.join(output_dir_base, "tx_" + suffix, "tx_" + idx_data + ".png"))
        # concatenate map and tx along channel-wisely
        arr_input = np.stack([arr_map, tx_layer], axis=0, dtype=np.float32)
        tensor_input = torch.from_numpy(arr_input).to(device)
        tensors.append(tensor_input)

    tensors = torch.stack(tensors, dim=0)
    # print(f"tensors shape: {tensors.shape}")

    return idx_map_tx, tensors, tx_layers


def inference_and_save(model: nn.Module, idx: list[str], tensors: torch.Tensor, batch_size: int = 256,
                       dir_base: str = "usc", dir_img: str = "pmap"):
    """Use PMNet to generate power maps from the given dataset and save them.

    """
    assert len(idx) == tensors.size(dim=0)
    # Set model to evaluation mode
    model.eval()

    n_batches = len(idx) // batch_size + 1 if len(idx) % batch_size != 0 else len(idx) // batch_size

    with torch.no_grad():
        for i in tqdm(range(n_batches)):
            start = i * batch_size
            end = min((i + 1) * batch_size, len(idx))

            batch_idx = idx[start: end]
            batch_tensors = tensors[start: end]

            preds = model(batch_tensors)
            # print(f"preds shape: {preds.shape}")
            # print(f"preds[0,0,:3,:3]: {preds[0,0,:3,:3]}")
            preds = torch.clip(preds, 0, 1)

            for j in range(len(preds)):
                file_name = 'pmap_' + batch_idx[j] + '.png'
                file_path = os.path.join(ROOT_DIR, dir_base, dir_img, file_name)
                arr = preds[j, 0].cpu().numpy()
                plt.imsave(file_path, arr, cmap='gray')
                # img_gray = Image.fromarray(arr).convert('L')
                # img_gray.save(file_path)


def inference(model: nn.Module, idx: list[str], tensors: torch.Tensor, batch_size: int = 256, save: bool = False,
              mark_tx: bool = True, tx_layers: dict[int, np.ndarray] | None = None, **kwargs) -> dict[int, np.ndarray]:
    """Use PMNet to generate power maps from the given dataset.

    """
    assert len(idx) == tensors.size(dim=0)
    # Set model to evaluation mode
    model.eval()

    n_batches = len(idx) // batch_size + 1 if len(idx) % batch_size != 0 else len(idx) // batch_size
    power_maps = {}

    with torch.no_grad():
        for i in range(n_batches):
            start = i * batch_size
            end = min((i + 1) * batch_size, len(idx))

            batch_idx = idx[start: end]
            batch_tensors = tensors[start: end]

            preds = model(batch_tensors)
            preds = torch.clip(preds, 0, 1)

            for j in range(len(preds)):
                pmap_idx = int(batch_idx[j].split('_')[-1])
                arr = preds[j, 0].cpu().numpy()
                if mark_tx:
                    # mark the tx location
                    tx_layer = tx_layers[pmap_idx]
                    arr = np.where(tx_layer > arr, tx_layer, arr)
                power_maps[pmap_idx] = arr
                # if j == 0:
                #     print(f"batch {i} first map:")
                #     print(arr[:5, :5])
                if save:
                    file_name = 'pmap_' + batch_idx[j] + '.png'
                    file_path = os.path.join(ROOT_DIR, kwargs['dir_base'], kwargs['dir_img'], file_name)
                    plt.imsave(file_path, arr, cmap='gray')

    return power_maps


def generate_pmaps(map_idx: int, upsampling_factor: int, mark_tx: bool, save: bool, **kwargs) -> dict[int, np.ndarray]:
    """Generate all path loss maps corresponding to some TX locations given a building map

    """
    # Load PMNet Model Parameters
    pretrained_model = os.path.join(ROOT_DIR, 'dataset_builder/checkpoints/summary_case4.pt')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = PMNet(n_blocks=[3, 3, 27, 3],
                  atrous_rates=[6, 12, 18],
                  multi_grids=[1, 2, 4],
                  output_stride=8, )
    model.load_state_dict(torch.load(pretrained_model, map_location=device))
    model = model.to(device)
    # Generate power maps using PMNet
    idx, tensors, tx_layers = create_dataset(input_dir_base='resource/usc_old', index=map_idx, tx_size=12,
                                             upsampling_factor=upsampling_factor, device=device)
    power_maps = inference(model=model, idx=idx, tensors=tensors, batch_size=64, mark_tx=mark_tx, tx_layers=tx_layers,
                           save=save, **kwargs)

    return power_maps


if __name__ == '__main__':
    for i in range(1, 1 + 32 * 500, 32):
        generate_pmaps(i, 8, mark_tx=True, save=True, dir_base='resource/usc_old_sparse', dir_img='pmap_train')

    for i in range(2, 2 + 32 * 100, 32):
        generate_pmaps(i, 8, mark_tx=True, save=True, dir_base='resource/usc_old_sparse', dir_img='pmap_test')

# if __name__ == "__main__":
#     # Load PMNet Model Parameters
#     pretrained_model = os.path.join(ROOT_DIR, 'dataset_builder/checkpoints/summary_case4.pt')
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     print(device)
#     model = PMNet(n_blocks=[3, 3, 27, 3],
#                   atrous_rates=[6, 12, 18],
#                   multi_grids=[1, 2, 4],
#                   output_stride=8, )
#     model.load_state_dict(torch.load(pretrained_model, map_location=device))
#     model = model.to(device)
#
#     # Save Power Map
#     # idx, tensors = create_dataset(input_dir_base='usc', indices=np.arange(1, dtype=int), device=device)
#     # inference_and_save(model=model, idx=idx, tensors=tensors, batch_size=8, dir_base='usc/pmap/')
#
#     idx_start, idx_end = 1, 1 + 32 * 100
#     # only do inference on one map at one time in case of OutOfMemoeryError
#     for idx_eval in tqdm(range(idx_start, idx_end, 32)):
#         idx, tensors = create_dataset(input_dir_base='resource/usc_old',
#                                       indices=np.arange(idx_eval, idx_eval + 1, dtype=int), tx_size=12,
#                                       upsampling_factor=8, device=device)
#         inference_and_save(model=model, idx=idx, tensors=tensors, batch_size=4, dir_base='resource/usc_old_2',
#                            dir_img='pmap_train')
#
#     idx_start, idx_end = 2, 2 + 32 * 50
#     # only do inference on one map at one time in case of OutOfMemoeryError
#     for idx_eval in tqdm(range(idx_start, idx_end, 32)):
#         idx, tensors = create_dataset(input_dir_base='resource/usc_old',
#                                       indices=np.arange(idx_eval, idx_eval + 1, dtype=int), tx_size=12,
#                                       upsampling_factor=8, device=device)
#         inference_and_save(model=model, idx=idx, tensors=tensors, batch_size=4, dir_base='resource/usc_old_2',
#                            dir_img='pmap_test')

# Do not pre-calculate and store environment stepping-related data in json because
# reading data from json files is slower than reading from images + calculating these data.
# # calculate optimal TX location based on power map and save all information to json
# calc_coverages_and_save(dataset_dir='resource/usc_old', output_dir='resource/usc_old_json',
#                         map_indices=np.arange(idx_eval, idx_eval + 1, dtype=int), map_suffix='test',
#                         coverage_threshold=220./255, upsampling_factor=256//64)
