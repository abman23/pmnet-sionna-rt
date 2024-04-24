#!/usr/bin/env python
# coding: utf-8
import json
# Power Map Generation
import os
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
# import cv2
from skimage import io

from dataset_builder.pmnet_v3 import PMNet
from env.utils_v1 import ROOT_DIR, calc_coverages_and_save, load_map_normalized


# Dataset preparation
def load_maps(dir_base: str = "usc", index: int = 1) -> np.ndarray:
    """Load pixel maps as np.ndarray from images.
    
    """
    dir_maps = os.path.join(ROOT_DIR, dir_base, "map")
    filename = os.path.join(dir_maps, str(index)) + ".png"
    # arr_map = Image.open(filename).convert('L')
    arr_map = np.asarray(io.imread(filename))
    # print(arr_map.shape)
    # print(arr_map.dtype)
    return arr_map / 255  # 256 x 256 matrix with value in [0,1] (grayscale)


def generate_tx_layer(arr_map: np.ndarray, tx_size: int = 5, upsampling_factor: int = 4,
                      non_building_pixel_value: float = 0.) -> dict[int, np.ndarray]:
    """Generate TX layers (same shape as the map) corresponding to all valid TX locations on a map.
    TX_SIZE better set as an odd number.

    """
    tx_layers = {}
    map_size = arr_map.shape[0]  # 256
    n_steps = map_size // upsampling_factor
    for row in range(n_steps):
        for col in range(n_steps):
            # only generate upsampled TX location corresponding to the action in auto BS
            y, x = row * upsampling_factor + (upsampling_factor - 1) // 2, col * upsampling_factor + (
                    upsampling_factor - 1) // 2
            if arr_map[y, x] != non_building_pixel_value:
                arr_tx = np.zeros((map_size, map_size), dtype=np.uint8)  # black background
                y_top, y_bottom = max(0, y - tx_size // 2), min(map_size, y + tx_size // 2 + 1),
                x_left, x_right = max(0, x - tx_size // 2), min(map_size, x + tx_size // 2 + 1)
                arr_tx[y_top: y_bottom, x_left: x_right] = 1.  # white tx location

                idx = map_size * y + x  # 1d index of TX location
                # arr_tx = cv2.resize(arr_tx, (256, 256))
                tx_layers[idx] = arr_tx
                # for debugging
                # tx_dir = os.path.join(ROOT_DIR, 'dataset_builder/figure/tx_map_sample')
                # os.makedirs(tx_dir, exist_ok=True)
                # # cv2.imwrite(os.path.join(tx_dir, str(idx) + ".png"), arr_tx)
                # plt.imsave(os.path.join(tx_dir, str(idx) + ".png"), arr_tx, cmap='gray')

    return tx_layers


def create_dataset(input_dir_base: str = "usc", index: int = 1, tx_size: int = 1, upsampling_factor: int = 4,
                   non_building_pixel_value: float = 0., device: str = "cpu") -> tuple[list[str], torch.Tensor, dict[int, np.ndarray]]:
    """Create dataset for PMNet (cropped maps + TX locations).

    """
    arr_map = load_maps(input_dir_base, index)
    idx_map_tx, tensors = [], []  # index (map index + tx index), tensor ([map, tx], ch=2)
    tx_layers = generate_tx_layer(arr_map, tx_size, upsampling_factor, non_building_pixel_value)
    for idx_tx, tx_layer in tx_layers.items():
        idx_data = str(index) + '_' + str(idx_tx)
        idx_map_tx.append(idx_data)
        # # save tx location as a separate image
        # tx_layer_grayscale = tx_layer * 255
        # img_tx = Image.fromarray(tx_layer_grayscale, mode='L')
        # img_tx.save(os.path.join(output_dir_base, "tx_" + suffix, "tx_" + idx_data + ".png"))
        # concatenate map and tx along channel-wisely
        arr_input = np.stack([arr_map, tx_layer], axis=0, dtype=np.float32)
        tensor_input = torch.from_numpy(arr_input)
        tensors.append(tensor_input)

    tensors = torch.stack(tensors, dim=0).to(device)
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
        for i in tqdm(range(n_batches)):
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
                    map_idx, loc_idx = batch_idx[j].split('_')[0], batch_idx[j].split('_')[1]
                    img_dir = os.path.join(ROOT_DIR, kwargs['dir_base'], kwargs['dir_img'], map_idx)
                    os.makedirs(img_dir, exist_ok=True)
                    file_path = os.path.join(img_dir, f'pmap_{map_idx}_{loc_idx}.png')
                    plt.imsave(file_path, arr, cmap='gray')

    return power_maps


def generate_pmaps(map_idx: int, upsampling_factor: int, mark_tx: bool, save: bool, **kwargs) -> dict[int, np.ndarray]:
    """Generate all path loss maps corresponding to some TX locations given a building map

    """
    # Load PMNet Model Parameters
    pretrained_model = os.path.join(ROOT_DIR, 'dataset_builder/checkpoints/model_0.00133.pt')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = PMNet(n_blocks=[3, 3, 27, 3],
                  atrous_rates=[6, 12, 18],
                  multi_grids=[1, 2, 4],
                  output_stride=8, )
    model.load_state_dict(torch.load(pretrained_model, map_location=device))
    model = model.to(device)
    # Generate power maps using PMNet
    idx, tensors, tx_layers = create_dataset(input_dir_base=kwargs['dir_base'], index=map_idx, tx_size=5,
                                             upsampling_factor=upsampling_factor, non_building_pixel_value=1.,
                                             device=device)
    power_maps = inference(model=model, idx=idx, tensors=tensors, mark_tx=mark_tx, tx_layers=tx_layers,
                           save=save, **kwargs)

    return power_maps


if __name__ == '__main__':
    # arr_map = load_maps('resource/new_usc', 1)
    # # print(arr_map.dtype)
    # tx_maps = generate_tx_layer(arr_map=arr_map, tx_size=5, upsampling_factor=8, non_building_pixel_value=1.)
    # tx_map = next(iter(tx_maps.values()))
    # print(tx_map.shape, tx_map.dtype)
    # print(tx_map[30:40, 85:95])
    #
    # sample_tx_map = np.asarray(io.imread('figure/0_0_0.png'))
    # print(sample_tx_map.shape, sample_tx_map.dtype)
    # print(sample_tx_map[65:75, 185:195])

    for i in range(1, 1101):
        generate_pmaps(i, 8, batch_size=32,
                       mark_tx=False, save=True, dir_base='resource/new_usc', dir_img='power_map')

    # for i in range(1001, 1101, 1):
    #     generate_pmaps(i, 8, batch_size=32,
    #                    mark_tx=False, save=True, dir_base='resource/new_usc', dir_img='pmap_test')

    # # Load PMNet Model Parameters
    # pretrained_model = os.path.join(ROOT_DIR, 'dataset_builder/checkpoints/model_0.00136.pt')
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # print(device)
    # model = PMNet(n_blocks=[3, 3, 27, 3],
    #               atrous_rates=[6, 12, 18],
    #               multi_grids=[1, 2, 4],
    #               output_stride=8, )
    # model.load_state_dict(torch.load(pretrained_model, map_location=device))
    # model = model.to(device)
    #
    # tensors = []
    # for i in range(4):
    #     bld_filename = os.path.join(ROOT_DIR, f'resource/cropped_downsample_512x512/city_map/0_0_{i}.png')
    #     arr_map = np.array(Image.open(bld_filename).convert('L'), dtype=np.float32)
    #     # print(arr_map.max(), arr_map.min())
    #     arr_map /= 255.
    #     tx_filename = os.path.join(ROOT_DIR, f'resource/cropped_downsample_512x512/tx_map/0_0_{i}.png')
    #     tx_layer = np.array(Image.open(tx_filename).convert('L'), dtype=np.float32) / 255.
    #     arr_input = np.stack([arr_map, tx_layer], axis=0, dtype=np.float32)
    #     tensor_input = torch.from_numpy(arr_input).to(device)
    #     tensors.append(tensor_input)
    # # tx_size, map_size = 3, 256
    # # y, x = 100, 100
    # # y_top, y_bottom = max(0, y - (tx_size - 1) // 2), min(map_size, y + tx_size // 2 + 1),
    # # x_left, x_right = max(0, x - (tx_size - 1) // 2), min(map_size, x + tx_size // 2 + 1)
    # # tx_layer2[y_top: y_bottom, x_left: x_right] = 1
    # # arr_input2 = np.stack([arr_map, tx_layer2], axis=0, dtype=np.float32)
    # # tensor_input2 = torch.from_numpy(arr_input2).to(device)
    # tensors = torch.stack(tensors, dim=0)
    # # print(tensors.shape)
    #
    # output = model(tensors)
    # output = torch.clip(output, 0, 1)
    # # print(output.shape)
    # arrs = []
    # thrs = []
    # for i in range(4):
    #     arr = output[i, 0].detach().numpy()
    #     arrs.append(arr)
    #     # print(arr.max())
    #     mask = arr > 0.  # only consider RoI area
    #     thr = np.percentile(arr[mask], 90)
    #     thrs.append(thr)
    #     print(f"thr {i}: {thr}")
    # # arr2 = mpimg.imread('../resource/cropped_downsample_512x512/power_map/0_0_0.png')
    # print(f"avg thr: {np.mean(thrs)}")
    #
    # fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    # # Plot the first image
    # axes[0].imshow(arrs[0], cmap='gray')
    # axes[0].set_title('From PMNet')
    #
    # # Plot the second image
    # axes[1].imshow(np.where(arrs[0] > 0.64, 1., 0.), cmap='gray')
    # axes[1].set_title('Covered Area')
    # # plt.imshow(arr2, cmap='gray')
    # plt.show()



    # # Generate reward matrices
    # for i in tqdm(range(1, 1 + 16 * 1000, 16)):
    #     generate_reward_matrix(map_idx=i, dir_dataset='new_usc', data_type='train',
    #                            map_size=256, upsampling_factor=8, coverage_thr=170./255)
    # for i in tqdm(range(2, 2 + 32 * 100, 32)):
    #     generate_reward_matrix(map_idx=i, dir_dataset='new_usc', data_type='test',
    #                            map_size=256, upsampling_factor=8, coverage_thr=170./255)
