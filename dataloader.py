
from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, datasets, models
import warnings

class PMnet_data_usc(Dataset):
    def __init__(self,
                 dir_dataset="",
                 transform= transforms.ToTensor()):

        self.dir_dataset = dir_dataset
        self.transform = transform
        self.png_list = os.listdir(dir_dataset + "Data/cropped/power_map/")

    def __len__(self):
        return len(self.png_list)

    def __getitem__(self, idx):
        #Load city map
        self.dir_buildings = self.dir_dataset+ "Data/cropped/city_map/"
        img_name_buildings = os.path.join(self.dir_buildings, f"{self.png_list[idx]}")
        image_buildings = np.asarray(io.imread(img_name_buildings))

        #Load Tx (transmitter):
        self.dir_Tx = self.dir_dataset+ "Data/cropped/tx_map/"
        img_name_Tx = os.path.join(self.dir_Tx, f"{self.png_list[idx]}")
        image_Tx = np.asarray(io.imread(img_name_Tx))

        #Load Power:
        self.dir_power = self.dir_dataset+ "Data/cropped/power_map/"
        img_name_power = os.path.join(self.dir_power, f"{self.png_list[idx]}")
        image_power = np.asarray(io.imread(img_name_power))

        inputs=np.stack([image_buildings, image_Tx], axis=2)

        if self.transform:
            inputs = self.transform(inputs).type(torch.float32)
            power = self.transform(image_power).type(torch.float32)

        return [inputs , power]
