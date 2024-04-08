import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
torch.set_default_dtype(torch.float32)
# torch.set_default_tensor_type('torch.cuda.FloatTensor')
# torch.backends.cudnn.enabled

from tqdm import tqdm
from datetime import datetime
#import pytorch_model_summary

import cv2

from Network.pmnet_v3 import PMNet
from config import config_USC_pmnetV3_V2
from dataloader import PMnet_data_usc
from loss import L1_loss, MSE, RMSE

import argparse

def eval_model(model, test_loader, error="RMSE", best_val=100, cfg=None, eval_mode=False, infer_img_path=''):
    # Set model to evaluate mode
    model.eval()

    n_samples = 0
    avg_loss = 0

    # check dataset type
    pred_cnt=1 # start from 1
    
    for inputs, targets in tqdm(test_loader):
        inputs = inputs.cuda()
        targets = targets.cuda()


        with torch.set_grad_enabled(False):
            if error == "MSE":
                criterion = MSE
            elif error == "RMSE":
                criterion = RMSE
            elif error == "L1_loss":
                criterion = L1_loss

            preds = model(inputs)
            preds = torch.clip(preds, 0, 1)

            loss = criterion(preds, targets)
            # NMSE

            avg_loss += (loss.item() * inputs.shape[0])
            n_samples += inputs.shape[0]

            # inference image
            if infer_img_path!='':
                for i in range(len(preds)):
                    tx_map = inputs[i, 1].cpu().detach().numpy() * 255
                    plt.figure(figsize=(15,10))
                    plt.subplot(1,3,1)
                    plt.axis("off")
                    plt.title("Ground Truth")
                    left=(targets[i].cpu().detach().numpy()*255 + tx_map)[0][:,::-1]
                    plt.imshow(left, cmap='gray', vmin=0, vmax=255)

                    plt.subplot(1,3,2)
                    plt.axis("off")
                    plt.title("Predicted Image")
                    right=(preds[i, 0].cpu().detach().numpy()*255 + tx_map)[:,::-1]
                    plt.imshow(right, cmap='gray', vmin=0, vmax=255)

                    img_name=os.path.join(infer_img_path,f'{pred_cnt}.png')
                    plt.savefig(img_name)
                    pred_cnt+=1
                    if pred_cnt%100==0:
                        print(f'{img_name} saved')


    avg_loss = avg_loss / (n_samples + 1e-7)

    model.train()
    return avg_loss, best_val


def createDirectory(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print("Error: Failed to create the directory.")

def helper(cfg, data_root = '', load_model=''):

    if cfg.sampling == 'exclusive':
        data_usc_train = PMnet_data_usc(dir_dataset=data_root)

        dataset_size = len(data_usc_train)

        train_size = int(dataset_size * cfg.train_ratio)
        # validation_size = int(dataset_size * 0.1)
        test_size = dataset_size - train_size
        train_dataset, test_dataset = random_split(data_usc_train, [train_size, test_size])

        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=True, num_workers=8)
    elif cfg.sampling == 'random':
        pass

    # init model
    model = PMNet(
        n_blocks=[3, 3, 27, 3],
        atrous_rates=[6, 12, 18],
        multi_grids=[1, 2, 4],
        output_stride=8,)
    model.cuda()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load(load_model))
    model.to(device)

    # create inference images directory if not exist
    createDirectory(RESULT_FOLDER)

    result = eval_model(model, test_loader, error="RMSE", best_val=100, cfg=None, eval_mode=True,
                        infer_img_path=RESULT_FOLDER)
    print('Evaluation score(RMSE): ', result)
    '''End evaluation region'''

if __name__ == "__main__":
    split_to_eval_score_dict = {}

    data_root = ""

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_to_eval', type=str, help='Pretrained model to evaluate.')
    args, unknown = parser.parse_known_args()
    RESULT_FOLDER = f'{data_root}PMNet_results/inference_images/{os.path.split(args.model_to_eval)[1]}/'

    print('start')
    cfg = config_USC_pmnetV3_V2()
    cfg.now = datetime.today().strftime("%Y%m%d%H%M") 


    os.makedirs(RESULT_FOLDER, exist_ok=True)
    

    helper(cfg, data_root, load_model=args.model_to_eval)

    print(f"Results are saved in folder: {RESULT_FOLDER}")


