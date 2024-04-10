import os
import time
from datetime import datetime

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, SequentialSampler
import matplotlib.pyplot as plt

from dataset_builder.reward_dataset import RewardDataset
from dataset_builder.reward_model import RewardModel, SimpleCNN
from env.utils_v1 import ROOT_DIR


def train(model: nn.Module, model_name: str, train_loader: DataLoader, test_loader: DataLoader,
          epochs: int = 10, learning_rate: float = 0.001, eval_interval: int = 5, device='cuda',
          save_model: bool = False, comment: str = ''):
    """Training method of the reward model.

    """
    model.to(device)  # Move model to GPU if available
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_losses = []
    test_losses = []
    train_epochs = []
    test_epochs = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for inputs, targets in tqdm(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)  # Move data to GPU if available
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)
        train_epochs.append(epoch)

        if (epoch+1) % eval_interval == 0 or epoch == 0:
            # Evaluate the model on the test set
            test_loss = evaluate(model, test_loader, criterion, device)
            test_losses.append(test_loss)
            test_epochs.append(epoch)

            print(f'Epoch [{epoch + 1}/{epochs}], '
                  f'Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}')

    timestamp = datetime.now().strftime('%m%d_%H%M')

    if save_model:
        # Save the model
        model_dir = os.path.join(ROOT_DIR, 'dataset_builder', 'reward_model')
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        file_path = os.path.join(model_dir, f'pmnet_{timestamp}.pth')
        torch.save(model.state_dict(), file_path)

    # Plotting the loss curves
    plt.figure(figsize=(10, 5))
    plt.plot(train_epochs, train_losses, label='Training Loss')
    plt.plot(test_epochs, test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Test Loss')
    plt.legend()

    plt.savefig(os.path.join(ROOT_DIR, f"figures/reward_model/{model_name}_{timestamp}_{comment}.png"))
    # plt.show()


def evaluate(model: nn.Module, test_loader: DataLoader, criterion: nn.MSELoss, device='cuda'):
    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        for inputs, targets in tqdm(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)  # Move data to GPU if available
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            running_loss += loss.item()

    test_loss = running_loss / len(test_loader)

    return test_loss


if __name__ == '__main__':
    # Prepare datasets and data loaders
    map_dim = 256
    action_dim = 32
    coverage_thr = 240. / 255
    data_train = RewardDataset(dir_dataset='usc_old_sparse', data_type='train',
                               map_indices=np.arange(1, 1 + 1000 * 16, 16),
                               map_dim=map_dim, matrix_dim=action_dim, reward_thr=coverage_thr)
    data_test = RewardDataset(dir_dataset='usc_old_sparse', data_type='test',
                              map_indices=np.arange(2, 2 + 100 * 32, 32),
                              map_dim=map_dim, matrix_dim=action_dim, reward_thr=coverage_thr)

    batch_size_train = 16
    batch_size_test = 16
    num_workers = 4
    sequential_sampler = SequentialSampler(data_test)

    train_loader = DataLoader(data_train, batch_size=batch_size_train, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(data_test, batch_size=batch_size_test, sampler=sequential_sampler, num_workers=num_workers)

    # Train reward model
    # model = RewardModel(n_blocks=[3, 3, 27, 3],
    #                     atrous_rates=[6, 12, 18],
    #                     multi_grids=[1, 2, 4],
    #                     output_stride=8, output_dim=action_dim)
    model = SimpleCNN(output_dim=action_dim**2)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train(model=model, model_name='simple_cnn', train_loader=train_loader, test_loader=test_loader, epochs=100,
          learning_rate=1e-4, eval_interval=5, device=device, save_model=False, comment='lessParam2')
