import json
import os

import numpy as np
import yaml
from matplotlib import pyplot as plt

from env.utils_v1 import ROOT_DIR

# miscellaneous script
if __name__ == '__main__':
    timestamp = '0312_1248'
    reward_data = json.load(open(os.path.join(ROOT_DIR, f'data/ppo_{timestamp}.json')))

    training_iterations = 300
    ep_train = reward_data['ep_train'][:training_iterations]
    ep_reward_mean_train = reward_data['ep_reward_mean'][:training_iterations]

    eval_iterations = training_iterations // 5
    ep_eval = reward_data['ep_eval'][:eval_iterations]
    ep_reward_mean = reward_data['ep_reward_mean'][:eval_iterations]
    ep_reward_std = reward_data['ep_reward_std'][:eval_iterations]

    # plot the mean reward in evaluation
    fig, ax = plt.subplots()
    fig.set_size_inches(10, 6)
    ax.plot(ep_eval, ep_reward_mean, color="blue")
    sup = list(map(lambda x, y: x + y, ep_reward_mean, ep_reward_std))
    inf = list(map(lambda x, y: x - y, ep_reward_mean, ep_reward_std))
    ax.fill_between(ep_eval, inf, sup, color="blue", alpha=0.2)
    ax.set(xlabel="training_step", ylabel="mean reward per step",
           title=f"PPO Evaluation Results")
    ax.grid()
    fig.savefig(f"./figures/v11_ppo_{timestamp}_eval.png")

    # plot mean reward in training
    fig, ax = plt.subplots()
    fig.set_size_inches(10, 6)
    ax.plot(ep_train, ep_reward_mean_train, color='red')
    ax.set(xlabel="training_step", ylabel="mean reward per step",
           title=f"PPO Training Results")
    ax.grid()
    fig.savefig(f"./figures/v11_ppo_{timestamp}_train.png")

    plt.show()
