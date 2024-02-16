import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import logging
from ray.rllib.algorithms.dqn import DQNConfig, DQN
from ray.rllib.algorithms.algorithm import Algorithm
from ray import air, tune
from ray.tune.logger import pretty_print

from env_v0 import BaseEnvironment
from config import config_run_train

# set a logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = logging.FileHandler(f"/Users/ylu/Documents/USC/WiDeS/BS_Deployment/log/runner_dqn.log", encoding='utf-8', mode='a')
formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

dqn_config = (
    DQNConfig()
    .environment(env=BaseEnvironment, env_config=config_run_train.get("env"))
    .framework("torch")
    .rollouts(num_rollout_workers=1, num_envs_per_worker=1)
    .resources(num_gpus=0)
    .exploration(
        explore=True,
        exploration_config=config_run_train["explore"].get("exploration_config", {})
    )
    .training(
        train_batch_size=config_run_train["train"].get("train_batch_size", 1),
        lr=1e-3,
        num_steps_sampled_before_learning_starts=config_run_train["train"].get("num_steps_sampled_before_learning_starts", 10000),
        replay_buffer_config=config_run_train["train"].get("replay_buffer_config"),
        double_q=True,
    )
    .evaluation(
        evaluation_interval=config_run_train["eval"].get("evaluation_interval", 1),
        evaluation_duration=config_run_train["eval"].get("evaluation_duration", 3),
        evaluation_config=config_run_train["eval"].get("evaluation_config", {}),
    )
)
dqn = dqn_config.build()
# dqn.restore('/Users/ylu/Documents/USC/WiDeS/BS_Deployment/checkpoint/dqn_0215_1703')

NUM_TRAINING_STEP = 100

if __name__ == "__main__":
    # eval_res = dqn.evaluate()
    # print(pretty_print(eval_res))

    # evaluation
    ep = np.arange(NUM_TRAINING_STEP)
    ep_reward_mean = np.empty(NUM_TRAINING_STEP, dtype=float)
    ep_reward_std = np.empty(NUM_TRAINING_STEP, dtype=float)
    # training
    ep_reward_mean_train = np.empty(NUM_TRAINING_STEP, dtype=float)

    for i in range(NUM_TRAINING_STEP):
        # one training step (may include multiple environment episodes)
        result = dqn.train()

        print("\n")
        print(f"================training # {i}================")
        print(f"timesteps_total: {result['timesteps_total']}")
        print(f"time_total_s: {result['time_total_s']}")

        if i == NUM_TRAINING_STEP - 1:
            # save the result and checkpoint
            logger.info("=============A WHOLE TRAINING PERIOD ENDED=============")
            logger.info(pretty_print(result))
            logger.debug(config_run_train)
            checkpoint_dir = dqn.save(f"./checkpoint/dqn_{datetime.now().strftime('%m%d_%H%M')}").checkpoint.path
            print(f"Checkpoint saved in directory {checkpoint_dir}")

        # calculate the training mean reward per step
        episodes_this_iter = result["sampler_results"]["episodes_this_iter"]
        ep_len_train = np.array(result["sampler_results"]["hist_stats"]["episode_lengths"][-episodes_this_iter:])
        ep_reward_train = np.array(result["sampler_results"]["hist_stats"]["episode_reward"][-episodes_this_iter:])
        ep_r_per_step = ep_reward_train / ep_len_train
        ep_reward_mean_train[i] = np.mean(ep_r_per_step)

        # calculate the evaluation mean reward per step
        ep_len = np.array(result["evaluation"]["hist_stats"]["episode_lengths"])
        ep_r_sum = np.array(result["evaluation"]["hist_stats"]["episode_reward"])
        ep_r_per_step = ep_r_sum / ep_len
        ep_r_mean, ep_r_std = np.mean(ep_r_per_step), np.std(ep_r_per_step)
        ep_reward_mean[i] = ep_r_mean
        ep_reward_std[i] = ep_r_std

    # plot the mean reward in evaluation
    fig, ax = plt.subplots()
    ax.plot(ep, ep_reward_mean, color="blue")
    sup = list(map(lambda x, y: x + y, ep_reward_mean, ep_reward_std))
    inf = list(map(lambda x, y: x - y, ep_reward_mean, ep_reward_std))
    ax.fill_between(ep, inf, sup, color="blue", alpha=0.2)
    ax.set(xlabel="training_step", ylabel="mean reward per step", title="DQN Evaluation Results")
    ax.grid()
    fig.savefig(f"./figures/dqn_eval_{datetime.now().strftime('%m%d_%H%M')}.png")

    # plot mean reward in training
    fig, ax = plt.subplots()
    ax.plot(ep, ep_reward_mean_train, color='red')
    ax.set(xlabel="training_step", ylabel="mean reward per step", title="DQN Training Results")
    ax.grid()
    fig.savefig(f"./figures/dqn_train_{datetime.now().strftime('%m%d_%H%M')}.png")

    # tuner = tune.Tuner(
    #     "DQN",
    #     param_space=dqn_config.to_dict(),
    #     run_config=air.RunConfig(
    #         stop=config_run_train["stop"]
    #     ),
    # )
    # tuner.fit()
