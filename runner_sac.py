import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import logging
from ray.rllib.algorithms.sac import SACConfig
from ray.rllib.algorithms.algorithm import Algorithm
from ray import air, tune
from ray.tune.logger import pretty_print

from env_v0 import BaseEnvironment
from config import config_run_sac

# set a logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = logging.FileHandler(f"/Users/ylu/Documents/USC/WiDeS/BS_Deployment/log/runner_sac.log", encoding='utf-8', mode='a+')
formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

sac_config = (
    SACConfig()
    .environment(env=BaseEnvironment, env_config=config_run_sac.get("env"))
    .framework("torch")
    .rollouts(num_rollout_workers=1, num_envs_per_worker=1)
    .resources(num_gpus=0)
    .exploration(
        explore=True,
        exploration_config=config_run_sac["explore"].get("exploration_config", {})
    )
    .training(
        train_batch_size=config_run_sac["train"].get("train_batch_size", 1),
        num_steps_sampled_before_learning_starts=config_run_sac["train"].get("num_steps_sampled_before_learning_starts", 10000),
        replay_buffer_config=config_run_sac["train"].get("replay_buffer_config"),
        target_network_update_freq=5,
    )
    .evaluation(
        evaluation_interval=config_run_sac["eval"].get("evaluation_interval", 1),
        evaluation_duration=config_run_sac["eval"].get("evaluation_duration", 3),
        evaluation_config=config_run_sac["eval"].get("evaluation_config", {}),
    )
    .reporting(min_sample_timesteps_per_iteration=config_run_sac["report"].get("min_sample_timesteps_per_iteration", 1000))
)
sac = sac_config.build()

NUM_TRAINING_STEP = 100

if __name__ == "__main__":
    # evaluation
    ep = np.arange(NUM_TRAINING_STEP)
    ep_reward_mean = np.empty(NUM_TRAINING_STEP, dtype=float)
    ep_reward_std = np.empty(NUM_TRAINING_STEP, dtype=float)
    # training
    ep_reward_mean_train = np.empty(NUM_TRAINING_STEP, dtype=float)

    for i in range(NUM_TRAINING_STEP):
        # one training step (may include multiple environment episodes)
        result = sac.train()

        print("\n")
        print(f"================training # {i}================")
        print(f"timesteps_total: {result['timesteps_total']}")
        print(f"time_total_s: {result['time_total_s']}")

        if i == NUM_TRAINING_STEP - 1:
            # save the result and checkpoint
            logger.info("=============A WHOLE TRAINING PERIOD ENDED=============")
            logger.info(pretty_print(result))
            logger.debug(config_run_sac)
            checkpoint_dir = sac.save(f"./checkpoint/sac_{datetime.now().strftime('%m%d_%H%M')}").checkpoint.path
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
    ax.set(xlabel="training_step", ylabel="mean reward per step", title="SAC Evaluation Results")
    ax.grid()
    fig.savefig(f"./figures/sac_eval_{datetime.now().strftime('%m%d_%H%M')}.png")

    # plot mean reward in training
    fig, ax = plt.subplots()
    ax.plot(ep, ep_reward_mean_train, color='red')
    ax.set(xlabel="training_step", ylabel="mean reward per step", title="SAC Training Results")
    ax.grid()
    fig.savefig(f"./figures/sac_train_{datetime.now().strftime('%m%d_%H%M')}.png")
