import time

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
handler = logging.FileHandler(f"./log/runner_dqn.log", encoding='utf-8', mode='a')
formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

dqn_config = (
    DQNConfig()
    .environment(env=BaseEnvironment, env_config=config_run_train.get("env"))
    .framework("torch")
    .rollouts(
        num_rollout_workers=config_run_train["rollout"].get("num_rollout_workers", 1),
        num_envs_per_worker=config_run_train["rollout"].get("num_envs_per_worker", 1),
    )
    .resources(
        num_gpus=config_run_train["resource"].get("num_gpus", 0),
    )
    .exploration(
        explore=True,
        exploration_config=config_run_train["explore"].get("exploration_config", {})
    )
    .training(
        train_batch_size=config_run_train["train"].get("train_batch_size", 1),
        lr=config_run_train["train"].get("lr", 3e-4),
        num_steps_sampled_before_learning_starts=config_run_train["train"].get(
            "num_steps_sampled_before_learning_starts", 10000),
        replay_buffer_config=config_run_train["train"].get("replay_buffer_config"),
    )
    .evaluation(
        evaluation_interval=config_run_train["eval"].get("evaluation_interval", 1),
        evaluation_duration=config_run_train["eval"].get("evaluation_duration", 3),
        evaluation_config=config_run_train["eval"].get("evaluation_config", {}),
    )
    .reporting(
        min_sample_timesteps_per_iteration=config_run_train["report"].get("min_sample_timesteps_per_iteration", 1000))
)
dqn = dqn_config.build()

NUM_TRAINING_STEP = config_run_train["stop"].get("training_iteration", 10)
EVAL_INTERVAL = config_run_train["eval"].get("evaluation_interval", 1)


def eval_plot(agent: Algorithm, env_config: dict, duration: int = 3):
    """Plot the agent action (TX location) and the optimal TX location in a same map, given a new env.

    """
    env_config["evaluation"] = True
    env_config["preset_map_path"] = None
    env_config["eval_plot"] = True
    env_config["n_maps"] = 3
    env_eval = BaseEnvironment(config=env_config)

    for i in range(duration):
        term, trunc = False, False
        obs, _ = env_eval.reset()
        while not (term or trunc):
            action = agent.compute_single_action(obs)
            obs, reward, term, trunc, info = env_eval.step(action)


def train_and_eval(agent: Algorithm, num_training_step: int, eval_interval: int, algo_name: str = 'dqn'):
    # evaluation data
    ep_eval = np.arange(0, num_training_step, num_training_step) + eval_interval
    print(f"ep_eval: {ep_eval}")
    ep_reward_mean = np.empty(num_training_step // eval_interval, dtype=float)
    ep_reward_std = np.empty(num_training_step // eval_interval, dtype=float)
    # training data
    ep_train = np.arange(num_training_step)
    ep_reward_mean_train = np.empty(num_training_step, dtype=float)

    for i in range(num_training_step):
        if i + 1 % eval_interval == 0:
            logger.info(f"================EVALUATION AT # {i + 1}================")

        # one training step (may include multiple environment episodes)
        result = agent.train()

        print("\n")
        print(f"================training # {i}================")
        print(f"timesteps_total: {result['timesteps_total']}")
        print(f"time_total_s: {result['time_total_s']}")

        if i == num_training_step - 1:
            # save the result and checkpoint
            logger.info("=============A WHOLE TRAINING PERIOD ENDED=============")
            logger.info(pretty_print(result))
            logger.debug(config_run_train)
            checkpoint_dir = agent.save(f"./checkpoint/{algo_name}_{datetime.now().strftime('%m%d_%H%M')}").checkpoint.path
            print(f"Checkpoint saved in directory {checkpoint_dir}")

        # calculate the training mean reward per step
        episodes_this_iter = result["sampler_results"]["episodes_this_iter"]
        ep_len_train = np.array(result["sampler_results"]["hist_stats"]["episode_lengths"][-episodes_this_iter:])
        ep_reward_train = np.array(result["sampler_results"]["hist_stats"]["episode_reward"][-episodes_this_iter:])
        ep_r_per_step = ep_reward_train / ep_len_train
        ep_reward_mean_train[i] = np.mean(ep_r_per_step)

        if (i + 1) % eval_interval == 0:
            # calculate the evaluation mean reward per step
            ep_len = np.array(result["evaluation"]["hist_stats"]["episode_lengths"])
            ep_r_sum = np.array(result["evaluation"]["hist_stats"]["episode_reward"])
            # print(f"ep_len: {ep_len}")
            # print(f"ep_r_sum: {ep_r_sum}")
            ep_r_per_step = ep_r_sum / ep_len
            ep_r_mean, ep_r_std = np.mean(ep_r_per_step), np.std(ep_r_per_step)
            idx = (i + 1) // eval_interval - 1
            ep_reward_mean[idx] = ep_r_mean
            ep_reward_std[idx] = ep_r_std

    # plot the mean reward in evaluation
    fig, ax = plt.subplots()
    ax.plot(ep_eval, ep_reward_mean, color="blue")
    sup = list(map(lambda x, y: x + y, ep_reward_mean, ep_reward_std))
    inf = list(map(lambda x, y: x - y, ep_reward_mean, ep_reward_std))
    ax.fill_between(ep_eval, inf, sup, color="blue", alpha=0.2)
    ax.set(xlabel="training_step", ylabel="mean reward per step", title=f"{algo_name.upper()} Evaluation Results")
    ax.grid()
    fig.savefig(f"./figures/{algo_name}_eval_{datetime.now().strftime('%m%d_%H%M')}.png")

    # plot mean reward in training
    fig, ax = plt.subplots()
    ax.plot(ep_train, ep_reward_mean_train, color='red')
    ax.set(xlabel="training_step", ylabel="mean reward per step", title=f"{algo_name.upper()} Training Results")
    ax.grid()
    fig.savefig(f"./figures/{algo_name}_train_{datetime.now().strftime('%m%d_%H%M')}.png")


if __name__ == "__main__":
    start = time.time()

    # plot the action (TX location) of the trained agent vs. the optimal TX location
    dqn.restore("./checkpoint/dqn_0220_1342")
    eval_plot(dqn, config_run_train.get("env"))

    # train the agent and evaluate every some steps
    train_and_eval(dqn, NUM_TRAINING_STEP, EVAL_INTERVAL, 'dqn')
    # tuner = tune.Tuner(
    #     "DQN",
    #     param_space=dqn_config.to_dict(),
    #     run_config=air.RunConfig(
    #         stop=config_run_train["stop"]
    #     ),
    # )
    # tuner.fit()

    end = time.time()
    print(f"total runtime: {end - start}s")
