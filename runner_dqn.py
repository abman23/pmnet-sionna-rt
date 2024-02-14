import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from ray.rllib.algorithms.dqn import DQNConfig
from ray import air, tune
from ray.tune.logger import pretty_print

from env_v0 import BaseEnvironment
from config import config_run, config_run2


dqn_config = (
    DQNConfig()
    .environment(env=BaseEnvironment, env_config=config_run.get("env"))
    .framework("torch")
    .rollouts(num_rollout_workers=1)
    .resources(num_gpus=0)
    # .evaluation(evaluation_interval=config_run["eval"].get("evaluation_interval", 5),
    #             evaluation_duration=config_run["eval"].get("evaluation_duration", 1))
    .training(train_batch_size=config_run["train"].get("train_batch_size"))
)
dqn = dqn_config.build()

dqn_config2 = (
    DQNConfig()
    .environment(env=BaseEnvironment, env_config=config_run2.get("env"))
    .framework("torch")
    .rollouts(num_rollout_workers=1)
    .resources(num_gpus=0)
    .training(train_batch_size=config_run["train"].get("train_batch_size"))
)
dqn2 = dqn_config2.build()

if __name__ == "__main__":
    # for i in range(2):
    #     print("================")
    #     print(f"training # {i}")
    #     print("================")

    # one training step (may include multiple environment episodes)
    result = dqn.train()
    print(pretty_print(result))

    # save the checkpoint
    checkpoint_dir = dqn.save(f"./checkpoint").checkpoint.path
    print(f"Checkpoint saved in directory {checkpoint_dir}")

    # plot the episode mean reward
    ep_len = np.array(result["sampler_results"]["hist_stats"]["episode_lengths"])
    ep_r = np.array(result["sampler_results"]["hist_stats"]["episode_reward"])
    ep_r_mean = ep_r / ep_len

    fig, ax = plt.subplots()
    ax.plot(np.arange(ep_r_mean.size), ep_r_mean)
    ax.set(xlabel="episode", ylabel="episode mean reward", title="DQN Training Results")
    ax.grid()
    fig.savefig(f"./figures/dqn_train_{datetime.now().strftime('%m%d_%H%M')}.png")

    # evaluate the trained model
    env = BaseEnvironment(config=config_run["env"])
    plt.close(fig)

    for i in range(3):
        obs, _ = env.reset()
        terminated, truncated = False, False
        r_ep = []
        n = 0

        while not terminated and not truncated:
            action = dqn.compute_single_action(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            n += 1
            r_ep.append(reward)

            # if info.get("detailed_rewards", None) is not None:
            #     print(info)

        ax.plot(list(range(n)), r_ep, label=f"map # {i}")

    ax.set(xlabel="step", ylabel="reward", title="DQN Evaluation")
    ax.legend()
    ax.grid()
    fig.savefig(f"./figures/dqn_eval_{datetime.now().strftime('%m%d_%H%M')}.png")

    # # one training step (may include multiple environment episodes)
    # result2 = dqn2.train()
    # print(pretty_print(result2))
    #
    # # plot the episode mean reward
    # ep_len = np.array(result2["sampler_results"]["hist_stats"]["episode_lengths"])
    # ep_r = np.array(result2["sampler_results"]["hist_stats"]["episode_reward"])
    # ep_r_mean = ep_r / ep_len
    #
    # fig, ax = plt.subplots()
    # ax.plot(np.arange(ep_r_mean.size), ep_r_mean)
    # ax.set(xlabel="episode", ylabel="episode mean reward", title="DQN Training Results")
    # ax.grid()
    # fig.savefig(f"./figures/dqn2_{datetime.now().strftime('%m%d_%H%M')}.png")
    #
    # # save the checkpoint
    # checkpoint_dir = dqn2.save(f"./checkpoint").checkpoint.path
    # print(f"Checkpoint saved in directory {checkpoint_dir}")

    # tuner = tune.Tuner(
    #     "DQN",
    #     param_space=dqn_config.to_dict(),
    #     run_config=air.RunConfig(
    #         stop=config_run["stop"]
    #     ),
    # )
    # tuner.fit()
