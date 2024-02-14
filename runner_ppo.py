import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from ray.rllib.algorithms.ppo import PPOConfig
from ray import air, tune
from ray.tune.logger import pretty_print

from env_v0 import BaseEnvironment
from config import config_run, config_run2


ppo_config = (
    PPOConfig()
    .environment(env=BaseEnvironment, env_config=config_run.get("env"))
    .framework("torch")
    .rollouts(num_rollout_workers=1)
    .resources(num_gpus=0)
    # .evaluation(evaluation_interval=config_run["eval"].get("evaluation_interval", 5),
    #             evaluation_duration=config_run["eval"].get("evaluation_duration", 1))
    .training(train_batch_size=config_run["train"].get("train_batch_size"))
)
ppo = ppo_config.build()

ppo_config2 = (
    PPOConfig()
    .environment(env=BaseEnvironment, env_config=config_run2.get("env"))
    .framework("torch")
    .rollouts(num_rollout_workers=1)
    .resources(num_gpus=0)
    .training(train_batch_size=config_run["train"].get("train_batch_size"))
)
ppo2 = ppo_config2.build()

if __name__ == "__main__":
    # for i in range(2):
    #     print("================")
    #     print(f"training # {i}")
    #     print("================")

    # one training step (may include multiple environment episodes)
    result = ppo.train()
    print(pretty_print(result))

    # save the checkpoint
    checkpoint_dir = ppo.save(f"./checkpoint").checkpoint.path
    print(f"Checkpoint saved in directory {checkpoint_dir}")

    # plot the episode mean reward
    ep_len = np.array(result["sampler_results"]["hist_stats"]["episode_lengths"])
    ep_r = np.array(result["sampler_results"]["hist_stats"]["episode_reward"])
    ep_r_mean = ep_r / ep_len

    fig, ax = plt.subplots()
    ax.plot(np.arange(ep_r_mean.size), ep_r_mean)
    ax.set(xlabel="episode", ylabel="episode mean reward", title="PPO Training Results")
    ax.grid()
    fig.savefig(f"./figures/ppo_train_{datetime.now().strftime('%m%d_%H%M')}.png")

    # evaluate the trained model
    env = BaseEnvironment(config=config_run["env"])
    plt.close(fig)

    for i in range(3):
        obs, _ = env.reset()
        terminated, truncated = False, False
        r_ep = []
        n = 0

        while not terminated and not truncated:
            action = ppo.compute_single_action(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            n += 1
            r_ep.append(reward)

            # if info.get("detailed_rewards", None) is not None:
            #     print(info)

        ax.plot(list(range(n)), r_ep, label=f"map # {i}")

    ax.set(xlabel="step", ylabel="reward", title="A2C Evaluation")
    ax.legend()
    ax.grid()
    fig.savefig(f"./figures/ppo_eval_{datetime.now().strftime('%m%d_%H%M')}.png")


