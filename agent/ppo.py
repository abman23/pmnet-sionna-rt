from ray.rllib.algorithms import PPOConfig

from agent.agent import Agent
from env_v0 import BaseEnvironment


class PPOAgent(Agent):
    def __init__(self, config: dict, log_file: str) -> None:
        super().__init__(config, log_file)

        ppo_config = (
            PPOConfig()
            .environment(env=BaseEnvironment, env_config=config.get("env"))
            .framework("torch")
            .rollouts(
                num_rollout_workers=config["rollout"].get("num_rollout_workers", 1),
                num_envs_per_worker=config["rollout"].get("num_envs_per_worker", 1),
            )
            .resources(
                num_gpus=config["resource"].get("num_gpus", 0),
            )
            .exploration(
                explore=True,
                exploration_config=config["explore"].get("exploration_config", {})
            )
            .training(
                train_batch_size=config["train"].get("train_batch_size", 5000),
                lr=config["train"].get("lr", 3e-4),
                gamma=config["train"].get("gamma", 0.95),
                grad_clip=config["train"].get("grad_clip"),
                sgd_minibatch_size=config["train"].get("sgd_minibatch_size", 128),
                num_sgd_iter=config["train"].get("num_sgd_iter", 30),
            )
            .evaluation(
                evaluation_interval=config["eval"].get("evaluation_interval", 1),
                evaluation_duration=config["eval"].get("evaluation_duration", 3),
                evaluation_config=config["eval"].get("evaluation_config", {}),
            )
            .reporting(
                min_sample_timesteps_per_iteration=config["report"].get("min_sample_timesteps_per_iteration", 1000)
            )
        )
        self.agent = ppo_config.build()
        self.algo_name = 'ppo'
