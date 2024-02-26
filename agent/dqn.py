from ray.rllib.algorithms import DQNConfig

from agent.agent import Agent


class DQNAgent(Agent):
    def __init__(self, config: dict, log_file: str) -> None:
        super().__init__(config, log_file)

        dqn_config = (
            DQNConfig()
            .environment(env=self.env_class, env_config=config.get("env"))
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
                train_batch_size=config["train"].get("train_batch_size", 1),
                tau=config["train"].get("tau", 5e-3),
                target_network_update_freq=config["train"].get("target_network_update_freq", 0),
                lr=config["train"].get("lr", 1e-4),
                gamma=config["train"].get("gamma", 0.95),
                num_steps_sampled_before_learning_starts=config["train"].get(
                    "num_steps_sampled_before_learning_starts", 10000),
                replay_buffer_config=config["train"].get("replay_buffer_config"),
            )
            .evaluation(
                evaluation_interval=config["eval"].get("evaluation_interval", 1),
                evaluation_duration=config["eval"].get("evaluation_duration", 3),
                evaluation_config=config["eval"].get("evaluation_config", {}),
                evaluation_num_workers=config["eval"].get("evaluation_num_workers", 3),
            )
            .reporting(
                min_sample_timesteps_per_iteration=config["report"].get("min_sample_timesteps_per_iteration", 1000)
            )
            .experimental(
                _enable_new_api_stack=True,
                _disable_preprocessor_api=True,
            )
        )
        self.agent = dqn_config.build()
        self.algo_name = 'dqn'

