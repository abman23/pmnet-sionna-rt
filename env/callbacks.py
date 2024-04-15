from typing import Optional, Union, Dict
import gymnasium as gym
import numpy as np

from ray.rllib import RolloutWorker, BaseEnv, Policy
from ray.rllib.algorithms import Algorithm
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.core.rl_module import RLModule
from ray.rllib.evaluation import Episode
from ray.rllib.evaluation.episode_v2 import EpisodeV2
from ray.rllib.utils.typing import EpisodeType, PolicyID
from ray.rllib.env.env_runner import EnvRunner
from ray.tune.logger import pretty_print


class AsyncActionCallbacks(DefaultCallbacks):
    """Custom callbacks used for asynchronous BS deployment,
    which mainly computes nad stores the sum of all BSs' rewards in one round.

    """
    def on_episode_start(
            self,
            *,
            episode: Union[EpisodeType, Episode, EpisodeV2],
            worker: Optional["EnvRunner"] = None,
            env_runner: Optional["EnvRunner"] = None,
            base_env: Optional[BaseEnv] = None,
            env: Optional[gym.Env] = None,
            policies: Optional[Dict[PolicyID, Policy]] = None,
            rl_module: Optional[RLModule] = None,
            env_index: int,
            **kwargs,
    ):
        episode.hist_data["reward_per_round"] = []


    def on_episode_end(
            self,
            *,
            episode: Union[EpisodeType, Episode, EpisodeV2],
            worker: Optional["EnvRunner"] = None,
            env_runner: Optional["EnvRunner"] = None,
            base_env: Optional[BaseEnv] = None,
            env: Optional[gym.Env] = None,
            policies: Optional[Dict[PolicyID, Policy]] = None,
            rl_module: Optional[RLModule] = None,
            env_index: int,
            **kwargs,
    ):
        accumulated_reward = episode.last_info_for()["accumulated_reward"]
        # print(f"accumulated_reward: {accumulated_reward}")
        n_bs, steps = episode.last_info_for()["n_bs"], episode.last_info_for()["steps"]
        n_rounds = np.ceil(steps / n_bs)
        # print(f"n_rounds: {n_rounds}")
        reward_per_round = accumulated_reward / n_rounds
        episode.custom_metrics["reward_per_round"] = reward_per_round
        episode.hist_data['reward_per_round'].append(reward_per_round)

    def on_train_result(self, *, algorithm, result: dict, **kwargs):
        # Normally, RLlib would aggregate any custom metric into a mean, max and min
        # of the given metric.
        # print(result["custom_metrics"])
        num_episodes = result["episodes_this_iter"]
        reward_per_round = result['sampler_results']["hist_stats"]["reward_per_round"][-num_episodes:]
        # print(f"reward_per_round: {reward_per_round}")
        std = np.std(reward_per_round)
        mean = np.mean(reward_per_round)
        result["custom_metrics"]["reward_per_round_std"] = std
        result["custom_metrics"]["reward_per_round_mean"] = mean
        # print(pretty_print(result))

    def on_evaluate_end(self, *, algorithm: "Algorithm", evaluation_metrics: dict, **kwargs) -> None:
        """Runs when the evaluation is done.

        Runs at the end of Algorithm.evaluate().

        Args:
            algorithm: Reference to the algorithm instance.
            evaluation_metrics: Results dict to be returned from algorithm.evaluate().
                You can mutate this object to add additional metrics.
            kwargs: Forward compatibility placeholder.
        """
        num_episodes = evaluation_metrics['evaluation']['sampler_results']["episodes_this_iter"]
        reward_per_round = evaluation_metrics['evaluation']['sampler_results']["hist_stats"]["reward_per_round"][-num_episodes:]
        std = np.std(reward_per_round)
        mean = np.mean(reward_per_round)
        evaluation_metrics['evaluation']["custom_metrics"]["reward_per_round_std"] = std
        evaluation_metrics['evaluation']["custom_metrics"]["reward_per_round_mean"] = mean
        # print(pretty_print(evaluation_metrics))
