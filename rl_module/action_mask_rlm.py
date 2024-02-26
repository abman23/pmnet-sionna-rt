from typing import Mapping, Any

import gymnasium as gym
import torch as th

from ray.rllib.algorithms.ppo.torch.ppo_torch_rl_module import PPOTorchRLModule
from ray.rllib.core.rl_module.rl_module import RLModuleConfig
from ray.rllib.core.rl_module.torch import TorchRLModule
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.nested_dict import NestedDict
from ray.rllib.utils.torch_utils import FLOAT_MIN


class BaseActionMaskRLM(TorchRLModule):
    """Base class for RL module with action masking.

    """

    def __init__(self, config: RLModuleConfig):
        if not isinstance(config.observation_space, gym.spaces.Dict):
            raise ValueError(
                "This model requires the environment to provide a "
                "gym.spaces.Dict observation space."
            )
        # extract only the observations part as the observation space for the default model
        config.observation_space = config.observation_space["observations"]

        super().__init__(config)
        self.mask_forward_fn = mask_forward_fn


class PPOActionMaskRLM(BaseActionMaskRLM, PPOTorchRLModule):
    """PPO RL module with action masking

    """

    def _forward_inference(self, batch: NestedDict, **kwargs) -> Mapping[str, Any]:
        return self.mask_forward_fn(super()._forward_inference, batch, **kwargs)

    def _forward_exploration(self, batch: NestedDict, **kwargs) -> Mapping[str, Any]:
        return self.mask_forward_fn(super()._forward_exploration, batch, **kwargs)

    def _forward_train(self, batch: NestedDict, **kwargs) -> Mapping[str, Any]:
        return self.mask_forward_fn(super()._forward_train, batch, **kwargs)


def mask_forward_fn(forward_fn, batch, **kwargs):
    """Forward function that masks actions

    Args:
        forward_fn: inference/exploration/train.
        batch: Input experience batch.
        **kwargs:

    Returns:
        The forward function output (logits corresponding to probability distribution of actions) with masking.

    """
    # Extract the available actions tensor from the observation.
    action_mask = batch[SampleBatch.OBS]["action_mask"]

    # Modify the incoming batch so that the default models can compute logits and values as usual.
    batch[SampleBatch.OBS] = batch[SampleBatch.OBS]["observations"]
    outputs = forward_fn(batch, **kwargs)

    # Mask logits
    logits = outputs[SampleBatch.ACTION_DIST_INPUTS]
    # Convert action_mask into a [0.0 || -inf]-type mask.
    inf_mask = th.clamp(th.log(action_mask), min=FLOAT_MIN)
    masked_logits = logits + inf_mask

    # Replace original values with masked values.
    outputs[SampleBatch.ACTION_DIST_INPUTS] = masked_logits

    return outputs
