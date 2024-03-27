import torch
import torch.nn as nn
from gymnasium.spaces import Dict, Tuple

from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC
from ray.rllib.utils.torch_utils import FLOAT_MIN


class ActionMaskPolicyModel(TorchModelV2, nn.Module):
    """Policy Model that handles simple discrete action masking.

    """
    def __init__(
        self,
        obs_space,
        action_space,
        num_outputs,
        model_config,
        name
    ):
        orig_space = getattr(obs_space, "original_space", obs_space)
        assert (
            isinstance(orig_space, Dict)
            and "action_mask" in orig_space.spaces
            and "observations" in orig_space.spaces
        )

        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        self.internal_model = TorchFC(
            orig_space["observations"],
            action_space,
            num_outputs,
            model_config,
            name + "_internal",
        )

        # disable action masking --> will likely lead to invalid actions
        self.masked_value = FLOAT_MIN
        if "masked_value" in model_config["custom_model_config"]:
            self.masked_value = model_config["custom_model_config"]["masked_value"]

    def forward(self, input_dict, state, seq_lens):
        # Extract the available actions tensor from the observation.
        action_mask = input_dict["obs"]["action_mask"]

        # Compute the unmasked logits.
        logits, _ = self.internal_model({"obs": input_dict["obs"]["observations"]})

        # Convert action_mask into a [0.0 || -inf]-type mask.
        inf_mask = torch.clamp(torch.log(action_mask), min=self.masked_value)
        masked_logits = logits + inf_mask

        # Return masked logits.
        return masked_logits, state

    def value_function(self):
        return self.internal_model.value_function()


class ActionMaskQModel(TorchModelV2, nn.Module):
    """Q model when using action masking.

    """
    def __init__(
            self,
            obs_space,
            action_space,
            num_outputs,
            model_config,
            name
    ):
        orig_space = getattr(obs_space, "original_space", obs_space)
        assert (
                isinstance(orig_space, Dict)
                and "action_mask" in orig_space.spaces
                and "observations" in orig_space.spaces
        )

        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        # Use only the observations part as observation space
        self.internal_model = TorchFC(
            orig_space["observations"],
            action_space,
            num_outputs,
            model_config,
            name + "_internal",
        )

    def forward(self, input_dict, state, seq_lens):
        # Compute the logits.
        logits, _ = self.internal_model({"obs": input_dict["obs"]["observations"]})

        return logits, state

    def value_function(self):
        return self.internal_model.value_function()
