from typing import Dict, List
import gymnasium as gym
import torch as th
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.typing import TensorType, ModelConfigDict


class MyDQN(TorchModelV2):
    """Custom model for the deep Q-learning algorithm, implementing action masking.

    """
    def __init__(self, obs_space: gym.spaces.Space, action_space: gym.spaces.Space, num_outputs: int,
                 model_config: ModelConfigDict, name: str):
        super().__init__(obs_space, action_space, num_outputs, model_config, name)

    def forward(self, input_dict: Dict[str, TensorType], state: List[TensorType], seq_lens: TensorType) -> (
    TensorType, List[TensorType]):
        pass

    def value_function(self) -> TensorType:
        pass