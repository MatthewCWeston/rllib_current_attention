import gymnasium as gym
import numpy as np

from ray.rllib.core.models.base import Encoder, Model
from ray.rllib.core.models.catalog import Catalog
from ray.rllib.core.models.configs import (
    MLPHeadConfig,
)
from ray.rllib.utils import override
from ray.rllib.utils.annotations import OverrideToImplementCustomLogic

import dataclasses
from ray.rllib.core.rl_module.default_model_config import DefaultModelConfig

class SharedCriticCatalog(Catalog):
    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,  # TODO: Remove?
        model_config_dict: dict,
    ):
        """Initializes the PPOCatalog.

        Args:
            observation_space: The observation space of the Encoder.
            action_space: The action space for the Pi Head.
            model_config_dict: The model config to use.
        """
        if dataclasses.is_dataclass(model_config_dict):
            model_config_dict = dataclasses.asdict(model_config_dict)
        default_config = dataclasses.asdict(DefaultModelConfig())
        self._model_config_dict = default_config | model_config_dict
        observation_spaces = self._model_config_dict["observation_spaces"]
        obs_size = 0
        low, high = [], []
        for agent, obs in observation_spaces.items():
            obs_size += obs.shape[0]  # Assume 1D Box observations
            low.append(obs.low)
            high.append(obs.high)
        # Join observation spaces together
        self.observation_space = gym.spaces.Box(np.hstack(low), np.hstack(high), (obs_size,),)
        self.action_space = action_space
        self._latent_dims = None
        self._determine_components_hook()
        # We only want one encoder, so we use the base encoder config.
        self.encoder_config = self._encoder_config
        # Value head architecture
        self.vf_head_hiddens = self._model_config_dict["head_fcnet_hiddens"]
        self.vf_head_activation = self._model_config_dict["head_fcnet_activation"]
        self.vf_head_config = MLPHeadConfig(
            input_dims=self.latent_dims,
            hidden_layer_dims=self.vf_head_hiddens,
            hidden_layer_activation=self.vf_head_activation,
            output_layer_activation="linear",
            output_layer_dim=len(observation_spaces),  # 1 value pred. per agent
        )

    @override(Catalog)
    def build_encoder(self, framework: str) -> Encoder:
        """Builds the encoder."""
        return self.encoder_config.build(framework=framework)

    @OverrideToImplementCustomLogic
    def build_vf_head(self, framework: str) -> Model:
        """Builds the value function head.

        The default behavior is to build the head from the vf_head_config.
        This can be overridden to build a custom value function head as a means of
        configuring the behavior of a MAPPORLModule implementation.

        Args:
            framework: The framework to use. Either "torch" or "tf2".

        Returns:
            The value function head.
        """
        return self.vf_head_config.build(framework=framework)


# __sphinx_doc_end__
