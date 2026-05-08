"""Custom features extractor for the GridWorld CPP Dict observation.

The observation has three pieces:
    - "local_view":  (k, k) int patch around the agent          -> small CNN
    - "agent_pos":   (2,) float, normalized                     -> small MLP
    - "coverage":    (1,) float                                 -> small MLP

We process the spatial patch with a CNN (which is shape-agnostic across
grid sizes, since the patch shape is fixed by the env), and concatenate it
with an MLP-encoded vector of the scalar features. The combined vector is
then handed off to the policy/value heads (and to the LSTM, if used).

This is the piece that makes the policy generalize across grids: the local
view CNN learns local patterns (walls, frontiers, recently visited cells)
that mean the same thing regardless of overall grid size.
"""

from __future__ import annotations

from typing import Dict

import gymnasium as gym
import torch as th
import torch.nn as nn

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class CPPFeaturesExtractor(BaseFeaturesExtractor):
    """CNN over local_view + MLP over (agent_pos, coverage)."""

    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        cnn_features_dim: int = 64,
        vec_features_dim: int = 32,
    ):
        # Total output size = CNN feats + vec feats. SB3 uses this as
        # `features_dim` for the rest of the policy.
        super().__init__(observation_space, features_dim=cnn_features_dim + vec_features_dim)

        # ---- CNN over the local patch ------------------------------------
        # local_view is (k, k) ints in {0,1,2,3}. We add a channel dim and
        # cast to float in `forward`.
        local_shape = observation_space["local_view"].shape  # e.g. (5, 5)
        n_channels_in = 1

        self.cnn = nn.Sequential(
            nn.Conv2d(n_channels_in, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute CNN flat size with a dummy forward pass.
        with th.no_grad():
            dummy = th.zeros(1, n_channels_in, *local_shape, dtype=th.float32)
            cnn_flat = self.cnn(dummy).shape[1]

        self.cnn_head = nn.Sequential(
            nn.Linear(cnn_flat, cnn_features_dim),
            nn.ReLU(),
        )

        # ---- MLP over (agent_pos, coverage) ------------------------------
        # 2 + 1 = 3 floats in.
        self.vec = nn.Sequential(
            nn.Linear(3, vec_features_dim),
            nn.ReLU(),
            nn.Linear(vec_features_dim, vec_features_dim),
            nn.ReLU(),
        )

    def forward(self, obs: Dict[str, th.Tensor]) -> th.Tensor:
        # local_view comes in as (B, k, k) ints; add channel dim and cast.
        lv = obs["local_view"].float().unsqueeze(1)
        cnn_feat = self.cnn_head(self.cnn(lv))

        vec_in = th.cat(
            [obs["agent_pos"].float(), obs["coverage"].float()], dim=-1
        )
        vec_feat = self.vec(vec_in)

        return th.cat([cnn_feat, vec_feat], dim=-1)
