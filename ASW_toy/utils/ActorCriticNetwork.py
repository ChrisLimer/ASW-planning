import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax.numpy as jnp
import jax
import chex
import numpy as np
from flax import struct
import abc
from jax import lax
import flax.linen as nn
import optax
from typing import NamedTuple

from jax import tree_util
import jax.tree_util as jtu

from dataclasses import dataclass

import flax.serialization
import pathlib
from typing import List
import re

class MLP(nn.Module):
    hidden_dim: int = 128  # Adjustable hidden size

    def get_dummy_input(self):
        N = 1
        return jnp.ones((N, 2, 13))


    def setup(self):

        self.linear_policy_pl1_1 = nn.Dense(features=self.hidden_dim)
        self.linear_policy_pl1_2 = nn.Dense(features=self.hidden_dim)
        self.linear_policy_pl1_3 = nn.Dense(features=13 * 13)

        self.linear_policy_pl2_1 = nn.Dense(features=self.hidden_dim)
        self.linear_policy_pl2_2 = nn.Dense(features=self.hidden_dim)
        self.linear_policy_pl2_3 = nn.Dense(features=1 * 13)

        self.linear_value_1 = nn.Dense(features=self.hidden_dim)
        self.linear_value_2 = nn.Dense(features=self.hidden_dim)
        self.linear_value_3 = nn.Dense(features=1)

        self.linear_value_map_1 = nn.Dense(features=self.hidden_dim)
        self.linear_value_map_2 = nn.Dense(features=self.hidden_dim)
        self.linear_value_map_3 = nn.Dense(features=13)



    @nn.compact
    def __call__(self, x):
        # x shape: (N, 2, 13)
        N = x.shape[0]

        # Flatten second and third dims: (N, 2, 13) → (N, 24)
        x = x.reshape((N, -1))

        # MLP layers
        p_pl1 = self.linear_policy_pl1_1(x)
        p_pl1 = nn.relu(p_pl1)
        p_pl1 = self.linear_policy_pl1_2(p_pl1)
        p_pl1 = nn.relu(p_pl1)
        p_pl1 = self.linear_policy_pl1_3(p_pl1)     # Final output: (N, 14*13)
        logits_pl1 = p_pl1.reshape((N, 13, 13))

        p_pl2 = self.linear_policy_pl2_1(x)
        p_pl2 = nn.relu(p_pl2)
        p_pl2 = self.linear_policy_pl2_2(p_pl2)
        p_pl2 = nn.relu(p_pl2)
        p_pl2 = self.linear_policy_pl2_3(p_pl2)     # Final output: (N, 14*13)
        logits_pl2 = p_pl2.reshape((N, 13))

        # value = nn.Dense(1)(x)       # shape (N, 1)
        v = self.linear_value_1(x)
        v = self.linear_value_2(v)
        v = self.linear_value_3(v)
        value = v.squeeze(-1)    # shape (N,)

        v_map = self.linear_value_map_1(x)
        v_map = self.linear_value_map_2(v_map)
        v_map = self.linear_value_map_3(v_map)

        return logits_pl1, logits_pl2, value, v_map
    
