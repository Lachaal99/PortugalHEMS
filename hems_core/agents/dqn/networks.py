"""
Neural network architectures for the HEMS DQN agent.

Architecture: Dueling Double DQN
─────────────────────────────────────────────────────────────────────────────
Input  (9,)
  │
  ▼
Shared Trunk  – two fully-connected layers with LayerNorm + LeakyReLU
  │
  ├──► Value stream    → scalar  V(s)
  │
  └──► Advantage stream → vector  A(s, a)  [N_ACTIONS]
  │
  ▼
Q(s, a) = V(s)  +  A(s, a)  −  mean_a[ A(s, a) ]

Why Dueling?
  Energy management rewards differ mostly in state value (e.g., cheap vs
  expensive hour) rather than action advantage.  The dueling split lets the
  value stream learn "this is a costly hour" without needing every (s,a)
  pair to be visited.

Why LayerNorm?
  Input features have very different scales even after normalization; LN
  stabilises gradient flow and removes the need for careful weight init.

Why LeakyReLU?
  Avoids dead neurons, which appear with saturating ReLU when rewards are
  predominantly negative (grid cost).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

from .config import (
    STATE_DIM, N_ACTIONS,
    HIDDEN_SIZES, VALUE_HIDDEN, ADVANTAGE_HIDDEN,
)


# ---------------------------------------------------------------------------
# Utility: build an MLP block
# ---------------------------------------------------------------------------

def _mlp_block(in_dim: int, out_dim: int, use_layer_norm: bool = True) -> nn.Sequential:
    layers: List[nn.Module] = [nn.Linear(in_dim, out_dim)]
    if use_layer_norm:
        layers.append(nn.LayerNorm(out_dim))
    layers.append(nn.LeakyReLU(0.1, inplace=True))
    return nn.Sequential(*layers)


# ---------------------------------------------------------------------------
# Dueling DQN
# ---------------------------------------------------------------------------

class DuelingDQN(nn.Module):
    """
    Dueling Double DQN for a discretized continuous action space.

    Parameters
    ----------
    state_dim   : dimension of the observation vector (default 9)
    n_actions   : total number of discrete joint actions  (default 49)
    hidden_sizes: list of hidden layer widths for the shared trunk
    value_hidden: hidden width for the value stream
    adv_hidden  : hidden width for the advantage stream
    """

    def __init__(
        self,
        state_dim:    int       = STATE_DIM,
        n_actions:    int       = N_ACTIONS,
        hidden_sizes: List[int] = HIDDEN_SIZES,
        value_hidden: int       = VALUE_HIDDEN,
        adv_hidden:   int       = ADVANTAGE_HIDDEN,
    ):
        super().__init__()

        # ── Shared trunk ──────────────────────────────────────────────────
        trunk_layers: List[nn.Module] = []
        in_dim = state_dim
        for h in hidden_sizes:
            trunk_layers.append(_mlp_block(in_dim, h, use_layer_norm=True))
            in_dim = h
        self.trunk = nn.Sequential(*trunk_layers)
        trunk_out = in_dim   # == hidden_sizes[-1]

        # ── Value stream  V(s) ────────────────────────────────────────────
        self.value_stream = nn.Sequential(
            _mlp_block(trunk_out, value_hidden, use_layer_norm=True),
            nn.Linear(value_hidden, 1),
        )

        # ── Advantage stream  A(s, a) ─────────────────────────────────────
        self.advantage_stream = nn.Sequential(
            _mlp_block(trunk_out, adv_hidden, use_layer_norm=True),
            nn.Linear(adv_hidden, n_actions),
        )

        # Weight initialisation (orthogonal for stability)
        self.apply(self._init_weights)

    # ------------------------------------------------------------------
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=1.0)
            nn.init.zeros_(module.bias)

    # ------------------------------------------------------------------
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        state : (batch, state_dim)  float32 tensor

        Returns
        -------
        q_values : (batch, n_actions)
        """
        features  = self.trunk(state)              # (B, trunk_out)
        value     = self.value_stream(features)    # (B, 1)
        advantage = self.advantage_stream(features)# (B, A)

        # Q = V + A - mean(A)  — zero-mean normalisation for identifiability
        q = value + advantage - advantage.mean(dim=1, keepdim=True)
        return q

    # ------------------------------------------------------------------
    def get_action(self, state: torch.Tensor) -> int:
        """Greedy action (no grad). State can be 1-D or batched."""
        with torch.no_grad():
            if state.dim() == 1:
                state = state.unsqueeze(0)
            q = self.forward(state)
            return int(q.argmax(dim=1).item())
