"""
Neural network architectures for the HEMS PPO agent.

Two separate networks are used (no shared trunk):
────────────────────────────────────────────────────────────────
  Actor  (policy π_θ)  : s → μ(s), log_σ(s)
                         action ~ clip(Normal(μ, σ), -1, 1)

  Critic (value V_φ)   : s → scalar V(s)
────────────────────────────────────────────────────────────────

Why separate networks?
  The actor and critic optimise different loss surfaces.  In energy
  management, V(s) correlates strongly with electricity price and load
  (large scale differences), while the policy gradient signal is much
  smaller.  Separate learning rates + separate trunks avoids the
  "dead gradient" problem that shared-trunk Actor-Critic suffers from.

Policy details
──────────────
  • Outputs μ ∈ [-1,1] via tanh squashing.
  • log_σ is a learned parameter vector (state-independent) clamped
    to [LOG_STD_MIN, LOG_STD_MAX].
  • log_prob is corrected for the tanh squash:
      log π(a|s) = log N(a_raw|μ_raw,σ) - Σ log(1 - tanh²(a_raw))
  • Actions are clipped hard to [-1, 1] after sampling for safety.

Architecture
────────────
  Both MLP trunks: Linear → LayerNorm → LeakyReLU (×2)
  Orthogonal init with scaled gain on the output layer.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from typing import List, Tuple

from .config import (
    STATE_DIM, ACTION_DIM,
    ACTOR_HIDDEN, CRITIC_HIDDEN,
    LOG_STD_INIT, LOG_STD_MIN, LOG_STD_MAX,
    ACTION_LOW, ACTION_HIGH,
)


# ---------------------------------------------------------------------------
# Shared MLP building block
# ---------------------------------------------------------------------------

def _build_mlp(
    in_dim:       int,
    hidden_sizes: List[int],
    use_layer_norm: bool = True,
) -> Tuple[nn.Sequential, int]:
    """Returns (sequential trunk, output_dim)."""
    layers: List[nn.Module] = []
    dim = in_dim
    for h in hidden_sizes:
        layers.append(nn.Linear(dim, h))
        if use_layer_norm:
            layers.append(nn.LayerNorm(h))
        layers.append(nn.LeakyReLU(0.1, inplace=True))
        dim = h
    return nn.Sequential(*layers), dim


def _init_weights(module: nn.Module, gain: float = 1.0):
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight, gain=gain)
        nn.init.zeros_(module.bias)


# ---------------------------------------------------------------------------
# Actor (Gaussian policy with tanh squashing)
# ---------------------------------------------------------------------------

class GaussianActor(nn.Module):
    """
    Diagonal Gaussian policy π_θ(a|s).

    Forward returns (action, log_prob, entropy).
    The mean is squashed through tanh → range (-1, 1).
    log_std is a learnable parameter (not state-dependent) — simpler and
    often just as good for low-dimensional continuous control.
    """

    def __init__(
        self,
        state_dim:    int       = STATE_DIM,
        action_dim:   int       = ACTION_DIM,
        hidden_sizes: List[int] = ACTOR_HIDDEN,
        log_std_init: float     = LOG_STD_INIT,
    ):
        super().__init__()
        self.trunk, trunk_out = _build_mlp(state_dim, hidden_sizes)
        self.mean_layer = nn.Linear(trunk_out, action_dim)

        # State-independent log_std parameter vector
        self.log_std = nn.Parameter(
            torch.full((action_dim,), log_std_init)
        )

        # Init: small gain on output layer → near-zero initial actions
        self.trunk.apply(lambda m: _init_weights(m, gain=1.0))
        _init_weights(self.mean_layer, gain=0.01)

    # ------------------------------------------------------------------
    def _distribution(self, state: torch.Tensor) -> Normal:
        features = self.trunk(state)
        mean_raw = self.mean_layer(features)          # un-squashed mean
        log_std  = self.log_std.clamp(LOG_STD_MIN, LOG_STD_MAX)
        std      = log_std.exp()
        return Normal(mean_raw, std)

    # ------------------------------------------------------------------
    @staticmethod
    def _squash_log_prob(dist: Normal, raw_action: torch.Tensor) -> torch.Tensor:
        """
        Compute log π(a|s) corrected for tanh squashing.
        log π = log N(raw|μ,σ)  -  Σ_i log(1 - tanh²(raw_i))
        """
        log_prob = dist.log_prob(raw_action).sum(dim=-1)
        # Numerically stable tanh correction
        correction = (2.0 * (
            torch.log(torch.tensor(2.0, device=raw_action.device))
            - raw_action
            - F.softplus(-2.0 * raw_action)
        )).sum(dim=-1)
        return log_prob - correction

    # ------------------------------------------------------------------
    def forward(
        self, state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample action and compute log_prob + entropy.

        Returns
        -------
        action   : (B, action_dim) ∈ (-1, 1)
        log_prob : (B,)
        entropy  : (B,)  — of the pre-squash Gaussian
        """
        dist       = self._distribution(state)
        raw_action = dist.rsample()                     # reparameterised sample
        action     = torch.tanh(raw_action)             # squash to (-1,1)
        log_prob   = self._squash_log_prob(dist, raw_action)
        entropy    = dist.entropy().sum(dim=-1)
        return action, log_prob, entropy

    # ------------------------------------------------------------------
    def evaluate_actions(
        self,
        state:  torch.Tensor,
        action: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Given stored (state, action) pairs, return log_prob + entropy
        under the CURRENT policy parameters.  Used during PPO update.

        Parameters
        ----------
        action : (B, action_dim) — squashed action stored in rollout buffer
        """
        dist       = self._distribution(state)
        # Recover raw (pre-tanh) action — clamp for numerical safety
        raw_action = torch.atanh(action.clamp(-1 + 1e-6, 1 - 1e-6))
        log_prob   = self._squash_log_prob(dist, raw_action)
        entropy    = dist.entropy().sum(dim=-1)
        return log_prob, entropy

    # ------------------------------------------------------------------
    @torch.no_grad()
    def get_action(
        self, state: torch.Tensor, deterministic: bool = False
    ) -> Tuple[torch.Tensor, float]:
        """
        Inference helper.

        Returns
        -------
        action   : numpy array [P_bat, P_ev] ∈ (-1,1)
        log_prob : scalar float
        """
        if state.dim() == 1:
            state = state.unsqueeze(0)
        dist       = self._distribution(state)
        raw_action = dist.mean if deterministic else dist.rsample()
        action     = torch.tanh(raw_action)
        log_prob   = self._squash_log_prob(dist, raw_action)
        return action.squeeze(0).cpu().numpy(), log_prob.item()


# ---------------------------------------------------------------------------
# Critic (state-value function)
# ---------------------------------------------------------------------------

class Critic(nn.Module):
    """
    State-value network V_φ(s) → scalar.
    Separate from the actor — see module docstring for rationale.
    """

    def __init__(
        self,
        state_dim:    int       = STATE_DIM,
        hidden_sizes: List[int] = CRITIC_HIDDEN,
    ):
        super().__init__()
        self.trunk, trunk_out = _build_mlp(state_dim, hidden_sizes)
        self.value_head = nn.Linear(trunk_out, 1)

        self.trunk.apply(lambda m: _init_weights(m, gain=1.0))
        _init_weights(self.value_head, gain=1.0)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        state : (B, STATE_DIM)

        Returns
        -------
        value : (B,)
        """
        return self.value_head(self.trunk(state)).squeeze(-1)
