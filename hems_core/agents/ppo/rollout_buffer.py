"""
Rollout Buffer for the HEMS PPO agent.

PPO is an on-policy algorithm: transitions collected under the CURRENT policy
are used for K epochs of gradient updates, then discarded.

The buffer also computes:
  • Generalised Advantage Estimates (GAE, Schulman et al. 2016)
  • Discounted returns used as value targets

Layout
──────
    push()  ×  ROLLOUT_STEPS
         ↓
    compute_returns_and_advantages()   ← called once per rollout
         ↓
    get_mini_batches()                 ← yields shuffled mini-batches for K epochs
"""

from __future__ import annotations

import numpy as np
from typing import Iterator, Tuple

from .config import (
    STATE_DIM, ACTION_DIM,
    GAMMA, GAE_LAMBDA,
    ROLLOUT_STEPS, MINI_BATCH_SIZE,
    NORMALIZE_ADVANTAGES,
)


class RolloutBuffer:
    """
    Fixed-size on-policy buffer.

    After `ROLLOUT_STEPS` transitions are stored, call
    `compute_returns_and_advantages(last_value, last_done)` once, then
    iterate over `get_mini_batches()` for training.  Call `reset()` before
    the next rollout.
    """

    def __init__(self, capacity: int = ROLLOUT_STEPS):
        self.capacity = capacity
        self._ptr     = 0
        self._full    = False

        # Pre-allocate arrays
        self.states      = np.zeros((capacity, STATE_DIM),  dtype=np.float32)
        self.actions     = np.zeros((capacity, ACTION_DIM), dtype=np.float32)
        self.rewards     = np.zeros(capacity,               dtype=np.float32)
        self.dones       = np.zeros(capacity,               dtype=np.float32)
        self.values      = np.zeros(capacity,               dtype=np.float32)
        self.log_probs   = np.zeros(capacity,               dtype=np.float32)

        # Computed after rollout ends
        self.advantages  = np.zeros(capacity, dtype=np.float32)
        self.returns     = np.zeros(capacity, dtype=np.float32)

    # ------------------------------------------------------------------
    def reset(self):
        self._ptr  = 0
        self._full = False

    # ------------------------------------------------------------------
    def push(
        self,
        state:    np.ndarray,
        action:   np.ndarray,
        reward:   float,
        done:     bool,
        value:    float,
        log_prob: float,
    ):
        """Store one transition."""
        i = self._ptr
        self.states[i]    = state
        self.actions[i]   = action
        self.rewards[i]   = reward
        self.dones[i]     = float(done)
        self.values[i]    = value
        self.log_probs[i] = log_prob
        self._ptr += 1
        if self._ptr >= self.capacity:
            self._full = True

    # ------------------------------------------------------------------
    def compute_returns_and_advantages(
        self,
        last_value: float,
        last_done:  bool,
    ):
        """
        GAE(λ) advantage estimation.

        δ_t = r_t + γ · V(s_{t+1}) · (1 - done_t) - V(s_t)
        A_t = δ_t + (γλ) · (1 - done_t) · A_{t+1}

        Returns are R_t = A_t + V(s_t).
        """
        n = self._ptr  # actual number of stored steps (may be < capacity)

        gae      = 0.0
        next_val = last_value
        next_done = float(last_done)

        for t in reversed(range(n)):
            mask    = 1.0 - self.dones[t]
            delta   = self.rewards[t] + GAMMA * next_val * mask - self.values[t]
            gae     = delta + GAMMA * GAE_LAMBDA * mask * gae
            self.advantages[t] = gae
            self.returns[t]    = gae + self.values[t]

            next_val  = self.values[t]
            next_done = self.dones[t]

    # ------------------------------------------------------------------
    def get_mini_batches(
        self,
        batch_size: int = MINI_BATCH_SIZE,
    ) -> Iterator[Tuple[np.ndarray, ...]]:
        """
        Yield shuffled mini-batches of stored transitions.
        Call `compute_returns_and_advantages` first.

        Yields
        ------
        (states, actions, old_log_probs, returns, advantages)
        """
        n    = self._ptr
        idxs = np.random.permutation(n)

        adv = self.advantages[:n].copy()
        if NORMALIZE_ADVANTAGES and adv.std() > 1e-8:
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        for start in range(0, n, batch_size):
            batch = idxs[start : start + batch_size]
            yield (
                self.states[batch],
                self.actions[batch],
                self.log_probs[batch],
                self.returns[batch],
                adv[batch],
            )

    # ------------------------------------------------------------------
    @property
    def size(self) -> int:
        return self._ptr

    def is_ready(self) -> bool:
        return self._ptr >= self.capacity
