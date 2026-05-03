"""
Experience Replay Buffer for the HEMS DQN agent.

Two implementations are provided:
  - ReplayBuffer          : uniform random sampling
  - PrioritizedReplayBuffer: proportional PER (Schaul et al. 2016)

The agent selects which one to use via config.USE_PER.
"""

import numpy as np
from collections import deque
import random
from typing import Tuple

from .config import (
    BUFFER_CAPACITY, BATCH_SIZE,
    PER_ALPHA, PER_BETA_START, PER_BETA_STEPS, PER_EPS,
    STATE_DIM,
)


# ---------------------------------------------------------------------------
# Uniform Replay Buffer
# ---------------------------------------------------------------------------

class ReplayBuffer:
    """Simple circular buffer with uniform sampling."""

    def __init__(self, capacity: int = BUFFER_CAPACITY):
        self.capacity = capacity
        self._states      = np.zeros((capacity, STATE_DIM), dtype=np.float32)
        self._actions     = np.zeros(capacity,              dtype=np.int64)
        self._rewards     = np.zeros(capacity,              dtype=np.float32)
        self._next_states = np.zeros((capacity, STATE_DIM), dtype=np.float32)
        self._dones       = np.zeros(capacity,              dtype=np.float32)
        self._ptr         = 0
        self._size        = 0

    def push(self, state, action: int, reward: float, next_state, done: bool):
        i = self._ptr
        self._states[i]      = state
        self._actions[i]     = action
        self._rewards[i]     = reward
        self._next_states[i] = next_state
        self._dones[i]       = float(done)
        self._ptr  = (i + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int = BATCH_SIZE):
        idxs = np.random.randint(0, self._size, size=batch_size)
        return (
            self._states[idxs],
            self._actions[idxs],
            self._rewards[idxs],
            self._next_states[idxs],
            self._dones[idxs],
            idxs,                        # indices (unused for uniform)
            np.ones(batch_size, dtype=np.float32),  # weights = 1
        )

    def update_priorities(self, idxs, priorities):
        """No-op for uniform buffer (keeps API compatible with PER)."""
        pass

    def __len__(self):
        return self._size


# ---------------------------------------------------------------------------
# Prioritised Experience Replay (Segment-tree implementation)
# ---------------------------------------------------------------------------

class SumTree:
    """Binary sum-tree for O(log n) priority updates and sampling."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree     = np.zeros(2 * capacity, dtype=np.float64)
        self.data_ptr = 0

    def update(self, idx: int, priority: float):
        """Update leaf at data index `idx`."""
        tree_idx = idx + self.capacity
        self.tree[tree_idx] = priority
        # Propagate upward
        tree_idx >>= 1
        while tree_idx >= 1:
            self.tree[tree_idx] = self.tree[2 * tree_idx] + self.tree[2 * tree_idx + 1]
            tree_idx >>= 1

    def get(self, value: float) -> Tuple[int, float]:
        """Sample a data index proportional to priority."""
        idx = 1
        while idx < self.capacity:
            left = 2 * idx
            if value <= self.tree[left]:
                idx = left
            else:
                value -= self.tree[left]
                idx = left + 1
        data_idx = idx - self.capacity
        return data_idx, self.tree[idx]

    @property
    def total(self):
        return self.tree[1]

    @property
    def max_priority(self):
        return self.tree[self.capacity:self.capacity + self.capacity].max()


class PrioritizedReplayBuffer:
    """
    Proportional Prioritized Experience Replay.
    Priorities p_i ∝ |δ_i|^α (TD-error magnitude).
    Importance-sampling weights w_i = (N·P_i)^{-β} / max_j(w_j).
    """

    def __init__(
        self,
        capacity:   int   = BUFFER_CAPACITY,
        alpha:      float = PER_ALPHA,
        beta_start: float = PER_BETA_START,
        beta_steps: int   = PER_BETA_STEPS,
        eps:        float = PER_EPS,
    ):
        self.capacity   = capacity
        self.alpha      = alpha
        self.beta       = beta_start
        self.beta_inc   = (1.0 - beta_start) / beta_steps
        self.eps        = eps
        self.tree       = SumTree(capacity)

        self._states      = np.zeros((capacity, STATE_DIM), dtype=np.float32)
        self._actions     = np.zeros(capacity,              dtype=np.int64)
        self._rewards     = np.zeros(capacity,              dtype=np.float32)
        self._next_states = np.zeros((capacity, STATE_DIM), dtype=np.float32)
        self._dones       = np.zeros(capacity,              dtype=np.float32)
        self._ptr         = 0
        self._size        = 0

    # Default priority for newly added transitions (max seen so far)
    def _default_priority(self):
        p = self.tree.max_priority
        return p if p > 0 else 1.0

    def push(self, state, action: int, reward: float, next_state, done: bool):
        i = self._ptr
        self._states[i]      = state
        self._actions[i]     = action
        self._rewards[i]     = reward
        self._next_states[i] = next_state
        self._dones[i]       = float(done)

        priority = self._default_priority() ** self.alpha
        self.tree.update(i, priority)

        self._ptr  = (i + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int = BATCH_SIZE):
        idxs      = np.empty(batch_size, dtype=np.int64)
        priorities = np.empty(batch_size, dtype=np.float64)

        segment = self.tree.total / batch_size
        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            v = random.uniform(a, b)
            idxs[i], priorities[i] = self.tree.get(v)

        # IS weights
        probs   = priorities / (self.tree.total + 1e-10)
        weights = (self._size * probs) ** (-self.beta)
        weights /= weights.max()

        # Anneal beta
        self.beta = min(1.0, self.beta + self.beta_inc)

        return (
            self._states[idxs],
            self._actions[idxs],
            self._rewards[idxs],
            self._next_states[idxs],
            self._dones[idxs],
            idxs,
            weights.astype(np.float32),
        )

    def update_priorities(self, idxs: np.ndarray, td_errors: np.ndarray):
        """Call after each gradient step with |δ| for sampled transitions."""
        for i, err in zip(idxs, td_errors):
            priority = (abs(float(err)) + self.eps) ** self.alpha
            self.tree.update(int(i), priority)

    def __len__(self):
        return self._size
