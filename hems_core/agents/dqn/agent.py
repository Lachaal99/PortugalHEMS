"""
DQN Agent for the HEMS EnergyEnv.

Algorithm  : Double DQN + Dueling architecture + Prioritized Experience Replay
Action space: 2D continuous [-1,1]² → discretized to N_ACTIONS joint actions
             (see config.ACTION_TABLE)

Training loop highlights
─────────────────────────
• Two networks: online (trained every TRAIN_FREQ steps) and target (soft-updated).
• Double DQN: online net selects next action; target net evaluates its Q-value.
• PER: TD-error priorities focus sampling on surprising transitions.
• Epsilon-greedy with linear decay.

Usage
─────
    from hems_core.agents.dqn.agent import DQNAgent, train

    agent = DQNAgent()
    train(agent, env, n_episodes=500)
"""

from __future__ import annotations

import os
import math
import random
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .config import (
    ACTION_TABLE, N_ACTIONS, STATE_DIM,
    GAMMA, LR, BATCH_SIZE, MIN_BUFFER_SIZE,
    TAU, EPS_START, EPS_END, EPS_DECAY_STEPS,
    TRAIN_FREQ, GRAD_CLIP,
    USE_PER,
    LOG_INTERVAL, SAVE_INTERVAL, CHECKPOINT_DIR,
)
from .networks import DuelingDQN
from .replay_buffer import ReplayBuffer, PrioritizedReplayBuffer

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)


# ---------------------------------------------------------------------------
# Action helpers
# ---------------------------------------------------------------------------

def decode_action(action_idx: int) -> np.ndarray:
    """Convert discrete index → continuous [P_bat, P_ev] vector."""
    return ACTION_TABLE[action_idx].copy()


def encode_action(continuous: np.ndarray) -> int:
    """Nearest-neighbour mapping: continuous → discrete index (for testing)."""
    dists = np.linalg.norm(ACTION_TABLE - continuous, axis=1)
    return int(np.argmin(dists))


# ---------------------------------------------------------------------------
# DQN Agent
# ---------------------------------------------------------------------------

class DQNAgent:
    """
    Double Dueling DQN agent for the HEMS EnergyEnv.

    Parameters
    ----------
    device : torch device string ('cuda', 'cpu', 'mps')
             Defaults to CUDA if available, then MPS, then CPU.
    seed   : random seed for reproducibility
    """

    def __init__(self,device: Optional[str] = None,seed:int = 42):
        # ── Device ────────────────────────────────────────────────────────
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = torch.device(device)
        logger.info(f"DQNAgent running on: {self.device}")

        # ── Reproducibility ───────────────────────────────────────────────
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        # ── Networks ──────────────────────────────────────────────────────
        self.online_net = DuelingDQN().to(self.device)
        self.target_net = DuelingDQN().to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        # ── Optimiser ─────────────────────────────────────────────────────
        self.optimizer = optim.Adam(self.online_net.parameters(), lr=LR)

        # ── Replay buffer ─────────────────────────────────────────────────
        self.buffer = PrioritizedReplayBuffer() if USE_PER else ReplayBuffer()

        # ── Hyperparameters (for logging) ─────────────────────────────────
        self.gamma = GAMMA
        self.tau = TAU
        self.lr = LR
        
        # ── Step counters ─────────────────────────────────────────────────
        self.total_steps   = 0
        self.train_steps   = 0

    # ------------------------------------------------------------------
    # Epsilon (linear decay)
    # ------------------------------------------------------------------

    @property
    def epsilon(self) -> float:
        progress = min(self.total_steps / EPS_DECAY_STEPS, 1.0)
        return EPS_START + progress * (EPS_END - EPS_START)

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def select_action(self, state: np.ndarray, explore: bool = True) -> int:
        """
        ε-greedy selection.

        Parameters
        ----------
        state   : raw numpy state from env.get_state()
        explore : if False, always greedy (for evaluation)

        Returns
        -------
        action_idx : integer index into ACTION_TABLE
        """
        if explore and random.random() < self.epsilon:
            return random.randrange(N_ACTIONS)

        s = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        return self.online_net.get_action(s)

    def get_continuous_action(self, state: np.ndarray, explore: bool = True) -> np.ndarray:
        """Convenience wrapper → returns [P_bat, P_ev]."""
        idx = self.select_action(state, explore=explore)
        return decode_action(idx)

    # ------------------------------------------------------------------
    # Store transition
    # ------------------------------------------------------------------

    def store(self,state: np.ndarray,action_idx: int,reward:float,next_state: np.ndarray,done: bool):
        self.buffer.push(state, action_idx, reward, next_state, done)
        self.total_steps += 1

    # ------------------------------------------------------------------
    # Training step (Double DQN + PER)
    # ------------------------------------------------------------------

    def train_step(self) -> Optional[float]:
        """
        One gradient update.  Returns the loss value, or None if the buffer
        is not yet warm.
        """
        if len(self.buffer) < MIN_BUFFER_SIZE:
            return None

        states, actions, rewards, next_states, dones, idxs, weights = \
            self.buffer.sample(BATCH_SIZE)

        # Move to device
        s  = torch.as_tensor(states,      device=self.device)
        a  = torch.as_tensor(actions,     device=self.device, dtype=torch.long)
        r  = torch.as_tensor(rewards,     device=self.device)
        ns = torch.as_tensor(next_states, device=self.device)
        d  = torch.as_tensor(dones,       device=self.device)
        w  = torch.as_tensor(weights,     device=self.device)

        # ── Current Q-values ──────────────────────────────────────────────
        q_values = self.online_net(s)                    # (B, A)
        q_pred   = q_values.gather(1, a.unsqueeze(1)).squeeze(1)  # (B,)

        # ── Double DQN target ─────────────────────────────────────────────
        with torch.no_grad():
            # Online net picks best next action
            next_q_online   = self.online_net(ns)
            next_actions    = next_q_online.argmax(dim=1)   # (B,)

            # Target net evaluates that action
            next_q_target   = self.target_net(ns)
            next_q_selected = next_q_target.gather(
                1, next_actions.unsqueeze(1)
            ).squeeze(1)

            target = r + GAMMA * next_q_selected * (1.0 - d)

        # ── Huber loss (IS-weighted) ───────────────────────────────────────
        td_errors = (q_pred - target)
        loss      = (w * F.huber_loss(q_pred, target, reduction='none')).mean()

        # ── Gradient step ─────────────────────────────────────────────────
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online_net.parameters(), GRAD_CLIP)
        self.optimizer.step()

        # ── Soft-update target network ────────────────────────────────────
        self._soft_update()

        # ── Update priorities ─────────────────────────────────────────────
        self.buffer.update_priorities(idxs, td_errors.detach().abs().cpu().numpy())

        self.train_steps += 1
        return loss.item()

    # ------------------------------------------------------------------
    # Soft target update (Polyak)
    # ------------------------------------------------------------------

    def _soft_update(self):
        for p_online, p_target in zip(
            self.online_net.parameters(), self.target_net.parameters()
        ):
            p_target.data.copy_(TAU * p_online.data + (1.0 - TAU) * p_target.data)

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------

    def save(self, path: str):
        """Save DQN agent networks and optimizer state."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "online_net":  self.online_net.state_dict(),
            "target_net":  self.target_net.state_dict(),
            "optimizer":   self.optimizer.state_dict(),
            "total_steps": self.total_steps,
            "train_steps": self.train_steps,
        }
        torch.save(checkpoint, path)
        print(f"Agent saved to {path}")

    def load(self, path: str):
        """Load DQN agent networks and optimizer state."""
        ckpt = torch.load(path, map_location=self.device)
        self.online_net.load_state_dict(ckpt["online_net"])
        self.target_net.load_state_dict(ckpt["target_net"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.total_steps = ckpt.get("total_steps", 0)
        self.train_steps = ckpt.get("train_steps", 0)
        
        # Set to eval mode
        self.online_net.eval()
        self.target_net.eval()
        print(f"Agent loaded from {path}")


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(
    agent:      DQNAgent,
    env,
    n_episodes: int         = 500,
    eval_env    = None,      # optional separate env for evaluation
    resume:     bool        = False,
    ckpt_dir:   str         = CHECKPOINT_DIR,
):
    """
    Main training loop.

    Parameters
    ----------
    agent      : DQNAgent instance
    env        : EnergyEnv (or any env with reset/step API)
    n_episodes : total training episodes
    eval_env   : if provided, run a greedy episode every LOG_INTERVAL episodes
    resume     : if True, look for the latest checkpoint and resume from it
    ckpt_dir   : directory for saving checkpoints
    """
    # ── Optionally resume ────────────────────────────────────────────────
    start_ep = 0
    if resume:
        ckpts = sorted(Path(ckpt_dir).glob("ep_*.pt"))
        if ckpts:
            agent.load(str(ckpts[-1]))
            start_ep = int(ckpts[-1].stem.split("_")[1])

    episode_rewards = []

    for ep in range(start_ep, n_episodes):
        state = env.reset()
        ep_reward   = 0.0
        ep_cost     = 0.0
        ep_loss_sum = 0.0
        ep_loss_cnt = 0
        done        = False

        while not done:
            # ── Action ───────────────────────────────────────────────────
            action_idx  = agent.select_action(state, explore=True)
            action_cont = decode_action(action_idx)

            # ── Env step ─────────────────────────────────────────────────
            next_state, reward, done, info = env.step(action_cont)

            # ── Store ────────────────────────────────────────────────────
            agent.store(state, action_idx, reward, next_state, done)
            state      = next_state
            ep_reward += reward
            ep_cost   += info.get("cost", 0.0)

            # ── Train ────────────────────────────────────────────────────
            if agent.total_steps % TRAIN_FREQ == 0:
                loss = agent.train_step()
                if loss is not None:
                    ep_loss_sum += loss
                    ep_loss_cnt += 1

        episode_rewards.append(ep_reward)

        # ── Logging ──────────────────────────────────────────────────────
        if (ep + 1) % LOG_INTERVAL == 0:
            mean_r    = np.mean(episode_rewards[-LOG_INTERVAL:])
            mean_loss = ep_loss_sum / max(ep_loss_cnt, 1)
            logger.info(
                f"Ep {ep+1:>5} | "
                f"Reward {mean_r:>8.2f} | "
                f"Cost {ep_cost:>7.4f} | "
                f"Loss {mean_loss:>7.4f} | "
                f"ε {agent.epsilon:.3f} | "
                f"Steps {agent.total_steps}"
            )

            # Optional evaluation episode (greedy)
            if eval_env is not None:
                eval_r = _evaluate(agent, eval_env)
                logger.info(f"         Eval reward: {eval_r:.2f}")

        # ── Checkpoint ───────────────────────────────────────────────────
        if (ep + 1) % SAVE_INTERVAL == 0:
            agent.save(os.path.join(ckpt_dir, f"ep_{ep+1:05d}.pt"))

    # Final save
    agent.save(os.path.join(ckpt_dir, "final.pt"))
    logger.info("Training complete.")
    return episode_rewards


# ---------------------------------------------------------------------------
# Greedy evaluation helper
# ---------------------------------------------------------------------------

def _evaluate(agent: DQNAgent, env, n_episodes: int = 1) -> float:
    total = 0.0
    for _ in range(n_episodes):
        state = env.reset()
        done  = False
        while not done:
            action_idx  = agent.select_action(state, explore=False)
            action_cont = decode_action(action_idx)
            state, r, done, _ = env.step(action_cont)
            total += r
    return total / n_episodes
