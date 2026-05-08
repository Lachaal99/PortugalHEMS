"""
PPO Agent for the HEMS EnergyEnv.

Algorithm : Proximal Policy Optimisation (Schulman et al. 2017)
            Clipped surrogate objective + GAE + separate Actor/Critic networks

Key differences vs DQN
───────────────────────
• On-policy: data collected under the current policy, used for K epochs,
  then thrown away.  No replay buffer.
• Continuous actions: Gaussian policy with tanh squashing → no discretization.
• Two losses optimised jointly:
    L = -L_CLIP + VALUE_COEF·L_VALUE - ENTROPY_COEF·H

Training loop
─────────────
    for each update:
        1. Collect ROLLOUT_STEPS transitions using current policy
        2. Compute GAE advantages + discounted returns
        3. For N_EPOCHS epochs:
               shuffle transitions → mini-batches
               compute clipped policy loss, value loss, entropy bonus
               gradient step on actor + critic

Usage
─────
    from hems_core.agents.ppo.agent import PPOAgent, train

    agent = PPOAgent()
    train(agent, env, n_updates=500)
"""

from __future__ import annotations

import os
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .config import (
    STATE_DIM, ACTION_DIM,
    GAMMA, CLIP_EPS, ENTROPY_COEF, VALUE_COEF, GRAD_CLIP,
    ROLLOUT_STEPS, N_EPOCHS, MINI_BATCH_SIZE,
    ACTOR_LR, CRITIC_LR, LR_ANNEAL,
    NORMALIZE_REWARDS, REWARD_NORM_CLIP,
    LOG_INTERVAL, SAVE_INTERVAL, CHECKPOINT_DIR,
)
from .networks import GaussianActor, Critic
from .rollout_buffer import RolloutBuffer

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)


# ---------------------------------------------------------------------------
# Running reward normaliser (Welford online algorithm)
# ---------------------------------------------------------------------------

class RunningMeanStd:
    """Tracks running mean and variance for reward normalisation."""

    def __init__(self, clip: float = REWARD_NORM_CLIP):
        self.mean  = 0.0
        self.var   = 1.0
        self.count = 1e-4
        self.clip  = clip

    def update(self, x: float):
        self.count += 1
        delta       = x - self.mean
        self.mean  += delta / self.count
        self.var   += delta * (x - self.mean)

    @property
    def std(self) -> float:
        return max(np.sqrt(self.var / self.count), 1e-8)

    def normalize(self, x: float) -> float:
        normed = (x - self.mean) / self.std
        return float(np.clip(normed, -self.clip, self.clip))


# ---------------------------------------------------------------------------
# PPO Agent
# ---------------------------------------------------------------------------

class PPOAgent:
    """
    Proximal Policy Optimisation agent for the HEMS EnergyEnv.

    Parameters
    ----------
    device : torch device string ('cuda', 'cpu', 'mps')
    seed   : random seed
    """

    def __init__(
        self,
        device: Optional[str] = None,
        seed:   int           = 42,
    ):
        # ── Device ────────────────────────────────────────────────────────
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = torch.device(device)
        logger.info(f"PPOAgent running on: {self.device}")

        # ── Reproducibility ───────────────────────────────────────────────
        np.random.seed(seed)
        torch.manual_seed(seed)

        # ── Networks ──────────────────────────────────────────────────────
        self.actor  = GaussianActor().to(self.device)
        self.critic = Critic().to(self.device)

        # ── Optimisers ────────────────────────────────────────────────────
        self.actor_opt  = optim.Adam(self.actor.parameters(),  lr=ACTOR_LR,  eps=1e-5)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=CRITIC_LR, eps=1e-5)

        # ── Rollout buffer (on-policy, discarded after each update) ───────
        self.buffer = RolloutBuffer(capacity=ROLLOUT_STEPS)

        # ── Reward normaliser ─────────────────────────────────────────────
        self.reward_norm = RunningMeanStd() if NORMALIZE_REWARDS else None

        # ── Counters ──────────────────────────────────────────────────────
        self.total_steps   = 0
        self.update_count  = 0

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    @torch.no_grad()
    def select_action(
        self, state: np.ndarray, deterministic: bool = False
    ):
        """
        Sample an action from the current policy.

        Returns
        -------
        action   : np.ndarray [P_bat, P_ev] ∈ (-1, 1)
        log_prob : float
        value    : float — V(s) from critic
        """
        s = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        if s.dim() == 1:
            s = s.unsqueeze(0)

        action, log_prob = self.actor.get_action(s, deterministic=deterministic)
        value            = self.critic(s).item()
        action[1] = (action[1] + 1) / 2  # Rescale EV action from [-1,1] to [0,1]
        return action, log_prob, value

    # ------------------------------------------------------------------
    # Store one transition
    # ------------------------------------------------------------------

    def store(
        self,
        state:    np.ndarray,
        action:   np.ndarray,
        reward:   float,
        done:     bool,
        value:    float,
        log_prob: float,
    ):
        if self.reward_norm is not None:
            self.reward_norm.update(reward)
            reward = self.reward_norm.normalize(reward)

        self.buffer.push(state, action, reward, done, value, log_prob)
        self.total_steps += 1

    # ------------------------------------------------------------------
    # PPO update
    # ------------------------------------------------------------------

    def update(self, last_value: float, last_done: bool) -> dict:
        """
        Run N_EPOCHS of PPO gradient updates on the current rollout.

        Parameters
        ----------
        last_value : V(s_T) — bootstrap value for open episode
        last_done  : whether the last step was terminal

        Returns
        -------
        metrics dict with mean losses
        """
        # ── GAE + returns ────────────────────────────────────────────────
        self.buffer.compute_returns_and_advantages(last_value, last_done)

        total_policy_loss  = 0.0
        total_value_loss   = 0.0
        total_entropy      = 0.0
        total_clip_frac    = 0.0
        n_batches          = 0

        for _ in range(N_EPOCHS):
            for (states, actions, old_log_probs,
                 returns, advantages) in self.buffer.get_mini_batches(MINI_BATCH_SIZE):

                s   = torch.as_tensor(states,       device=self.device)
                a   = torch.as_tensor(actions,      device=self.device)
                olp = torch.as_tensor(old_log_probs,device=self.device)
                ret = torch.as_tensor(returns,      device=self.device)
                adv = torch.as_tensor(advantages,   device=self.device)

                # ── Policy loss (clipped surrogate) ───────────────────────
                log_probs, entropy = self.actor.evaluate_actions(s, a)
                ratio  = (log_probs - olp).exp()           # π_new / π_old
                clip_r = ratio.clamp(1.0 - CLIP_EPS, 1.0 + CLIP_EPS)

                policy_loss = -torch.min(ratio * adv, clip_r * adv).mean()

                # Fraction of samples where clipping was active (diagnostic)
                clip_frac   = ((ratio - 1.0).abs() > CLIP_EPS).float().mean().item()

                # ── Value loss ────────────────────────────────────────────
                values      = self.critic(s)
                value_loss  = nn.functional.mse_loss(values, ret)

                # ── Combined loss ─────────────────────────────────────────
                entropy_mean = entropy.mean()
                loss = policy_loss + VALUE_COEF * value_loss - ENTROPY_COEF * entropy_mean

                # ── Gradient steps ────────────────────────────────────────
                self.actor_opt.zero_grad()
                self.critic_opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(),  GRAD_CLIP)
                nn.utils.clip_grad_norm_(self.critic.parameters(), GRAD_CLIP)
                self.actor_opt.step()
                self.critic_opt.step()

                total_policy_loss += policy_loss.item()
                total_value_loss  += value_loss.item()
                total_entropy     += entropy_mean.item()
                total_clip_frac   += clip_frac
                n_batches         += 1

        self.buffer.reset()
        self.update_count += 1

        return {
            "policy_loss" : total_policy_loss / max(n_batches, 1),
            "value_loss"  : total_value_loss  / max(n_batches, 1),
            "entropy"     : total_entropy     / max(n_batches, 1),
            "clip_frac"   : total_clip_frac   / max(n_batches, 1),
        }

    # ------------------------------------------------------------------
    # LR annealing
    # ------------------------------------------------------------------

    def anneal_lr(self, progress: float):
        """
        Linearly decay learning rates.
        progress ∈ [0, 1] — fraction of training completed.
        """
        if not LR_ANNEAL:
            return
        factor = 1.0 - progress
        for g in self.actor_opt.param_groups:
            g["lr"] = ACTOR_LR * factor
        for g in self.critic_opt.param_groups:
            g["lr"] = CRITIC_LR * factor

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "actor":        self.actor.state_dict(),
                "critic":       self.critic.state_dict(),
                "actor_opt":    self.actor_opt.state_dict(),
                "critic_opt":   self.critic_opt.state_dict(),
                "total_steps":  self.total_steps,
                "update_count": self.update_count,
            },
            path,
        )
        logger.info(f"Checkpoint saved → {path}")

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self.actor_opt.load_state_dict(ckpt["actor_opt"])
        self.critic_opt.load_state_dict(ckpt["critic_opt"])
        self.total_steps  = ckpt.get("total_steps",  0)
        self.update_count = ckpt.get("update_count", 0)
        logger.info(f"Checkpoint loaded ← {path}  (update {self.update_count})")


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(
    agent:      PPOAgent,
    env,
    n_updates:  int         = 500,
    eval_env                = None,
    resume:     bool        = False,
    ckpt_dir:   str         = CHECKPOINT_DIR,
):
    """
    Main PPO training loop.

    The loop alternates between:
      • Rollout phase : collect ROLLOUT_STEPS transitions
      • Update phase  : N_EPOCHS of mini-batch PPO gradient steps

    Parameters
    ----------
    agent     : PPOAgent instance
    env       : EnergyEnv
    n_updates : total number of PPO updates
    eval_env  : optional separate env for greedy evaluation
    resume    : load the latest checkpoint before starting
    ckpt_dir  : directory for checkpoints
    """
    # ── Optionally resume ────────────────────────────────────────────────
    start_update = 0
    if resume:
        ckpts = sorted(Path(ckpt_dir).glob("update_*.pt"))
        if ckpts:
            agent.load(str(ckpts[-1]))
            start_update = agent.update_count

    state = env.reset()
    done  = False

    ep_reward    = 0.0
    ep_cost      = 0.0
    ep_count     = 0
    all_ep_rewards: list = []

    for update in range(start_update, n_updates):

        # ── LR annealing ─────────────────────────────────────────────────
        agent.anneal_lr(progress=(update - start_update) / n_updates)

        # ── Rollout phase ─────────────────────────────────────────────────
        agent.buffer.reset()

        for _ in range(ROLLOUT_STEPS):
            action, log_prob, value = agent.select_action(state)
            next_state, reward, done, info = env.step(action)

            agent.store(state, action, reward, done, value, log_prob)

            ep_reward += reward
            ep_cost   += info.get("cost", 0.0)
            state      = next_state

            if done:
                all_ep_rewards.append(ep_reward)
                ep_reward = 0.0
                ep_cost   = 0.0
                ep_count += 1
                state     = env.reset()
                done      = False

        # Bootstrap value for the last unfinished episode
        if not done:
            _, _, last_value = agent.select_action(state, deterministic=True)
        else:
            last_value = 0.0

        # ── PPO update ────────────────────────────────────────────────────
        metrics = agent.update(last_value=last_value, last_done=done)

        # ── Logging ───────────────────────────────────────────────────────
        if (update + 1) % LOG_INTERVAL == 0:
            recent = all_ep_rewards[-10:] if all_ep_rewards else [0.0]
            mean_r = np.mean(recent)
            logger.info(
                f"Update {update+1:>5} | "
                f"Ep {ep_count:>5} | "
                f"Reward {mean_r:>8.2f} | "
                f"π-loss {metrics['policy_loss']:>7.4f} | "
                f"V-loss {metrics['value_loss']:>7.4f} | "
                f"Entropy {metrics['entropy']:>5.3f} | "
                f"Clip% {metrics['clip_frac']*100:>4.1f} | "
                f"Steps {agent.total_steps}"
            )

            if eval_env is not None:
                eval_r = _evaluate(agent, eval_env)
                logger.info(f"               Eval reward: {eval_r:.2f}")

        # ── Checkpoint ────────────────────────────────────────────────────
        if (update + 1) % SAVE_INTERVAL == 0:
            agent.save(os.path.join(ckpt_dir, f"update_{update+1:05d}.pt"))

    agent.save(os.path.join(ckpt_dir, "final.pt"))
    logger.info("Training complete.")
    return all_ep_rewards


# ---------------------------------------------------------------------------
# Greedy evaluation helper
# ---------------------------------------------------------------------------

def _evaluate(agent: PPOAgent, env, n_episodes: int = 1) -> float:
    total = 0.0
    for _ in range(n_episodes):
        state = env.reset()
        done  = False
        while not done:
            action, _, _ = agent.select_action(state, deterministic=True)
            state, r, done, _ = env.step(action)
            total += r
    return total / n_episodes
