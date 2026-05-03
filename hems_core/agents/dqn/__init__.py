"""
HEMS DQN Agent package.

Quick start
-----------
    from hems_core.agents.dqn import DQNAgent, train
    from hems_core.env.energy_env import EnergyEnv

    env   = EnergyEnv()
    agent = DQNAgent()
    rewards = train(agent, env, n_episodes=300)
"""

from .agent        import DQNAgent, train, decode_action, encode_action
from .networks     import DuelingDQN
from .replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
from .config       import ACTION_TABLE, N_ACTIONS, STATE_DIM

__all__ = [
    "DQNAgent",
    "train",
    "decode_action",
    "encode_action",
    "DuelingDQN",
    "ReplayBuffer",
    "PrioritizedReplayBuffer",
    "ACTION_TABLE",
    "N_ACTIONS",
    "STATE_DIM",
]
