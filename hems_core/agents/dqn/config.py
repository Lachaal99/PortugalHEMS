"""
Hyperparameters and action discretization for the HEMS DQN agent.

Action space: [P_bat, P_ev] each in [-1.0, 1.0]
We discretize each dimension independently → joint action = cartesian product.
"""

import numpy as np

# ---------------------------------------------------------------------------
# Action discretization
# ---------------------------------------------------------------------------
# Number of discrete levels per actuator.
# 7 levels → 49 joint actions  (good balance of resolution vs Q-table size)
# 11 levels → 121 joint actions (finer, but slower to converge)
N_BAT_ACTIONS = 7   # battery: discharge … charge
N_EV_ACTIONS  = 7   # EV:      discharge … charge  (negative = V2H if supported)

BAT_LEVELS = np.linspace(-1.0, 1.0, N_BAT_ACTIONS, dtype=np.float32)
EV_LEVELS  = np.linspace(0, 1.0, N_EV_ACTIONS,  dtype=np.float32)

# Joint action table: shape (N_BAT * N_EV, 2)
# action_table[i] = [bat_power, ev_power]
ACTION_TABLE = np.array(
    [[b, e] for b in BAT_LEVELS for e in EV_LEVELS],
    dtype=np.float32,
)
N_ACTIONS = len(ACTION_TABLE)          # 49


# ---------------------------------------------------------------------------
# State space
# ---------------------------------------------------------------------------
STATE_DIM = 9   # matches EnergyEnv.get_state()


# ---------------------------------------------------------------------------
# Network architecture
# ---------------------------------------------------------------------------
HIDDEN_SIZES   = [256, 256]    # shared trunk layers
VALUE_HIDDEN   = 128           # value stream hidden size
ADVANTAGE_HIDDEN = 128         # advantage stream hidden size


# ---------------------------------------------------------------------------
# Training hyperparameters
# ---------------------------------------------------------------------------
GAMMA           = 0.99         # discount factor
LR              = 3e-4         # Adam learning rate
BATCH_SIZE      = 256
BUFFER_CAPACITY = 100_000
MIN_BUFFER_SIZE = 2_000        # warm-up steps before training starts

# Target network soft-update (Polyak) coefficient
TAU             = 0.005        # θ_target ← τ·θ_online + (1-τ)·θ_target

# Epsilon-greedy exploration
EPS_START       = 1.0
EPS_END         = 0.05
EPS_DECAY_STEPS = 50_000       # linear decay over this many env steps

# Training schedule
TRAIN_FREQ      = 4            # update network every N env steps
TARGET_UPDATE   = 1            # target net updated every training step (via Polyak)

# Gradient clipping
GRAD_CLIP       = 10.0

# ---------------------------------------------------------------------------
# Prioritized Experience Replay (PER)
# ---------------------------------------------------------------------------
USE_PER         = True
PER_ALPHA       = 0.6          # priority exponent
PER_BETA_START  = 0.4          # IS-weight exponent (annealed to 1)
PER_BETA_STEPS  = 100_000      # steps to anneal beta
PER_EPS         = 1e-6         # small constant to avoid zero priority

# ---------------------------------------------------------------------------
# Logging / checkpointing
# ---------------------------------------------------------------------------
LOG_INTERVAL    = 10           # episodes between console logs
SAVE_INTERVAL   = 100          # episodes between checkpoint saves
CHECKPOINT_DIR  = "checkpoints"
