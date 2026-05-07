"""
Hyperparameters for the HEMS PPO agent.

Action space : [P_bat, P_ev] — continuous, each in [-1.0, 1.0]
PPO operates directly in continuous space using a Gaussian policy,
so NO discretization is required (unlike DQN).
Actions are sampled from N(μ(s), σ(s)) then clipped to [-1, 1].
"""

# ---------------------------------------------------------------------------
# State / Action space
# ---------------------------------------------------------------------------
STATE_DIM  = 9    # matches EnergyEnv.get_state()
ACTION_DIM = 2    # [P_bat, P_ev]
ACTION_LOW  = -1.0
ACTION_HIGH =  1.0

# ---------------------------------------------------------------------------
# Network architecture
# ---------------------------------------------------------------------------
# Shared trunk (actor + critic each have their own trunk — NOT shared)
# Sharing trunk can hurt in energy tasks where value scale differs a lot
ACTOR_HIDDEN  = [256, 256]   # actor MLP hidden layers
CRITIC_HIDDEN = [256, 256]   # critic MLP hidden layers

# Initial log-std for the Gaussian policy
# exp(-0.5) ≈ 0.6 → moderate initial exploration
LOG_STD_INIT = -0.5
LOG_STD_MIN  = -4.0   # clamp: min std ≈ 0.018 (near-deterministic)
LOG_STD_MAX  =  1.0   # clamp: max std ≈ 2.7   (very exploratory)

# ---------------------------------------------------------------------------
# PPO core hyperparameters
# ---------------------------------------------------------------------------
GAMMA          = 0.99     # discount factor
GAE_LAMBDA     = 0.95     # GAE λ — bias/variance trade-off
CLIP_EPS       = 0.2      # PPO clipping ε
ENTROPY_COEF   = 0.01     # entropy bonus coefficient (encourages exploration)
VALUE_COEF     = 0.5      # critic loss coefficient in combined loss
GRAD_CLIP      = 0.5      # max gradient norm

# ---------------------------------------------------------------------------
# Rollout & update schedule
# ---------------------------------------------------------------------------
ROLLOUT_STEPS  = 96 * 4   # steps collected per update (4 full days)
N_EPOCHS       = 10       # gradient epochs per rollout
MINI_BATCH_SIZE = 128     # mini-batch size within each epoch

# Learning rates (separate for actor/critic; set equal to use same)
ACTOR_LR  = 3e-4
CRITIC_LR = 1e-3          # critic typically benefits from a higher LR

# LR annealing: linearly decay to 0 over training
LR_ANNEAL = True

# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------
# Normalize advantages to zero mean / unit std within each mini-batch
NORMALIZE_ADVANTAGES = True

# Reward normalisation via running statistics (helps with scale changes)
NORMALIZE_REWARDS = True
REWARD_NORM_CLIP  = 10.0  # clip normalised reward to ±clip

# ---------------------------------------------------------------------------
# Logging / checkpointing
# ---------------------------------------------------------------------------
LOG_INTERVAL   = 10    # updates between console logs
SAVE_INTERVAL  = 50    # updates between checkpoint saves
CHECKPOINT_DIR = "checkpoints_ppo"
