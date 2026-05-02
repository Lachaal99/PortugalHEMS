# SAC Configuration
ACTOR_LR = 3e-4
CRITIC_LR = 3e-4
ALPHA_LR = 3e-4

BUFFER_CAPACITY = 10000
BATCH_SIZE = 256

GAMMA = 0.99  # Discount factor
TAU = 0.005   # Soft update coefficient

HIDDEN_DIM = 256
LOG_STD_MIN = -20
LOG_STD_MAX = 2

# Target entropy for automatic entropy tuning
# Typically: -action_dim
TARGET_ENTROPY = None  # Will be set dynamically based on action_dim
