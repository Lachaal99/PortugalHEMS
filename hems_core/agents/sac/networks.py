import torch
import torch.nn as nn
from .config import HIDDEN_DIM, LOG_STD_MIN, LOG_STD_MAX

# Determine device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Actor(nn.Module):
    """Policy network that outputs mean and std of action distribution."""
    
    def __init__(self, state_dim, action_dim, hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        self.mu = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)

        self.LOG_STD_MIN = LOG_STD_MIN
        self.LOG_STD_MAX = LOG_STD_MAX

    def forward(self, state):
        """Return mean and std of action distribution."""
        x = self.net(state)
        mu = self.mu(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)
        std = torch.exp(log_std)
        return mu, std

    def sample(self, state):
        """Sample action and compute log probability."""
        mu, std = self(state)
        normal = torch.distributions.Normal(mu, std)
        z = normal.rsample()
        action = torch.tanh(z)

        # Correct log probability for tanh transformation
        log_prob = normal.log_prob(z) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=1, keepdim=True)

        return action, log_prob


class Critic(nn.Module):
    """Q-value network that takes state and action as input."""
    
    def __init__(self, state_dim, action_dim, hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state, action):
        """Return Q-value for state-action pair."""
        x = torch.cat([state, action], dim=1)
        return self.net(x)
