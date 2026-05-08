import torch
import torch.nn as nn
import torch.optim as optim
from .networks import Actor, Critic
from .replay_buffer import ReplayBuffer
from .config import (
    ACTOR_LR, CRITIC_LR, ALPHA_LR, GAMMA, TAU, TARGET_ENTROPY
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

class SACAgent:
    """Soft Actor-Critic (SAC) agent for continuous control."""
    
    def __init__(self, state_dim, action_dim, buffer_capacity=10000):
        """
        Initialize SAC agent.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            buffer_capacity: Capacity of replay buffer
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device

        # Actor network
        self.actor = Actor(state_dim, action_dim).to(device)
        
        # Q-value networks (critic)
        self.q1 = Critic(state_dim, action_dim).to(device)
        self.q2 = Critic(state_dim, action_dim).to(device)
        self.q1_target = Critic(state_dim, action_dim).to(device)
        self.q2_target = Critic(state_dim, action_dim).to(device)

        # Copy initial parameters to target networks
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        # Optimizers
        self.actor_opt = optim.Adam(self.actor.parameters(), lr=ACTOR_LR)
        self.q1_opt = optim.Adam(self.q1.parameters(), lr=CRITIC_LR)
        self.q2_opt = optim.Adam(self.q2.parameters(), lr=CRITIC_LR)

        # Entropy coefficient (automatic entropy tuning)
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha_opt = optim.Adam([self.log_alpha], lr=ALPHA_LR)
        self.target_entropy = TARGET_ENTROPY if TARGET_ENTROPY is not None else -action_dim

        # Hyperparameters
        self.gamma = GAMMA
        self.tau = TAU

        # Replay buffer
        self.buffer = ReplayBuffer(capacity=buffer_capacity)

    @property
    def alpha(self):
        """Temperature coefficient for entropy regularization."""
        return self.log_alpha.exp()

    def select_action(self, state, training=True):
        """
        Select action from policy.
        
        Args:
            state: Current state
            training: If True, sample from distribution. If False, use mean.
        
        Returns:
            Action (numpy array)
        """
        state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        
        with torch.no_grad():
            if training:
                action, _ = self.actor.sample(state_tensor)
            else:
                action, _ = self.actor.sample(state_tensor)
                action = self.actor(state_tensor)[0]
        actions = action.cpu().numpy().squeeze(0)
        actions[1]= (actions[1]+1) /2
        return actions

    def store_transition(self, s, a, r, s_next, done):
        """Store transition in replay buffer."""
        self.buffer.push(s, a, r, s_next, float(done))

    def update(self, batch_size=256):
        """
        Update actor and critic networks.
        
        Args:
            batch_size: Batch size for training
        """
        if len(self.buffer) < batch_size:
            return None, None, None

        s, a, r, s2, d = self.buffer.sample(batch_size)

        # ========== Critic Update ==========
        with torch.no_grad():
            a2, logp2 = self.actor.sample(s2)
            q1_t = self.q1_target(s2, a2)
            q2_t = self.q2_target(s2, a2)
            q_target = torch.min(q1_t, q2_t) - self.alpha * logp2
            y = r + self.gamma * (1 - d) * q_target

        q1 = self.q1(s, a)
        q2 = self.q2(s, a)

        q1_loss = nn.MSELoss()(q1, y)
        q2_loss = nn.MSELoss()(q2, y)

        # Update Q1
        self.q1_opt.zero_grad()
        q1_loss.backward()
        self.q1_opt.step()

        # Update Q2
        self.q2_opt.zero_grad()
        q2_loss.backward()
        self.q2_opt.step()

        # ========== Actor Update ==========
        a_new, logp = self.actor.sample(s)
        q1_new = self.q1(s, a_new)
        q2_new = self.q2(s, a_new)
        q_new = torch.min(q1_new, q2_new)

        actor_loss = (self.alpha * logp - q_new).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        # ========== Entropy (Alpha) Update ==========
        alpha_loss = -(self.log_alpha * (logp + self.target_entropy).detach()).mean()

        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()

        # ========== Target Network Update ==========
        self.soft_update(self.q1, self.q1_target)
        self.soft_update(self.q2, self.q2_target)

        return q1_loss.item(), actor_loss.item(), alpha_loss.item()

    def soft_update(self, net, target_net):
        """Soft update target network parameters."""
        for p, tp in zip(net.parameters(), target_net.parameters()):
            tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)

    def save(self, path="sac_agent.pt"):
        """Save agent weights."""
        checkpoint = {
            "actor": self.actor.state_dict(),
            "critic1": self.q1.state_dict(),
            "critic2": self.q2.state_dict(),
            "critic1_target": self.q1_target.state_dict(),
            "critic2_target": self.q2_target.state_dict(),
            "optimizerActor": self.actor_opt.state_dict(),
            "optimizerCritic1": self.q1_opt.state_dict(),
            "optimizerCritic2": self.q2_opt.state_dict(),
        }
        torch.save(checkpoint, path)
        print(f"Agent saved to {path}")

    def load(self, path="sac_agent.pt"):
        """Load agent weights."""
        checkpoint = torch.load(path, weights_only=True)
        self.actor.load_state_dict(checkpoint["actor"])
        self.q1.load_state_dict(checkpoint["critic1"])
        self.q2.load_state_dict(checkpoint["critic2"])
        self.q1_target.load_state_dict(checkpoint["critic1_target"])
        self.q2_target.load_state_dict(checkpoint["critic2_target"])
        self.actor_opt.load_state_dict(checkpoint["optimizerActor"])
        self.q1_opt.load_state_dict(checkpoint["optimizerCritic1"])
        self.q2_opt.load_state_dict(checkpoint["optimizerCritic2"])
        
        # Set to eval mode
        self.actor.eval()
        self.q1.eval()
        self.q2.eval()
        self.q1_target.eval()
        self.q2_target.eval()
        print(f"Agent loaded from {path}")
