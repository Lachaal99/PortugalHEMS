import numpy as np
import random
import torch
from collections import deque

# Determine device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ReplayBuffer:
    """Experience replay buffer for storing and sampling trajectories."""
    
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, s, a, r, s2, d):
        """Add experience to buffer."""
        self.buffer.append((s, a, r, s2, d))

    def sample(self, batch_size):
        """Sample a batch of experiences."""
        batch = random.sample(self.buffer, batch_size)
        s, a, r, s2, d = map(np.array, zip(*batch))
        return (
            torch.tensor(s, dtype=torch.float32, device=device),
            torch.tensor(a, dtype=torch.float32, device=device),
            torch.tensor(r, dtype=torch.float32, device=device).unsqueeze(1),
            torch.tensor(s2, dtype=torch.float32, device=device),
            torch.tensor(d, dtype=torch.float32, device=device).unsqueeze(1)
        )

    def __len__(self):
        return len(self.buffer)
