# Home Energy Management System with Reinforcement Learning (HEMS-RL)

<div align="center">

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

**An intelligent energy management system using Soft Actor-Critic (SAC) reinforcement learning to optimize household energy consumption, battery charging, and EV charging.**

[Quick Start](#-quick-start) • [Installation](#-installation) • [Usage](#-usage) • [Results](#-results) • [Documentation](#-documentation)

</div>

---

## 🎯 Overview

This project implements a **Home Energy Management System (HEMS)** powered by **Soft Actor-Critic (SAC)** reinforcement learning. The system learns optimal control policies for:

- 🔋 **Battery Storage** - Charge/discharge decisions to minimize costs
- 🚗 **EV Charging** - Smart charging when prices are low
- ⚡ **Grid Interaction** - Minimize peak load and electricity costs
- 📊 **Real-time Learning** - Adapts to variable energy prices, PV generation, and consumption

### Key Features

✅ **Real-world Data** - Portuguese electricity prices, weather, PV generation, household loads  
✅ **SAC Agent** - State-of-the-art continuous control algorithm  
✅ **Modular Design** - Easily extend with new components (HVAC, heat pump, etc.)  
✅ **Comprehensive Logging** - Track all training metrics and visualize results  
✅ **Fast Training** - GPU support for efficient learning  

---

## 📋 Requirements

- **Python**: 3.9 or higher
- **OS**: Windows, macOS, Linux
- **GPU** (optional): NVIDIA GPU for faster training

---

## 🚀 Installation

### Option 1: Quick Install (Recommended)

```bash
# Clone the repository
git clone https://github.com/yourusername/home-energy-rl.git
cd home-energy-rl

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements-core.txt
```

### Option 2: Development Install

```bash
# Clone and setup
git clone https://github.com/yourusername/home-energy-rl.git
cd home-energy-rl

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install with development tools
pip install -r requirements.txt

# Install in editable mode
pip install -e .
```

### Option 3: Using Poetry (Alternative)

```bash
# Install poetry if you haven't already
pip install poetry

# Install dependencies
poetry install

# Activate environment
poetry shell
```

### Verify Installation

```bash
python -c "import torch; import pandas; import numpy; print('✓ All dependencies installed!')"
```

---

## 📂 Project Structure

```
home-energy-rl/
├── pyproject.toml                # Modern Python packaging
├── requirements.txt              # All dependencies
├── requirements-core.txt         # Production-only dependencies
├── README.md                     # This file
├── TRAINING_GUIDE.md            # Detailed training documentation
│
├── configs/                      # Configuration files
│   ├── env.yaml                  # Environment parameters (battery, EV specs)
│   └── agent.yaml                # Agent hyperparameters
│
├── data/                         # Data directory
│   ├── raw/                      # Original CSV files (excluded from git)
│   │   ├── Load House 1.csv
│   │   ├── Price Portugal.csv
│   │   ├── PV Generation House 1.csv
│   │   └── Weather House 1.csv
│   └── processed/                # Processed data
│
├── hems_core/                    # Main package
│   ├── __init__.py
│   ├── env/                      # Environment module
│   │   ├── __init__.py
│   │   ├── engine.py            # Energy environment simulator
│   │   ├── home_env.py          # Gymnasium wrapper
│   │   ├── rewards.py           # Reward functions
│   │   └── components/          # Physical components
│   │       ├── __init__.py
│   │       ├── battery.py       # Battery storage model
│   │       └── EV.py            # Electric vehicle model
│   │
│   ├── agents/                   # RL Agents
│   │   ├── __init__.py
│   │   ├── sac/                 # SAC Agent implementation
│   │   │   ├── __init__.py
│   │   │   ├── agent.py         # SAC agent
│   │   │   ├── networks.py      # Actor & Critic networks
│   │   │   ├── replay_buffer.py # Experience replay
│   │   │   └── config.py        # Hyperparameters
│   │   ├── dqn/                 # DQN (future)
│   │   └── ppo/                 # PPO (future)
│   │
│   └── utils/                    # Utilities
│       ├── __init__.py
│       └── data_functions.py    # Data loading & preprocessing
│
├── notebooks/                    # Jupyter notebooks
│   └── trial.ipynb              # Experimentation notebook
│
├── logs/                         # Training outputs (created after training)
│   └── YYYYMMDD_HHMMSS/
│       ├── config.json          # Training config
│       ├── training_episodes.csv # Episode summaries
│       ├── training_steps.csv   # Step-by-step data
│       ├── sac_agent_final.pt   # Trained model
│       └── plots/               # Visualizations
│
├── tests/                        # Test suite (future)
│
├── main.py                       # Training entry point
└── plot_results.py              # Visualization script
```

---

## ⚡ Quick Start

### 1. Prepare Data

Ensure your data files are in `data/raw/`:
- `Load House 1.csv` - Household consumption (kW)
- `Price Portugal.csv` - Electricity prices (EUR/MWh)
- `PV Generation House 1.csv` - Solar generation (W)
- `Weather House 1.csv` - Temperature and weather

### 2. Configure Environment

Edit `configs/env.yaml` to set battery and EV parameters:

```yaml
battery:
  capacity: 10.0              # kWh
  max_charge_rate: 5.0        # kW
  max_discharge_rate: 5.0     # kW
  efficiency: 0.9

ev:
  capacity: 60.0              # kWh
  max_charge_rate: 7.0        # kW
  max_discharge_rate: 7.0     # kW (V2G capable)
  efficiency: 0.9
  departure_time: 18.0        # Hours (6 PM)
  arrival_time: 8.0           # Hours (8 AM)
```

### 3. Train the Agent

```bash
python main.py
```

**Training Output:**
```
======================================================================
Starting SAC Training on HEMS Environment
======================================================================

[1] Initializing environment and agent...
✓ Environment initialized (state_dim=9, action_dim=2)
✓ Agent initialized on device: cuda
✓ Logs will be saved to: logs/20260428_120000

[2] Starting training loop...
----------------------------------------------------------------------
Episode    0 | Reward:     -98.45 | Cost:    45.23 EUR | Length: 96
Episode    1 | Reward:     -85.32 | Cost:    42.15 EUR | Length: 96
...
```

A timestamped directory `logs/YYYYMMDD_HHMMSS/` is created with:
- Training metrics CSV files
- Model checkpoints
- Configuration snapshot

### 4. Visualize Results

After training completes:

```bash
python plot_results.py logs/20260428_120000
```

Generates 5 plots:
1. 📈 Rewards & Costs over time
2. 📉 Training losses (Q1, Actor, Entropy)
3. 🎮 Actions & States (last 5 episodes)
4. 📊 Environment state variables
5. 📋 Statistics summary

All plots saved to `logs/YYYYMMDD_HHMMSS/plots/`

---

## 📖 Usage

### Training Configuration

In `main.py`, customize training:

```python
train_sac(
    num_episodes=100,        # Total training episodes (days)
    batch_size=256,          # Neural network batch size
    update_freq=1,           # Update every N steps
    model_save_freq=10       # Checkpoint every N episodes
)
```

### Load and Evaluate Trained Agent

```python
from hems_core.env.engine import EnergyEnv
from hems_core.agents.sac import SACAgent

# Setup
env = EnergyEnv()
agent = SACAgent(state_dim=9, action_dim=2)
agent.load("logs/YYYYMMDD_HHMMSS/sac_agent_final.pt")

# Test (no exploration, use mean action)
state = env.reset()
episode_reward = 0

for step in range(96):
    action = agent.select_action(state, training=False)
    next_state, reward, done, info = env.step(action)
    episode_reward += reward
    print(f"Step {step}: Cost={info['cost']:.4f} EUR, Grid={info['P_grid']:.2f} kW")
    state = next_state

print(f"Total Episode Reward: {episode_reward:.2f}")
```

### Access Training Data

```python
import pandas as pd

# Load episode summaries
episodes_df = pd.read_csv("logs/YYYYMMDD_HHMMSS/training_episodes.csv")
print(episodes_df.describe())

# Load step-by-step data
steps_df = pd.read_csv("logs/YYYYMMDD_HHMMSS/training_steps.csv")
print(steps_df.head())

# Analyze costs over training
import matplotlib.pyplot as plt
plt.plot(episodes_df['episode'], episodes_df['total_cost'])
plt.xlabel('Episode')
plt.ylabel('Daily Cost (EUR)')
plt.show()
```

---

## 📊 Results

Training results are automatically saved and can be analyzed:

### CSV Data Columns

**training_episodes.csv:**
- `episode` - Episode number
- `total_reward` - Sum of rewards for the day
- `total_cost` - Electricity cost (EUR)
- `avg_q1_loss`, `avg_actor_loss`, `avg_alpha_loss` - Training losses

**training_steps.csv:**
- `episode`, `step` - Episode and step indices
- `action_battery`, `action_ev` - Control actions
- `reward`, `cost` - Immediate feedback
- `state_*` - All 9 state variables
- `P_grid` - Grid power (kW)
- `q1_loss`, `actor_loss`, `alpha_loss` - Network losses

### Example Analysis

```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("logs/20260428_120000/training_episodes.csv")

print(f"Average Daily Cost: €{df['total_cost'].mean():.2f}")
print(f"Cost Reduction: {(1 - df['total_cost'].iloc[-10:].mean() / df['total_cost'].iloc[:10].mean()) * 100:.1f}%")
```

---

## 🔧 Advanced Configuration

### Adjust Hyperparameters

Edit `hems_core/agents/sac/config.py`:

```python
ACTOR_LR = 3e-4              # Actor learning rate
CRITIC_LR = 3e-4             # Critic learning rate
GAMMA = 0.99                 # Discount factor
TAU = 0.005                  # Target network update rate
BUFFER_CAPACITY = 10000      # Replay buffer size
```

### GPU Training

The system automatically detects and uses GPU if available:

```bash
# Force CPU
CUDA_VISIBLE_DEVICES=-1 python main.py

# Force GPU device 0
CUDA_VISIBLE_DEVICES=0 python main.py
```

---

## 📚 Documentation

- [TRAINING_GUIDE.md](TRAINING_GUIDE.md) - Detailed training instructions
- [SAC Algorithm](https://arxiv.org/abs/1801.01290) - Original paper

---

## 🤝 Contributing

Contributions welcome! Areas for enhancement:

- [ ] Additional agents (DQN, PPO, DDPG)
- [ ] More physical components (HVAC, heat pump)
- [ ] Multi-household simulations
- [ ] Real-time API integration
- [ ] Web dashboard for visualization

---

## 📝 License

MIT License - see LICENSE file for details

---

## 👤 Author

**Amine** - PFA2 Project

---

## 🙏 Acknowledgments

- Real data from Portuguese grid and household
- SAC algorithm implementation based on [Spinning Up](https://spinningup.openai.com/)
- PyTorch and the open-source ML community

---

## ❓ Troubleshooting

### Import Error: "No module named 'hems_core'"

```bash
# Install in editable mode
pip install -e .

# Or add to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### CUDA Memory Error

Reduce batch size in `main.py`:
```python
train_sac(num_episodes=100, batch_size=128)  # 256 → 128
```

### Data File Not Found

Ensure CSV files are in `data/raw/` with correct names matching `data_functions.py`

### Slow Training

- Use GPU: NVIDIA drivers installed + PyTorch GPU version
- Reduce `num_episodes` for testing
- Increase `batch_size` for stability

---

<div align="center">

**Questions?** Open an issue on GitHub!

⭐ If this project helped you, consider starring it! ⭐

</div>
