# Home Energy Management System with Reinforcement Learning (HEMS-RL)

**Complete Project Documentation**

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Quick Start](#quick-start)
3. [Installation](#installation)
4. [Project Structure](#project-structure)
5. [Environment Design](#environment-design)
6. [Data Sources](#data-sources)
7. [Reinforcement Learning Agents](#reinforcement-learning-agents)
8. [Training Guide](#training-guide)
9. [Data Logging & Analysis](#data-logging--analysis)
10. [Visualization & Results](#visualization--results)
11. [Troubleshooting](#troubleshooting)
12. [References](#references)

---

# Project Overview

## 🎯 Objective

This project implements a **Home Energy Management System (HEMS)** powered by **Reinforcement Learning** to optimize household energy consumption. The system learns optimal control policies for:

- 🔋 **Battery Storage** - Charge/discharge decisions to minimize costs
- 🚗 **EV Charging** - Smart charging when prices are low
- ⚡ **Grid Interaction** - Minimize peak load and electricity costs
- 📊 **Real-time Learning** - Adapts to variable energy prices, PV generation, and consumption

## ✨ Key Features

- ✅ **Real-world Data** - Portuguese electricity prices, weather, PV generation, household loads
- ✅ **Multiple Algorithms** - SAC (Soft Actor-Critic), DQN, and PPO agents
- ✅ **Modular Design** - Easily extend with new components
- ✅ **Comprehensive Logging** - Track all training metrics and visualize results
- ✅ **GPU Support** - Fast training with NVIDIA GPU
- ✅ **Continuous Action Space** - Smooth, realistic control

## 📋 Requirements

- **Python**: 3.9 or higher
- **OS**: Windows, macOS, Linux
- **GPU** (optional): NVIDIA GPU for faster training

---

# Quick Start

## 1. Basic Setup

```bash
# Clone or navigate to project
cd home-energy-rl

# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # macOS/Linux

# Install dependencies
pip install -r requirements-core.txt
```

## 2. Train an Agent (SAC)

```bash
python main.py
```

Training will create a timestamped directory in `logs/YYYYMMDD_HHMMSS/` containing:
- Model checkpoints
- Training metrics (CSV files)
- Configuration snapshot

## 3. Visualize Results

```bash
python plot_results.py logs/YYYYMMDD_HHMMSS
```

Generates 5 comprehensive plots:
1. Rewards & Costs over time
2. Agent-specific losses
3. Actions & States (last episodes)
4. State Variables Distribution
5. Statistics Summary

---

# Installation

## Option 1: Quick Install (Recommended)

```bash
git clone https://github.com/Lachaal99/PortugalHEMS.git
cd PortugalHEMS

python -m venv venv
venv\Scripts\activate  # or `source venv/bin/activate` on macOS/Linux

pip install -r requirements-core.txt
```

## Option 2: Development Install

```bash
python -m venv venv
venv\Scripts\activate  # or `source venv/bin/activate`

pip install -r requirements.txt
pip install -e .
```

## Option 3: Poetry

```bash
pip install poetry
poetry install
poetry shell
```

## Verify Installation

```bash
python -c "import torch; import pandas; import numpy; print('✓ All dependencies installed!')"
```

---

# Project Structure

```
home-energy-rl/
├── DOCUMENTATION.md              # This file - Complete reference
├── pyproject.toml                # Modern Python packaging
├── requirements.txt              # All dependencies
├── requirements-core.txt         # Production dependencies
├── README.md                     # Quick overview
│
├── configs/                      # Configuration files
│   ├── env.yaml                  # Battery, EV specifications
│   └── agent.yaml                # Agent hyperparameters
│
├── data/                         # Data directory
│   ├── raw/                      # Original CSV files
│   │   ├── Load House 1.csv
│   │   ├── Price Portugal.csv
│   │   ├── PV Generation House 1.csv
│   │   └── Weather House 1.csv
│   └── processed/                # Processed data
│
├── analysis/                     # Data analysis scripts
│   ├── analyze_electricity_prices.py
│   ├── analyze_load_profile.py
│   └── analyze_pv_generation.py
│
├── hems_core/                    # Main package
│   ├── __init__.py
│   ├── env/                      # Environment module
│   │   ├── __init__.py
│   │   ├── engine.py             # Energy environment simulator
│   │   ├── rewards.py            # Reward functions
│   │   └── components/
│   │       ├── battery.py        # Battery storage model
│   │       └── EV.py             # Electric vehicle model
│   │
│   ├── agents/                   # RL Agents
│   │   ├── __init__.py
│   │   ├── sac/                  # SAC Agent
│   │   │   ├── agent.py
│   │   │   ├── networks.py
│   │   │   ├── replay_buffer.py
│   │   │   └── config.py
│   │   ├── dqn/                  # DQN Agent
│   │   └── ppo/                  # PPO Agent
│   │
│   └── utils/
│       ├── __init__.py
│       └── data_functions.py     # Data loading
│
├── notebooks/                    # Jupyter notebooks
│   └── trial.ipynb
│
├── logs/                         # Training outputs (auto-created)
│   └── YYYYMMDD_HHMMSS/
│       ├── config.json
│       ├── training_episodes.csv
│       ├── training_steps.csv
│       ├── agent_model.pt        # Final model
│       └── plots/                # Generated visualizations
│
├── tests/                        # Test suite
│
├── main.py                       # Training entry point
└── plot_results.py               # Visualization script
```

---

# Environment Design

## Overview

The HEMS environment simulates a household with:
- **Battery storage** (10 kWh, 5 kW charge/discharge)
- **Electric vehicle** (60 kWh, 7 kW charging)
- **Variable electricity prices** (15-minute resolution)
- **Solar PV generation** (weather-dependent)
- **Household consumption** (time-varying loads)

## Temporal Resolution

- **Time step**: 0.25 hours = 15 minutes
- **Episode length**: 96 steps = 1 full day
- **Data frequency**: 15-minute intervals

## State Space (Dimension: 9)

The environment provides a normalized state vector `s ∈ [0,1]^9`:

| Index | Component | Range | Meaning |
|-------|-----------|-------|---------|
| 0 | Temperature | [0,1] | Outdoor temperature (-10°C to 40°C) |
| 1 | Load | [0,1] | Household consumption (normalized) |
| 2 | Battery SOC | [0,1] | Battery state of charge |
| 3 | EV SOC | [0,1] | EV battery state of charge |
| 4 | PV Generation | [0,1] | Solar power available (normalized) |
| 5 | Electricity Price | [0,1] | Grid electricity price (normalized) |
| 6 | Season | [0,1] | Time of year (winter/spring/summer/autumn) |
| 7 | Day Type | [0,1] | Weekday vs Weekend |
| 8 | Hour of Day | [0,1] | Time within the day |

**Normalization Strategy:**
- Temperature: Linear mapping from [-10°C, 40°C]
- Load, PV, Price: 90th percentile as maximum
- SOC: Direct state of charge (0 = empty, 1 = full)
- Temporal: Sinusoidal/circular encoding

## Action Space

### Continuous Actions (SAC, PPO)

- **Dimension**: 2-dimensional continuous vector
- **Raw space**: `a = [a_bat, a_ev] ∈ [-1, 1]^2`
- **Mapping**:
  - `a_bat ∈ [-1, 1]`: Battery power (-5 kW to +5 kW)
  - `a_ev ∈ [0, 1]`: EV charging power (0 to +7 kW)

### Discrete Actions (DQN)

- **Battery levels**: 7 values (linspace from -1 to 1)
- **EV levels**: 7 values (linspace from -1 to 1)
- **Total actions**: 7 × 7 = 49 discrete actions

## Battery Component

| Property | Value |
|----------|-------|
| Capacity | 10 kWh |
| Max Charge Rate | 5 kW |
| Max Discharge Rate | 5 kW |
| Round-trip Efficiency | 90% |
| Initial SOC | 0 (empty) |

**Dynamics:**
```
Energy(t) = action(t) × max_charge_rate × dt
SOC(t+1) = clip([0,1], SOC(t) + Energy(t) × efficiency / capacity)
```

**Reward:**
```
bat_reward = -penalty_for_SOC_violations
Penalties for exceeding bounds or violating constraints
```

## Electric Vehicle Component

| Property | Value |
|----------|-------|
| Capacity | 60 kWh |
| Max Charge Rate | 7 kW |
| V2H Capability | Yes (can discharge) |
| Efficiency | 90% |
| Arrival Time | 6 PM (hour 72) |
| Departure Time | 8 AM (hour 32) |

**Constraints:**
- Only charges when plugged in (6 PM to 8 AM)
- Discharge attempts penalized: -5.0
- Charge when not plugged penalized: -5.0
- Departure with SOC < 80%: -10.0 × (1 - SOC)

## Power Balance & Reward

**Power Balance Equation:**
```
P_grid(t) = max(0, P_bat(t) + P_load(t) + P_ev(t) - P_pv(t))
```

Where:
- `P_bat`: Battery power (discharge > 0, charge < 0)
- `P_load`: Non-flexible household load
- `P_ev`: EV charging power
- `P_pv`: Solar PV generation

**Cost Calculation:**
```
cost(t) = P_grid(t) × price(t) × dt  [EUR per 15-min interval]
```

**Step Reward:**
```
reward(t) = -cost(t) + bat_reward(t) + ev_reward(t)
```

**Episode Objective:**
```
J = Σ(t=0 to 95) reward(t) = minimize total daily cost while respecting constraints
```

---

# Data Sources

## Location

All data files are in `data/raw/`:

```
data/raw/
├── Load House 1.csv              # Household consumption
├── PV Generation House 1.csv     # Solar generation
├── Price Portugal.csv            # Electricity prices
└── Weather House 1.csv           # Temperature & weather
```

## Data Files

### Load House 1.csv
- **Description**: Household electricity consumption (non-flexible baseline)
- **Resolution**: 15-minute intervals
- **Columns**: DateTime, Consumption (kW)
- **Unit**: kilowatts (kW)
- **Usage**: Determines baseline household demand

### PV Generation House 1.csv
- **Description**: Solar panel generation data
- **Resolution**: 15-minute intervals
- **Columns**: Timestamp, PV Power Generation (W)
- **Unit**: Watts (W)
- **Usage**: Available renewable energy at each step

### Price Portugal.csv
- **Description**: Portuguese electricity market prices
- **Resolution**: Hourly (interpolated to 15-minute)
- **Columns**: Timestamp, Price (EUR/MWh)
- **Unit**: EUR per megawatt-hour
- **Usage**: Determines cost, drives optimization

### Weather House 1.csv
- **Description**: Meteorological data
- **Resolution**: 15-minute intervals
- **Columns**: Timestamp, Temperature (°C), [other weather]
- **Unit**: Celsius
- **Usage**: Environmental context for decisions

## Data Characteristics

- **Date Range**: Starting March 1, 2021
- **Duration**: Continuous time series
- **Frequency**: 15-minute intervals (96 observations per day)
- **Normalization**: 90th percentile normalization per dataset
- **Missing Data**: Fallback to default values

## Data Access (Code)

```python
from hems_core.utils.data_functions import (
    load_pv_data, load_price_data, 
    load_weather_data, load_load_data,
    pv_profile, price_profile,
    outdoor_temperature, load_profile
)

# Load complete datasets
pv_df = load_pv_data('data/raw/PV Generation House 1.csv')
price_df = load_price_data('data/raw/Price Portugal.csv')

# Access by 15-minute index
pv_watts = pv_profile(idx=24)  # 24 = 6 hours into day
price_eur_per_mwh = price_profile(idx=24)
temp_celsius = outdoor_temperature(idx=24)
load_kw = load_profile(idx=24)
```

## Data Indexing

All data indexed by 15-minute intervals:
- `idx=0`: March 1, 2021 00:00
- `idx=1`: March 1, 2021 00:15
- `idx=95`: March 1, 2021 23:45
- `idx=96`: March 2, 2021 00:00

---

# Reinforcement Learning Agents

## Overview

Three RL agents are implemented, supporting both continuous and discrete action spaces:

### Agent Comparison

| Feature | SAC | PPO | DQN |
|---------|-----|-----|-----|
| **Action Space** | Continuous | Continuous | Discrete |
| **Algorithm Type** | Off-Policy | On-Policy | Off-Policy |
| **Exploration** | Entropy Regularization | Stochastic Policy | ε-Greedy |
| **Experience Replay** | Yes | Rollout Only | Yes |
| **Update Frequency** | Every Step | Every 384 Steps | Every Step |
| **Sample Efficiency** | High | Medium | High |
| **Training Stability** | Very Stable | Stable | Can be Unstable |
| **Best For** | Most Cases | Resource-Constrained | Discrete Spaces |

## SAC - Soft Actor-Critic

### Overview
Maximum entropy RL algorithm that learns a stochastic policy while maximizing entropy for exploration.

### Architecture
- **Actor Network**: Outputs mean and std dev of Gaussian policy
- **Critic Networks**: 2 independent Q-function networks (for double Q-learning)
- **Automatic Entropy Tuning**: Learns the entropy coefficient dynamically

### Hyperparameters (in `hems_core/agents/sac/config.py`)
```
ACTOR_LR = 3e-4              # Actor learning rate
CRITIC_LR = 3e-4             # Critic learning rate
ALPHA_LR = 3e-4              # Entropy coefficient learning rate
GAMMA = 0.99                 # Discount factor
TAU = 5e-3                   # Target network update rate
BATCH_SIZE = 256             # Mini-batch size
REPLAY_BUFFER_SIZE = 1e6     # Experience replay capacity
INITIAL_ALPHA = 0.2          # Initial entropy coefficient
```

### Training Workflow
```python
for each step:
    1. Select action: a ~ π(·|s)
    2. Execute in environment: (r, s')
    3. Store transition in replay buffer
    4. Sample batch from buffer
    5. Update Q-networks (MSE loss)
    6. Update actor (policy gradient)
    7. Update entropy coefficient α
    8. Update target networks (soft update)
```

## PPO - Proximal Policy Optimization

### Overview
On-policy algorithm using trust region optimization to ensure stable policy updates.

### Architecture
- **Actor Network**: Outputs mean of Gaussian policy
- **Critic Network**: Single value function network
- **GAE**: Generalized Advantage Estimation for low-variance advantage

### Hyperparameters (in `hems_core/agents/ppo/config.py`)
```
ROLLOUT_STEPS = 384          # Collect 4 full days before update
N_EPOCHS = 10                # Gradient epochs per rollout
MINI_BATCH_SIZE = 128        # Mini-batch size within epochs
ACTOR_LR = 3e-4              # Actor learning rate
CRITIC_LR = 1e-3             # Critic learning rate
CLIP_EPS = 0.2               # PPO clipping coefficient
ENTROPY_COEF = 0.01          # Entropy bonus coefficient
GAMMA = 0.99                 # Discount factor
GAE_LAMBDA = 0.95            # GAE lambda
```

### Key Characteristics
- **On-Policy**: Uses data only once, then discards (sample efficient)
- **Rollout Mechanism**: Collects 384 steps (4 days) before each update
- **Trust Region**: Clips policy ratio to prevent large updates
- **Update Frequency**: Every 384 steps (every ~4 episodes)

### Training Workflow
```python
for each update cycle (384 steps):
    1. Collect 384 transitions into rollout buffer
    2. Compute advantages using GAE
    3. For 10 epochs:
        - Sample mini-batches from buffer
        - Compute policy gradient (with clipping)
        - Compute value function loss
        - Update actor and critic
    4. Compute metrics (entropy, clip_frac)
    5. Decay learning rate
    6. Clear rollout buffer
```

## DQN - Double Deep Q-Network

### Overview
Value-based algorithm with dueling architecture for discrete action spaces.

### Architecture
- **Dueling Network**: Shared trunk + separate value and advantage streams
- **Double Q-Learning**: Addresses overestimation bias
- **Target Network**: Stable targets for training

### Hyperparameters (in `hems_core/agents/dqn/config.py`)
```
LEARNING_RATE = 1e-4         # Network learning rate
GAMMA = 0.99                 # Discount factor
TAU = 1e-2                   # Target network update rate
EPSILON_START = 1.0          # Initial exploration rate
EPSILON_END = 0.05           # Final exploration rate
EPSILON_DECAY = 0.9995       # Decay rate per episode
BATCH_SIZE = 32              # Mini-batch size
REPLAY_BUFFER_SIZE = 1e5     # Experience replay capacity
```

### Key Characteristics
- **Discrete Actions**: 49 discrete actions (7×7 grid)
- **Dueling Architecture**: Value(s) + Advantage(s,a) formulation
- **Double Q-Learning**: Mitigates value overestimation
- **ε-Greedy Exploration**: Decays from 1.0 to 0.05

---

# Training Guide

## Basic Training (SAC)

### Configuration

In `main.py`, set training parameters:

```python
if __name__ == "__main__":
    train_sac(
        num_episodes=100,        # Training episodes
        batch_size=256,          # Batch size
        update_freq=1,           # Update frequency
        model_save_freq=10       # Save every N episodes
    )
```

### Running Training

```bash
python main.py
```

**Expected Console Output:**
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
Episode    2 | Reward:     -78.91 | Cost:    39.80 EUR | Length: 96
...
Episode   99 | Reward:     -25.15 | Cost:    15.50 EUR | Length: 96
----------------------------------------------------------------------

[3] Training completed!
✓ Final model saved: logs/20260428_120000/sac_agent_final.pt

[4] Saving training data...
✓ Training data saved:
  - logs/20260428_120000/training_episodes.csv
  - logs/20260428_120000/training_steps.csv

[5] Training Summary:
  - Total Episodes: 100
  - Total Steps: 9600
  - Avg Episode Reward: -55.32
  - Best Episode Reward: -20.15
  - Total Cost: 4150.00 EUR
```

## PPO Training

### Configuration

```python
if __name__ == "__main__":
    train_ppo(
        num_episodes=200,        # Training episodes
        model_save_freq=20       # Save every 20 episodes
    )
```

### Key Differences from SAC

1. **Update Frequency**: Updates every 384 steps (~4 episodes), not every step
2. **Data Handling**: Discards data after update (on-policy)
3. **Metrics**: Policy loss, value loss, entropy, clip_frac
4. **Performance Dips**: Expect performance variations every 4 episodes (normal!)

### Interpreting PPO Metrics

- **Policy Loss**: Should gradually decrease (lower = better policy)
- **Value Loss**: Should decrease (critic learning state values)
- **Entropy**: Starts high, decreases over time (exploring → exploiting)
- **Clip Fraction**: Diagnostic metric; ~10-30% is typical

## DQN Training

### Configuration

```python
if __name__ == "__main__":
    train_dqn(
        num_episodes=300,        # More episodes often needed
        model_save_freq=50       # Save every 50 episodes
    )
```

### Special Considerations

- **Discrete Actions**: Only 49 possible actions
- **ε-Greedy Exploration**: Decays over time (linear schedule)
- **Slower Convergence**: May need more episodes than SAC/PPO
- **Overestimation**: Double Q-learning mitigates this issue

## Training Tips

### General
- **GPU Support**: Training is faster with NVIDIA GPU
- **Long Runs**: For 1000+ episodes, use background execution (tmux, nohup)
- **Batch Size**: Larger = more stable but slower; smaller = faster but noisier
- **Update Frequency**: More frequent updates = longer training but better learning

### SAC-Specific
- **Entropy Tuning**: Check that α is reasonable (typically 0.1-0.3)
- **Learning Rates**: Reduce if training becomes unstable
- **Replay Buffer**: Large buffer = slower updates but better data mixing

### PPO-Specific
- **Rollout Size**: Increase ROLLOUT_STEPS for more stable updates (trade-off: less frequent)
- **Clip Coefficient**: Increase CLIP_EPS for more aggressive updates (default 0.2 is usually good)
- **Entropy Coefficient**: Increase to encourage exploration, decrease to encourage exploitation

### DQN-Specific
- **Epsilon Decay**: Adjust EPSILON_DECAY for faster/slower exploration decay
- **Target Network Update**: Slower updates (higher TAU) = more stable but slower convergence
- **Network Size**: Larger networks = better approximation but more prone to overfitting

---

# Data Logging & Analysis

## What Gets Saved

Each training run creates a timestamped directory in `logs/YYYYMMDD_HHMMSS/`:

```
logs/20260504_143200/
├── config.json                          # Training hyperparameters
├── training_episodes.csv                # Daily metrics (1 row per episode)
├── training_steps.csv                   # Step data (96 rows per episode)
├── sac_agent_ep10.pt                    # Model checkpoint every N episodes
├── sac_agent_ep20.pt
├── sac_agent_final.pt                   # Final trained model
└── plots/                               # Generated visualizations
    ├── 01_rewards_costs.png
    ├── 02_losses.png
    ├── 03_actions_states_last_episodes.png
    ├── 04_state_variables.png
    └── 05_statistics_summary.txt.png
```

## CSV Data Format

### training_episodes.csv
One row per episode with daily aggregates:

```csv
episode,total_reward,total_cost,length,avg_q1_loss,avg_actor_loss,avg_alpha_loss
0,-98.45,45.23,96,0.823,0.456,0.123
1,-85.32,42.15,96,0.812,0.445,0.119
2,-78.91,39.80,96,0.798,0.432,0.115
...
```

**Columns:**
- `episode`: Episode number (0-indexed)
- `total_reward`: Sum of rewards for the day
- `total_cost`: Daily electricity cost (EUR)
- `length`: Episode length (always 96 for full day)
- `avg_*_loss`: Average training losses for the day

**Agent-Specific Loss Columns:**
- SAC: `avg_q1_loss`, `avg_actor_loss`, `avg_alpha_loss`
- PPO: `avg_policy_loss`, `avg_value_loss`, `avg_entropy`, `avg_clip_frac`
- DQN: `avg_q_loss`, `avg_epsilon`

### training_steps.csv
One row per step (96 rows per episode):

```csv
episode,step,action_battery,action_ev,reward,cost,state_temperature,...,q1_loss,actor_loss,alpha_loss
0,0,0.234,-0.156,-1.234,0.123,0.45,...,0.823,0.456,0.123
0,1,0.145,-0.234,-0.945,0.098,0.42,...,0.812,0.445,0.119
...
```

**State Columns (state_*):**
- `state_temperature`, `state_load`, `state_battery_soc`
- `state_ev_soc`, `state_pv`, `state_price`
- `state_season`, `state_day_type`, `state_hour`

### config.json
Training configuration snapshot:

```json
{
    "algorithm": "SAC",
    "num_episodes": 100,
    "batch_size": 256,
    "learning_rate": 3e-4,
    "device": "cuda",
    "timestamp": "2026-05-04 14:32:00"
}
```

## Analyzing Training Data

### Python Analysis

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load episode data
episodes_df = pd.read_csv("logs/20260504_143200/training_episodes.csv")

# Basic statistics
print(episodes_df.describe())

# Cost improvement
cost_improvement = episodes_df['total_cost'].iloc[0] - episodes_df['total_cost'].iloc[-1]
print(f"Cost savings: {cost_improvement:.2f} EUR ({cost_improvement/episodes_df['total_cost'].iloc[0]*100:.1f}%)")

# Average final cost
avg_final_cost = episodes_df['total_cost'].tail(10).mean()
print(f"Average final daily cost: {avg_final_cost:.2f} EUR")

# Load step data
steps_df = pd.read_csv("logs/20260504_143200/training_steps.csv")

# Analyze actions
print(f"\nBattery action range: [{steps_df['action_battery'].min():.2f}, {steps_df['action_battery'].max():.2f}]")
print(f"EV action range: [{steps_df['action_ev'].min():.2f}, {steps_df['action_ev'].max():.2f}]")
```

### Excel Analysis

Open CSV files in Excel for:
- Pivot tables of costs by season
- Charts of cost trends
- Filtering by specific episodes or time periods

---

# Visualization & Results

## Automatic Plotting

After training, generate visualizations:

```bash
python plot_results.py logs/YYYYMMDD_HHMMSS
```

## Plot Details

### Plot 1: Rewards & Costs
- **Top**: Episode rewards over training
- **Bottom**: Daily electricity costs
- **Interpretation**:
  - Rewards increasing = better policy learned
  - Costs decreasing = cost optimization working

### Plot 2: Training Losses (Algorithm-Specific)

**SAC:**
- **Q1 Loss**: Q-network MSE loss (should plateau)
- **Actor Loss**: Policy gradient loss
- **Alpha Loss**: Entropy coefficient learning

**PPO:**
- **Policy Loss**: Actor network loss (decreasing = better policy)
- **Value Loss**: Critic network loss
- **Entropy**: Exploration metric (should decrease over time)
- **Clip Frac**: Fraction of updates with clipping (diagnostic)

**DQN:**
- **Q Loss**: Value network MSE loss
- **Epsilon**: Exploration rate (should decay)

### Plot 3: Actions & States (Last 5 Episodes)
- **Top 2 Rows**: Battery and EV actions over time
- **Bottom 3 Rows**: Battery SOC, EV SOC, Grid Power
- **Interpretation**: Shows learned control policy in action

### Plot 4: State Variables Distribution
- Histograms of each state variable over training
- Shows what states the agent experienced

### Plot 5: Statistics Summary
- Summary statistics table
- Key metrics visualization

## Comparing Algorithms

```bash
# After training all three algorithms:
python plot_results.py logs/YYYYMMDD_HHMMSS1  # SAC results
python plot_results.py logs/YYYYMMDD_HHMMSS2  # PPO results
python plot_results.py logs/YYYYMMDD_HHMMSS3  # DQN results
```

Then compare:
- Final costs achieved
- Training stability (loss curves)
- Convergence speed
- Sample efficiency

---

# Troubleshooting

## Training Issues

### Training is Very Slow
**Possible causes:**
- Using CPU instead of GPU
- Batch size too large
- Network too large

**Solutions:**
```python
# Check device
print(torch.cuda.is_available())  # Should be True

# Reduce batch size in config
BATCH_SIZE = 128  # from 256

# Reduce network size in networks.py
hidden_dim = 128  # from 256
```

### Training is Unstable (Rewards fluctuating)
**Possible causes:**
- Learning rates too high
- Batch size too small
- Discount factor incorrect

**Solutions:**
```python
# Reduce learning rates
ACTOR_LR = 1e-4  # from 3e-4
CRITIC_LR = 1e-4

# Increase batch size
BATCH_SIZE = 512

# SAC specific: check entropy coefficient
print(f"Current alpha: {agent.alpha:.4f}")
```

### Out of Memory Error
**Solution:**
```python
# Reduce batch size
BATCH_SIZE = 64

# Reduce replay buffer size (SAC)
REPLAY_BUFFER_SIZE = 5e5

# Reduce rollout steps (PPO)
ROLLOUT_STEPS = 192  # from 384
```

### Low Entropy (Policy Too Deterministic)
**For PPO:**
```python
ENTROPY_COEF = 0.05  # increase from 0.01
```

**For SAC:**
```python
INITIAL_ALPHA = 0.5  # increase from 0.2
```

### High Value Loss (Critic Not Learning)
**Solution:**
```python
CRITIC_LR = 5e-3  # increase learning rate
```

## Data Issues

### Missing Data Files
```
FileNotFoundError: data/raw/Load House 1.csv not found
```

**Solution:**
- Ensure all CSV files are in `data/raw/` directory
- Check filenames match exactly
- Use data analysis scripts in `analysis/` folder to verify

### Data Not Loading Correctly
**Solution:**
```python
# Debug data loading
from hems_core.utils.data_functions import load_pv_data

df = load_pv_data('data/raw/PV Generation House 1.csv')
print(df.head())
print(df.dtypes)
print(df.isnull().sum())
```

## Plotting Issues

### ImportError for Matplotlib
```
ModuleNotFoundError: No module named 'matplotlib'
```

**Solution:**
```bash
pip install matplotlib
```

### Plot Generation Failed
**Solution:**
```bash
# Check log directory exists
ls logs/YYYYMMDD_HHMMSS/

# Check CSV files exist
ls logs/YYYYMMDD_HHMMSS/training_*.csv

# Try with verbose output
python -u plot_results.py logs/YYYYMMDD_HHMMSS
```

---

# Advanced Usage

## Load and Resume Training

```python
from hems_core.agents.sac import SACAgent
from hems_core.env.engine import EnergyEnv

# Load existing agent
agent = SACAgent(state_dim=9, action_dim=2)
agent.load("logs/YYYYMMDD_HHMMSS/sac_agent_ep100.pt")

# Continue training
env = EnergyEnv()
# ... continue training loop
```

## Test Trained Agent

```python
from hems_core.env.engine import EnergyEnv
from hems_core.agents.sac import SACAgent

env = EnergyEnv()
agent = SACAgent(state_dim=9, action_dim=2)
agent.load("logs/YYYYMMDD_HHMMSS/sac_agent_final.pt")

# Evaluate for one episode
state = env.reset()
episode_reward = 0
episode_cost = 0

for step in range(96):
    # Use mean action (no exploration)
    action = agent.select_action(state, training=False)
    
    next_state, reward, done, info = env.step(action)
    episode_reward += reward
    episode_cost += info['cost']
    
    print(f"Step {step:2d}: Cost={info['cost']:.4f} EUR, "
          f"Grid={info['P_grid']:.2f} kW, "
          f"SOC={info['battery_soc']:.2f}")
    
    state = next_state

print(f"\nEpisode Total Cost: {episode_cost:.2f} EUR")
print(f"Episode Total Reward: {episode_reward:.2f}")
```

## Custom Reward Function

Edit `hems_core/env/rewards.py` to implement custom rewards:

```python
def compute_reward(cost, battery_soc, ev_soc, P_grid, ...):
    """Implement custom reward function"""
    
    # Default: minimize cost + penalties
    reward = -cost
    
    # Custom: prioritize peak shaving
    peak_penalty = max(0, P_grid - 3.0) * 2.0
    reward -= peak_penalty
    
    # Custom: battery degradation cost
    battery_degradation_penalty = abs(action_battery) * 0.01
    reward -= battery_degradation_penalty
    
    return reward
```

---

# References

## Academic Papers

- **PPO**: https://arxiv.org/abs/1707.06347 - Proximal Policy Optimization Algorithms
- **SAC**: https://arxiv.org/abs/1801.01290 - Soft Actor-Critic Algorithms
- **DQN**: https://arxiv.org/abs/1312.5602 - Playing Atari with Deep Reinforcement Learning
- **Double DQN**: https://arxiv.org/abs/1509.06461 - Deep Reinforcement Learning with Double Q-learning
- **GAE**: https://arxiv.org/abs/1506.02438 - High-Dimensional Continuous Control Using GAE

## Resources

- **OpenAI Spinning Up**: https://spinningup.openai.com/
- **PyTorch**: https://pytorch.org/
- **Gymnasium**: https://gymnasium.farama.org/
- **Python RL**: https://github.com/rll/rllabTorch

## Portuguese Energy Market

- **INESC**: https://www.inesc-id.pt/ - Energy research institute
- **IPCED**: Portuguese electricity pricing data
- **REN**: Renewable energy statistics (Portugal)

---

## Document History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2026-05-07 | Initial consolidated documentation |

---

**Last Updated**: May 7, 2026

For questions or issues, check the specific agent documentation or run examples in `analysis/` folder.
