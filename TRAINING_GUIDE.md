# HEMS Training Guide

## 🚀 Quick Start

### 1. Train the SAC Agent

Run from the project root:
```bash
python main.py
```

**Training Configuration** (edit in main.py):
- `num_episodes=100` - Number of training episodes
- `batch_size=256` - Batch size for network updates  
- `update_freq=1` - Update every N steps
- `model_save_freq=10` - Save model every N episodes

### 2. What Gets Saved

Training creates a timestamped directory in `logs/YYYYMMDD_HHMMSS/`:

```
logs/20240428_120000/
├── config.json                      # Training hyperparameters
├── training_episodes.csv            # Episode-level metrics
├── training_steps.csv               # Step-level data (detailed)
├── sac_agent_ep10.pt                # Model checkpoint every N episodes
├── sac_agent_ep20.pt
├── sac_agent_final.pt               # Final trained model
└── plots/                           # Generated after visualization
    ├── 01_rewards_costs.png
    ├── 02_losses.png
    ├── 03_actions_states_last_episodes.png
    ├── 04_state_variables.png
    └── 05_statistics_summary.txt.png
```

### 3. Visualize Training Results

After training completes, plot the results:
```bash
python plot_results.py logs/20240428_120000
```

This generates 5 comprehensive plots:
1. **Rewards & Costs** - Episode rewards and daily costs
2. **Losses** - Q1, Actor, and Entropy losses during training
3. **Actions & States (Last 5 Episodes)** - Battery/EV control and SOC
4. **State Variables** - Environment inputs over training
5. **Statistics Summary** - Summary metrics

## 📊 Saved Data Fields

### `training_episodes.csv`
- `episode` - Episode number
- `total_reward` - Sum of rewards for the episode
- `total_cost` - Daily electricity cost (EUR)
- `length` - Episode length (96 steps)
- `avg_q1_loss`, `avg_actor_loss`, `avg_alpha_loss` - Training losses

### `training_steps.csv`
- `episode`, `step` - Episode and step indices
- `action_battery` - Battery action [-1, 1]
- `action_ev` - EV action [0, 1]
- `reward` - Immediate reward
- `cost` - Step cost (EUR)
- `ev_reward`, `bat_reward` - Component rewards
- `P_grid` - Grid power draw (kW)
- `state_*` - All state variables (temp, load, SOC, PV, price, etc.)
- `q1_loss`, `actor_loss`, `alpha_loss` - Training losses (if updated)

## 🎯 Key Metrics to Monitor

1. **Total Reward** - Should increase over training
2. **Total Cost** - Should decrease (goal: minimize electricity cost)
3. **Actor Loss** - Should converge to stable value
4. **Grid Power** - Want to minimize peak grid power usage
5. **Battery SOC** - Should actively manage charge/discharge

## 💡 Tips

- **Tune Learning Rates** - Modify in `hems_core/agents/sac/config.py`
- **Adjust Update Frequency** - More frequent updates = longer training but better learning
- **Batch Size** - Larger batch = more stable but slower; smaller = faster but noisier
- **Model Checkpoints** - Save frequently to monitor learning progress
- **Long Runs** - For 1000+ episodes, consider running with `nohup` or in tmux

## 🔧 Advanced Usage

### Load and Continue Training
```python
from main import train_sac
from hems_core.agents.sac import SACAgent

agent = SACAgent(state_dim=9, action_dim=2)
agent.load("logs/20240428_120000/sac_agent_ep100.pt")
# Continue training...
```

### Test Trained Agent
```python
from hems_core.env.engine import EnergyEnv
from hems_core.agents.sac import SACAgent

env = EnergyEnv()
agent = SACAgent(state_dim=9, action_dim=2)
agent.load("logs/20240428_120000/sac_agent_final.pt")

state = env.reset()
for _ in range(96):
    action = agent.select_action(state, training=False)  # Use mean, no exploration
    next_state, reward, done, info = env.step(action)
    state = next_state
```

---

**Questions?** Check the code comments or inspect CSV files in your favorite notebook tool (Jupyter, Excel, pandas)!
