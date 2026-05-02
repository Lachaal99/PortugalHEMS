import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import json
from datetime import datetime
import argparse

# Add project root to path
root = Path(__file__).parent
sys.path.insert(0, str(root))

from hems_core.env.engine import EnergyEnv
from hems_core.agents.sac.agent import SACAgent


class InferenceLogger:
    """Logger for inference metrics and visualization data."""
    
    def __init__(self, log_dir="logs/inference", model_name="inference"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True, parents=True)
        
        # Create timestamped subdirectory
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = self.log_dir / f"{model_name}_{self.timestamp}"
        self.run_dir.mkdir(exist_ok=True)
        
        # Data storage
        self.episodes_data = []
        self.steps_data = []
        
    def log_step(self, episode, step, state, action, reward, info, next_state):
        """Log a single inference step."""
        step_record = {
            'episode': episode,
            'step': step,
            'action_battery': float(action[0]),
            'action_ev': float(action[1]),
            'reward': float(reward),
            'cost': float(info.get('cost', 0)),
            'ev_reward': float(info.get('ev_reward', 0)),
            'bat_reward': float(info.get('bat_reward', 0)),
            'P_grid': float(info.get('P_grid', 0)),
            'state_temp': float(state[0]),
            'state_load': float(state[1]),
            'state_bat_soc': float(state[2]),
            'state_ev_soc': float(state[3]),
            'state_pv': float(state[4]),
            'state_price': float(state[5]),
            'state_season': float(state[6]),
            'state_day_type': float(state[7]),
            'state_hour': float(state[8]),
        }
        self.steps_data.append(step_record)
    
    def log_episode(self, episode, episode_reward, episode_cost, episode_length, 
                    total_grid_power, avg_battery_soc, avg_ev_soc):
        """Log episode summary."""
        episode_record = {
            'episode': episode,
            'total_reward': float(episode_reward),
            'total_cost': float(episode_cost),
            'length': episode_length,
            'total_grid_power': float(total_grid_power),
            'avg_battery_soc': float(avg_battery_soc),
            'avg_ev_soc': float(avg_ev_soc),
        }
        self.episodes_data.append(episode_record)
        
        # Print progress
        print(f"Episode {episode:4d} | Reward: {episode_reward:10.2f} | "
              f"Cost: {episode_cost:8.2f} EUR | Avg Grid Power: {total_grid_power:8.2f} kW")
    
    def save_inference_data(self):
        """Save all inference data to CSV files."""
        steps_df = pd.DataFrame(self.steps_data)
        episodes_df = pd.DataFrame(self.episodes_data)
        
        steps_file = self.run_dir / "inference_steps.csv"
        episodes_file = self.run_dir / "inference_episodes.csv"
        
        steps_df.to_csv(steps_file, index=False)
        episodes_df.to_csv(episodes_file, index=False)
        
        print(f"\n✓ Inference data saved:")
        print(f"  - {steps_file}")
        print(f"  - {episodes_file}")
        
        return steps_file, episodes_file
    
    def save_summary_stats(self, config_dict):
        """Save summary statistics and configuration."""
        stats = {
            'config': config_dict,
            'summary': {
                'total_episodes': len(self.episodes_data),
                'avg_reward': float(np.mean([e['total_reward'] for e in self.episodes_data])),
                'avg_cost': float(np.mean([e['total_cost'] for e in self.episodes_data])),
                'avg_grid_power': float(np.mean([e['total_grid_power'] for e in self.episodes_data])),
                'avg_battery_soc': float(np.mean([e['avg_battery_soc'] for e in self.episodes_data])),
                'avg_ev_soc': float(np.mean([e['avg_ev_soc'] for e in self.episodes_data])),
            }
        }
        
        stats_file = self.run_dir / "summary_stats.json"
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=4)
        
        print(f"✓ Summary stats saved: {stats_file}")
        return stats_file


def plot_inference_results(run_dir):
    """
    Generate plots from inference results.
    
    Args:
        run_dir: Path to the inference run directory
    """
    run_path = Path(run_dir)
    
    if not run_path.exists():
        print(f"Error: Run directory not found: {run_dir}")
        return
    
    # Load data
    print(f"\nGenerating plots from {run_dir}...")
    episodes_df = pd.read_csv(run_path / "inference_episodes.csv")
    steps_df = pd.read_csv(run_path / "inference_steps.csv")
    
    # Create output directory for plots
    plots_dir = run_path / "plots"
    plots_dir.mkdir(exist_ok=True)
    
    # Set style
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # ========== Figure 1: Episode Rewards and Costs ==========
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 10))
    
    # Total Reward
    ax1.plot(episodes_df['episode'], episodes_df['total_reward'], linewidth=2, 
             color='#2E86AB', marker='o', markersize=4, label='Total Reward')
    ax1.fill_between(episodes_df['episode'], episodes_df['total_reward'], alpha=0.3, color='#2E86AB')
    ax1.set_ylabel('Total Reward', fontsize=12)
    ax1.set_title('Inference Episode Rewards', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Total Cost
    ax2.plot(episodes_df['episode'], episodes_df['total_cost'], linewidth=2, 
             color='#A23B72', marker='o', markersize=4, label='Daily Cost')
    ax2.fill_between(episodes_df['episode'], episodes_df['total_cost'], alpha=0.3, color='#A23B72')
    ax2.set_ylabel('Total Cost (EUR)', fontsize=12)
    ax2.set_title('Daily Electricity Cost', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Average Grid Power
    ax3.plot(episodes_df['episode'], episodes_df['total_grid_power'], linewidth=2, 
             color='#F18F01', marker='o', markersize=4, label='Total Grid Power')
    ax3.fill_between(episodes_df['episode'], episodes_df['total_grid_power'], alpha=0.3, color='#F18F01')
    ax3.set_xlabel('Episode', fontsize=12)
    ax3.set_ylabel('Total Grid Power (kWh)', fontsize=12)
    ax3.set_title('Daily Grid Power Consumption', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    plt.tight_layout()
    plt.savefig(plots_dir / "01_episode_metrics.png", dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {plots_dir / '01_episode_metrics.png'}")
    plt.close()
    
    # ========== Figure 2: Battery and EV State of Charge ==========
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))
    
    # Battery SOC
    ax1.plot(episodes_df['episode'], episodes_df['avg_battery_soc'], linewidth=2, 
             color='#06A77D', marker='s', markersize=4, label='Avg Battery SOC')
    ax1.fill_between(episodes_df['episode'], episodes_df['avg_battery_soc'], alpha=0.3, color='#06A77D')
    ax1.set_ylabel('SOC (%)', fontsize=12)
    ax1.set_title('Average Battery State of Charge', fontsize=14, fontweight='bold')
    ax1.set_ylim([0, 1])
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # EV SOC
    ax2.plot(episodes_df['episode'], episodes_df['avg_ev_soc'], linewidth=2, 
             color='#D62246', marker='^', markersize=4, label='Avg EV SOC')
    ax2.fill_between(episodes_df['episode'], episodes_df['avg_ev_soc'], alpha=0.3, color='#D62246')
    ax2.set_xlabel('Episode', fontsize=12)
    ax2.set_ylabel('SOC (%)', fontsize=12)
    ax2.set_title('Average EV State of Charge', fontsize=14, fontweight='bold')
    ax2.set_ylim([0, 1])
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(plots_dir / "02_soc_evolution.png", dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {plots_dir / '02_soc_evolution.png'}")
    plt.close()
    
    # ========== Figure 3: Action Distribution ==========
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Battery action distribution
    battery_actions = steps_df['action_battery'].values
    ax1.hist(battery_actions, bins=50, color='#2E86AB', alpha=0.7, edgecolor='black')
    ax1.axvline(np.mean(battery_actions), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(battery_actions):.3f}')
    ax1.set_xlabel('Action Value', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title('Battery Action Distribution', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # EV action distribution
    ev_actions = steps_df['action_ev'].values
    ax2.hist(ev_actions, bins=50, color='#D62246', alpha=0.7, edgecolor='black')
    ax2.axvline(np.mean(ev_actions), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(ev_actions):.3f}')
    ax2.set_xlabel('Action Value', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title('EV Action Distribution', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(plots_dir / "03_action_distributions.png", dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {plots_dir / '03_action_distributions.png'}")
    plt.close()
    
    # ========== Figure 4: Example Day Trajectory (First Episode) ==========
    first_episode_data = steps_df[steps_df['episode'] == 0]
    
    fig, axes = plt.subplots(3, 2, figsize=(15, 10))
    
    steps = first_episode_data['step'].values
    
    # Battery SOC
    axes[0, 0].plot(steps, first_episode_data['state_bat_soc'].values, linewidth=2, color='#06A77D')
    axes[0, 0].fill_between(steps, first_episode_data['state_bat_soc'].values, alpha=0.3, color='#06A77D')
    axes[0, 0].set_ylabel('Battery SOC', fontsize=11)
    axes[0, 0].set_title('Example Day: Battery State of Charge', fontsize=12, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    
    # EV SOC
    axes[0, 1].plot(steps, first_episode_data['state_ev_soc'].values, linewidth=2, color='#D62246')
    axes[0, 1].fill_between(steps, first_episode_data['state_ev_soc'].values, alpha=0.3, color='#D62246')
    axes[0, 1].set_ylabel('EV SOC', fontsize=11)
    axes[0, 1].set_title('Example Day: EV State of Charge', fontsize=12, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Grid Power
    axes[1, 0].bar(steps, first_episode_data['P_grid'].values, color='#F18F01', alpha=0.7, edgecolor='black')
    axes[1, 0].set_ylabel('Grid Power (kW)', fontsize=11)
    axes[1, 0].set_title('Example Day: Grid Power', fontsize=12, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # Price
    axes[1, 1].plot(steps, first_episode_data['state_price'].values, linewidth=2, color='#A23B72', marker='o', markersize=3)
    axes[1, 1].fill_between(steps, first_episode_data['state_price'].values, alpha=0.3, color='#A23B72')
    axes[1, 1].set_ylabel('Price (Normalized)', fontsize=11)
    axes[1, 1].set_title('Example Day: Electricity Price', fontsize=12, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Actions
    axes[2, 0].plot(steps, first_episode_data['action_battery'].values, linewidth=2, label='Battery', marker='o', markersize=3)
    axes[2, 0].plot(steps, first_episode_data['action_ev'].values, linewidth=2, label='EV', marker='s', markersize=3)
    axes[2, 0].set_ylabel('Action Value', fontsize=11)
    axes[2, 0].set_title('Example Day: Agent Actions', fontsize=12, fontweight='bold')
    axes[2, 0].legend()
    axes[2, 0].grid(True, alpha=0.3)
    
    # Load and PV
    axes[2, 1].plot(steps, first_episode_data['state_load'].values, linewidth=2, label='Load', marker='o', markersize=3)
    axes[2, 1].plot(steps, first_episode_data['state_pv'].values, linewidth=2, label='PV Generation', marker='s', markersize=3)
    axes[2, 1].set_xlabel('Time Step', fontsize=11)
    axes[2, 1].set_ylabel('Power (Normalized)', fontsize=11)
    axes[2, 1].set_title('Example Day: Load & PV Generation', fontsize=12, fontweight='bold')
    axes[2, 1].legend()
    axes[2, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(plots_dir / "04_example_day_trajectory.png", dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {plots_dir / '04_example_day_trajectory.png'}")
    plt.close()
    
    # ========== Figure 5: Summary Statistics ==========
    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Prepare data for statistics
    ax_text = fig.add_subplot(gs[0, :])
    ax_text.axis('off')
    
    stats_text = f"""
    INFERENCE SUMMARY STATISTICS
    
    Episodes:                    {len(episodes_df)}
    Total Steps:                 {len(steps_df)}
    
    Average Reward:              {episodes_df['total_reward'].mean():.2f}
    Average Daily Cost:          {episodes_df['total_cost'].mean():.2f} EUR
    Average Grid Power:          {episodes_df['total_grid_power'].mean():.2f} kWh
    
    Battery SOC (Avg):           {episodes_df['avg_battery_soc'].mean():.2%}
    EV SOC (Avg):                {episodes_df['avg_ev_soc'].mean():.2%}
    
    Min Daily Cost:              {episodes_df['total_cost'].min():.2f} EUR
    Max Daily Cost:              {episodes_df['total_cost'].max():.2f} EUR
    """
    
    ax_text.text(0.05, 0.95, stats_text, transform=ax_text.transAxes, fontsize=12,
                verticalalignment='top', family='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Reward distribution
    ax1 = fig.add_subplot(gs[1, 0])
    ax1.hist(episodes_df['total_reward'], bins=30, color='#2E86AB', alpha=0.7, edgecolor='black')
    ax1.set_xlabel('Total Reward', fontsize=10)
    ax1.set_ylabel('Frequency', fontsize=10)
    ax1.set_title('Reward Distribution', fontsize=11, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Cost distribution
    ax2 = fig.add_subplot(gs[1, 1])
    ax2.hist(episodes_df['total_cost'], bins=30, color='#A23B72', alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Total Cost (EUR)', fontsize=10)
    ax2.set_ylabel('Frequency', fontsize=10)
    ax2.set_title('Cost Distribution', fontsize=11, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Grid Power distribution
    ax3 = fig.add_subplot(gs[1, 2])
    ax3.hist(episodes_df['total_grid_power'], bins=30, color='#F18F01', alpha=0.7, edgecolor='black')
    ax3.set_xlabel('Total Grid Power (kWh)', fontsize=10)
    ax3.set_ylabel('Frequency', fontsize=10)
    ax3.set_title('Grid Power Distribution', fontsize=11, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Step rewards
    ax4 = fig.add_subplot(gs[2, :])
    ax4.plot(steps_df['step'], steps_df['reward'].rolling(96).mean(), linewidth=2, color='#2E86AB', label='Step Reward (MA-96)')
    ax4.fill_between(range(len(steps_df)), steps_df['reward'].rolling(96).mean(), alpha=0.3, color='#2E86AB')
    ax4.set_xlabel('Step', fontsize=10)
    ax4.set_ylabel('Average Reward', fontsize=10)
    ax4.set_title('Step-wise Reward Trend', fontsize=11, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    
    plt.savefig(plots_dir / "05_summary_statistics.png", dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {plots_dir / '05_summary_statistics.png'}")
    plt.close()
    
    print(f"\n✓ All plots saved to: {plots_dir}")


def run_inference(model_path, num_episodes=10, render_plots=True):
    """
    Run inference with a trained SAC agent.
    
    Args:
        model_path: Path to the trained model checkpoint
        num_episodes: Number of episodes to run inference on
        render_plots: Whether to generate plots
    """
    # Initialize logger
    model_name = Path(model_path).stem
    logger = InferenceLogger(model_name=model_name)
    
    print(f"\n{'='*70}")
    print(f"INFERENCE WITH TRAINED MODEL")
    print(f"{'='*70}")
    print(f"Model: {model_path}")
    print(f"Episodes: {num_episodes}")
    print(f"Output Directory: {logger.run_dir}")
    print(f"{'='*70}\n")
    
    # Initialize environment and agent
    env = EnergyEnv()
    agent = SACAgent(state_dim=9, action_dim=2)
    
    # Load trained model
    try:
        agent.load(model_path)
        print(f"✓ Model loaded successfully from {model_path}\n")
    except FileNotFoundError:
        print(f"✗ Error: Model file not found at {model_path}")
        return
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return
    
    # Run inference episodes
    print("Running inference episodes...\n")
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0.0
        episode_cost = 0.0
        episode_grid_power = 0.0
        battery_socs = []
        ev_socs = []
        step_count = 0
        
        done = False
        while not done:
            # Select action (deterministic for inference)
            action = agent.select_action(state, training=False)
            
            # Take step in environment
            next_state, reward, done, info = env.step(action)
            
            # Log step
            logger.log_step(episode, step_count, state, action, reward, info, next_state)
            
            # Accumulate metrics
            episode_reward += reward
            episode_cost += info['cost']
            episode_grid_power += info['P_grid']
            battery_socs.append(state[2])
            ev_socs.append(state[3])
            
            # Update state
            state = next_state
            step_count += 1
        
        # Log episode summary
        avg_battery_soc = np.mean(battery_socs) if battery_socs else 0
        avg_ev_soc = np.mean(ev_socs) if ev_socs else 0
        logger.log_episode(episode, episode_reward, episode_cost, step_count, 
                         episode_grid_power, avg_battery_soc, avg_ev_soc)
    
    # Save all data
    logger.save_inference_data()
    
    # Save configuration
    config = {
        'model_path': str(model_path),
        'num_episodes': num_episodes,
        'state_dim': 9,
        'action_dim': 2,
        'environment': 'EnergyEnv',
        'agent': 'SACAgent',
        'timestamp': logger.timestamp,
    }
    logger.save_summary_stats(config)
    
    # Generate plots
    if render_plots:
        plot_inference_results(logger.run_dir)
    
    print(f"\n{'='*70}")
    print(f"INFERENCE COMPLETED SUCCESSFULLY")
    print(f"Results saved to: {logger.run_dir}")
    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(description="Run inference with trained SAC agent")
    parser.add_argument("--model", type=str, required=True, 
                       help="Path to trained model checkpoint (e.g., logs/20260502_121146/sac_agent_final.pt)")
    parser.add_argument("--episodes", type=int, default=10,
                       help="Number of inference episodes (default: 10)")
    parser.add_argument("--no-plots", action="store_true",
                       help="Skip plot generation")
    
    args = parser.parse_args()
    
    run_inference(
        model_path=args.model,
        num_episodes=args.episodes,
        render_plots=not args.no_plots
    )


if __name__ == "__main__":
    main()
