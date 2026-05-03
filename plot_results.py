import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse


def plot_training_results(run_dir):
    """
    Plot training results from a completed training run.
    
    Args:
        run_dir: Path to the training run directory (e.g., logs/20240428_120000)
    """
    run_path = Path(run_dir)
    
    if not run_path.exists():
        print(f"Error: Run directory not found: {run_dir}")
        return
    
    # Load data
    print(f"Loading data from {run_dir}...")
    episodes_df = pd.read_csv(run_path / "training_episodes.csv")
    steps_df = pd.read_csv(run_path / "training_steps.csv")
    
    # Detect agent type based on available columns
    has_actor_loss = 'avg_actor_loss' in episodes_df.columns
    has_dqn_loss = 'avg_dqn_loss' in episodes_df.columns
    agent_type = 'SAC' if has_actor_loss else 'DQN' if has_dqn_loss else 'Unknown'
    
    print(f"Detected agent type: {agent_type}")
    
    # Create output directory for plots
    plots_dir = run_path / "plots"
    plots_dir.mkdir(exist_ok=True)
    
    # Set style
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # ========== Figure 1: Individual Rewards and Costs ==========
    # Aggregate rewards by episode
    battery_reward_per_episode = steps_df.groupby('episode')['bat_reward'].sum().reset_index()
    ev_reward_per_episode = steps_df.groupby('episode')['ev_reward'].sum().reset_index()
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
    
    # Battery Reward
    ax1.plot(battery_reward_per_episode['episode'], battery_reward_per_episode['bat_reward'], 
             linewidth=2, color='green')
    ax1.fill_between(battery_reward_per_episode['episode'], battery_reward_per_episode['bat_reward'], 
                     alpha=0.3, color='green')
    ax1.set_ylabel('Battery Reward')
    ax1.set_title('Battery Reward Over Time')
    ax1.grid(True, alpha=0.3)
    
    # EV Reward
    ax2.plot(ev_reward_per_episode['episode'], ev_reward_per_episode['ev_reward'], 
             linewidth=2, color='orange')
    ax2.fill_between(ev_reward_per_episode['episode'], ev_reward_per_episode['ev_reward'], 
                     alpha=0.3, color='orange')
    ax2.set_ylabel('EV Reward')
    ax2.set_title('EV Reward Over Time')
    ax2.grid(True, alpha=0.3)
    
    # Total Cost
    ax3.plot(episodes_df['episode'], episodes_df['total_cost'], linewidth=2, color='red')
    ax3.fill_between(episodes_df['episode'], episodes_df['total_cost'], alpha=0.3, color='red')
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Total Cost (EUR)')
    ax3.set_title('Daily Cost Over Time')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(plots_dir / "01_rewards_costs.png", dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {plots_dir / '01_rewards_costs.png'}")
    plt.close()
    
    # ========== Figure 2: Training Losses ==========
    if agent_type == 'SAC':
        # SAC: Plot Q1 Loss, Actor Loss, Alpha Loss
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        
        ax = axes[0]
        ax.plot(episodes_df['episode'], episodes_df['avg_q1_loss'], linewidth=2, color='green')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Q1 Loss')
        ax.set_title('Critic Q1 Loss')
        ax.grid(True, alpha=0.3)
        
        ax = axes[1]
        ax.plot(episodes_df['episode'], episodes_df['avg_actor_loss'], linewidth=2, color='orange')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Actor Loss')
        ax.set_title('Actor Loss')
        ax.grid(True, alpha=0.3)
        
        ax = axes[2]
        ax.plot(episodes_df['episode'], episodes_df['avg_alpha_loss'], linewidth=2, color='purple')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Alpha Loss')
        ax.set_title('Entropy Loss')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(plots_dir / "02_losses.png", dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {plots_dir / '02_losses.png'}")
        plt.close()
        
    elif agent_type == 'DQN':
        # DQN: Plot DQN Loss and Epsilon Decay
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        ax = axes[0]
        ax.plot(episodes_df['episode'], episodes_df['avg_dqn_loss'], linewidth=2, color='blue')
        ax.fill_between(episodes_df['episode'], episodes_df['avg_dqn_loss'], alpha=0.3, color='blue')
        ax.set_xlabel('Episode')
        ax.set_ylabel('DQN Loss')
        ax.set_title('DQN Loss Over Training')
        ax.grid(True, alpha=0.3)
        
        ax = axes[1]
        ax.plot(episodes_df['episode'], episodes_df['epsilon'], linewidth=2, color='red')
        ax.fill_between(episodes_df['episode'], episodes_df['epsilon'], alpha=0.3, color='red')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Epsilon (Exploration Rate)')
        ax.set_title('Epsilon Decay Over Training')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(plots_dir / "02_losses.png", dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {plots_dir / '02_losses.png'}")
        plt.close()
    
    # ========== Figure 3: Step-by-step Actions & States (Last Episode) ==========
    last_episode = sorted(steps_df['episode'].unique())[-1]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    
    
    episode_data = steps_df[steps_df['episode'] == last_episode]
        
    axes[0, 0].plot(episode_data['step'], episode_data['action_battery'], 
                       label=f"Ep {last_episode}", linewidth=1.5, alpha=0.7)
    axes[0, 1].plot(episode_data['step'], episode_data['action_ev'], 
                       label=f"Ep {last_episode}", linewidth=1.5, alpha=0.7)
    axes[1, 0].plot(episode_data['step'], episode_data['state_bat_soc'], 
                       label=f"Ep {last_episode}", linewidth=1.5, alpha=0.7)
    axes[1, 1].plot(episode_data['step'], episode_data['state_ev_soc'], 
                       label=f"Ep {last_episode}", linewidth=1.5, alpha=0.7)
    
    axes[0, 0].set_title('Battery Action')
    axes[0, 0].set_ylabel('Action [-1, 1]')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].set_title('EV Action')
    axes[0, 1].set_ylabel('Action [-1, 1]')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].set_title('Battery SOC')
    axes[1, 0].set_xlabel('Step')
    axes[1, 0].set_ylabel('SOC [0, 1]')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].set_title('EV SOC')
    axes[1, 1].set_xlabel('Step')
    axes[1, 1].set_ylabel('SOC [0, 1]')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(plots_dir / "03_actions_states_last_episodes.png", dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {plots_dir / '03_actions_states_last_episodes.png'}")
    plt.close()
    
    # ========== Figure 4: State Variables Over Training ==========
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    
    # Sample every 10th step to reduce noise
    sampled_steps = steps_df[steps_df['step'] % 10 == 0]
    
    axes[0, 0].scatter(sampled_steps['episode'], sampled_steps['state_temp'], 
                      s=5, alpha=0.5, c=sampled_steps['episode'], cmap='viridis')
    axes[0, 0].set_title('Temperature')
    axes[0, 0].set_ylabel('Normalized Temp [0, 1]')
    
    axes[0, 1].scatter(sampled_steps['episode'], sampled_steps['state_load'], 
                      s=5, alpha=0.5, c=sampled_steps['episode'], cmap='viridis')
    axes[0, 1].set_title('Load')
    axes[0, 1].set_ylabel('Normalized Load [0, 1]')
    
    axes[0, 2].scatter(sampled_steps['episode'], sampled_steps['state_pv'], 
                      s=5, alpha=0.5, c=sampled_steps['episode'], cmap='viridis')
    axes[0, 2].set_title('PV Generation')
    axes[0, 2].set_ylabel('Normalized PV [0, 1]')
    
    axes[1, 0].scatter(sampled_steps['episode'], sampled_steps['state_price'], 
                      s=5, alpha=0.5, c=sampled_steps['episode'], cmap='viridis')
    axes[1, 0].set_title('Electricity Price')
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Normalized Price [0, 1]')
    
    axes[1, 1].scatter(sampled_steps['episode'], sampled_steps['P_grid'], 
                      s=5, alpha=0.5, c=sampled_steps['episode'], cmap='viridis')
    axes[1, 1].set_title('Grid Power')
    axes[1, 1].set_xlabel('Episode')
    axes[1, 1].set_ylabel('Power (kW)')
    
    axes[1, 2].scatter(sampled_steps['episode'], sampled_steps['cost'], 
                      s=5, alpha=0.5, c=sampled_steps['episode'], cmap='viridis')
    axes[1, 2].set_title('Step Cost')
    axes[1, 2].set_xlabel('Episode')
    axes[1, 2].set_ylabel('Cost (EUR)')
    
    plt.tight_layout()
    plt.savefig(plots_dir / "04_state_variables.png", dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {plots_dir / '04_state_variables.png'}")
    plt.close()
    
    # ========== Figure 5: Statistics Summary ==========
    fig = plt.figure(figsize=(12, 6))
    
    # Create text summary
    summary_text = f"""
TRAINING SUMMARY STATISTICS

Episodes: {len(episodes_df)}
Total Steps: {len(steps_df)}

REWARDS:
  Mean: {episodes_df['total_reward'].mean():.2f}
  Std:  {episodes_df['total_reward'].std():.2f}
  Min:  {episodes_df['total_reward'].min():.2f}
  Max:  {episodes_df['total_reward'].max():.2f}

COSTS (EUR):
  Mean Daily: {episodes_df['total_cost'].mean():.4f}
  Std:        {episodes_df['total_cost'].std():.4f}
  Min Daily:  {episodes_df['total_cost'].min():.4f}
  Max Daily:  {episodes_df['total_cost'].max():.4f}
  Total:      {episodes_df['total_cost'].sum():.4f}

GRID POWER (kW):
  Mean:  {steps_df['P_grid'].mean():.2f}
  Std:   {steps_df['P_grid'].std():.2f}
  Min:   {steps_df['P_grid'].min():.2f}
  Max:   {steps_df['P_grid'].max():.2f}

BATTERY SOC:
  Mean: {steps_df['state_bat_soc'].mean():.3f}
  Std:  {steps_df['state_bat_soc'].std():.3f}

EV SOC:
  Mean: {steps_df['state_ev_soc'].mean():.3f}
  Std:  {steps_df['state_ev_soc'].std():.3f}
"""
    
    ax = fig.add_subplot(111)
    ax.text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
            verticalalignment='center', transform=ax.transAxes)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(plots_dir / "05_statistics_summary.txt.png", dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {plots_dir / '05_statistics_summary.txt.png'}")
    plt.close()
    
    print(f"\n✓ All plots saved to: {plots_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot training results")
    parser.add_argument("run_dir", help="Path to training run directory (e.g., logs/20240428_120000)")
    args = parser.parse_args()
    
    plot_training_results(args.run_dir)
