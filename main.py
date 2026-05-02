import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from datetime import datetime
import json

# Add project root to path
root = Path(__file__).parent
sys.path.insert(0, str(root))

from hems_core.env.engine import EnergyEnv
from hems_core.agents.sac.agent import SACAgent


class TrainingLogger:
    """Logger for training metrics and visualization data."""
    
    def __init__(self, log_dir="logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Create timestamped subdirectory
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = self.log_dir / self.timestamp
        self.run_dir.mkdir(exist_ok=True)
        
        # Data storage
        self.episodes_data = []
        self.steps_data = []
        
    def log_step(self, episode, step, state, action, reward, info, loss_dict=None):
        """Log a single training step."""
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
        }
        
        if loss_dict is not None:
            step_record.update({
                'q1_loss': float(loss_dict.get('q1_loss', 0)),
                'actor_loss': float(loss_dict.get('actor_loss', 0)),
                'alpha_loss': float(loss_dict.get('alpha_loss', 0)),
            })
        
        self.steps_data.append(step_record)
    
    def log_episode(self, episode, episode_reward, episode_cost, episode_length, 
                    avg_q1_loss, avg_actor_loss, avg_alpha_loss):
        """Log episode summary."""
        episode_record = {
            'episode': episode,
            'total_reward': float(episode_reward),
            'total_cost': float(episode_cost),
            'length': episode_length,
            'avg_q1_loss': float(avg_q1_loss),
            'avg_actor_loss': float(avg_actor_loss),
            'avg_alpha_loss': float(avg_alpha_loss),
        }
        self.episodes_data.append(episode_record)
        
        # Print progress
        print(f"Episode {episode:4d} | Reward: {episode_reward:10.2f} | "
              f"Cost: {episode_cost:8.2f} EUR | Length: {episode_length}")
    
    def save_training_data(self):
        """Save all training data to CSV files."""
        steps_df = pd.DataFrame(self.steps_data)
        episodes_df = pd.DataFrame(self.episodes_data)
        
        steps_file = self.run_dir / "training_steps.csv"
        episodes_file = self.run_dir / "training_episodes.csv"
        
        steps_df.to_csv(steps_file, index=False)
        episodes_df.to_csv(episodes_file, index=False)
        
        print(f"\n✓ Training data saved:")
        print(f"  - {steps_file}")
        print(f"  - {episodes_file}")
        
        return steps_file, episodes_file
    
    def save_config(self, config_dict):
        """Save training configuration."""
        config_file = self.run_dir / "config.json"
        with open(config_file, 'w') as f:
            json.dump(config_dict, f, indent=2)
        print(f"  - {config_file}")


def train_sac(num_episodes=100, batch_size=256, update_freq=1, model_save_freq=20):
    """
    Train SAC agent on HEMS environment.
    
    Args:
        num_episodes: Number of episodes to train
        batch_size: Batch size for training
        update_freq: Update frequency (every N steps)
        model_save_freq: Save model every N episodes
    """
    
    print("=" * 70)
    print("Starting SAC Training on HEMS Environment")
    print("=" * 70)
    
    # Initialize environment and agent
    print("\n[1] Initializing environment and agent...")
    env = EnergyEnv()
    agent = SACAgent(state_dim=9, action_dim=2)  # Based on your get_state()
    
    # Initialize logger
    logger = TrainingLogger(log_dir="logs")
    
    # Training configuration
    config = {
        'num_episodes': num_episodes,
        'batch_size': batch_size,
        'update_freq': update_freq,
        'model_save_freq': model_save_freq,
        'device': str(agent.device),
        'gamma': agent.gamma,
        'tau': agent.tau,
        'target_entropy': float(agent.target_entropy),
    }
    logger.save_config(config)
    
    # Training tracking
    episode_rewards = []
    episode_costs = []
    
    print(f"✓ Environment initialized (state_dim=9, action_dim=2)")
    print(f"✓ Agent initialized on device: {agent.device}")
    print(f"✓ Logs will be saved to: {logger.run_dir}")
    
    # Training loop
    print("\n[2] Starting training loop...")
    print("-" * 70)
    
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        episode_cost = 0
        episode_losses = {'q1': [], 'actor': [], 'alpha': []}
        
        # Episode loop (96 steps per day)
        for step in range(96):
            # Select action
            action = agent.select_action(state, training=True)
            # Take step in environment
            next_state, reward, done, info = env.step(action)
            
            
            # Store transition in replay buffer
            agent.store_transition(state, action, reward, next_state, done)
            
            # Update agent
            if step % update_freq == 0:
                q1_loss, actor_loss, alpha_loss = agent.update(batch_size=batch_size)
                if q1_loss is not None:
                    episode_losses['q1'].append(q1_loss)
                    episode_losses['actor'].append(actor_loss)
                    episode_losses['alpha'].append(alpha_loss)
            
            # Accumulate metrics
            episode_reward += reward
            episode_cost += info['cost']
            
            # Log step data
            q1_loss, actor_loss, alpha_loss = agent.update(batch_size=batch_size)
                
            if q1_loss is not None:
                episode_losses['q1'].append(q1_loss)
                episode_losses['actor'].append(actor_loss)
                episode_losses['alpha'].append(alpha_loss)
                loss_dict = {
                    'q1_loss': q1_loss,
                    'actor_loss': actor_loss,
                    'alpha_loss': alpha_loss
                }
            else:
                loss_dict = None
                
            logger.log_step(episode, step, state, action, reward, info, loss_dict=loss_dict)
            
            # Move to next state
            state = next_state
        
        # Episode summary
        avg_q1_loss = np.mean(episode_losses['q1']) if episode_losses['q1'] else 0
        avg_actor_loss = np.mean(episode_losses['actor']) if episode_losses['actor'] else 0
        avg_alpha_loss = np.mean(episode_losses['alpha']) if episode_losses['alpha'] else 0
        
        logger.log_episode(episode, episode_reward, episode_cost, 96,
                          avg_q1_loss, avg_actor_loss, avg_alpha_loss)
        
        episode_rewards.append(episode_reward)
        episode_costs.append(episode_cost)
        
        # Save model periodically
        if (episode + 1) % model_save_freq == 0:
            model_path = logger.run_dir / f"sac_agent_ep{episode+1}.pt"
            agent.save(str(model_path))
            print(f"  → Model saved: {model_path}")
    
    print("-" * 70)
    print(f"\n[3] Training completed!")
    
    # Save final model
    final_model_path = logger.run_dir / "sac_agent_final.pt"
    agent.save(str(final_model_path))
    print(f"✓ Final model saved: {final_model_path}")
    
    # Save training data
    print(f"\n[4] Saving training data...")
    logger.save_training_data()
    
    # Print summary statistics
    print(f"\n[5] Training Summary:")
    print(f"  - Total Episodes: {num_episodes}")
    print(f"  - Avg Episode Reward: {np.mean(episode_rewards):.2f}")
    print(f"  - Best Episode Reward: {np.max(episode_rewards):.2f}")
    print(f"  - Worst Episode Reward: {np.min(episode_rewards):.2f}")
    print(f"  - Avg Daily Cost: {np.mean(episode_costs):.4f} EUR")
    print(f"  - Total Cost: {np.sum(episode_costs):.4f} EUR")
    
    print("\n" + "=" * 70)
    print("Training session complete!")
    print("=" * 70)


if __name__ == "__main__":
    # Train for 100 episodes
    # Adjust hyperparameters as needed:
    # - num_episodes: Number of training episodes
    # - batch_size: Batch size for updates
    # - update_freq: How often to update (every N steps)
    # - model_save_freq: Save model every N episodes
    
    train_sac(
        num_episodes=800,
        batch_size=256,
        update_freq=1,
        model_save_freq=500
    )
