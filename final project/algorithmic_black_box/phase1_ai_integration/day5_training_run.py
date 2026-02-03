"""
Day 5: The "Sanity Check" Training Run
Objective: Validate sustained learning over 50,000 timesteps
"""

import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3.common.callbacks import BaseCallback
from day4_ppo_agent import TradingPPOAgent

class TrainingMonitorCallback(BaseCallback):
    """Custom callback to monitor training progress"""
    
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.entropies = []
        self.current_episode_reward = 0
        self.current_episode_length = 0
    
    def _on_step(self) -> bool:
        # Track episode progress
        self.current_episode_reward += self.locals['rewards'][0]
        self.current_episode_length += 1
        
        # Check if episode ended
        if self.locals['dones'][0]:
            self.episode_rewards.append(self.current_episode_reward)
            self.episode_lengths.append(self.current_episode_length)
            
            # Reset counters
            self.current_episode_reward = 0
            self.current_episode_length = 0
            
            # Log progress every 10 episodes
            if len(self.episode_rewards) % 10 == 0:
                recent_rewards = self.episode_rewards[-10:]
                print(f"Episodes {len(self.episode_rewards)-9}-{len(self.episode_rewards)}: "
                      f"Mean Reward = {np.mean(recent_rewards):.2f}")
        
        return True

def extended_training_run():
    """Run extended 50k timestep training with monitoring"""
    print("=== Day 5: Extended Training Run (50,000 timesteps) ===")
    
    # Create agent
    agent = TradingPPOAgent(
        env_kwargs={'risk_aversion': 0.1, 'max_steps': 1000},
        model_kwargs={
            'learning_rate': 3e-4,
            'gamma': 0.99,
            'ent_coef': 0.01,
            'verbose': 1
        }
    )
    
    # Setup monitoring
    monitor_callback = TrainingMonitorCallback()
    
    # Extended training
    print("Starting extended training...")
    agent.model.learn(
        total_timesteps=50000,
        callback=monitor_callback
    )
    
    # Analyze training progression
    analyze_training_progression(monitor_callback)
    
    # Final evaluation
    print("\n=== Final Evaluation ===")
    final_stats = agent.evaluate(n_episodes=10)
    
    # Save final model
    agent.save_model("day5_extended_ppo")
    
    return monitor_callback, final_stats

def analyze_training_progression(monitor_callback):
    """Analyze training progression and create plots"""
    print("\n=== Training Analysis ===")
    
    rewards = monitor_callback.episode_rewards
    if len(rewards) < 10:
        print("Insufficient episodes for analysis")
        return
    
    # Calculate moving averages
    window = min(20, len(rewards) // 4)
    moving_avg = []
    for i in range(len(rewards)):
        start_idx = max(0, i - window + 1)
        moving_avg.append(np.mean(rewards[start_idx:i+1]))
    
    # Statistical tests
    early_rewards = rewards[:len(rewards)//3]
    late_rewards = rewards[-len(rewards)//3:]
    
    improvement = np.mean(late_rewards) - np.mean(early_rewards)
    trend_slope = np.polyfit(range(len(rewards)), rewards, 1)[0]
    
    print(f"Total Episodes: {len(rewards)}")
    print(f"Early Mean Reward: {np.mean(early_rewards):.2f}")
    print(f"Late Mean Reward: {np.mean(late_rewards):.2f}")
    print(f"Improvement: {improvement:.2f}")
    print(f"Trend Slope: {trend_slope:.4f}")
    
    # Create training plot
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.plot(rewards, alpha=0.3, label='Episode Rewards')
    plt.plot(moving_avg, label=f'Moving Average ({window})')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Training Progress: Episode Rewards')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 2, 2)
    plt.hist(early_rewards, alpha=0.7, label='Early Episodes', bins=20)
    plt.hist(late_rewards, alpha=0.7, label='Late Episodes', bins=20)
    plt.xlabel('Reward')
    plt.ylabel('Frequency')
    plt.title('Reward Distribution: Early vs Late')
    plt.legend()
    
    plt.subplot(2, 2, 3)
    episode_lengths = monitor_callback.episode_lengths
    plt.plot(episode_lengths)
    plt.xlabel('Episode')
    plt.ylabel('Episode Length')
    plt.title('Episode Lengths Over Time')
    plt.grid(True)
    
    plt.subplot(2, 2, 4)
    # Action entropy proxy (episode length variation)
    entropy_proxy = [np.std(episode_lengths[max(0, i-10):i+1]) 
                     for i in range(len(episode_lengths))]
    plt.plot(entropy_proxy)
    plt.xlabel('Episode')
    plt.ylabel('Length Std Dev (Entropy Proxy)')
    plt.title('Policy Entropy Proxy')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('day5_training_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Learning verdict
    learning_criteria = {
        'positive_trend': trend_slope > 0,
        'significant_improvement': improvement > 1.0,
        'stable_episodes': np.std(late_rewards) < 50
    }
    
    print(f"\n=== Learning Verdict ===")
    for criterion, passed in learning_criteria.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{criterion}: {status}")
    
    overall_success = all(learning_criteria.values())
    
    if overall_success:
        print("\n🎉 LEARNING CONFIRMED: Agent shows sustained improvement")
        print("→ Proceed to Phase II: Multi-Agent Dynamics")
    else:
        print("\n⚠️  LEARNING ISSUES: Review reward/environment design")
        print("→ Return to reward engineering or environment tuning")
    
    return overall_success

if __name__ == "__main__":
    monitor_callback, final_stats = extended_training_run()
    
    print(f"\n=== Day 5 Deliverables ===")
    print("✓ 50,000 timestep training completed")
    print("✓ Training progression analyzed")
    print("✓ Learning verdict determined")
    print("✓ Model saved for Phase II")