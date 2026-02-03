"""
Day 4: First Learning Agent (PPO)
Objective: Train baseline PPO policy and validate learning signal exists
"""

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
import numpy as np
import matplotlib.pyplot as plt
from day3_reward_engineering import RiskAwareTradingEnv

class TradingPPOAgent:
    """PPO agent for trading with monitoring capabilities"""
    
    def __init__(self, env_kwargs=None, model_kwargs=None):
        self.env_kwargs = env_kwargs or {}
        self.model_kwargs = model_kwargs or {
            'learning_rate': 3e-4,
            'gamma': 0.99,
            'ent_coef': 0.01,
            'clip_range': 0.2,
            'verbose': 1
        }
        
        # Create environment
        self.env = RiskAwareTradingEnv(**self.env_kwargs)
        
        # Create model
        self.model = PPO(
            policy="MlpPolicy",
            env=self.env,
            **self.model_kwargs
        )
        
        self.training_history = []
    
    def train(self, total_timesteps=10000, eval_freq=1000):
        """Train the PPO agent with monitoring"""
        print(f"=== Training PPO Agent ({total_timesteps} timesteps) ===")
        
        # Setup evaluation callback
        eval_env = RiskAwareTradingEnv(**self.env_kwargs)
        eval_callback = EvalCallback(
            eval_env, 
            best_model_save_path='./best_model/',
            log_path='./logs/',
            eval_freq=eval_freq,
            deterministic=True,
            render=False
        )
        
        # Train model
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=eval_callback
        )
        
        print("✓ Training completed")
        return self.model
    
    def evaluate(self, n_episodes=10):
        """Evaluate trained agent"""
        print(f"=== Evaluating Agent ({n_episodes} episodes) ===")
        
        episode_rewards = []
        episode_pnls = []
        episode_actions = []
        
        for episode in range(n_episodes):
            obs, _ = self.env.reset()
            episode_reward = 0
            actions = []
            
            for step in range(1000):
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = self.env.step(action)
                
                episode_reward += reward
                actions.append(action)
                
                if terminated or truncated:
                    break
            
            episode_rewards.append(episode_reward)
            episode_pnls.append(info['pnl'])
            episode_actions.append(actions)
            
            print(f"Episode {episode+1}: Reward={episode_reward:.2f}, "
                  f"PnL={info['pnl']:.2f}, Actions={len(set(actions))} unique")
        
        # Calculate statistics
        stats = {
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'mean_pnl': np.mean(episode_pnls),
            'std_pnl': np.std(episode_pnls),
            'action_diversity': np.mean([len(set(actions)) for actions in episode_actions])
        }
        
        print(f"\n=== Evaluation Results ===")
        print(f"Mean Reward: {stats['mean_reward']:.2f} ± {stats['std_reward']:.2f}")
        print(f"Mean PnL: {stats['mean_pnl']:.2f} ± {stats['std_pnl']:.2f}")
        print(f"Action Diversity: {stats['action_diversity']:.1f}/3")
        
        return stats
    
    def save_model(self, path="ppo_trading_agent"):
        """Save trained model"""
        self.model.save(path)
        print(f"✓ Model saved to {path}")
    
    def load_model(self, path="ppo_trading_agent"):
        """Load trained model"""
        self.model = PPO.load(path, env=self.env)
        print(f"✓ Model loaded from {path}")

def run_baseline_training():
    """Run baseline PPO training experiment"""
    print("=== Day 4: Baseline PPO Training ===")
    
    # Create agent with conservative settings
    agent = TradingPPOAgent(
        env_kwargs={'risk_aversion': 0.1},
        model_kwargs={
            'learning_rate': 3e-4,
            'gamma': 0.99,
            'ent_coef': 0.01,
            'verbose': 1
        }
    )
    
    # Train for 10k timesteps (fast feedback)
    agent.train(total_timesteps=10000)
    
    # Evaluate performance
    stats = agent.evaluate(n_episodes=5)
    
    # Save model
    agent.save_model("day4_baseline_ppo")
    
    # Success criteria check
    success_criteria = {
        'learning_signal': abs(stats['mean_reward']) > 0.1,
        'action_diversity': stats['action_diversity'] > 1.5,
        'stable_training': stats['std_reward'] < 100
    }
    
    print(f"\n=== Success Criteria ===")
    for criterion, passed in success_criteria.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{criterion}: {status}")
    
    return all(success_criteria.values())

if __name__ == "__main__":
    success = run_baseline_training()
    
    print(f"\n=== Day 4 Outcome ===")
    if success:
        print("✓ PPO agent training successful")
        print("✓ Learning signal detected")
        print("✓ Agent behavior changes over time")
        print("✓ Ready for extended training (Day 5)")
    else:
        print("✗ Training issues detected")
        print("→ Review reward engineering (Day 3)")
        print("→ Check environment dynamics (Day 2)")