"""
Day 2: Custom Gymnasium Trading Environment
Objective: Create Gymnasium-compliant environment for Stable-Baselines3
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces
import random

class TradingEnv(gym.Env):
    """
    Custom trading environment implementing Gymnasium interface
    State: [bid, ask, spread, inventory, cash] (normalized)
    Actions: 0=Hold, 1=Buy, 2=Sell
    """
    
    def __init__(self, initial_cash=10000, max_steps=1000):
        super().__init__()
        
        # Environment parameters
        self.initial_cash = initial_cash
        self.max_steps = max_steps
        
        # Observation space: [bid, ask, spread, inventory, cash] normalized to [0,1]
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(5,), dtype=np.float32
        )
        
        # Action space: 0=Hold, 1=Buy, 2=Sell
        self.action_space = spaces.Discrete(3)
        
        # Market simulation parameters
        self.price_volatility = 0.02
        self.spread_width = 0.002
        
        self.reset()
    
    def reset(self, seed=None):
        """Reset environment to initial state"""
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # Reset state
        self.cash = self.initial_cash
        self.inventory = 0
        self.mid_price = 100.0
        self.step_count = 0
        self.portfolio_history = [self.initial_cash]
        
        return self._get_observation(), {}
    
    def step(self, action):
        """Execute one trading step"""
        # Update market prices (random walk)
        self.mid_price += np.random.normal(0, self.price_volatility)
        self.mid_price = max(self.mid_price, 1.0)  # Prevent negative prices
        
        # Calculate bid/ask
        spread = self.mid_price * self.spread_width
        bid = self.mid_price - spread/2
        ask = self.mid_price + spread/2
        
        # Execute action
        transaction_cost = 0
        if action == 1 and self.cash >= ask:  # Buy
            self.inventory += 1
            self.cash -= ask
            transaction_cost = 0.001 * ask  # 0.1% transaction cost
        elif action == 2 and self.inventory > 0:  # Sell
            self.inventory -= 1
            self.cash += bid
            transaction_cost = 0.001 * bid
        
        # Calculate reward
        portfolio_value = self.cash + self.inventory * self.mid_price
        reward = self._calculate_reward(portfolio_value, transaction_cost)
        
        # Update step count
        self.step_count += 1
        self.portfolio_history.append(portfolio_value)
        
        # Check termination
        terminated = self.step_count >= self.max_steps
        
        return self._get_observation(), reward, terminated, False, self._get_info()
    
    def _calculate_reward(self, portfolio_value, transaction_cost):
        """Calculate risk-adjusted reward"""
        # Base reward: change in portfolio value
        if len(self.portfolio_history) > 1:
            pnl = portfolio_value - self.portfolio_history[-1]
        else:
            pnl = 0
        
        # Risk penalties
        inventory_penalty = 0.01 * abs(self.inventory)  # Inventory risk
        
        # Total reward
        reward = pnl - transaction_cost - inventory_penalty
        
        return reward
    
    def _get_observation(self):
        """Get normalized observation"""
        spread = self.mid_price * self.spread_width
        bid = self.mid_price - spread/2
        ask = self.mid_price + spread/2
        
        # Normalize to [0,1]
        obs = np.array([
            bid / 200.0,           # Normalized bid
            ask / 200.0,           # Normalized ask  
            spread / 2.0,          # Normalized spread
            (self.inventory + 10) / 20.0,  # Normalized inventory [-10,10] -> [0,1]
            self.cash / (2 * self.initial_cash)  # Normalized cash
        ], dtype=np.float32)
        
        return np.clip(obs, 0, 1)
    
    def _get_info(self):
        """Return diagnostic information"""
        portfolio_value = self.cash + self.inventory * self.mid_price
        return {
            'portfolio_value': portfolio_value,
            'cash': self.cash,
            'inventory': self.inventory,
            'mid_price': self.mid_price,
            'pnl': portfolio_value - self.initial_cash
        }

# Test environment
if __name__ == "__main__":
    env = TradingEnv()
    
    print("=== Environment Test ===")
    obs, _ = env.reset()
    print(f"Initial observation: {obs}")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    
    # Run a few steps
    for i in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Step {i+1}: Action={action}, Reward={reward:.4f}, PnL={info['pnl']:.2f}")
        
        if terminated:
            break
    
    print("\n✓ Day 2 Complete: Trading environment created and tested")