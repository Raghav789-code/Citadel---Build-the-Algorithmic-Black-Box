"""
Day 6: Multi-Agent Market Simulation
Objective: Scale up to multi-agent environment and collect market data
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from stable_baselines3 import PPO
import random
from collections import defaultdict

class NoiseTrader:
    """Random trader that provides market noise"""
    
    def __init__(self, agent_id):
        self.agent_id = agent_id
        self.inventory = 0
        self.max_inventory = 10
    
    def get_action(self, market_state):
        """Random buy/sell with inventory limits"""
        if self.inventory >= self.max_inventory:
            return 2  # Forced sell
        elif self.inventory <= -self.max_inventory:
            return 1  # Forced buy
        else:
            return random.choice([0, 1, 2])  # Random action
    
    def update_inventory(self, action):
        if action == 1:  # Buy
            self.inventory += 1
        elif action == 2:  # Sell
            self.inventory -= 1

class MarketMaker:
    """Simple market maker that provides liquidity"""
    
    def __init__(self, agent_id, spread_target=0.02):
        self.agent_id = agent_id
        self.inventory = 0
        self.spread_target = spread_target
        self.max_inventory = 20
    
    def get_action(self, market_state):
        """Market making strategy: provide liquidity, manage inventory"""
        mid_price = market_state.get('mid_price', 100)
        
        # Inventory management
        if self.inventory > self.max_inventory // 2:
            return 2  # Sell to reduce inventory
        elif self.inventory < -self.max_inventory // 2:
            return 1  # Buy to reduce inventory
        else:
            # Provide liquidity (alternate buy/sell)
            return 1 if random.random() < 0.5 else 2
    
    def update_inventory(self, action):
        if action == 1:  # Buy
            self.inventory += 1
        elif action == 2:  # Sell
            self.inventory -= 1

class MultiAgentMarket:
    """Multi-agent market simulation environment"""
    
    def __init__(self, rl_model_path=None):
        # Load RL agent if available
        self.rl_agent = None
        if rl_model_path:
            try:
                from day3_reward_engineering import RiskAwareTradingEnv
                dummy_env = RiskAwareTradingEnv()
                self.rl_agent = PPO.load(rl_model_path, env=dummy_env)
                print("✓ RL agent loaded successfully")
            except:
                print("⚠️  Could not load RL agent, using random agent instead")
        
        # Create agent population
        self.noise_traders = [NoiseTrader(f"NT_{i}") for i in range(50)]
        self.market_makers = [MarketMaker(f"MM_{i}") for i in range(10)]
        
        # Market state
        self.mid_price = 100.0
        self.price_history = [self.mid_price]
        self.volume_history = []
        self.spread_history = []
        self.trade_history = []
        
        # RL agent state (if available)
        self.rl_inventory = 0
        self.rl_cash = 10000
        
        print(f"Market initialized with {len(self.noise_traders)} noise traders, "
              f"{len(self.market_makers)} market makers, and 1 RL agent")
    
    def step(self, step_num):
        """Execute one market step"""
        # Collect all agent actions
        actions = defaultdict(list)
        
        # Get RL agent action
        if self.rl_agent:
            market_obs = self._get_rl_observation()
            rl_action, _ = self.rl_agent.predict(market_obs, deterministic=True)
            actions['RL'].append(rl_action)
            self._update_rl_agent(rl_action)
        
        # Get noise trader actions
        market_state = {'mid_price': self.mid_price, 'step': step_num}
        for trader in self.noise_traders:
            action = trader.get_action(market_state)
            actions['Noise'].append(action)
            trader.update_inventory(action)
        
        # Get market maker actions
        for mm in self.market_makers:
            action = mm.get_action(market_state)
            actions['MM'].append(action)
            mm.update_inventory(action)
        
        # Calculate market impact
        total_buy_pressure = sum(actions['RL'].count(1) + actions['Noise'].count(1) + actions['MM'].count(1))
        total_sell_pressure = sum(actions['RL'].count(2) + actions['Noise'].count(2) + actions['MM'].count(2))
        
        net_pressure = total_buy_pressure - total_sell_pressure
        total_volume = total_buy_pressure + total_sell_pressure
        
        # Update market price (with noise)
        price_impact = net_pressure * 0.01  # 1% impact per net order
        noise = np.random.normal(0, 0.005)  # Market noise
        
        self.mid_price += price_impact + noise
        self.mid_price = max(self.mid_price, 1.0)  # Prevent negative prices
        
        # Calculate spread (function of volume and volatility)
        base_spread = 0.002 * self.mid_price
        volume_spread = max(0, (10 - total_volume) * 0.001 * self.mid_price)
        spread = base_spread + volume_spread
        
        # Record market data
        self.price_history.append(self.mid_price)
        self.volume_history.append(total_volume)
        self.spread_history.append(spread)
        
        # Record trades
        if total_volume > 0:
            self.trade_history.append({
                'step': step_num,
                'price': self.mid_price,
                'volume': total_volume,
                'buy_pressure': total_buy_pressure,
                'sell_pressure': total_sell_pressure,
                'rl_action': actions['RL'][0] if actions['RL'] else 0
            })
        
        return {
            'price': self.mid_price,
            'volume': total_volume,
            'spread': spread,
            'actions': actions
        }
    
    def _get_rl_observation(self):
        """Get observation for RL agent"""
        spread = 0.002 * self.mid_price
        bid = self.mid_price - spread/2
        ask = self.mid_price + spread/2
        
        obs = np.array([
            bid / 200.0,
            ask / 200.0,
            spread / 2.0,
            (self.rl_inventory + 10) / 20.0,
            self.rl_cash / 20000.0
        ], dtype=np.float32)
        
        return np.clip(obs, 0, 1)
    
    def _update_rl_agent(self, action):
        """Update RL agent state"""
        if action == 1 and self.rl_cash >= self.mid_price:  # Buy
            self.rl_inventory += 1
            self.rl_cash -= self.mid_price
        elif action == 2 and self.rl_inventory > 0:  # Sell
            self.rl_inventory -= 1
            self.rl_cash += self.mid_price
    
    def run_simulation(self, steps=5000):
        """Run full market simulation"""
        print(f"Running {steps}-step market simulation...")
        
        for step in range(steps):
            self.step(step)
            
            if step % 1000 == 0:
                print(f"Step {step}: Price={self.mid_price:.2f}, "
                      f"Volume={self.volume_history[-1] if self.volume_history else 0}")
        
        # Create results DataFrame
        results = pd.DataFrame({
            'step': range(len(self.price_history)),
            'price': self.price_history,
            'volume': [0] + self.volume_history,  # Pad to match length
            'spread': [self.spread_history[0]] + self.spread_history  # Pad to match length
        })
        
        trades_df = pd.DataFrame(self.trade_history)
        
        print(f"✓ Simulation completed: {len(results)} price points, {len(trades_df)} trades")
        
        return results, trades_df

def create_order_book_heatmap(results, trades_df):
    """Create order book liquidity heatmap"""
    print("Creating order book heatmap...")
    
    # Simulate liquidity levels around mid-price
    time_points = results['step'].values[::50]  # Sample every 50 steps
    price_levels = []
    liquidity_matrix = []
    
    for i, step in enumerate(time_points):
        mid_price = results.loc[results['step'] == step, 'price'].iloc[0]
        
        # Create price levels around mid-price
        price_range = np.linspace(mid_price * 0.98, mid_price * 1.02, 20)
        price_levels.append(price_range)
        
        # Simulate liquidity (higher near mid-price)
        liquidity = np.exp(-((price_range - mid_price) / (mid_price * 0.01))**2)
        liquidity += np.random.normal(0, 0.1, len(liquidity))  # Add noise
        liquidity = np.maximum(liquidity, 0)  # Ensure non-negative
        
        liquidity_matrix.append(liquidity)
    
    # Create heatmap
    plt.figure(figsize=(15, 8))
    
    # Plot 1: Order book heatmap
    plt.subplot(2, 1, 1)
    liquidity_matrix = np.array(liquidity_matrix).T
    
    plt.imshow(liquidity_matrix, aspect='auto', cmap='YlOrRd', origin='lower')
    plt.colorbar(label='Liquidity Intensity')
    plt.title('Order Book Liquidity Heatmap')
    plt.xlabel('Time Steps (sampled)')
    plt.ylabel('Price Levels')
    
    # Plot 2: Price and volume
    plt.subplot(2, 1, 2)
    plt.plot(results['step'], results['price'], label='Mid Price', alpha=0.8)
    plt.xlabel('Time Step')
    plt.ylabel('Price')
    plt.title('Price Discovery Over Time')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('day6_market_simulation.png', dpi=150, bbox_inches='tight')
    plt.show()

def run_day6_simulation():
    """Run complete Day 6 simulation"""
    print("=== Day 6: Multi-Agent Market Simulation ===")
    
    # Try to load trained RL agent from Day 5
    rl_model_path = "day5_extended_ppo"
    
    # Create market
    market = MultiAgentMarket(rl_model_path)
    
    # Run simulation
    results, trades_df = market.run_simulation(steps=5000)
    
    # Create visualizations
    create_order_book_heatmap(results, trades_df)
    
    # Save data
    results.to_csv('day6_market_data.csv', index=False)
    trades_df.to_csv('day6_trade_data.csv', index=False)
    
    # Analysis
    print(f"\n=== Market Analysis ===")
    print(f"Price Range: {results['price'].min():.2f} - {results['price'].max():.2f}")
    print(f"Average Volume: {results['volume'].mean():.1f}")
    print(f"Price Volatility: {results['price'].pct_change().std():.4f}")
    print(f"Total Trades: {len(trades_df)}")
    
    return results, trades_df

if __name__ == "__main__":
    results, trades_df = run_day6_simulation()
    
    print(f"\n=== Day 6 Deliverables ===")
    print("✓ Multi-agent simulation completed")
    print("✓ Market microstructure data collected")
    print("✓ Order book heatmap generated")
    print("✓ Data saved for Phase II analysis")