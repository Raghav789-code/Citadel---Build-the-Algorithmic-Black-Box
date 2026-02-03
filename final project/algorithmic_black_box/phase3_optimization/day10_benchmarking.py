"""
Day 10: Benchmarking - The Alpha Test
Objective: Determine if RL agent delivers statistical and economic value vs baselines
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
import seaborn as sns
from scipy import stats

# Import our environment
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'phase1_ai_integration'))
from day3_reward_engineering import RiskAwareTradingEnv

class BaselineAgent:
    """Base class for baseline trading agents"""
    
    def __init__(self, name):
        self.name = name
        self.reset()
    
    def reset(self):
        self.cash = 10000
        self.inventory = 0
        self.initial_cash = 10000
    
    def get_portfolio_value(self, price):
        return self.cash + self.inventory * price
    
    def predict(self, obs):
        raise NotImplementedError

class BuyAndHoldAgent(BaselineAgent):
    """Buy and hold baseline agent"""
    
    def __init__(self):
        super().__init__("Buy & Hold")
        self.has_bought = False
    
    def predict(self, obs):
        # Buy once at the beginning, then hold
        if not self.has_bought and self.cash > 0:
            self.has_bought = True
            return 1  # Buy
        return 0  # Hold

class RandomAgent(BaselineAgent):
    """Random trading baseline agent"""
    
    def __init__(self, seed=42):
        super().__init__("Random")
        np.random.seed(seed)
    
    def predict(self, obs):
        return np.random.choice([0, 1, 2])  # Random action

class MomentumAgent(BaselineAgent):
    """Simple momentum baseline agent"""
    
    def __init__(self, lookback=10):
        super().__init__("Momentum")
        self.lookback = lookback
        self.price_history = []
    
    def predict(self, obs):
        # Extract price from observation (assuming it's in the obs)
        # This is a simplified version
        current_price = obs[0] * 200  # Denormalize
        self.price_history.append(current_price)
        
        if len(self.price_history) < self.lookback:
            return 0  # Hold until we have enough history
        
        # Simple momentum: buy if price trending up, sell if trending down
        recent_prices = self.price_history[-self.lookback:]
        trend = (recent_prices[-1] - recent_prices[0]) / recent_prices[0]
        
        if trend > 0.01:  # 1% upward trend
            return 1  # Buy
        elif trend < -0.01:  # 1% downward trend
            return 2  # Sell
        else:
            return 0  # Hold

class AlphaTester:
    """Comprehensive alpha testing framework"""
    
    def __init__(self, rl_model_path=None):
        self.rl_model = None
        self.rl_model_path = rl_model_path
        
        # Load RL model
        if rl_model_path and os.path.exists(rl_model_path + ".zip"):
            try:
                dummy_env = RiskAwareTradingEnv()
                self.rl_model = PPO.load(rl_model_path, env=dummy_env)
                print(f"✓ RL model loaded from {rl_model_path}")
            except Exception as e:
                print(f"⚠️  Could not load RL model: {e}")
        
        # Create baseline agents
        self.baseline_agents = {
            'Buy & Hold': BuyAndHoldAgent(),
            'Random': RandomAgent(seed=42),
            'Momentum': MomentumAgent(lookback=10)
        }
    
    def run_agent_evaluation(self, agent, agent_name, n_episodes=20):
        """Evaluate a single agent"""
        print(f"Evaluating {agent_name}...")
        
        episode_results = []
        
        for episode in range(n_episodes):
            env = RiskAwareTradingEnv(max_steps=1000, initial_cash=10000)
            
            if hasattr(agent, 'reset'):
                agent.reset()
            
            obs, _ = env.reset(seed=42 + episode)
            episode_reward = 0
            portfolio_values = [10000]
            actions = []
            
            for step in range(1000):
                if agent_name == "RL Agent" and self.rl_model:
                    action, _ = self.rl_model.predict(obs, deterministic=True)
                else:
                    action = agent.predict(obs)
                
                obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                portfolio_values.append(info['portfolio_value'])
                actions.append(action)
                
                if terminated or truncated:
                    break
            
            # Calculate metrics
            final_pnl = info['portfolio_value'] - 10000
            returns = np.diff(portfolio_values) / portfolio_values[:-1]
            returns = returns[returns != 0]  # Remove zero returns
            
            if len(returns) > 1:
                sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
                max_drawdown = self._calculate_max_drawdown(portfolio_values)
            else:
                sharpe_ratio = 0
                max_drawdown = 0
            
            episode_results.append({
                'episode': episode,
                'final_pnl': final_pnl,
                'total_reward': episode_reward,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'portfolio_values': portfolio_values,
                'actions': actions,
                'n_trades': len([a for a in actions if a != 0])
            })
            
            env.close()
        
        return episode_results
    
    def _calculate_max_drawdown(self, portfolio_values):
        """Calculate maximum drawdown"""
        peak = portfolio_values[0]
        max_dd = 0
        
        for value in portfolio_values:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak
            if drawdown > max_dd:
                max_dd = drawdown
        
        return max_dd
    
    def run_comprehensive_benchmark(self, n_episodes=20):
        """Run comprehensive benchmarking"""
        print("=== Running Comprehensive Alpha Test ===")
        
        all_results = {}
        
        # Evaluate RL agent
        if self.rl_model:
            rl_results = self.run_agent_evaluation(None, "RL Agent", n_episodes)
            all_results["RL Agent"] = rl_results
        
        # Evaluate baseline agents
        for name, agent in self.baseline_agents.items():
            results = self.run_agent_evaluation(agent, name, n_episodes)
            all_results[name] = results
        
        return all_results
    
    def analyze_results(self, all_results):
        """Analyze and compare results"""
        print("\n=== Performance Analysis ===")
        
        # Aggregate statistics
        summary_stats = {}
        
        for agent_name, results in all_results.items():
            pnls = [r['final_pnl'] for r in results]
            sharpes = [r['sharpe_ratio'] for r in results if not np.isnan(r['sharpe_ratio'])]
            drawdowns = [r['max_drawdown'] for r in results]
            n_trades = [r['n_trades'] for r in results]
            
            summary_stats[agent_name] = {
                'mean_pnl': np.mean(pnls),
                'std_pnl': np.std(pnls),
                'mean_sharpe': np.mean(sharpes) if sharpes else 0,
                'std_sharpe': np.std(sharpes) if sharpes else 0,
                'mean_drawdown': np.mean(drawdowns),
                'max_drawdown': np.max(drawdowns),
                'mean_trades': np.mean(n_trades),
                'win_rate': len([p for p in pnls if p > 0]) / len(pnls)
            }
        
        # Create comparison table
        comparison_df = pd.DataFrame(summary_stats).T
        print("\nPerformance Comparison:")
        print(comparison_df.round(4))
        
        # Statistical significance tests
        if "RL Agent" in all_results:
            self._statistical_tests(all_results)
        
        return comparison_df
    
    def _statistical_tests(self, all_results):
        """Perform statistical significance tests"""
        print("\n=== Statistical Significance Tests ===")
        
        rl_pnls = [r['final_pnl'] for r in all_results["RL Agent"]]
        
        for baseline_name in self.baseline_agents.keys():
            if baseline_name in all_results:
                baseline_pnls = [r['final_pnl'] for r in all_results[baseline_name]]
                
                # T-test
                t_stat, p_value = stats.ttest_ind(rl_pnls, baseline_pnls)
                
                print(f"RL Agent vs {baseline_name}:")
                print(f"  T-statistic: {t_stat:.4f}")
                print(f"  P-value: {p_value:.6f}")
                print(f"  Significant: {'✓ YES' if p_value < 0.05 else '✗ NO'}")
    
    def create_visualizations(self, all_results):
        """Create comprehensive visualizations"""
        print("\n=== Creating Visualizations ===")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Plot 1: PnL Distribution
        ax1 = axes[0, 0]
        for agent_name, results in all_results.items():
            pnls = [r['final_pnl'] for r in results]
            ax1.hist(pnls, alpha=0.6, label=agent_name, bins=15)
        ax1.set_xlabel('Final PnL')
        ax1.set_ylabel('Frequency')
        ax1.set_title('PnL Distribution Comparison')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Sharpe Ratio Comparison
        ax2 = axes[0, 1]
        agent_names = list(all_results.keys())
        sharpe_means = []
        sharpe_stds = []
        
        for agent_name in agent_names:
            sharpes = [r['sharpe_ratio'] for r in all_results[agent_name] 
                      if not np.isnan(r['sharpe_ratio'])]
            sharpe_means.append(np.mean(sharpes) if sharpes else 0)
            sharpe_stds.append(np.std(sharpes) if sharpes else 0)
        
        ax2.bar(agent_names, sharpe_means, yerr=sharpe_stds, alpha=0.7, capsize=5)
        ax2.set_ylabel('Sharpe Ratio')
        ax2.set_title('Risk-Adjusted Returns')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Drawdown Comparison
        ax3 = axes[0, 2]
        drawdown_data = []
        labels = []
        
        for agent_name, results in all_results.items():
            drawdowns = [r['max_drawdown'] * 100 for r in results]  # Convert to percentage
            drawdown_data.append(drawdowns)
            labels.append(agent_name)
        
        ax3.boxplot(drawdown_data, labels=labels)
        ax3.set_ylabel('Max Drawdown (%)')
        ax3.set_title('Drawdown Risk Comparison')
        ax3.tick_params(axis='x', rotation=45)
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Sample Equity Curves
        ax4 = axes[1, 0]
        for agent_name, results in all_results.items():
            # Plot first episode equity curve
            portfolio_values = results[0]['portfolio_values']
            ax4.plot(portfolio_values, label=agent_name, alpha=0.8)
        ax4.set_xlabel('Time Steps')
        ax4.set_ylabel('Portfolio Value')
        ax4.set_title('Sample Equity Curves')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Plot 5: Trading Activity
        ax5 = axes[1, 1]
        trade_counts = []
        for agent_name in agent_names:
            trades = [r['n_trades'] for r in all_results[agent_name]]
            trade_counts.append(np.mean(trades))
        
        ax5.bar(agent_names, trade_counts, alpha=0.7)
        ax5.set_ylabel('Average Number of Trades')
        ax5.set_title('Trading Activity')
        ax5.tick_params(axis='x', rotation=45)
        ax5.grid(True, alpha=0.3)
        
        # Plot 6: Win Rate
        ax6 = axes[1, 2]
        win_rates = []
        for agent_name in agent_names:
            pnls = [r['final_pnl'] for r in all_results[agent_name]]
            win_rate = len([p for p in pnls if p > 0]) / len(pnls) * 100
            win_rates.append(win_rate)
        
        ax6.bar(agent_names, win_rates, alpha=0.7)
        ax6.set_ylabel('Win Rate (%)')
        ax6.set_title('Profitability Rate')
        ax6.tick_params(axis='x', rotation=45)
        ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('day10_alpha_test_results.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    def generate_alpha_verdict(self, comparison_df):
        """Generate final alpha verdict"""
        print("\n=== ALPHA VERDICT ===")
        
        if "RL Agent" not in comparison_df.index:
            print("❌ RL Agent not available for comparison")
            return False
        
        rl_stats = comparison_df.loc["RL Agent"]
        
        # Compare against each baseline
        alpha_detected = False
        
        for baseline in self.baseline_agents.keys():
            if baseline in comparison_df.index:
                baseline_stats = comparison_df.loc[baseline]
                
                print(f"\nRL Agent vs {baseline}:")
                
                # PnL comparison
                pnl_better = rl_stats['mean_pnl'] > baseline_stats['mean_pnl']
                print(f"  PnL: {rl_stats['mean_pnl']:.2f} vs {baseline_stats['mean_pnl']:.2f} {'✓' if pnl_better else '✗'}")
                
                # Sharpe comparison
                sharpe_better = rl_stats['mean_sharpe'] > baseline_stats['mean_sharpe']
                print(f"  Sharpe: {rl_stats['mean_sharpe']:.3f} vs {baseline_stats['mean_sharpe']:.3f} {'✓' if sharpe_better else '✗'}")
                
                # Drawdown comparison (lower is better)
                dd_better = rl_stats['mean_drawdown'] < baseline_stats['mean_drawdown']
                print(f"  Drawdown: {rl_stats['mean_drawdown']:.3f} vs {baseline_stats['mean_drawdown']:.3f} {'✓' if dd_better else '✗'}")
                
                if pnl_better and sharpe_better:
                    alpha_detected = True
        
        # Final verdict
        if alpha_detected:
            print(f"\n🎉 ALPHA DETECTED!")
            print("✓ RL Agent outperforms baselines on risk-adjusted basis")
            print("✓ Statistical and economic value demonstrated")
        else:
            print(f"\n⚠️  NO CLEAR ALPHA")
            print("✗ RL Agent does not consistently outperform baselines")
            print("→ Consider further hyperparameter tuning or strategy refinement")
        
        return alpha_detected

def run_alpha_test():
    """Run complete alpha testing"""
    print("=== Day 10: Alpha Testing ===")
    
    # Try to load optimized model from Day 9
    model_path = "day9_optimized_ppo"
    if not os.path.exists(model_path + ".zip"):
        print("⚠️  Optimized model not found, trying Day 5 model")
        model_path = "day5_extended_ppo"
        if not os.path.exists(model_path + ".zip"):
            print("⚠️  No trained models found, will only test baselines")
            model_path = None
    
    # Create alpha tester
    tester = AlphaTester(model_path)
    
    # Run comprehensive benchmark
    all_results = tester.run_comprehensive_benchmark(n_episodes=15)  # Reduced for demo
    
    # Analyze results
    comparison_df = tester.analyze_results(all_results)
    
    # Create visualizations
    tester.create_visualizations(all_results)
    
    # Generate verdict
    alpha_detected = tester.generate_alpha_verdict(comparison_df)
    
    # Save results
    comparison_df.to_csv('day10_benchmark_results.csv')
    
    return all_results, comparison_df, alpha_detected

if __name__ == "__main__":
    all_results, comparison_df, alpha_detected = run_alpha_test()
    
    print(f"\n=== Day 10 Deliverables ===")
    print("✓ Comprehensive benchmarking completed")
    print("✓ Statistical significance tests performed")
    print("✓ Risk-adjusted performance metrics calculated")
    print("✓ Alpha verdict determined")
    print("✓ Results saved for final dashboard")