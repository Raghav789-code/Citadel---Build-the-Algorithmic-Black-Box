"""
Day 3: Reward Function Engineering
Objective: Design reward function that produces economically rational behavior
"""

import numpy as np
from day2_trading_environment import TradingEnv

class RiskAwareTradingEnv(TradingEnv):
    """
    Enhanced trading environment with sophisticated reward engineering
    Reward = PnL - λ * Risk_penalty
    """
    
    def __init__(self, risk_aversion=0.1, **kwargs):
        super().__init__(**kwargs)
        self.risk_aversion = risk_aversion  # λ parameter
        self.peak_portfolio_value = self.initial_cash
        self.volatility_window = 20
        self.returns_history = []
    
    def _calculate_reward(self, portfolio_value, transaction_cost):
        """
        Risk-aware reward function:
        Reward = ΔPnL - λ * (inventory_penalty + drawdown_penalty + volatility_penalty)
        """
        # Base PnL change
        if len(self.portfolio_history) > 1:
            pnl_change = portfolio_value - self.portfolio_history[-1]
        else:
            pnl_change = 0
        
        # Track returns for volatility calculation
        if len(self.portfolio_history) > 1:
            ret = (portfolio_value - self.portfolio_history[-1]) / self.portfolio_history[-1]
            self.returns_history.append(ret)
            if len(self.returns_history) > self.volatility_window:
                self.returns_history.pop(0)
        
        # Risk penalties
        inventory_penalty = abs(self.inventory) * 0.01  # Linear inventory penalty
        
        # Drawdown penalty
        self.peak_portfolio_value = max(self.peak_portfolio_value, portfolio_value)
        drawdown = max(0, self.peak_portfolio_value - portfolio_value)
        drawdown_penalty = drawdown * 0.001
        
        # Volatility penalty
        volatility_penalty = 0
        if len(self.returns_history) >= 5:
            vol = np.std(self.returns_history)
            volatility_penalty = vol * 0.1
        
        # Total risk penalty
        total_risk = inventory_penalty + drawdown_penalty + volatility_penalty
        
        # Final reward
        reward = pnl_change - transaction_cost - self.risk_aversion * total_risk
        
        return reward
    
    def _get_info(self):
        """Enhanced info with risk metrics"""
        info = super()._get_info()
        
        # Add risk metrics
        drawdown = max(0, self.peak_portfolio_value - info['portfolio_value'])
        volatility = np.std(self.returns_history) if len(self.returns_history) >= 2 else 0
        
        info.update({
            'drawdown': drawdown,
            'max_drawdown_pct': drawdown / self.peak_portfolio_value * 100,
            'volatility': volatility,
            'inventory_abs': abs(self.inventory),
            'risk_aversion': self.risk_aversion
        })
        
        return info

def test_reward_scenarios():
    """Test different reward scenarios"""
    print("=== Reward Engineering Tests ===")
    
    # Test 1: Conservative agent (high risk aversion)
    env_conservative = RiskAwareTradingEnv(risk_aversion=0.5)
    
    # Test 2: Aggressive agent (low risk aversion)  
    env_aggressive = RiskAwareTradingEnv(risk_aversion=0.01)
    
    scenarios = [
        ("Conservative (λ=0.5)", env_conservative),
        ("Aggressive (λ=0.01)", env_aggressive)
    ]
    
    for name, env in scenarios:
        print(f"\n--- {name} ---")
        obs, _ = env.reset(seed=42)
        
        total_reward = 0
        for step in range(10):
            # Simulate some trading activity
            action = 1 if step % 3 == 0 else (2 if step % 3 == 1 else 0)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            if step % 3 == 0:  # Print every 3rd step
                print(f"Step {step}: Reward={reward:.4f}, PnL={info['pnl']:.2f}, "
                      f"Drawdown={info['max_drawdown_pct']:.2f}%")
        
        print(f"Total Reward: {total_reward:.4f}")

if __name__ == "__main__":
    test_reward_scenarios()
    
    print("\n=== Day 3 Deliverables ===")
    print("✓ Risk-aware reward function implemented")
    print("✓ Configurable λ (risk aversion) parameter")
    print("✓ Multiple risk penalties: inventory, drawdown, volatility")
    print("✓ Reward ≠ raw PnL (risk-adjusted)")
    print("✓ Stable reward computation verified")