# Citadel - Algorithmic Black Box

## 11-Day Implementation Roadmap

### Phase I: Integrating the Artificial Intelligence (Days 1-5)
- **Day 1**: MDP Framework (Conceptual) ✓
- **Day 2**: Custom Gymnasium Environment ✓
- **Day 3**: Reward Function Engineering ✓
- **Day 4**: First Learning Agent (PPO) ✓
- **Day 5**: Sanity Check Training Run ✓

### Phase II: Emergent Market Dynamics (Days 6-8)
- **Day 6**: Multi-Agent Market Simulation ✓
- **Day 7**: Stylized Facts Analysis ✓
- **Day 8**: Behavioral Analysis (Herding) ✓

### Phase III: Optimization & Black Box Report (Days 9-11)
- **Day 9**: Hyperparameter Tuning ✓
- **Day 10**: Benchmarking & Alpha Test ✓
- **Day 11**: Interactive Visualization Dashboard ✓

## Tech Stack
- **Core**: Python, Gymnasium (OpenAI Gym), Stable-Baselines3
- **Data**: NumPy, Pandas, SciPy
- **Visualization**: Matplotlib, Plotly, Seaborn
- **Optimization**: Optuna
- **ML**: Scikit-learn

## Project Structure
```
algorithmic_black_box/
├── phase1_ai_integration/
│   ├── day1_mdp_framework.py          # MDP conceptual framework
│   ├── day2_trading_environment.py    # Gymnasium trading environment
│   ├── day3_reward_engineering.py     # Risk-aware reward function
│   ├── day4_ppo_agent.py             # PPO agent implementation
│   └── day5_training_run.py          # Extended training validation
├── phase2_market_dynamics/
│   ├── day6_multi_agent_sim.py       # Multi-agent market simulation
│   ├── day7_stylized_facts.py        # Volatility clustering & fat tails
│   └── day8_herding_analysis.py      # Behavioral herding detection
├── phase3_optimization/
│   ├── day9_hyperparameter_tuning.py # Optuna optimization
│   ├── day10_benchmarking.py         # Alpha testing vs baselines
│   └── day11_dashboard.py            # Interactive Plotly dashboard
├── requirements.txt                   # Dependencies
└── README.md                         # This file
```

## Installation & Setup

1. **Install Dependencies**:
```bash
pip install -r requirements.txt
```

2. **Run Phase I (AI Integration)**:
```bash
cd phase1_ai_integration
python day1_mdp_framework.py
python day2_trading_environment.py
python day3_reward_engineering.py
python day4_ppo_agent.py
python day5_training_run.py
```

3. **Run Phase II (Market Dynamics)**:
```bash
cd ../phase2_market_dynamics
python day6_multi_agent_sim.py
python day7_stylized_facts.py
python day8_herding_analysis.py
```

4. **Run Phase III (Optimization)**:
```bash
cd ../phase3_optimization
python day9_hyperparameter_tuning.py
python day10_benchmarking.py
python day11_dashboard.py
```

## Key Features

### Phase I: AI Integration
- **MDP Framework**: Rigorous mathematical foundation for trading as RL problem
- **Custom Environment**: Gymnasium-compliant trading environment with realistic market dynamics
- **Risk-Aware Rewards**: Sophisticated reward engineering with inventory, drawdown, and volatility penalties
- **PPO Agent**: Stable policy optimization with comprehensive monitoring
- **Training Validation**: Extended 50k timestep runs with learning verification

### Phase II: Market Dynamics
- **Multi-Agent Simulation**: RL agent + noise traders + market makers ecosystem
- **Stylized Facts**: Volatility clustering and fat-tail distribution analysis
- **Herding Detection**: Agent correlation analysis and behavioral synchronization
- **Order Book Visualization**: Liquidity heatmaps and market microstructure

### Phase III: Optimization
- **Hyperparameter Tuning**: Optuna-based systematic optimization
- **Alpha Testing**: Statistical significance testing vs Buy & Hold, Random, Momentum baselines
- **Interactive Dashboard**: Executive-ready Plotly dashboard with full interactivity

## Results & Deliverables

Each day produces specific deliverables:

- **Day 1**: MDP validation checklist
- **Day 2**: Tested Gymnasium environment
- **Day 3**: Risk-aware reward function
- **Day 4**: Trained PPO baseline model
- **Day 5**: Learning progression analysis + plots
- **Day 6**: Multi-agent market data + order book heatmap
- **Day 7**: Stylized facts validation report
- **Day 8**: Herding behavior analysis + visualizations
- **Day 9**: Optimized hyperparameters + importance analysis
- **Day 10**: Alpha test results + statistical significance
- **Day 11**: Interactive HTML dashboard

## Success Criteria

### Phase I Success:
- ✅ Environment runs without errors
- ✅ Reward correlates with sensible behavior
- ✅ Agent shows learning over 50k timesteps
- ✅ Policy entropy decreases gradually

### Phase II Success:
- ✅ Volatility clustering detected
- ✅ Fat-tail distributions present
- ✅ Herding events identified
- ✅ Realistic market microstructure

### Phase III Success:
- ✅ Hyperparameter optimization improves performance
- ✅ RL agent beats at least one baseline on risk-adjusted basis
- ✅ Interactive dashboard fully functional

## Technical Notes

### Environment Design
- State space: [bid, ask, spread, inventory, cash] normalized to [0,1]
- Action space: Discrete {Hold, Buy, Sell}
- Reward: PnL - λ × (inventory_penalty + drawdown_penalty + volatility_penalty)

### Agent Architecture
- Policy: MLP (Multi-Layer Perceptron)
- Algorithm: PPO (Proximal Policy Optimization)
- Key hyperparameters: learning_rate, gamma, ent_coef, clip_range

### Evaluation Metrics
- **Return Metrics**: Mean PnL, Total Return
- **Risk Metrics**: Sharpe Ratio, Maximum Drawdown, Volatility
- **Trading Metrics**: Win Rate, Number of Trades, Action Diversity

## Extensions & Future Work

1. **Advanced Environments**:
   - Continuous action spaces (price + quantity)
   - Multi-asset trading
   - Transaction cost models
   - Market impact functions

2. **Advanced Agents**:
   - Actor-Critic methods (A2C, SAC)
   - Transformer-based policies
   - Multi-agent reinforcement learning

3. **Advanced Analysis**:
   - Regime detection
   - Risk factor decomposition
   - Portfolio optimization integration
   - Real market data backtesting

## License
MIT License - See LICENSE file for details

## Contact
For questions or contributions, please open an issue or submit a pull request.

---

**🎯 Project Status: COMPLETE**  
All 11 days implemented with full functionality and deliverables.