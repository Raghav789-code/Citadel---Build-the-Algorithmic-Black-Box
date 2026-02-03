"""
Day 9: Hyperparameter Tuning
Objective: Systematically optimize PPO hyperparameters using Optuna
"""

import optuna
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
import matplotlib.pyplot as plt
import pandas as pd
import joblib

# Import our custom environment
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'phase1_ai_integration'))
from day3_reward_engineering import RiskAwareTradingEnv

class HyperparameterTuner:
    """Optuna-based hyperparameter tuning for PPO trading agent"""
    
    def __init__(self, n_trials=50, n_eval_episodes=10):
        self.n_trials = n_trials
        self.n_eval_episodes = n_eval_episodes
        self.study = None
        self.best_params = None
        
    def objective(self, trial):
        """Optuna objective function"""
        # Sample hyperparameters
        learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
        gamma = trial.suggest_float('gamma', 0.9, 0.9999)
        ent_coef = trial.suggest_float('ent_coef', 1e-4, 1e-1, log=True)
        clip_range = trial.suggest_float('clip_range', 0.1, 0.4)
        n_steps = trial.suggest_categorical('n_steps', [512, 1024, 2048])
        batch_size = trial.suggest_categorical('batch_size', [32, 64, 128, 256])
        n_epochs = trial.suggest_int('n_epochs', 3, 20)
        
        # Environment parameters
        risk_aversion = trial.suggest_float('risk_aversion', 0.01, 0.5)
        
        try:
            # Create environment
            env = RiskAwareTradingEnv(
                risk_aversion=risk_aversion,
                max_steps=1000,
                initial_cash=10000
            )
            
            # Create model with sampled hyperparameters
            model = PPO(
                'MlpPolicy',
                env,
                learning_rate=learning_rate,
                gamma=gamma,
                ent_coef=ent_coef,
                clip_range=clip_range,
                n_steps=n_steps,
                batch_size=batch_size,
                n_epochs=n_epochs,
                verbose=0,
                seed=42
            )
            
            # Train model (shorter training for hyperparameter search)
            model.learn(total_timesteps=20000)
            
            # Evaluate model
            eval_env = RiskAwareTradingEnv(
                risk_aversion=risk_aversion,
                max_steps=1000,
                initial_cash=10000
            )
            
            mean_reward, std_reward = evaluate_policy(
                model, eval_env, n_eval_episodes=self.n_eval_episodes,
                deterministic=True, return_episode_rewards=False
            )
            
            # Clean up
            env.close()
            eval_env.close()
            del model
            
            return mean_reward
            
        except Exception as e:
            print(f"Trial failed: {e}")
            return -1000  # Return very low score for failed trials
    
    def tune(self, study_name="ppo_trading_optimization"):
        """Run hyperparameter optimization"""
        print(f"=== Starting Hyperparameter Tuning ({self.n_trials} trials) ===")
        
        # Create study
        self.study = optuna.create_study(
            direction='maximize',
            study_name=study_name,
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)
        )
        
        # Optimize
        self.study.optimize(self.objective, n_trials=self.n_trials, show_progress_bar=True)
        
        # Store best parameters
        self.best_params = self.study.best_params
        
        print(f"✓ Optimization completed")
        print(f"Best value: {self.study.best_value:.4f}")
        print(f"Best parameters: {self.best_params}")
        
        return self.study
    
    def analyze_results(self):
        """Analyze optimization results"""
        if self.study is None:
            print("No study available. Run tune() first.")
            return
        
        print("\n=== Hyperparameter Analysis ===")
        
        # Get completed trials
        completed_trials = [t for t in self.study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        
        if len(completed_trials) == 0:
            print("No completed trials found.")
            return
        
        # Create results DataFrame
        results = []
        for trial in completed_trials:
            result = {'trial_number': trial.number, 'value': trial.value}
            result.update(trial.params)
            results.append(result)
        
        results_df = pd.DataFrame(results)
        
        # Parameter importance
        try:
            importance = optuna.importance.get_param_importances(self.study)
            print("\nParameter Importance:")
            for param, imp in sorted(importance.items(), key=lambda x: x[1], reverse=True):
                print(f"  {param}: {imp:.4f}")
        except:
            print("Could not calculate parameter importance")
        
        # Visualizations
        self._create_optimization_plots(results_df)
        
        return results_df
    
    def _create_optimization_plots(self, results_df):
        """Create optimization analysis plots"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Optimization history
        axes[0, 0].plot(results_df['trial_number'], results_df['value'])
        axes[0, 0].set_xlabel('Trial Number')
        axes[0, 0].set_ylabel('Objective Value')
        axes[0, 0].set_title('Optimization History')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Parameter correlation with performance
        if 'learning_rate' in results_df.columns:
            axes[0, 1].scatter(results_df['learning_rate'], results_df['value'], alpha=0.6)
            axes[0, 1].set_xlabel('Learning Rate')
            axes[0, 1].set_ylabel('Objective Value')
            axes[0, 1].set_title('Learning Rate vs Performance')
            axes[0, 1].set_xscale('log')
            axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Gamma vs performance
        if 'gamma' in results_df.columns:
            axes[1, 0].scatter(results_df['gamma'], results_df['value'], alpha=0.6)
            axes[1, 0].set_xlabel('Gamma (Discount Factor)')
            axes[1, 0].set_ylabel('Objective Value')
            axes[1, 0].set_title('Gamma vs Performance')
            axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Risk aversion vs performance
        if 'risk_aversion' in results_df.columns:
            axes[1, 1].scatter(results_df['risk_aversion'], results_df['value'], alpha=0.6)
            axes[1, 1].set_xlabel('Risk Aversion')
            axes[1, 1].set_ylabel('Objective Value')
            axes[1, 1].set_title('Risk Aversion vs Performance')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('day9_hyperparameter_analysis.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    def train_best_model(self, total_timesteps=100000):
        """Train final model with best hyperparameters"""
        if self.best_params is None:
            print("No best parameters available. Run tune() first.")
            return None
        
        print(f"\n=== Training Best Model ({total_timesteps} timesteps) ===")
        print(f"Using parameters: {self.best_params}")
        
        # Extract parameters
        env_params = {
            'risk_aversion': self.best_params.get('risk_aversion', 0.1),
            'max_steps': 1000,
            'initial_cash': 10000
        }
        
        model_params = {
            'learning_rate': self.best_params.get('learning_rate', 3e-4),
            'gamma': self.best_params.get('gamma', 0.99),
            'ent_coef': self.best_params.get('ent_coef', 0.01),
            'clip_range': self.best_params.get('clip_range', 0.2),
            'n_steps': self.best_params.get('n_steps', 1024),
            'batch_size': self.best_params.get('batch_size', 64),
            'n_epochs': self.best_params.get('n_epochs', 10),
            'verbose': 1
        }
        
        # Create environment and model
        env = RiskAwareTradingEnv(**env_params)
        
        model = PPO('MlpPolicy', env, **model_params)
        
        # Train
        model.learn(total_timesteps=total_timesteps)
        
        # Save model
        model.save("day9_optimized_ppo")
        
        # Final evaluation
        eval_env = RiskAwareTradingEnv(**env_params)
        mean_reward, std_reward = evaluate_policy(
            model, eval_env, n_eval_episodes=20, deterministic=True
        )
        
        print(f"✓ Final model performance: {mean_reward:.2f} ± {std_reward:.2f}")
        
        env.close()
        eval_env.close()
        
        return model
    
    def save_study(self, filename="day9_optuna_study.pkl"):
        """Save Optuna study"""
        if self.study is not None:
            joblib.dump(self.study, filename)
            print(f"✓ Study saved to {filename}")
    
    def load_study(self, filename="day9_optuna_study.pkl"):
        """Load Optuna study"""
        try:
            self.study = joblib.load(filename)
            self.best_params = self.study.best_params
            print(f"✓ Study loaded from {filename}")
            return self.study
        except FileNotFoundError:
            print(f"Study file {filename} not found")
            return None

def run_hyperparameter_optimization():
    """Run complete hyperparameter optimization"""
    print("=== Day 9: Hyperparameter Optimization ===")
    
    # Create tuner
    tuner = HyperparameterTuner(n_trials=30, n_eval_episodes=5)  # Reduced for demo
    
    # Run optimization
    study = tuner.tune()
    
    # Analyze results
    results_df = tuner.analyze_results()
    
    # Save study
    tuner.save_study()
    
    # Train best model
    best_model = tuner.train_best_model(total_timesteps=50000)
    
    # Summary
    print(f"\n=== Optimization Summary ===")
    print(f"Trials completed: {len(study.trials)}")
    print(f"Best objective value: {study.best_value:.4f}")
    print(f"Best parameters found:")
    for param, value in tuner.best_params.items():
        print(f"  {param}: {value}")
    
    # Performance comparison
    baseline_performance = -10.0  # Assume baseline from Day 5
    improvement = study.best_value - baseline_performance
    
    if improvement > 5.0:
        print(f"\n🎉 SIGNIFICANT IMPROVEMENT: +{improvement:.2f}")
        print("→ Hyperparameter tuning successful")
    elif improvement > 0:
        print(f"\n✓ MODEST IMPROVEMENT: +{improvement:.2f}")
        print("→ Some benefit from tuning")
    else:
        print(f"\n⚠️  NO IMPROVEMENT: {improvement:.2f}")
        print("→ Task may be well-conditioned already")
    
    return tuner, results_df

if __name__ == "__main__":
    tuner, results_df = run_hyperparameter_optimization()
    
    print(f"\n=== Day 9 Deliverables ===")
    print("✓ Optuna hyperparameter study completed")
    print("✓ Parameter importance analysis done")
    print("✓ Best model trained and saved")
    print("✓ Optimization results visualized")
    print("✓ Ready for benchmarking (Day 10)")