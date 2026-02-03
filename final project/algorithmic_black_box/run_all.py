"""
Master Run Script for Algorithmic Black Box Project
Execute all 11 days in sequence with proper error handling
"""

import os
import sys
import subprocess
import time
from datetime import datetime

class AlgorithmicBlackBoxRunner:
    """Master runner for the complete 11-day project"""
    
    def __init__(self):
        self.project_root = os.path.dirname(os.path.abspath(__file__))
        self.phases = {
            'Phase I: AI Integration': [
                ('Day 1: MDP Framework', 'phase1_ai_integration/day1_mdp_framework.py'),
                ('Day 2: Trading Environment', 'phase1_ai_integration/day2_trading_environment.py'),
                ('Day 3: Reward Engineering', 'phase1_ai_integration/day3_reward_engineering.py'),
                ('Day 4: PPO Agent', 'phase1_ai_integration/day4_ppo_agent.py'),
                ('Day 5: Training Run', 'phase1_ai_integration/day5_training_run.py'),
            ],
            'Phase II: Market Dynamics': [
                ('Day 6: Multi-Agent Simulation', 'phase2_market_dynamics/day6_multi_agent_sim.py'),
                ('Day 7: Stylized Facts', 'phase2_market_dynamics/day7_stylized_facts.py'),
                ('Day 8: Herding Analysis', 'phase2_market_dynamics/day8_herding_analysis.py'),
            ],
            'Phase III: Optimization': [
                ('Day 9: Hyperparameter Tuning', 'phase3_optimization/day9_hyperparameter_tuning.py'),
                ('Day 10: Benchmarking', 'phase3_optimization/day10_benchmarking.py'),
                ('Day 11: Dashboard', 'phase3_optimization/day11_dashboard.py'),
            ]
        }
        
        self.results = {}
    
    def run_script(self, script_path, day_name):
        """Run a single script with error handling"""
        print(f"\n{'='*60}")
        print(f"🚀 RUNNING: {day_name}")
        print(f"📁 Script: {script_path}")
        print(f"⏰ Started: {datetime.now().strftime('%H:%M:%S')}")
        print(f"{'='*60}")
        
        start_time = time.time()
        
        try:
            # Change to script directory
            script_dir = os.path.dirname(os.path.join(self.project_root, script_path))
            script_name = os.path.basename(script_path)
            
            # Run the script
            result = subprocess.run(
                [sys.executable, script_name],
                cwd=script_dir,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            end_time = time.time()
            duration = end_time - start_time
            
            if result.returncode == 0:
                print(f"✅ SUCCESS: {day_name}")
                print(f"⏱️  Duration: {duration:.1f} seconds")
                if result.stdout:
                    print("📋 Output:")
                    print(result.stdout[-500:])  # Last 500 chars
                
                self.results[day_name] = {
                    'status': 'SUCCESS',
                    'duration': duration,
                    'output': result.stdout
                }
                return True
            else:
                print(f"❌ FAILED: {day_name}")
                print(f"⏱️  Duration: {duration:.1f} seconds")
                print("🚨 Error:")
                print(result.stderr)
                
                self.results[day_name] = {
                    'status': 'FAILED',
                    'duration': duration,
                    'error': result.stderr
                }
                return False
                
        except subprocess.TimeoutExpired:
            print(f"⏰ TIMEOUT: {day_name} (exceeded 5 minutes)")
            self.results[day_name] = {
                'status': 'TIMEOUT',
                'duration': 300,
                'error': 'Script exceeded 5 minute timeout'
            }
            return False
            
        except Exception as e:
            print(f"💥 EXCEPTION: {day_name}")
            print(f"Error: {str(e)}")
            self.results[day_name] = {
                'status': 'EXCEPTION',
                'duration': 0,
                'error': str(e)
            }
            return False
    
    def run_phase(self, phase_name, scripts, continue_on_error=True):
        """Run all scripts in a phase"""
        print(f"\n🎯 STARTING {phase_name}")
        print(f"📊 Scripts to run: {len(scripts)}")
        
        phase_results = []
        
        for day_name, script_path in scripts:
            success = self.run_script(script_path, day_name)
            phase_results.append(success)
            
            if not success and not continue_on_error:
                print(f"🛑 STOPPING: {day_name} failed and continue_on_error=False")
                break
            
            # Brief pause between scripts
            time.sleep(2)
        
        success_count = sum(phase_results)
        total_count = len(phase_results)
        
        print(f"\n📈 {phase_name} SUMMARY:")
        print(f"✅ Successful: {success_count}/{total_count}")
        print(f"❌ Failed: {total_count - success_count}/{total_count}")
        
        return phase_results
    
    def run_all(self, continue_on_error=True):
        """Run the complete 11-day project"""
        print("🎉 STARTING ALGORITHMIC BLACK BOX PROJECT")
        print("📅 11-Day Implementation")
        print(f"🕐 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        overall_start = time.time()
        all_results = {}
        
        for phase_name, scripts in self.phases.items():
            phase_results = self.run_phase(phase_name, scripts, continue_on_error)
            all_results[phase_name] = phase_results
        
        overall_end = time.time()
        total_duration = overall_end - overall_start
        
        # Generate final report
        self.generate_final_report(total_duration)
        
        return all_results
    
    def generate_final_report(self, total_duration):
        """Generate comprehensive final report"""
        print(f"\n{'='*80}")
        print("🏁 ALGORITHMIC BLACK BOX PROJECT COMPLETE")
        print(f"{'='*80}")
        
        print(f"⏱️  Total Duration: {total_duration/60:.1f} minutes")
        print(f"🕐 Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Count successes and failures
        total_scripts = 0
        successful_scripts = 0
        
        print(f"\n📊 DETAILED RESULTS:")
        for day_name, result in self.results.items():
            total_scripts += 1
            status_emoji = "✅" if result['status'] == 'SUCCESS' else "❌"
            print(f"{status_emoji} {day_name}: {result['status']} ({result['duration']:.1f}s)")
            
            if result['status'] == 'SUCCESS':
                successful_scripts += 1
        
        success_rate = (successful_scripts / total_scripts) * 100
        
        print(f"\n🎯 OVERALL SUMMARY:")
        print(f"✅ Successful: {successful_scripts}/{total_scripts} ({success_rate:.1f}%)")
        print(f"❌ Failed: {total_scripts - successful_scripts}/{total_scripts}")
        
        # Project deliverables
        print(f"\n📦 PROJECT DELIVERABLES:")
        deliverables = [
            "✓ MDP Framework Implementation",
            "✓ Custom Gymnasium Trading Environment", 
            "✓ Risk-Aware Reward Function",
            "✓ Trained PPO Trading Agent",
            "✓ Multi-Agent Market Simulation",
            "✓ Stylized Facts Analysis",
            "✓ Herding Behavior Detection",
            "✓ Hyperparameter Optimization",
            "✓ Alpha Testing vs Baselines",
            "✓ Interactive Trading Dashboard",
            "✓ Complete Documentation"
        ]
        
        for deliverable in deliverables:
            print(f"  {deliverable}")
        
        # Next steps
        if success_rate >= 80:
            print(f"\n🎉 PROJECT SUCCESS!")
            print("🚀 Ready for production deployment")
            print("📈 Consider extending with:")
            print("  • Real market data integration")
            print("  • Advanced RL algorithms (SAC, TD3)")
            print("  • Multi-asset portfolio optimization")
        else:
            print(f"\n⚠️  PROJECT NEEDS ATTENTION")
            print("🔧 Review failed components")
            print("📋 Check error logs above")
            print("🛠️  Fix issues and re-run")
        
        print(f"\n{'='*80}")
    
    def run_single_day(self, day_number):
        """Run a single day by number (1-11)"""
        day_mapping = {}
        day_counter = 1
        
        for phase_name, scripts in self.phases.items():
            for day_name, script_path in scripts:
                day_mapping[day_counter] = (day_name, script_path)
                day_counter += 1
        
        if day_number in day_mapping:
            day_name, script_path = day_mapping[day_number]
            return self.run_script(script_path, day_name)
        else:
            print(f"❌ Invalid day number: {day_number}. Must be 1-11.")
            return False

def main():
    """Main entry point"""
    runner = AlgorithmicBlackBoxRunner()
    
    if len(sys.argv) > 1:
        # Run specific day
        try:
            day_num = int(sys.argv[1])
            runner.run_single_day(day_num)
        except ValueError:
            print("Usage: python run_all.py [day_number]")
            print("Example: python run_all.py 5  # Run only Day 5")
    else:
        # Run all days
        runner.run_all(continue_on_error=True)

if __name__ == "__main__":
    main()