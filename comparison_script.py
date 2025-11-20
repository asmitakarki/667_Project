"""
Comprehensive comparison script for RL agents vs A* baseline
Tests across different map types and generates comparison plots
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from stable_baselines3 import PPO, DQN, A2C
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from enhanced_pathfinding_env import EnhancedPathfindingEnv
from astar_baseline import AStarPathfinder
import json
import time

class PathfindingComparison:
    """Compare RL agents with A* baseline across different map types"""
    
    def __init__(self, grid_size=20):
        self.grid_size = grid_size
        self.results = {}
        
    def train_rl_agent(self, map_type, algorithm='PPO', timesteps=50000, **map_kwargs):
        """Train an RL agent on a specific map type"""
        print(f"\n{'='*60}")
        print(f"Training {algorithm} on {map_type} map")
        print(f"{'='*60}\n")
        
        # Create environment
        def make_env():
            env = EnhancedPathfindingEnv(
                grid_size=self.grid_size,
                map_type=map_type,
                **map_kwargs
            )
            return Monitor(env)
        
        env = DummyVecEnv([make_env])
        
        # Initialize model
        if algorithm == 'PPO':
            model = PPO('MlpPolicy', env, verbose=1, learning_rate=3e-4)
        elif algorithm == 'DQN':
            model = DQN('MlpPolicy', env, verbose=1, learning_rate=1e-4)
        elif algorithm == 'A2C':
            model = A2C('MlpPolicy', env, verbose=1, learning_rate=7e-4)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        
        # Train
        start_time = time.time()
        model.learn(total_timesteps=timesteps)
        training_time = time.time() - start_time
        
        print(f"\nTraining completed in {training_time:.2f} seconds")
        
        return model, training_time
    
    def evaluate_rl_agent(self, model, map_type, num_episodes=20, **map_kwargs):
        """Evaluate RL agent"""
        print(f"\nEvaluating RL agent on {map_type} map...")
        
        env = EnhancedPathfindingEnv(
            grid_size=self.grid_size,
            map_type=map_type,
            **map_kwargs
        )
        
        successes = 0
        episode_lengths = []
        episode_rewards = []
        
        for episode in range(num_episodes):
            obs, info = env.reset()
            episode_reward = 0
            steps = 0
            
            while True:
                action, _ = model.predict(obs, deterministic=True)
                # Convert action to scalar if it's an array
                if isinstance(action, np.ndarray):
                    action = int(action.item())
                obs, reward, terminated, truncated, info = env.step(action)
                
                episode_reward += reward
                steps += 1
                
                if terminated or truncated:
                    if terminated:
                        successes += 1
                    episode_lengths.append(steps)
                    episode_rewards.append(episode_reward)
                    break
        
        env.close()
        
        return {
            'success_rate': successes / num_episodes,
            'avg_steps': np.mean(episode_lengths),
            'std_steps': np.std(episode_lengths),
            'avg_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards)
        }
    
    def evaluate_astar(self, map_type, num_episodes=20, **map_kwargs):
        """Evaluate A* baseline"""
        print(f"\nEvaluating A* on {map_type} map...")
        
        env = EnhancedPathfindingEnv(
            grid_size=self.grid_size,
            map_type=map_type,
            **map_kwargs
        )
        
        pathfinder = AStarPathfinder(grid_size=self.grid_size, grid_resolution=0.5)
        
        successes = 0
        episode_lengths = []
        computation_times = []
        path_lengths = []
        
        for episode in range(num_episodes):
            obs, info = env.reset()
            start = env.robot_pos.copy()
            goal = env.goal_pos.copy()
            obstacles = env.obstacles
            
            # Find path
            path, stats = pathfinder.find_path(start, goal, obstacles)
            
            if path is None:
                continue
            
            computation_times.append(stats['computation_time'])
            path_lengths.append(len(path))
            
            # Execute path
            steps = 0
            for waypoint in path[1:]:
                while np.linalg.norm(env.robot_pos - waypoint) > 0.3:
                    direction = waypoint - env.robot_pos
                    direction = direction / np.linalg.norm(direction)
                    
                    if abs(direction[0]) > abs(direction[1]):
                        action = 1 if direction[0] > 0 else 3
                    else:
                        action = 0 if direction[1] > 0 else 2
                    
                    obs, reward, terminated, truncated, info = env.step(action)
                    steps += 1
                    
                    if terminated:
                        successes += 1
                        episode_lengths.append(steps)
                        break
                    if truncated:
                        break
                
                if terminated or truncated:
                    break
        
        env.close()
        
        return {
            'success_rate': successes / num_episodes if episode_lengths else 0,
            'avg_steps': np.mean(episode_lengths) if episode_lengths else 0,
            'std_steps': np.std(episode_lengths) if episode_lengths else 0,
            'avg_computation_time': np.mean(computation_times) if computation_times else 0,
            'avg_path_length': np.mean(path_lengths) if path_lengths else 0
        }
    
    def run_full_comparison(self, map_configs, algorithms=['PPO'], num_eval_episodes=20):
        """
        Run complete comparison across all map types and algorithms
        
        Args:
            map_configs: List of (map_type, map_kwargs) tuples
            algorithms: List of RL algorithms to test
            num_eval_episodes: Number of episodes for evaluation
        """
        results = {}
        
        for map_type, map_kwargs in map_configs:
            print(f"\n{'#'*70}")
            print(f"# Testing on {map_type.upper()} map")
            print(f"{'#'*70}")
            
            map_results = {}
            
            # Test A* baseline
            astar_results = self.evaluate_astar(map_type, num_eval_episodes, **map_kwargs)
            map_results['A*'] = astar_results
            
            # Test RL algorithms
            for algorithm in algorithms:
                # Train agent
                model, training_time = self.train_rl_agent(
                    map_type, algorithm, timesteps=50000, **map_kwargs
                )
                
                # Evaluate agent
                rl_results = self.evaluate_rl_agent(model, map_type, num_eval_episodes, **map_kwargs)
                rl_results['training_time'] = training_time
                
                map_results[algorithm] = rl_results
            
            results[map_type] = map_results
        
        self.results = results
        return results
    
    def generate_comparison_plots(self):
        """Generate comparison plots"""
        if not self.results:
            print("No results to plot. Run comparison first.")
            return
        
        map_types = list(self.results.keys())
        methods = list(self.results[map_types[0]].keys())
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('RL vs A* Pathfinding Comparison', fontsize=16, fontweight='bold')
        
        # 1. Success Rate Comparison
        ax = axes[0, 0]
        x = np.arange(len(map_types))
        width = 0.8 / len(methods)
        
        for i, method in enumerate(methods):
            success_rates = [self.results[mt][method]['success_rate'] * 100 for mt in map_types]
            ax.bar(x + i * width, success_rates, width, label=method)
        
        ax.set_xlabel('Map Type')
        ax.set_ylabel('Success Rate (%)')
        ax.set_title('Success Rate by Map Type')
        ax.set_xticks(x + width * (len(methods) - 1) / 2)
        ax.set_xticklabels(map_types)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Average Steps Comparison
        ax = axes[0, 1]
        for i, method in enumerate(methods):
            avg_steps = [self.results[mt][method]['avg_steps'] for mt in map_types]
            std_steps = [self.results[mt][method]['std_steps'] for mt in map_types]
            ax.bar(x + i * width, avg_steps, width, yerr=std_steps, label=method, capsize=5)
        
        ax.set_xlabel('Map Type')
        ax.set_ylabel('Average Steps')
        ax.set_title('Path Length Efficiency')
        ax.set_xticks(x + width * (len(methods) - 1) / 2)
        ax.set_xticklabels(map_types)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. Training Time (RL only)
        ax = axes[1, 0]
        rl_methods = [m for m in methods if m != 'A*']
        if rl_methods:
            training_times = {method: [] for method in rl_methods}
            for mt in map_types:
                for method in rl_methods:
                    training_times[method].append(self.results[mt][method].get('training_time', 0))
            
            for i, method in enumerate(rl_methods):
                ax.bar(x + i * width, training_times[method], width, label=method)
            
            ax.set_xlabel('Map Type')
            ax.set_ylabel('Training Time (seconds)')
            ax.set_title('RL Training Time')
            ax.set_xticks(x + width * (len(rl_methods) - 1) / 2)
            ax.set_xticklabels(map_types)
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # 4. Summary Table
        ax = axes[1, 1]
        ax.axis('tight')
        ax.axis('off')
        
        # Create summary data
        table_data = []
        for method in methods:
            avg_success = np.mean([self.results[mt][method]['success_rate'] * 100 for mt in map_types])
            avg_steps = np.mean([self.results[mt][method]['avg_steps'] for mt in map_types])
            table_data.append([method, f"{avg_success:.1f}%", f"{avg_steps:.1f}"])
        
        table = ax.table(
            cellText=table_data,
            colLabels=['Method', 'Avg Success Rate', 'Avg Steps'],
            cellLoc='center',
            loc='center',
            bbox=[0, 0, 1, 1]
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        plt.tight_layout()
        plt.savefig('comparison_results.png', dpi=150, bbox_inches='tight')
        print("\nComparison plots saved as 'comparison_results.png'")
        plt.show()
    
    def save_results(self, filename='comparison_results.json'):
        """Save results to JSON file"""
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"\nResults saved to {filename}")


if __name__ == "__main__":
    print("Starting Comprehensive Pathfinding Comparison")
    print("=" * 70)
    
    # Define map configurations to test
    map_configs = [
        ('random', {'num_obstacles': 5}),
        ('grid', {'spacing': 3}),
        ('maze', {'cell_size': 2}),
        ('spiral', {'num_spirals': 2}),
    ]
    
    # Initialize comparison
    comparison = PathfindingComparison(grid_size=20)
    
    # Run comparison with PPO (you can add more algorithms)
    results = comparison.run_full_comparison(
        map_configs=map_configs,
        algorithms=['PPO'],  # Add 'DQN', 'A2C' to test more
        num_eval_episodes=10  # Increase for more robust results
    )
    
    # Generate plots
    comparison.generate_comparison_plots()
    
    # Save results
    comparison.save_results()
    
    print("\n" + "=" * 70)
    print("Comparison completed!")