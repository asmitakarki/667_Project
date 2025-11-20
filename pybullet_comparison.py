"""
Comprehensive comparison script for 3D RL agents vs A* baseline
Tests in PyBullet with realistic physics
"""

import numpy as np
import matplotlib.pyplot as plt
import json
import time
from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from pybullet_pathfinding_env import PyBulletPathfindingEnv
from astar_baseline import AStarPathfinder
import pybullet as p

class PyBulletComparison:
    """Compare RL agents with A* baseline in 3D PyBullet environment"""
    
    def __init__(self, grid_size=20):
        self.grid_size = grid_size
        self.results = {}
        
    def train_rl_agent(self, map_type, algorithm='PPO', timesteps=100000, **map_kwargs):
        """Train an RL agent on a specific map type"""
        print(f"\n{'='*60}")
        print(f"Training {algorithm} on {map_type} map (3D)")
        print(f"{'='*60}\n")
        
        # Create environment
        def make_env():
            env = PyBulletPathfindingEnv(
                grid_size=self.grid_size,
                map_type=map_type,
                render_mode=None,
                **map_kwargs
            )
            return Monitor(env)
        
        # Use parallel environments
        n_envs = 4
        env = DummyVecEnv([make_env for _ in range(n_envs)])
        
        # Initialize model
        if algorithm == 'PPO':
            model = PPO(
                'MlpPolicy', 
                env, 
                verbose=1, 
                learning_rate=3e-4,
                n_steps=2048,
                batch_size=64,
                ent_coef=0.01
            )
        elif algorithm == 'SAC':
            model = SAC('MlpPolicy', env, verbose=1, learning_rate=3e-4)
        elif algorithm == 'TD3':
            model = TD3('MlpPolicy', env, verbose=1, learning_rate=3e-4)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        
        # Train
        start_time = time.time()
        model.learn(total_timesteps=timesteps, progress_bar=True)
        training_time = time.time() - start_time
        
        print(f"\nTraining completed in {training_time:.2f} seconds")
        
        return model, training_time
    
    def evaluate_rl_agent(self, model, map_type, num_episodes=10, **map_kwargs):
        """Evaluate RL agent in 3D"""
        print(f"\nEvaluating RL agent on {map_type} map (3D)...")
        
        env = PyBulletPathfindingEnv(
            grid_size=self.grid_size,
            map_type=map_type,
            render_mode=None,
            **map_kwargs
        )
        
        successes = 0
        episode_lengths = []
        episode_rewards = []
        
        for episode in range(num_episodes):
            obs, info = env.reset()
            episode_reward = 0
            steps = 0
            
            while steps < 500:
                action, _ = model.predict(obs, deterministic=True)
                if isinstance(action, np.ndarray):
                    action = int(action.item()) if action.size == 1 else action
                
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
            'avg_steps': np.mean(episode_lengths) if episode_lengths else 0,
            'std_steps': np.std(episode_lengths) if episode_lengths else 0,
            'avg_reward': np.mean(episode_rewards) if episode_rewards else 0,
            'std_reward': np.std(episode_rewards) if episode_rewards else 0
        }
    
    def evaluate_astar(self, map_type, num_episodes=10, **map_kwargs):
        """Evaluate A* baseline in 3D"""
        print(f"\nEvaluating A* on {map_type} map (3D)...")
        
        env = PyBulletPathfindingEnv(
            grid_size=self.grid_size,
            map_type=map_type,
            render_mode=None,
            **map_kwargs
        )
        
        pathfinder = AStarPathfinder(grid_size=self.grid_size, grid_resolution=0.5)
        
        successes = 0
        episode_lengths = []
        computation_times = []
        path_lengths = []
        
        for episode in range(num_episodes):
            obs, info = env.reset()
            
            # Get 2D positions
            start_2d = obs[:2]
            goal_2d = obs[3:5]
            
            # Get obstacles
            obstacles_2d = []
            for obs_id in env.obstacle_ids:
                pos, _ = p.getBasePositionAndOrientation(obs_id)
                obstacles_2d.append({
                    'pos': np.array([pos[0], pos[1]]),
                    'size': 0.5
                })
            
            # Plan path
            path, stats = pathfinder.find_path(start_2d, goal_2d, obstacles_2d)
            
            if path is None:
                continue
            
            computation_times.append(stats['computation_time'])
            path_lengths.append(len(path))
            
            # Execute path
            steps = 0
            for waypoint in path[1:]:
                max_attempts = 50
                attempts = 0
                
                while attempts < max_attempts:
                    current_pos = obs[:2]
                    distance = np.linalg.norm(waypoint - current_pos)
                    
                    if distance < 0.5:
                        break
                    
                    # Determine action
                    robot_yaw = obs[2]
                    direction = waypoint - current_pos
                    target_angle = np.arctan2(direction[1], direction[0])
                    angle_diff = target_angle - robot_yaw
                    
                    # Normalize
                    while angle_diff > np.pi:
                        angle_diff -= 2 * np.pi
                    while angle_diff < -np.pi:
                        angle_diff += 2 * np.pi
                    
                    if abs(angle_diff) > 0.3:
                        action = 1 if angle_diff < 0 else 2
                    else:
                        action = 0
                    
                    obs, reward, terminated, truncated, info = env.step(action)
                    steps += 1
                    attempts += 1
                    
                    if terminated:
                        successes += 1
                        episode_lengths.append(steps)
                        break
                    if truncated:
                        episode_lengths.append(steps)
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
    
    def run_full_comparison(self, map_configs, algorithms=['PPO'], num_eval_episodes=10, training_timesteps=100000):
        """Run complete comparison across all map types and algorithms"""
        results = {}
        
        for map_type, map_kwargs in map_configs:
            print(f"\n{'#'*70}")
            print(f"# Testing on {map_type.upper()} map (3D)")
            print(f"{'#'*70}")
            
            map_results = {}
            
            # Test A* baseline
            astar_results = self.evaluate_astar(map_type, num_eval_episodes, **map_kwargs)
            map_results['A*'] = astar_results
            
            # Test RL algorithms
            for algorithm in algorithms:
                # Train agent
                model, training_time = self.train_rl_agent(
                    map_type, algorithm, timesteps=training_timesteps, **map_kwargs
                )
                
                # Evaluate agent
                rl_results = self.evaluate_rl_agent(model, map_type, num_eval_episodes, **map_kwargs)
                rl_results['training_time'] = training_time
                
                map_results[algorithm] = rl_results
                
                # Save model
                model.save(f"models/{algorithm}_{map_type}_pybullet")
            
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
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('3D RL vs A* Pathfinding Comparison (PyBullet)', fontsize=16, fontweight='bold')
        
        # 1. Success Rate
        ax = axes[0, 0]
        x = np.arange(len(map_types))
        width = 0.8 / len(methods)
        
        for i, method in enumerate(methods):
            success_rates = [self.results[mt][method]['success_rate'] * 100 for mt in map_types]
            ax.bar(x + i * width, success_rates, width, label=method)
        
        ax.set_xlabel('Map Type')
        ax.set_ylabel('Success Rate (%)')
        ax.set_title('Success Rate by Map Type (3D)')
        ax.set_xticks(x + width * (len(methods) - 1) / 2)
        ax.set_xticklabels(map_types)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Average Steps
        ax = axes[0, 1]
        for i, method in enumerate(methods):
            avg_steps = [self.results[mt][method]['avg_steps'] for mt in map_types]
            std_steps = [self.results[mt][method]['std_steps'] for mt in map_types]
            ax.bar(x + i * width, avg_steps, width, yerr=std_steps, label=method, capsize=5)
        
        ax.set_xlabel('Map Type')
        ax.set_ylabel('Average Steps')
        ax.set_title('Path Length Efficiency (3D)')
        ax.set_xticks(x + width * (len(methods) - 1) / 2)
        ax.set_xticklabels(map_types)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. Training Time
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
            ax.set_title('RL Training Time (3D)')
            ax.set_xticks(x + width * (len(rl_methods) - 1) / 2)
            ax.set_xticklabels(map_types)
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # 4. Summary Table
        ax = axes[1, 1]
        ax.axis('tight')
        ax.axis('off')
        
        table_data = []
        for method in methods:
            avg_success = np.mean([self.results[mt][method]['success_rate'] * 100 for mt in map_types])
            avg_steps = np.mean([self.results[mt][method]['avg_steps'] for mt in map_types])
            table_data.append([method, f"{avg_success:.1f}%", f"{avg_steps:.1f}"])
        
        table = ax.table(
            cellText=table_data,
            colLabels=['Method', 'Avg Success', 'Avg Steps'],
            cellLoc='center',
            loc='center',
            bbox=[0, 0, 1, 1]
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        plt.tight_layout()
        plt.savefig('pybullet_comparison_results.png', dpi=150, bbox_inches='tight')
        print("\nComparison plots saved as 'pybullet_comparison_results.png'")
        plt.show()
    
    def save_results(self, filename='pybullet_comparison_results.json'):
        """Save results to JSON file"""
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"\nResults saved to {filename}")


if __name__ == "__main__":
    print("Starting 3D Pathfinding Comparison (PyBullet)")
    print("=" * 70)
    
    map_configs = [
        ('random', {'num_obstacles': 5}),
        ('maze', {'cell_size': 2}),
        ('grid', {'spacing': 3}),
    ]
    
    comparison = PyBulletComparison(grid_size=20)
    
    results = comparison.run_full_comparison(
        map_configs=map_configs,
        algorithms=['PPO'],
        num_eval_episodes=5,
        training_timesteps=100000
    )
    
    comparison.generate_comparison_plots()
    comparison.save_results()
    
    print("\n" + "=" * 70)
    print("Comparison completed!")