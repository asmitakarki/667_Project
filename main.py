"""
Master Script for CS 667 Pathfinding Project
Run this script to execute different parts of your project

Usage:
    python main.py --mode [visualize|train|evaluate|compare|demo]
"""

import argparse
import sys

def visualize_maps():
    """Visualize all different map types"""
    print("\n=== Visualizing Map Types ===\n")
    from map_generators import (
        RandomObstaclesGenerator, SpiralMapGenerator, MazeMapGenerator,
        GridMapGenerator, CorridorMapGenerator
    )
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Rectangle
    
    maps = {
        'Random': RandomObstaclesGenerator(grid_size=20, num_obstacles=8),
        'Spiral': SpiralMapGenerator(grid_size=20, num_spirals=3),
        'Maze': MazeMapGenerator(grid_size=20, cell_size=2),
        'Grid': GridMapGenerator(grid_size=20, spacing=3),
        'Corridor': CorridorMapGenerator(grid_size=20, num_segments=6)
    }
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, (name, map_gen) in enumerate(maps.items()):
        ax = axes[idx]
        ax.set_xlim(0, map_gen.grid_size)
        ax.set_ylim(0, map_gen.grid_size)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_title(name, fontweight='bold')
        
        for obs in map_gen.obstacles:
            if 'width' in obs and 'height' in obs:
                rect = Rectangle(
                    (obs['pos'][0] - obs['width']/2, obs['pos'][1] - obs['height']/2),
                    obs['width'], obs['height'],
                    color='gray', alpha=0.7
                )
                ax.add_patch(rect)
            else:
                circle = Circle(obs['pos'], obs['size'], color='gray', alpha=0.7)
                ax.add_patch(circle)
    
    axes[-1].axis('off')
    plt.tight_layout()
    plt.savefig('map_types_overview.png', dpi=150, bbox_inches='tight')
    print("Map visualization saved as 'map_types_overview.png'")
    plt.show()

def train_agent(map_type='random', algorithm='PPO', timesteps=100000):
    """Train an RL agent"""
    print(f"\n=== Training {algorithm} on {map_type} map ===\n")
    
    from stable_baselines3 import PPO, DQN, A2C
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.monitor import Monitor
    from enhanced_pathfinding_env import EnhancedPathfindingEnv
    
    def make_env():
        env = EnhancedPathfindingEnv(grid_size=20, map_type=map_type)
        return Monitor(env)
    
    env = DummyVecEnv([make_env])
    
    if algorithm == 'PPO':
        model = PPO('MlpPolicy', env, verbose=1, learning_rate=3e-4)
    elif algorithm == 'DQN':
        model = DQN('MlpPolicy', env, verbose=1, learning_rate=1e-4)
    elif algorithm == 'A2C':
        model = A2C('MlpPolicy', env, verbose=1, learning_rate=7e-4)
    
    model.learn(total_timesteps=timesteps)
    
    model_name = f'models/{algorithm}_{map_type}_final'
    model.save(model_name)
    print(f"\nModel saved as '{model_name}'")
    
    return model

def evaluate_agent(model_path, map_type='random', num_episodes=10):
    """Evaluate a trained agent"""
    print(f"\n=== Evaluating Agent on {map_type} map ===\n")
    
    from stable_baselines3 import PPO, DQN, A2C
    from enhanced_pathfinding_env import EnhancedPathfindingEnv
    import numpy as np
    
    # Load model (try different algorithms)
    try:
        model = PPO.load(model_path)
    except:
        try:
            model = DQN.load(model_path)
        except:
            model = A2C.load(model_path)
    
    env = EnhancedPathfindingEnv(grid_size=20, map_type=map_type, render_mode='human')
    
    successes = 0
    episode_lengths = []
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        steps = 0
        
        while True:
            action, _ = model.predict(obs, deterministic=True)
            # Convert action to scalar if it's an array
            if isinstance(action, np.ndarray):
                action = int(action.item())
            obs, reward, terminated, truncated, info = env.step(action)
            env.render()
            steps += 1
            
            if terminated or truncated:
                if terminated:
                    successes += 1
                    print(f"Episode {episode+1}: SUCCESS in {steps} steps")
                else:
                    print(f"Episode {episode+1}: TIMEOUT in {steps} steps")
                episode_lengths.append(steps)
                break
    
    env.close()
    
    print(f"\n=== Results ===")
    print(f"Success rate: {successes}/{num_episodes} ({100*successes/num_episodes:.1f}%)")
    print(f"Average steps: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")

def test_astar(map_type='random', num_episodes=10):
    """Test A* algorithm"""
    print(f"\n=== Testing A* on {map_type} map ===\n")
    
    from enhanced_pathfinding_env import EnhancedPathfindingEnv
    from astar_baseline import AStarPathfinder, test_astar_on_env
    
    env = EnhancedPathfindingEnv(
        grid_size=20,
        map_type=map_type,
        render_mode='human'
    )
    
    pathfinder = AStarPathfinder(grid_size=20, grid_resolution=0.5)
    test_astar_on_env(env, pathfinder, num_episodes=num_episodes, render=True)
    
    env.close()

def run_comparison():
    """Run full comparison between RL and A*"""
    print("\n=== Running Comprehensive Comparison ===\n")
    print("This will take some time (training + evaluation)...\n")
    
    from comparison_script import PathfindingComparison
    
    map_configs = [
        ('random', {'num_obstacles': 5}),
        ('grid', {'spacing': 3}),
        ('maze', {'cell_size': 2}),
        ('spiral', {'num_spirals': 2}),
    ]
    
    comparison = PathfindingComparison(grid_size=20)
    
    results = comparison.run_full_comparison(
        map_configs=map_configs,
        algorithms=['PPO'],
        num_eval_episodes=10
    )
    
    comparison.generate_comparison_plots()
    comparison.save_results()

def demo_mode():
    """Interactive demo"""
    print("\n=== Interactive Demo ===\n")
    print("Choose a demo:")
    print("1. Test random agent on random map")
    print("2. Test random agent on maze")
    print("3. Test random agent on spiral")
    print("4. Quick A* demonstration")
    
    choice = input("\nEnter choice (1-4): ")
    
    from enhanced_pathfinding_env import EnhancedPathfindingEnv
    
    if choice == '1':
        env = EnhancedPathfindingEnv(grid_size=20, map_type='random', render_mode='human')
    elif choice == '2':
        env = EnhancedPathfindingEnv(grid_size=20, map_type='maze', render_mode='human')
    elif choice == '3':
        env = EnhancedPathfindingEnv(grid_size=20, map_type='spiral', render_mode='human')
    elif choice == '4':
        test_astar('random', num_episodes=3)
        return
    else:
        print("Invalid choice")
        return
    
    obs, info = env.reset()
    
    print("\nPress Ctrl+C to stop...")
    try:
        for _ in range(200):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            env.render()
            
            if terminated or truncated:
                print(f"\nEpisode finished. Distance to goal: {info['distance_to_goal']:.2f}")
                obs, info = env.reset()
    except KeyboardInterrupt:
        print("\nDemo stopped")
    
    env.close()

def main():
    parser = argparse.ArgumentParser(description='CS 667 Pathfinding Project')
    parser.add_argument('--mode', type=str, required=True,
                       choices=['visualize', 'train', 'evaluate', 'astar', 'compare', 'demo'],
                       help='Operation mode')
    parser.add_argument('--map-type', type=str, default='random',
                       choices=['random', 'spiral', 'maze', 'grid', 'corridor'],
                       help='Map type to use')
    parser.add_argument('--algorithm', type=str, default='PPO',
                       choices=['PPO', 'DQN', 'A2C'],
                       help='RL algorithm')
    parser.add_argument('--timesteps', type=int, default=100000,
                       help='Training timesteps')
    parser.add_argument('--model-path', type=str, default=None,
                       help='Path to trained model for evaluation')
    parser.add_argument('--episodes', type=int, default=10,
                       help='Number of evaluation episodes')
    
    args = parser.parse_args()
    
    if args.mode == 'visualize':
        visualize_maps()
    
    elif args.mode == 'train':
        train_agent(args.map_type, args.algorithm, args.timesteps)
    
    elif args.mode == 'evaluate':
        if args.model_path is None:
            print("Error: --model-path required for evaluation")
            sys.exit(1)
        evaluate_agent(args.model_path, args.map_type, args.episodes)
    
    elif args.mode == 'astar':
        test_astar(args.map_type, args.episodes)
    
    elif args.mode == 'compare':
        run_comparison()
    
    elif args.mode == 'demo':
        demo_mode()

if __name__ == "__main__":
    # Quick start guide
    if len(sys.argv) == 1:
        print("\n" + "="*70)
        print("CS 667 Pathfinding Project - Quick Start Guide")
        print("="*70)
        print("\nUsage: python main.py --mode [MODE] [OPTIONS]\n")
        print("Modes:")
        print("  visualize  - Show all map types")
        print("  train      - Train an RL agent")
        print("  evaluate   - Evaluate a trained agent")
        print("  astar      - Test A* baseline")
        print("  compare    - Run full comparison (RL vs A*)")
        print("  demo       - Interactive demo")
        print("\nExamples:")
        print("  python main.py --mode visualize")
        print("  python main.py --mode train --map-type maze --algorithm PPO")
        print("  python main.py --mode astar --map-type spiral --episodes 5")
        print("  python main.py --mode compare")
        print("  python main.py --mode demo")
        print("\nFor detailed help:")
        print("  python main.py --help")
        print("="*70 + "\n")
    else:
        main()