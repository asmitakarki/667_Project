"""
Master Script for CS 667 3D Pathfinding Project (PyBullet Only)
Run this script to execute different parts of your project

Usage:
    python main.py --mode [visualize|train|evaluate|compare|demo]
"""

import argparse
import sys

def visualize_maps():
    """Visualize all different map types in PyBullet"""
    print("\n=== Visualizing 3D Map Types ===\n")
    from pybullet_pathfinding_env import PyBulletPathfindingEnv
    import time
    
    map_types = ['random', 'maze', 'grid']
    
    for map_type in map_types:
        print(f"\nShowing {map_type} map in PyBullet...")
        env = PyBulletPathfindingEnv(
            grid_size=20,
            map_type=map_type,
            render_mode='human',
            num_obstacles=5 if map_type == 'random' else None,
            cell_size=2 if map_type == 'maze' else None,
            spacing=3 if map_type == 'grid' else None
        )
        
        obs, info = env.reset()
        
        # Let user view the map
        print(f"  Viewing {map_type} map. Robot will move randomly for 100 steps...")
        for _ in range(100):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                obs, info = env.reset()
        
        env.close()
        print(f"  {map_type} map shown.")
        time.sleep(1)

def train_agent(map_type='random', algorithm='PPO', timesteps=200000):
    """Train an RL agent in PyBullet"""
    print(f"\n=== Training {algorithm} on {map_type} map (3D) ===\n")
    
    from stable_baselines3 import PPO, SAC, TD3
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.monitor import Monitor
    from pybullet_pathfinding_env import PyBulletPathfindingEnv
    
    def make_env():
        env = PyBulletPathfindingEnv(
            grid_size=20, 
            map_type=map_type,
            render_mode=None,  # No rendering during training
            num_obstacles=5
        )
        return Monitor(env)
    
    # Use multiple parallel environments for faster training
    n_envs = 4
    env = DummyVecEnv([make_env for _ in range(n_envs)])
    
    if algorithm == 'PPO':
        model = PPO(
            'MlpPolicy', 
            env, 
            verbose=1, 
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            ent_coef=0.01
        )
    elif algorithm == 'SAC':
        model = SAC('MlpPolicy', env, verbose=1, learning_rate=3e-4)
    elif algorithm == 'TD3':
        model = TD3('MlpPolicy', env, verbose=1, learning_rate=3e-4)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    
    print(f"\nTraining for {timesteps} timesteps with {n_envs} parallel environments...")
    model.learn(total_timesteps=timesteps, progress_bar=True)
    
    model_name = f'models/{algorithm}_{map_type}_pybullet'
    model.save(model_name)
    print(f"\nModel saved as '{model_name}'")
    
    return model

def evaluate_agent(model_path, map_type='random', num_episodes=10):
    """Evaluate a trained agent in PyBullet"""
    print(f"\n=== Evaluating Agent on {map_type} map (3D) ===\n")
    
    from stable_baselines3 import PPO, SAC, TD3
    from pybullet_pathfinding_env import PyBulletPathfindingEnv
    import numpy as np
    
    # Load model
    try:
        model = PPO.load(model_path)
    except:
        try:
            model = SAC.load(model_path)
        except:
            model = TD3.load(model_path)
    
    env = PyBulletPathfindingEnv(
        grid_size=20, 
        map_type=map_type, 
        render_mode='human',
        num_obstacles=5
    )
    
    successes = 0
    episode_lengths = []
    episode_rewards = []
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        steps = 0
        episode_reward = 0
        
        print(f"\nEpisode {episode+1}:")
        
        while steps < 500:
            action, _ = model.predict(obs, deterministic=True)
            if isinstance(action, np.ndarray):
                action = int(action.item()) if action.size == 1 else action
            
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            steps += 1
            
            if steps % 50 == 0:
                print(f"  Step {steps}: Distance = {info['distance_to_goal']:.2f}")
            
            if terminated:
                successes += 1
                print(f"  ✓ SUCCESS in {steps} steps! Reward: {episode_reward:.1f}")
                episode_lengths.append(steps)
                episode_rewards.append(episode_reward)
                break
            
            if truncated:
                print(f"  ✗ TIMEOUT after {steps} steps. Reward: {episode_reward:.1f}")
                episode_lengths.append(steps)
                episode_rewards.append(episode_reward)
                break
    
    env.close()
    
    print(f"\n=== Results ===")
    print(f"Success rate: {successes}/{num_episodes} ({100*successes/num_episodes:.1f}%)")
    if episode_lengths:
        print(f"Average steps: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")
        print(f"Average reward: {np.mean(episode_rewards):.1f} ± {np.std(episode_rewards):.1f}")

def test_astar(map_type='random', num_episodes=5):
    """Test A* algorithm in PyBullet environment"""
    print(f"\n=== Testing A* on {map_type} map (3D) ===\n")
    print("Note: A* plans in 2D but executes in 3D physics simulation\n")
    
    from pybullet_pathfinding_env import PyBulletPathfindingEnv
    from astar_baseline import AStarPathfinder
    import numpy as np
    
    env = PyBulletPathfindingEnv(
        grid_size=20,
        map_type=map_type,
        render_mode='human',
        num_obstacles=5
    )
    
    pathfinder = AStarPathfinder(grid_size=20, grid_resolution=0.5)
    
    successes = 0
    episode_lengths = []
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        
        # Get 2D positions for A*
        start_2d = obs[:2]
        goal_2d = obs[3:5]
        
        # Get obstacles (need to access from environment)
        obstacles_2d = []
        for obs_id in env.obstacle_ids:
            import pybullet as p
            pos, _ = p.getBasePositionAndOrientation(obs_id)
            # Approximate obstacle size
            obstacles_2d.append({
                'pos': np.array([pos[0], pos[1]]),
                'size': 0.5  # Approximate
            })
        
        print(f"\nEpisode {episode + 1}:")
        print(f"  Planning path with A*...")
        
        # Plan path
        path, stats = pathfinder.find_path(start_2d, goal_2d, obstacles_2d)
        
        if path is None:
            print(f"  ✗ A* couldn't find a path")
            continue
        
        print(f"  Path found: {len(path)} waypoints")
        print(f"  Executing in 3D physics...")
        
        # Execute path in 3D
        steps = 0
        for i, waypoint in enumerate(path[1:], 1):
            # Move towards waypoint
            max_attempts = 50
            attempts = 0
            
            while attempts < max_attempts:
                current_pos = obs[:2]
                direction = waypoint - current_pos
                distance = np.linalg.norm(direction)
                
                if distance < 0.5:
                    break
                
                # Determine action based on robot orientation
                robot_yaw = obs[2]
                target_angle = np.arctan2(direction[1], direction[0])
                angle_diff = target_angle - robot_yaw
                
                # Normalize angle
                while angle_diff > np.pi:
                    angle_diff -= 2 * np.pi
                while angle_diff < -np.pi:
                    angle_diff += 2 * np.pi
                
                # Choose action
                if abs(angle_diff) > 0.3:
                    action = 1 if angle_diff < 0 else 2  # Turn
                else:
                    action = 0  # Forward
                
                obs, reward, terminated, truncated, info = env.step(action)
                steps += 1
                attempts += 1
                
                if terminated:
                    successes += 1
                    print(f"  ✓ SUCCESS in {steps} steps!")
                    episode_lengths.append(steps)
                    break
                
                if truncated:
                    print(f"  ✗ TIMEOUT after {steps} steps")
                    episode_lengths.append(steps)
                    break
            
            if terminated or truncated:
                break
    
    env.close()
    
    print(f"\n=== A* Results ===")
    print(f"Success rate: {successes}/{num_episodes} ({100*successes/num_episodes:.1f}%)")
    if episode_lengths:
        print(f"Average steps: {np.mean(episode_lengths):.1f}")

def run_comparison():
    """Run full comparison between RL and A* in 3D"""
    print("\n=== Running 3D Comparison (RL vs A*) ===\n")
    print("This will train agents and test them. This takes time!\n")
    
    from pybullet_comparison import PyBulletComparison
    
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
        training_timesteps=100000  # Reduced for faster testing
    )
    
    comparison.generate_comparison_plots()
    comparison.save_results()

def demo_mode():
    """Interactive demo"""
    print("\n=== Interactive 3D Demo ===\n")
    print("Choose a demo:")
    print("1. Random agent on random map")
    print("2. Random agent on maze")
    print("3. Random agent on grid")
    print("4. View all map types")
    
    choice = input("\nEnter choice (1-4): ")
    
    from pybullet_pathfinding_env import PyBulletPathfindingEnv
    
    if choice == '1':
        map_type = 'random'
    elif choice == '2':
        map_type = 'maze'
    elif choice == '3':
        map_type = 'grid'
    elif choice == '4':
        visualize_maps()
        return
    else:
        print("Invalid choice")
        return
    
    env = PyBulletPathfindingEnv(
        grid_size=20, 
        map_type=map_type, 
        render_mode='human',
        num_obstacles=5
    )
    
    obs, info = env.reset()
    
    print(f"\nRunning random agent on {map_type} map...")
    print("Watch the 3D simulation. It will auto-reset.\n")
    
    for i in range(500):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            result = "GOAL REACHED!" if terminated else "Timeout"
            print(f"Episode finished at step {i}: {result}")
            obs, info = env.reset()
    
    env.close()

def main():
    parser = argparse.ArgumentParser(description='CS 667 3D Pathfinding Project')
    parser.add_argument('--mode', type=str, required=True,
                       choices=['visualize', 'train', 'evaluate', 'astar', 'compare', 'demo'],
                       help='Operation mode')
    parser.add_argument('--map-type', type=str, default='random',
                       choices=['random', 'maze', 'grid'],
                       help='Map type to use')
    parser.add_argument('--algorithm', type=str, default='PPO',
                       choices=['PPO', 'SAC', 'TD3'],
                       help='RL algorithm')
    parser.add_argument('--timesteps', type=int, default=200000,
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
    if len(sys.argv) == 1:
        print("\n" + "="*70)
        print("CS 667 3D Pathfinding Project (PyBullet) - Quick Start")
        print("="*70)
        print("\nUsage: python main.py --mode [MODE] [OPTIONS]\n")
        print("Modes:")
        print("  visualize  - Show all 3D map types")
        print("  train      - Train an RL agent in 3D")
        print("  evaluate   - Evaluate a trained agent")
        print("  astar      - Test A* baseline in 3D")
        print("  compare    - Run full comparison (RL vs A*)")
        print("  demo       - Interactive 3D demo")
        print("\nExamples:")
        print("  python main.py --mode visualize")
        print("  python main.py --mode train --map-type maze --algorithm PPO")
        print("  python main.py --mode demo")
        print("  python main.py --mode compare")
        print("\nFor detailed help:")
        print("  python main.py --help")
        print("="*70 + "\n")
    else:
        main()