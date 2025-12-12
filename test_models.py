"""
Test trained RL models and generate results for paper
Separate from training for clean workflow
"""

import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO, SAC, TD3
from robot_pov_env import RobotPOVEnv, RobotPOVContinuousEnv
import time
import os


def test_single_algorithm(algo_name, model_path, n_episodes=20, render=False, verbose=True):
    """
    Test a single trained model
    
    Returns:
        dict: Statistics including avg_reward, success_rate, avg_length, etc.
    """
    
    if verbose:
        print("\n" + "="*70)
        print(f"Testing {algo_name}")
        print("="*70 + "\n")
    
    # Load model
    if algo_name == "PPO":
        model = PPO.load(model_path)
        EnvClass = RobotPOVEnv
    elif algo_name == "SAC":
        model = SAC.load(model_path)
        EnvClass = RobotPOVContinuousEnv
    elif algo_name == "TD3":
        model = TD3.load(model_path)
        EnvClass = RobotPOVContinuousEnv
    else:
        raise ValueError(f"Unknown algorithm: {algo_name}")
    
    # Create test environment
    env = EnvClass(
        grid_size=20,
        map_type="city",
        render_mode="human" if render else None,
        use_camera_obs=False,
        num_obstacles=4,
    )
    
    episode_rewards = []
    episode_lengths = []
    successes = []
    collisions = []
    
    for ep in range(n_episodes):
        obs, info = env.reset()
        done = False
        truncated = False
        total_reward = 0.0
        step = 0
        
        while not (done or truncated) and step < 500:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            total_reward += float(reward)
            step += 1
            
            if render:
                time.sleep(0.01)
        
        # Determine outcome
        success = (done and total_reward > 50)  # Got success bonus
        collision = (done and total_reward < -20)  # Got collision penalty
        
        episode_rewards.append(total_reward)
        episode_lengths.append(step)
        successes.append(success)
        collisions.append(collision)
        
        if verbose:
            outcome = "SUCCESS" if success else ("COLLISION" if collision else "TIMEOUT")
            print(f"Episode {ep+1:2d}: {step:3d} steps, reward={total_reward:6.1f} [{outcome}]")
    
    env.close()
    
    # Calculate statistics
    stats = {
        "avg_reward": float(np.mean(episode_rewards)),
        "std_reward": float(np.std(episode_rewards)),
        "avg_length": float(np.mean(episode_lengths)),
        "std_length": float(np.std(episode_lengths)),
        "success_rate": float(np.mean(successes) * 100),  # Percentage
        "collision_rate": float(np.mean(collisions) * 100),
        "min_reward": float(np.min(episode_rewards)),
        "max_reward": float(np.max(episode_rewards)),
    }
    
    if verbose:
        print(f"\n{algo_name} Test Summary:")
        print(f"  Average Reward:   {stats['avg_reward']:6.1f} ± {stats['std_reward']:.1f}")
        print(f"  Average Length:   {stats['avg_length']:6.1f} ± {stats['std_length']:.1f} steps")
        print(f"  Success Rate:     {stats['success_rate']:5.1f}%")
        print(f"  Collision Rate:   {stats['collision_rate']:5.1f}%")
        print(f"  Reward Range:     [{stats['min_reward']:.1f}, {stats['max_reward']:.1f}]")
    
    return stats


def compare_all_algorithms(n_episodes=20, model_dir="models"):
    """
    Test all trained algorithms and generate comparison
    """
    
    print("\n" + "="*70)
    print("TESTING ALL ALGORITHMS")
    print("="*70)
    
    algorithms = ["PPO", "SAC", "TD3"]
    results = {}
    
    for algo in algorithms:
        model_path = f"{model_dir}/{algo}/final_model"
        
        if not os.path.exists(f"{model_path}.zip"):
            print(f"\n{algo} model not found at {model_path}.zip")
            print(f"   Skipping {algo}...")
            continue
        
        results[algo] = test_single_algorithm(
            algo,
            model_path,
            n_episodes=n_episodes,
            render=False,
            verbose=True
        )
    
    if not results:
        print("\nNo trained models found!")
        print("Train models first using:")
        print("  python train_optimized.py --algo PPO --timesteps 200000 --n-envs 4")
        return
    
    # Print comparison table
    print("\n" + "="*70)
    print("COMPARISON TABLE")
    print("="*70)
    print(f"\n{'Algorithm':<12} {'Avg Reward':<15} {'Success Rate':<15} {'Avg Length':<15}")
    print("-" * 70)
    
    for algo in algorithms:
        if algo not in results:
            continue
        r = results[algo]
        print(f"{algo:<12} {r['avg_reward']:>6.1f} ± {r['std_reward']:<5.1f} "
              f"{r['success_rate']:>6.1f}%         "
              f"{r['avg_length']:>6.1f} ± {r['std_length']:<5.1f}")
    
    # Generate plots
    if len(results) > 0:
        generate_comparison_plots(results, algorithms)
    
    return results


def generate_comparison_plots(results, algorithms):
    """
    Generate comparison plots for paper
    """
    
    print("\n" + "="*70)
    print("GENERATING PLOTS")
    print("="*70)
    
    # Filter to only algorithms with results
    algos_with_results = [a for a in algorithms if a in results]
    
    if len(algos_with_results) == 0:
        print("No results to plot!")
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    colors = ['#2ecc71', '#3498db', '#e74c3c']  # Green, Blue, Red
    
    # Plot 1: Average Rewards
    avg_rewards = [results[algo]['avg_reward'] for algo in algos_with_results]
    std_rewards = [results[algo]['std_reward'] for algo in algos_with_results]
    axes[0].bar(algos_with_results, avg_rewards, 
                yerr=std_rewards, capsize=5,
                color=colors[:len(algos_with_results)])
    axes[0].set_ylabel('Average Reward', fontsize=12)
    axes[0].set_title('Average Episode Reward', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3, axis='y')
    axes[0].axhline(y=0, color='black', linestyle='--', linewidth=0.8)
    
    # Plot 2: Success Rates
    success_rates = [results[algo]['success_rate'] for algo in algos_with_results]
    axes[1].bar(algos_with_results, success_rates,
                color=colors[:len(algos_with_results)])
    axes[1].set_ylabel('Success Rate (%)', fontsize=12)
    axes[1].set_title('Goal Achievement Rate', fontsize=14, fontweight='bold')
    axes[1].set_ylim([0, 100])
    axes[1].grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Episode Lengths
    avg_lengths = [results[algo]['avg_length'] for algo in algos_with_results]
    std_lengths = [results[algo]['std_length'] for algo in algos_with_results]
    axes[2].bar(algos_with_results, avg_lengths,
                yerr=std_lengths, capsize=5,
                color=colors[:len(algos_with_results)])
    axes[2].set_ylabel('Episode Length (steps)', fontsize=12)
    axes[2].set_title('Average Episode Length', fontsize=14, fontweight='bold')
    axes[2].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('algorithm_comparison.png', dpi=150, bbox_inches='tight')
    print(f"✓ Saved: algorithm_comparison.png")
    
    # Generate individual metric plots for paper
    fig2, ax = plt.subplots(figsize=(8, 6))
    
    metrics = ['avg_reward', 'success_rate', 'avg_length']
    metric_names = ['Avg Reward', 'Success Rate (%)', 'Avg Episode Length']
    
    x = np.arange(len(algos_with_results))
    width = 0.25
    
    for i, (metric, name) in enumerate(zip(metrics, metric_names)):
        values = [results[algo][metric] for algo in algos_with_results]
        # Normalize to 0-100 scale for comparison
        if metric == 'avg_reward':
            values = [(v + 50) / 1.5 for v in values]  # Rough normalization
        elif metric == 'avg_length':
            values = [100 - (v / 3) for v in values]  # Lower is better
        
        ax.bar(x + i*width, values, width, label=name, alpha=0.8)
    
    ax.set_ylabel('Normalized Score', fontsize=12)
    ax.set_title('Algorithm Performance Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels(algos_with_results)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('normalized_comparison.png', dpi=150, bbox_inches='tight')
    print(f"✓ Saved: normalized_comparison.png")
    
    plt.close('all')


def watch_trained_agent(algo_name, model_path, n_episodes=5):
    """
    Watch a trained agent perform (with rendering)
    """
    
    print(f"\n{'='*70}")
    print(f"WATCHING {algo_name} AGENT")
    print(f"{'='*70}\n")
    print("Close the window to continue to next episode...")
    
    test_single_algorithm(
        algo_name,
        model_path,
        n_episodes=n_episodes,
        render=True,
        verbose=True
    )


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test trained RL models')
    parser.add_argument('--mode', type=str, default='compare',
                       choices=['compare', 'single', 'watch'],
                       help='compare=test all, single=test one, watch=visualize')
    parser.add_argument('--algo', type=str, default='PPO',
                       choices=['PPO', 'SAC', 'TD3'],
                       help='Algorithm to test (for single/watch modes)')
    parser.add_argument('--episodes', type=int, default=20,
                       help='Number of test episodes')
    parser.add_argument('--model-dir', type=str, default='models',
                       help='Directory containing trained models')
    
    args = parser.parse_args()
    
    if args.mode == 'compare':
        # Test all algorithms and generate comparison
        results = compare_all_algorithms(
            n_episodes=args.episodes,
            model_dir=args.model_dir
        )
    
    elif args.mode == 'single':
        # Test single algorithm
        model_path = f"{args.model_dir}/{args.algo}/final_model"
        if not os.path.exists(f"{model_path}.zip"):
            print(f"Model not found: {model_path}.zip")
            print(f"Train first: python train_optimized.py --algo {args.algo}")
        else:
            test_single_algorithm(
                args.algo,
                model_path,
                n_episodes=args.episodes,
                render=False,
                verbose=True
            )
    
    elif args.mode == 'watch':
        # Watch agent with visualization
        model_path = f"{args.model_dir}/{args.algo}/final_model"
        if not os.path.exists(f"{model_path}.zip"):
            print(f"Model not found: {model_path}.zip")
            print(f"Train first: python train_optimized.py --algo {args.algo}")
        else:
            watch_trained_agent(
                args.algo,
                model_path,
                n_episodes=args.episodes
            )
    
    print("\nDone!")