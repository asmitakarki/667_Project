"""
Test trained RL models and generate results for report
"""

import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO, SAC, TD3
from robot_pov_env import RobotPOVContinuousEnv
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
    
    # Load model - ALL algorithms now use continuous actions!
    if algo_name == "PPO":
        model = PPO.load(model_path)
        EnvClass = RobotPOVContinuousEnv
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
        num_obstacles=6,  # CHANGED: Match training difficulty (was 4)
    )
    
    episode_rewards = []
    episode_lengths = []
    successes = []
    collisions = []
    timeouts = []
    steps_to_goal = []  # Track steps for successful episodes only
    
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
        
        # Determine outcome using info dict
        success = info.get("success", False)
        collision = info.get("collision", False)
        timeout = truncated and not done
        
        episode_rewards.append(total_reward)
        episode_lengths.append(step)
        successes.append(success)
        collisions.append(collision)
        timeouts.append(timeout)
        
        # Track steps-to-goal only for successes
        if success:
            steps_to_goal.append(step)
        
        if verbose:
            if success:
                outcome = "SUCCESS"
            elif collision:
                outcome = "COLLISION"
            elif timeout:
                outcome = "TIMEOUT"
            else:
                outcome = "UNKNOWN"
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
        "timeout_rate": float(np.mean(timeouts) * 100),
        "min_reward": float(np.min(episode_rewards)),
        "max_reward": float(np.max(episode_rewards)),
        "avg_steps_to_goal": float(np.mean(steps_to_goal)) if len(steps_to_goal) > 0 else 0,
        "std_steps_to_goal": float(np.std(steps_to_goal)) if len(steps_to_goal) > 0 else 0,
    }
    
    if verbose:
        print(f"\n{algo_name} Test Summary:")
        print(f"  Average Reward:   {stats['avg_reward']:6.1f} ± {stats['std_reward']:.1f}")
        print(f"  Average Length:   {stats['avg_length']:6.1f} ± {stats['std_length']:.1f} steps")
        print(f"  Success Rate:     {stats['success_rate']:5.1f}%")
        print(f"  Collision Rate:   {stats['collision_rate']:5.1f}%")
        print(f"  Timeout Rate:     {stats['timeout_rate']:5.1f}%")
        if stats['avg_steps_to_goal'] > 0:
            print(f"  Avg Steps-to-Goal: {stats['avg_steps_to_goal']:5.1f} ± {stats['std_steps_to_goal']:.1f} (successes only)")
        print(f"  Reward Range:     [{stats['min_reward']:.1f}, {stats['max_reward']:.1f}]")
    
    return stats


def compare_all_algorithms(n_episodes=20, model_dir="models"):
    """
    Test all trained algorithms and generate comparison
    """
    
    print("\n" + "="*70)
    print("TESTING ALL ALGORITHMS (All using continuous actions)")
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
        print("  python train.py --algo PPO --timesteps 5000000 --n-envs 4")
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
        plot_outcomes(results, algorithms)
    
    return results


def generate_comparison_plots(results, algorithms):
    """
    Generate comparison plots for paper
    """
    
    print("\n" + "="*70)
    print("GENERATING PLOTS")
    print("="*70)
    
    # Create graphs directory if it doesn't exist
    os.makedirs("graphs", exist_ok=True)
    
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
    axes[1].axhline(y=50, color='orange', linestyle='--', linewidth=0.8, alpha=0.5, label='50% baseline')
    axes[1].legend()
    
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
    save_path = "graphs/algorithm_comparison.png"
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    print(f"Saved: {save_path}")
    plt.close()

    # Generate normalized comparison plot
    fig2, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(algos_with_results))
    width = 0.25
    
    # Normalize metrics to 0-100 scale
    normalized_rewards = []
    normalized_success = []
    normalized_lengths = []
    
    for algo in algos_with_results:
        # Reward: scale from [-100, 200] to [0, 100]
        norm_reward = (results[algo]['avg_reward'] + 100) / 3.0
        normalized_rewards.append(max(0, min(100, norm_reward)))
        
        # Success rate: already in [0, 100]
        normalized_success.append(results[algo]['success_rate'])
        
        # Length: invert (lower is better), scale from [0, 400] to [100, 0]
        norm_length = 100 - (results[algo]['avg_length'] / 4.0)
        normalized_lengths.append(max(0, min(100, norm_length)))
    
    ax.bar(x - width, normalized_rewards, width, label='Avg Reward (normalized)', alpha=0.8, color='#3498db')
    ax.bar(x, normalized_success, width, label='Success Rate (%)', alpha=0.8, color='#2ecc71')
    ax.bar(x + width, normalized_lengths, width, label='Efficiency (inverted length)', alpha=0.8, color='#e74c3c')
    
    ax.set_ylabel('Score (0-100)', fontsize=12)
    ax.set_title('Algorithm Performance Comparison (Normalized)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(algos_with_results)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 105])
    
    plt.tight_layout()
    save_path = 'graphs/normalized_comparison.png'
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()

def watch_trained_agent(algo_name, model_path, n_episodes=5):
    """
    Watch a trained agent perform (with rendering)
    """
    
    print(f"\n{'='*70}")
    print(f"WATCHING {algo_name} AGENT (Continuous Actions)")
    print(f"{'='*70}\n")
    print("Close the window to continue to next episode...")
    
    test_single_algorithm(
        algo_name,
        model_path,
        n_episodes=n_episodes,
        render=True,
        verbose=True
    )

def plot_outcomes(results, algorithms, save_path="graphs/outcome_rates.png"):
    """
    Plot and save Success / Collision / Timeout rates for each algorithm
    """

    # Create graphs directory if it doesn't exist
    os.makedirs("graphs", exist_ok=True)

    algos_with_results = [a for a in algorithms if a in results]
    if not algos_with_results:
        print("No outcome data to plot!")
        return

    success_rates = [results[a]["success_rate"] for a in algos_with_results]
    collision_rates = [results[a]["collision_rate"] for a in algos_with_results]
    timeout_rates = [results[a]["timeout_rate"] for a in algos_with_results]

    x = np.arange(len(algos_with_results))
    width = 0.25

    plt.figure(figsize=(10, 6))

    plt.bar(x - width, success_rates, width, label="Success", color="#239653")
    plt.bar(x, collision_rates, width, label="Collision", color="#89261b")
    plt.bar(x + width, timeout_rates, width, label="Timeout", color="#2a61eb")

    plt.ylabel("Rate (%)", fontsize=12)
    plt.title("Episode Outcomes by Algorithm", fontsize=14, fontweight="bold")
    plt.xticks(x, algos_with_results)
    plt.ylim(0, 100)
    plt.grid(True, alpha=0.3, axis="y")
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    print(f"Saved: {save_path}")
    plt.close()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test trained RL models (all use continuous actions)')
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
        print("\nTesting all algorithms (continuous actions)")
        print("   This will test PPO, SAC, and TD3\n")
        results = compare_all_algorithms(
            n_episodes=args.episodes,
            model_dir=args.model_dir
        )
    
    elif args.mode == 'single':
        # Test single algorithm
        model_path = f"{args.model_dir}/{args.algo}/final_model"
        if not os.path.exists(f"{model_path}.zip"):
            print(f"Model not found: {model_path}.zip")
            print(f"Train first: python train.py --algo {args.algo} --timesteps 5000000")
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
            print(f"Train first: python train.py --algo {args.algo} --timesteps 5000000")
        else:
            watch_trained_agent(
                args.algo,
                model_path,
                n_episodes=args.episodes
            )
    
    print("\nDone!")