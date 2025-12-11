"""
Compare PPO, DDPG, and SAC using Stable-Baselines3
Much cleaner and more reliable than custom implementations!
"""

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO, SAC, TD3  # TD3 is better than DDPG
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
import time
import os
import torch

from robot_pov_env import RobotPOVEnv
from robot_pov_env import RobotPOVContinuousEnv



def make_discrete_env(render_mode=None):
    """Env factory for PPO (discrete actions)."""
    def _init():
        env = RobotPOVEnv(
            grid_size=20,
            map_type="city",
            render_mode=render_mode,
            use_camera_obs=False,  # position-based observations
            num_obstacles=4,
        )
        return Monitor(env)
    return _init


def make_continuous_env(render_mode=None):
    """Env factory for SAC/TD3 (continuous actions)."""
    def _init():
        env = RobotPOVContinuousEnv(
            grid_size=20,
            map_type="city",
            render_mode=render_mode,
            use_camera_obs=False,  # still position-based for now
            num_obstacles=4,
        )
        return Monitor(env)
    return _init


def train_algorithm(algo_name, total_timesteps=200000, save_dir="models", render_training=False):
    """
    Train a single algorithm
    
    Args:
        algo_name: 'PPO', 'SAC', or 'TD3'
        total_timesteps: Total training steps
        save_dir: Where to save models
        render_training: Show robot's camera view during training (slower!)
    """
    
    print(f"\n{'='*70}")
    print(f"Training {algo_name}")
    print(f"{'='*70}\n")
    
    if render_training:
        print(" RENDERING ENABLED - Training will be 5-10x slower!")
        print("    Press Ctrl+C to stop early if needed\n")
    
    # Create directories
    os.makedirs(f"{save_dir}/{algo_name}", exist_ok=True)
    os.makedirs(f"logs/{algo_name}", exist_ok=True)
    
    # Create vectorized environment
    render_mode = "human" if render_training else None

    if algo_name == "PPO":
        # Discrete control env
        env = DummyVecEnv([make_discrete_env(render_mode=render_mode)])
    else:
        # Continuous control env for SAC / TD3
        env = DummyVecEnv([make_continuous_env(render_mode=render_mode)])
    
    # Initialize algorithm with tuned hyperparameters
    start_time = time.time()
    
    # Check for GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if device == "cpu":
        print("GPU not detected")
    
    # Initialize algorithm - NO CALLBACKS to avoid hanging
    start_time = time.time()
    
    if algo_name == "PPO":
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            verbose=1,
            device=device,
            tensorboard_log=f"logs/{algo_name}"
        )
    
    elif algo_name == "SAC":
        model = SAC(
            "MlpPolicy",
            env,
            learning_rate=3e-4,
            buffer_size=100000,
            learning_starts=1000,
            batch_size=256,
            tau=0.005,
            gamma=0.99,
            train_freq=1,
            gradient_steps=1,
            ent_coef='auto',
            verbose=1,
            device=device,
            tensorboard_log=f"logs/{algo_name}"
        )
    
    elif algo_name == "TD3":
        model = TD3(
            "MlpPolicy",
            env,
            learning_rate=3e-4,
            buffer_size=100000,
            learning_starts=1000,
            batch_size=256,
            tau=0.005,
            gamma=0.99,
            train_freq=1,
            gradient_steps=1,
            policy_delay=2,
            verbose=1,
            device=device,
            tensorboard_log=f"logs/{algo_name}"
        )
    
    else:
        raise ValueError(f"Unknown algorithm: {algo_name}")
    
    # Train WITHOUT callbacks - no hanging!
    print(f"Starting training for {total_timesteps} timesteps...")
    print(f"No evaluation during training (prevents freezing)\n")
    model.learn(
        total_timesteps=total_timesteps,
        progress_bar=True,
        log_interval=10  # Print stats every 10 rollouts
    )
    
    training_time = time.time() - start_time
    
    # Save final model (no VecNormalize to save)
    model.save(f"{save_dir}/{algo_name}/final_model")
    
    print(f"\n{algo_name} Training Complete!")
    print(f"Time: {training_time/60:.1f} minutes")
    print(f"Model saved to: {save_dir}/{algo_name}/final_model.zip\n")
    
    env.close()
    
    return model, training_time

def test_algorithm(algo_name, model_path, n_episodes=5, render=True):
    """
    Simple test for a trained model.
    Uses the discrete env for PPO and the continuous env for SAC/TD3.
    Returns simple stats for comparison.
    """

    print("\n" + "="*70)
    print(f"Testing {algo_name}")
    print("="*70 + "\n")

    # ---- Load model ----
    if algo_name == "PPO":
        model = PPO.load(model_path)
    elif algo_name == "SAC":
        model = SAC.load(model_path)
    elif algo_name == "TD3":
        model = TD3.load(model_path)
    else:
        raise ValueError(f"Unknown algo in test_algorithm: {algo_name}")

    # ---- Pick environment class ----
    EnvClass = RobotPOVEnv if algo_name == "PPO" else RobotPOVContinuousEnv

    env = EnvClass(
        grid_size=20,
        map_type="city",
        render_mode="human" if render else None,
        use_camera_obs=False,
        num_obstacles=4,
    )

    episode_rewards = []
    episode_lengths = []

    for ep in range(n_episodes):
        obs, info = env.reset()   # Gymnasium: (obs, info)
        done = False
        truncated = False
        total_reward = 0.0

        for step in range(500):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)

            total_reward += float(reward)

            if render:
                import time
                time.sleep(0.01)

            if done or truncated:
                print(f"Episode {ep+1}: finished in {step+1} steps, reward={total_reward:.1f}")
                episode_rewards.append(total_reward)
                episode_lengths.append(step + 1)
                break

    env.close()

    # Basic stats
    avg_rew = float(np.mean(episode_rewards)) if episode_rewards else 0.0
    std_rew = float(np.std(episode_rewards)) if episode_rewards else 0.0
    avg_len = float(np.mean(episode_lengths)) if episode_lengths else 0.0
    std_len = float(np.std(episode_lengths)) if episode_lengths else 0.0

    print(f"\n{algo_name} test summary:")
    print(f"  Avg reward: {avg_rew:.2f} ± {std_rew:.2f}")
    print(f"  Avg length: {avg_len:.1f} ± {std_len:.1f}\n")

    return {
        "avg_reward": avg_rew,
        "std_reward": std_rew,
        "avg_length": avg_len,
        "std_length": std_len,
    }



def compare_all_algorithms(timesteps=200000, render_training=False):
    """
    Train and compare PPO, SAC, and TD3
    """
    
    print("\n" + "="*70)
    print("COMPARING: PPO vs SAC vs TD3")
    print("="*70)
    
    algorithms = ["PPO", "SAC", "TD3"]
    results = {}
    training_times = {}
    
    # Train all algorithms
    for algo in algorithms:
        model, train_time = train_algorithm(
            algo,
            total_timesteps=timesteps,
            render_training=render_training,
        )
        training_times[algo] = train_time
        
        # Test using the saved final model
        print(f"\nQuick test of {algo}...")
        model_path = f"models/{algo}/final_model"
        test_stats = test_algorithm(
            algo,
            model_path,
            n_episodes=5,
            render=False,
        )
        results[algo] = test_stats
    
    # Print comparison
    print("\n" + "="*70)
    print("FINAL COMPARISON")
    print("="*70)
    print(f"\n{'Algorithm':<10} {'Avg Reward':<15} {'Avg Length':<15} {'Train Time':<15}")
    print("-" * 70)
    
    for algo in algorithms:
        print(f"{algo:<10} "
              f"{results[algo]['avg_reward']:>8.1f}        "
              f"{results[algo]['avg_length']:>8.1f}        "
              f"{training_times[algo]/60:>6.1f} min")
    
    # Optional: update plot_comparison to use avg_reward/avg_length
    # or keep it as-is and map appropriately.
    
    return results

def plot_comparison(results, algorithms):
    """Plot comparison of all algorithms"""
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Success rates
    success_rates = [results[algo]['success_rate'] for algo in algorithms]
    colors = ['#2ecc71', '#3498db', '#e74c3c']
    axes[0].bar(algorithms, success_rates, color=colors)
    axes[0].set_ylabel('Success Rate (%)')
    axes[0].set_title('Success Rate Comparison')
    axes[0].set_ylim([0, 100])
    axes[0].grid(True, alpha=0.3)
    
    # Average rewards
    avg_rewards = [results[algo]['avg_reward'] for algo in algorithms]
    axes[1].bar(algorithms, avg_rewards, color=colors)
    axes[1].set_ylabel('Average Reward')
    axes[1].set_title('Average Reward Comparison')
    axes[1].grid(True, alpha=0.3)
    
    # Episode lengths
    avg_lengths = [results[algo]['avg_length'] for algo in algorithms]
    axes[2].bar(algorithms, avg_lengths, color=colors)
    axes[2].set_ylabel('Average Steps')
    axes[2].set_title('Average Episode Length')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('algorithm_comparison.png', dpi=150, bbox_inches='tight')
    print(f"\nComparison plot saved: algorithm_comparison.png")


def interactive_test():
    """Let user choose which algorithm to test"""
    
    print("\n" + "="*70)
    print("INTERACTIVE TESTING")
    print("="*70)
    
    algorithms = ["PPO", "SAC", "TD3"]
    
    print("\nAvailable trained models:")
    for i, algo in enumerate(algorithms, 1):
        model_path = f"models/{algo}/best/best_model.zip"
        if os.path.exists(model_path):
            print(f"  {i}. {algo} ✓")
        else:
            print(f"  {i}. {algo} ✗ (not trained)")
    
    print("\nWhich algorithm would you like to test?")
    choice = input("Enter number (1-3) or algorithm name: ").strip()
    
    if choice.isdigit() and 1 <= int(choice) <= 3:
        algo = algorithms[int(choice) - 1]
    elif choice.upper() in algorithms:
        algo = choice.upper()
    else:
        print("Invalid choice, testing PPO by default")
        algo = "PPO"
    
    print(f"\nTesting {algo}...")
    test_algorithm(
        algo,
        f"models/{algo}/best/best_model",
        n_episodes=10,
        render=True
    )


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train and compare RL algorithms')
    parser.add_argument('--mode', type=str, default='compare',
                       choices=['compare', 'train', 'test', 'interactive'],
                       help='Mode: compare all, train single, test single, or interactive test')
    parser.add_argument('--algo', type=str, default='PPO',
                       choices=['PPO', 'SAC', 'TD3'],
                       help='Algorithm to train/test (for train/test modes)')
    parser.add_argument('--timesteps', type=int, default=200000,
                       help='Total training timesteps')
    parser.add_argument('--test-episodes', type=int, default=20,
                       help='Number of test episodes')
    parser.add_argument('--render', action='store_true',
                       help='Show robot camera view during training (5-10x slower!)')
    
    args = parser.parse_args()
    
    if args.mode == 'compare':
        # Train and compare all algorithms
        compare_all_algorithms(timesteps=args.timesteps, render_training=args.render)
    
    elif args.mode == 'train':
        # Train single algorithm
        train_algorithm(args.algo, total_timesteps=args.timesteps, render_training=args.render)
    

    elif args.mode == 'test':
        model_path = f"models/{args.algo}/final_model"
        test_algorithm(
            args.algo,
            model_path,
            n_episodes=args.test_episodes,
            render=True,
        )
    
    elif args.mode == 'interactive':
        # Interactive testing
        interactive_test()
    
    print("\nDone!")