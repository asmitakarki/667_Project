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

from robot_pov_env import RobotPOVEnv


def make_env(render_mode=None):
    """Create and wrap environment"""
    def _init():
        env = RobotPOVEnv(
            grid_size=20,
            map_type="city",
            render_mode=render_mode,
            use_camera_obs=False,  # Position-based observations
            num_obstacles=4
        )
        env = Monitor(env)  # Track episode stats
        return env
    return _init


def train_algorithm(algo_name, total_timesteps=200000, save_dir="models"):
    """
    Train a single algorithm
    
    Args:
        algo_name: 'PPO', 'SAC', or 'TD3'
        total_timesteps: Total training steps
        save_dir: Where to save models
    """
    
    print(f"\n{'='*70}")
    print(f"Training {algo_name}")
    print(f"{'='*70}\n")
    
    # Create directories
    os.makedirs(f"{save_dir}/{algo_name}", exist_ok=True)
    os.makedirs(f"logs/{algo_name}", exist_ok=True)
    
    # Create vectorized environment
    env = DummyVecEnv([make_env()])
    env = VecNormalize(env, norm_obs=True, norm_reward=True)
    
    # Create evaluation environment
    eval_env = DummyVecEnv([make_env()])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, training=False)
    
    # Callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"{save_dir}/{algo_name}/best",
        log_path=f"logs/{algo_name}",
        eval_freq=5000,
        deterministic=True,
        render=False,
        n_eval_episodes=10
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=f"{save_dir}/{algo_name}/checkpoints",
        name_prefix=f"{algo_name.lower()}_model"
    )
    
    # Initialize algorithm with tuned hyperparameters
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
            tensorboard_log=f"logs/{algo_name}"
        )
    
    elif algo_name == "TD3":
        # TD3 is an improved version of DDPG
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
            tensorboard_log=f"logs/{algo_name}"
        )
    
    else:
        raise ValueError(f"Unknown algorithm: {algo_name}")
    
    # Train
    print(f"Starting training for {total_timesteps} timesteps...")
    model.learn(
        total_timesteps=total_timesteps,
        callback=[eval_callback, checkpoint_callback],
        progress_bar=True
    )
    
    training_time = time.time() - start_time
    
    # Save final model
    model.save(f"{save_dir}/{algo_name}/final_model")
    env.save(f"{save_dir}/{algo_name}/vec_normalize.pkl")
    
    print(f"\n{algo_name} Training Complete!")
    print(f"Time: {training_time/60:.1f} minutes")
    print(f"Model saved to: {save_dir}/{algo_name}/\n")
    
    env.close()
    eval_env.close()
    
    return model, training_time


def test_algorithm(algo_name, model_path, n_episodes=20, render=True):
    """
    Test a trained algorithm
    
    Args:
        algo_name: 'PPO', 'SAC', or 'TD3'
        model_path: Path to saved model (without .zip extension)
        n_episodes: Number of test episodes
        render: Show visualization
    """
    
    print(f"\n{'='*70}")
    print(f"Testing {algo_name}")
    print(f"{'='*70}\n")
    
    # Load model
    if algo_name == "PPO":
        model = PPO.load(model_path)
    elif algo_name == "SAC":
        model = SAC.load(model_path)
    elif algo_name == "TD3":
        model = TD3.load(model_path)
    
    # Create test environment
    env = RobotPOVEnv(
        grid_size=20,
        map_type="city",
        render_mode="human" if render else None,
        use_camera_obs=False,
        num_obstacles=4
    )
    
    # Load normalization stats
    vec_norm_path = model_path.replace("final_model", "vec_normalize.pkl").replace("best_model", "vec_normalize.pkl")
    if os.path.exists(vec_norm_path):
        print(f"Loading normalization stats from {vec_norm_path}")
        env = DummyVecEnv([lambda: env])
        env = VecNormalize.load(vec_norm_path, env)
        env.training = False
        env.norm_reward = False
    
    # Test
    successes = 0
    total_rewards = []
    episode_lengths = []
    
    for ep in range(n_episodes):
        obs = env.reset()
        episode_reward = 0
        steps = 0
        
        for step in range(500):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            episode_reward += reward[0] if isinstance(reward, np.ndarray) else reward
            steps += 1
            
            if render:
                time.sleep(0.01)
            
            if done:
                if episode_reward > 50:  # Success threshold
                    successes += 1
                    print(f"Episode {ep+1}: ✓ SUCCESS in {steps} steps (Reward: {episode_reward:.1f})")
                else:
                    print(f"Episode {ep+1}: ✗ Failed (Reward: {episode_reward:.1f})")
                
                total_rewards.append(episode_reward)
                episode_lengths.append(steps)
                
                if render:
                    time.sleep(1)
                break
    
    # Results
    avg_reward = np.mean(total_rewards)
    avg_length = np.mean(episode_lengths)
    success_rate = successes / n_episodes * 100
    
    print(f"\n{algo_name} Test Results:")
    print(f"  Success Rate: {successes}/{n_episodes} ({success_rate:.1f}%)")
    print(f"  Avg Reward: {avg_reward:.2f} ± {np.std(total_rewards):.2f}")
    print(f"  Avg Length: {avg_length:.1f} ± {np.std(episode_lengths):.1f}")
    
    env.close()
    
    return {
        'success_rate': success_rate,
        'avg_reward': avg_reward,
        'avg_length': avg_length,
        'rewards': total_rewards,
        'lengths': episode_lengths
    }


def compare_all_algorithms(timesteps=200000):
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
        model, train_time = train_algorithm(algo, total_timesteps=timesteps)
        training_times[algo] = train_time
        
        # Quick test after training
        print(f"\nQuick test of {algo}...")
        test_results = test_algorithm(
            algo,
            f"models/{algo}/best/best_model",
            n_episodes=10,
            render=False
        )
        results[algo] = test_results
    
    # Print comparison
    print("\n" + "="*70)
    print("FINAL COMPARISON")
    print("="*70)
    print(f"\n{'Algorithm':<10} {'Success Rate':<15} {'Avg Reward':<15} {'Training Time':<15}")
    print("-" * 70)
    
    for algo in algorithms:
        print(f"{algo:<10} "
              f"{results[algo]['success_rate']:>6.1f}%        "
              f"{results[algo]['avg_reward']:>8.1f}        "
              f"{training_times[algo]/60:>6.1f} min")
    
    # Plot comparison
    plot_comparison(results, algorithms)
    
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
    
    args = parser.parse_args()
    
    if args.mode == 'compare':
        # Train and compare all algorithms
        compare_all_algorithms(timesteps=args.timesteps)
    
    elif args.mode == 'train':
        # Train single algorithm
        train_algorithm(args.algo, total_timesteps=args.timesteps)
    
    elif args.mode == 'test':
        # Test single algorithm
        test_algorithm(
            args.algo,
            f"models/{args.algo}/best/best_model",
            n_episodes=args.test_episodes,
            render=True
        )
    
    elif args.mode == 'interactive':
        # Interactive testing
        interactive_test()
    
    print("\nDone!")