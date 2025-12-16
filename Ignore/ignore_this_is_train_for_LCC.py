"""
OPTIMIZED training - CONTINUOUS actions for all algorithms (PPO, SAC, TD3)
With improved reward shaping and hyperparameters
"""

import os
import sys
import types
# Create figure module with Figure class
figure_module = types.ModuleType('figure')
figure_module.Figure = type('Figure', (), {})
# Create matplotlib mock to avoid import errors
matplotlib_mock = types.ModuleType('matplotlib')
matplotlib_mock.use = lambda x: None
matplotlib_mock.figure = figure_module
# Register all matplotlib modules
sys.modules['matplotlib'] = matplotlib_mock
sys.modules['matplotlib.pyplot'] = types.ModuleType('pyplot')
sys.modules['matplotlib.figure'] = figure_module
import gymnasium as gym
import numpy as np
#import matplotlib.pyplot as plt
from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.callbacks import EveryNTimesteps
import time
import torch
from robot_pov_env import RobotPOVContinuousEnv
# for plotting
import glob
import pandas as pd

from stable_baselines3.common.callbacks import BaseCallback


class CurriculumCallback(BaseCallback):
    """
    REAL Curriculum Learning: Progressively increases difficulty during training
    """
    def __init__(self, schedule, verbose=0):
        """
        Args:
            schedule: list of (timestep, num_obstacles) tuples
                      e.g., [(0, 2), (1_000_000, 4), (3_000_000, 6)]
        """
        super().__init__(verbose)
        self.schedule = schedule
        self.idx = 0
    
    def _on_step(self) -> bool:
        t = self.num_timesteps
        while self.idx < len(self.schedule) and t >= self.schedule[self.idx][0]:
            nobs = self.schedule[self.idx][1]
            self.training_env.env_method("set_num_obstacles", nobs)
            if self.verbose > 0:
                print(f"\n{'='*70}")
                print(f"[Curriculum] Timesteps: {t:,} → Obstacles: {nobs}")
                print(f"{'='*70}\n")
            self.idx += 1
        return True


def make_continuous_env(render_mode=None, log_dir=None, rank=0, num_obstacles=6):
    """Env factory for ALL algorithms (PPO, SAC, TD3) - all use continuous actions now."""
    def _init():
        env = RobotPOVContinuousEnv(
            grid_size=20,
            map_type="city",
            render_mode=render_mode,
            use_camera_obs=False,
            num_obstacles=num_obstacles,
        )
        if log_dir is not None:
            os.makedirs(log_dir, exist_ok=True)
            return Monitor(env, filename=os.path.join(log_dir, f"monitor_{rank}.csv"))
        return Monitor(env)
    return _init

def plot_training_curve(log_dir, algo_name, window=50, save_path=None):
    files = glob.glob(os.path.join(log_dir, "monitor_*.csv"))
    if len(files) == 0:
        print("No monitor files found:", log_dir)
        return

    dfs = []
    for f in files:
        # Monitor CSV has a comment header line starting with '#'
        df = pd.read_csv(f, comment="#")
        dfs.append(df)

    data = pd.concat(dfs, ignore_index=True)

    # data columns: r (episode reward), l (episode length), t (time)
    data = data.sort_values("t")

    rewards = data["r"].to_numpy()
    steps = np.cumsum(data["l"].to_numpy())  # approximate x-axis as env steps

    # moving average
    if len(rewards) >= window:
        ma = np.convolve(rewards, np.ones(window)/window, mode="valid")
        ma_x = steps[window-1:]
    else:
        ma = rewards
        ma_x = steps

    plt.figure(figsize=(10, 6))
    plt.plot(steps, rewards, alpha=0.3, label='Episode Reward', linewidth=0.5)
    plt.plot(ma_x, ma, linewidth=2, label=f'{window}-episode Moving Average', color='red')
    plt.xlabel("Environment steps (approx)")
    plt.ylabel("Episode reward")
    plt.title(f"{algo_name} Training Reward")
    plt.legend()
    plt.grid(True, alpha=0.3)
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved training curve to {save_path}")


def train_algorithm(algo_name, total_timesteps=5000000, save_dir="models", 
                    render_training=False, n_envs=4, use_curriculum=True):
    """
    Training with parallel environments - ALL algorithms use CONTINUOUS actions
    
    Key changes:
    - PPO now uses continuous actions (smooth control)
    - Increased training timesteps default to 5M
    - Better hyperparameters for all algorithms
    - Improved reward shaping in environment
    - REAL CURRICULUM LEARNING: obstacles increase during training
    """
    
    print(f"\n{'='*70}")
    print(f"Training {algo_name} with CONTINUOUS actions")
    print(f"{'='*70}\n")
    
    # REAL CURRICULUM: Start with obstacle count, will increase during training
    initial_obstacles = 2 if use_curriculum else 6
    
    if use_curriculum:
        print(f"REAL Curriculum Learning Enabled:")
        print(f"   0 steps      → 2 obstacles (EASY)")
        print(f"   1M steps     → 4 obstacles (MEDIUM)")
        print(f"   3M steps     → 6 obstacles (HARD)")
        print(f"   Difficulty increases DURING training!\n")
    else:
        print(f"Fixed difficulty: {initial_obstacles} obstacles\n")
    
    # Create directories
    os.makedirs(f"{save_dir}/{algo_name}", exist_ok=True)
    os.makedirs(f"{save_dir}/{algo_name}/checkpoints", exist_ok=True)
    os.makedirs(f"logs/{algo_name}", exist_ok=True)

    render_mode = "human" if render_training else None
    log_dir = f"logs/{algo_name}/monitor"

    # All algorithms now use continuous env with initial obstacle count
    if n_envs > 1 and not render_training:
        env = SubprocVecEnv([make_continuous_env(log_dir=log_dir, rank=i, num_obstacles=initial_obstacles) for i in range(n_envs)])
    else:
        env = DummyVecEnv([make_continuous_env(render_mode=render_mode, log_dir=log_dir, rank=0, num_obstacles=initial_obstacles)])

    start_time = time.time()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Number of environments: {n_envs}")
    print(f"Action space: {env.action_space}")
    
    if algo_name == "PPO":
        # PPO for CONTINUOUS actions
        # Lower entropy coefficient for continuous control
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=3e-4,
            n_steps=2048 // n_envs,  # Adjust for parallel envs
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,  # CHANGED: Lower for continuous (was 0.1 for discrete)
            max_grad_norm=0.5,  # Added gradient clipping for stability
            verbose=1,
            device=device,
            tensorboard_log=f"logs/{algo_name}"
        )
    
    elif algo_name == "SAC":
        # SAC with IMPROVED hyperparameters for better exploration
        model = SAC(
            "MlpPolicy",
            env,
            learning_rate=3e-4,
            buffer_size=500000,  # INCREASED from 200k to 500k
            learning_starts=5000,  # INCREASED from 1000 for better exploration
            batch_size=512,  # INCREASED from 256 to 512 for larger batches
            tau=0.005,
            gamma=0.99,
            train_freq=1,
            gradient_steps=1,
            ent_coef='auto_0.5',  # Initial entropy target (was just 'auto')
            verbose=1,
            device=device,
            tensorboard_log=f"logs/{algo_name}"
        )
    
    elif algo_name == "TD3":
        # TD3 with improved hyperparameters
        model = TD3(
            "MlpPolicy",
            env,
            learning_rate=3e-4,
            buffer_size=500000,  # INCREASED from 200k to 500k
            learning_starts=5000,  # INCREASED from 1000
            batch_size=512,  # INCREASED from 256 to 512
            tau=0.005,
            gamma=0.99,
            train_freq=1,
            gradient_steps=1,
            policy_delay=2,
            verbose=1,
            device=device,
            tensorboard_log=f"logs/{algo_name}"
        )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=100000,  # Save every 100k steps (was 50k)
        save_path=f"{save_dir}/{algo_name}/checkpoints/",
        name_prefix=f"{algo_name}",
        save_replay_buffer=(algo_name in ["SAC", "TD3"]),
        save_vecnormalize=True,
    )
    
    # REAL Curriculum callback - obstacles increase during training
    callbacks = [checkpoint_callback]
    
    if use_curriculum:
        curriculum_callback = CurriculumCallback(
            schedule=[
                (0, 2),           # Start easy: 2 obstacles
                (1_000_000, 4),   # Medium: 4 obstacles at 1M steps
                (3_000_000, 6),   # Hard: 6 obstacles at 3M steps
            ],
            verbose=1
        )
        callbacks.append(curriculum_callback)
    
    print(f"Starting training for {total_timesteps:,} timesteps...")
    print(f"Expected time: ~{total_timesteps / (500 * 60):.1f} minutes at 500 steps/sec\n")
    
    model.learn(
        total_timesteps=total_timesteps,
        progress_bar=True,
        log_interval=10,
        callback=callbacks  # Now includes curriculum callback!
    )

    training_time = time.time() - start_time
    
    model.save(f"{save_dir}/{algo_name}/final_model")
    
    print(f"\n{algo_name} Training Complete!")
    print(f"Time: {training_time/60:.1f} minutes")
    print(f"Timesteps/sec: {total_timesteps/training_time:.1f}")
    print(f"Model saved to: {save_dir}/{algo_name}/final_model.zip\n")

    plot_training_curve(
        log_dir=f"logs/{algo_name}/monitor",
        algo_name=algo_name,
        window=50,
        save_path=f"logs/{algo_name}/reward_curve.png"
    )
    
    env.close()
    
    return model, training_time


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='RL training with CONTINUOUS actions for all algorithms')
    parser.add_argument('--mode', type=str, default='train',
                       choices=['train', 'test'],
                       help='Mode: train or test')
    parser.add_argument('--algo', type=str, default='PPO',
                       choices=['PPO', 'SAC', 'TD3'],
                       help='Algorithm to train/test')
    parser.add_argument('--timesteps', type=int, default=5000000,
                       help='Total training timesteps (default: 5M)')
    parser.add_argument('--n-envs', type=int, default=4,
                       help='Number of parallel environments (speedup)')
    parser.add_argument('--no-curriculum', action='store_true',
                       help='Disable curriculum learning (always use 6 obstacles)')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        print(f"\nTraining {args.algo} with CONTINUOUS actions")
        print(f"   Timesteps: {args.timesteps:,}")
        print(f"   Parallel envs: {args.n_envs}")
        print(f"   Curriculum learning: {'Disabled' if args.no_curriculum else 'Enabled'}")
        print(f"   Key improvements:")
        print(f"   - PPO uses continuous actions (smooth control)")
        print(f"   - Improved reward shaping")
        print(f"   - Better hyperparameters")
        print(f"   - Longer training by default")
        if not args.no_curriculum:
            print(f"   - Progressive difficulty (2→4→6 obstacles)\n")
        else:
            print(f"   - Fixed 6 obstacles\n")
        
        train_algorithm(
            args.algo, 
            total_timesteps=args.timesteps,
            n_envs=args.n_envs,
            use_curriculum=not args.no_curriculum
        )

    print("\nDone!")