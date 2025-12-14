"""
OPTIMIZED training - fixes for slow FPS and poor performance
"""

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback

import time
import os
import torch

# for plotting
import glob
import pandas as pd

from robot_pov_env import RobotPOVEnv, RobotPOVContinuousEnv


def make_discrete_env(render_mode=None, log_dir=None, rank=0):
    """Env factory for PPO (discrete actions)."""
    def _init():
        env = RobotPOVEnv(
            grid_size=20,
            map_type="city",
            render_mode=render_mode,
            use_camera_obs=False,
            num_obstacles=4,
        )
        if log_dir is not None:
            os.makedirs(log_dir, exist_ok=True)
            return Monitor(env, filename=os.path.join(log_dir, f"monitor_{rank}.csv"))
        return Monitor(env)
    return _init


def make_continuous_env(render_mode=None, log_dir=None, rank=0):
    """Env factory for SAC/TD3 (continuous actions)."""
    def _init():
        env = RobotPOVContinuousEnv(
            grid_size=20,
            map_type="city",
            render_mode=render_mode,
            use_camera_obs=False,
            num_obstacles=4,
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

    plt.figure()
    plt.plot(steps, rewards)
    plt.plot(ma_x, ma)
    plt.xlabel("Environment steps (approx)")
    plt.ylabel("Episode reward")
    plt.title(f"{algo_name} Training Reward")
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()


def train_algorithm(algo_name, total_timesteps=200000, save_dir="models", 
                    render_training=False, n_envs=1):
    """
    Training with parallel environments
    """
    
    print(f"\n{'='*70}")
    print(f"Training {algo_name}")
    print(f"{'='*70}\n")
    
    # Create directories
    os.makedirs(f"{save_dir}/{algo_name}", exist_ok=True)
    os.makedirs(f"{save_dir}/{algo_name}/checkpoints", exist_ok=True)
    os.makedirs(f"logs/{algo_name}", exist_ok=True)

    render_mode = "human" if render_training else None
    log_dir = f"logs/{algo_name}/monitor"

    if n_envs > 1 and not render_training:
        if algo_name == "PPO":
            env = SubprocVecEnv([make_discrete_env(log_dir=log_dir, rank=i) for i in range(n_envs)])
        else:
            env = SubprocVecEnv([make_continuous_env(log_dir=log_dir, rank=i) for i in range(n_envs)])
    else:
        if algo_name == "PPO":
            env = DummyVecEnv([make_discrete_env(render_mode=render_mode, log_dir=log_dir, rank=0)])
        else:
            env = DummyVecEnv([make_continuous_env(render_mode=render_mode, log_dir=log_dir, rank=0)])

    start_time = time.time()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Number of environments: {n_envs}")
    
    if algo_name == "PPO":
        # OPTIMIZED PPO hyperparameters
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
            ent_coef=0.1,  # Encourage exploration. changed from 0.01 to 0.05
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
    # Create checkpoint callback
    '''
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,  # Save every 10k steps
        save_path=f"{save_dir}/{algo_name}/checkpoints/",
        name_prefix=f"{algo_name}_model",
        save_replay_buffer=True,  # For SAC/TD3
        save_vecnormalize=True,
    )
    '''
    checkpoint_callback = CheckpointCallback(
    save_freq=50000,
    save_path=f"{save_dir}/{algo_name}/checkpoints/",
    name_prefix=f"{algo_name}",
    save_replay_buffer=(algo_name in ["SAC", "TD3"]),
    save_vecnormalize=True,
    )
    """
    time_save_callback = EveryNTimesteps(
        n_steps=50000,
        callback=CheckpointCallback(
            save_freq=1,
            save_path=f"{save_dir}/{algo_name}/checkpoints/",
            name_prefix=f"{algo_name}_autosave",
            save_replay_buffer=True,
            save_vecnormalize=True,
        )
    )
    """
    # use checkpoint callback during learning
    print(f"Starting training for {total_timesteps} timesteps...")
    model.learn(
        total_timesteps=total_timesteps,
        progress_bar=True,
        log_interval=10,
        callback=[checkpoint_callback] # will save checkpoints during training every 50k steps
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
    
    parser = argparse.ArgumentParser(description='OPTIMIZED RL training')
    parser.add_argument('--mode', type=str, default='train',
                       choices=['train', 'test'],
                       help='Mode: train or test')
    parser.add_argument('--algo', type=str, default='PPO',
                       choices=['PPO', 'SAC', 'TD3'],
                       help='Algorithm to train/test')
    parser.add_argument('--timesteps', type=int, default=200000,
                       help='Total training timesteps')
    parser.add_argument('--n-envs', type=int, default=4,
                       help='Number of parallel environments (speedup)')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        train_algorithm(
            args.algo, 
            total_timesteps=args.timesteps,
            n_envs=args.n_envs
        )
    
    print("\nDone!")

    