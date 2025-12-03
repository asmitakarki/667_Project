"""
Train RL agents on PyBullet 3D environment
"""

from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from grid_bounded_env import GridBoundedEnv as PyBulletPathfindingEnv
import numpy as np
import os

def make_env(grid_size=20, map_type='random', rank=0, **map_kwargs):
    """Create environment"""
    def _init():
        env = PyBulletPathfindingEnv(
            grid_size=grid_size,
            map_type=map_type,
            render_mode=None,  # No rendering during training
            **map_kwargs
        )
        env = Monitor(env)
        return env
    return _init

def train_pybullet(algorithm='PPO', map_type='random', timesteps=5000):
    """
    Train an agent on PyBullet environment
    
    Args:
        algorithm: 'PPO', 'SAC', or 'TD3'
        map_type: 'random', 'maze', or 'grid'
        timesteps: Training timesteps
    """
    print("\n" + "="*60)
    print(f"Training {algorithm} on PyBullet {map_type} environment")
    print("="*60 + "\n")
    
    # Create directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('logs/pybullet', exist_ok=True)
    os.makedirs('checkpoints/pybullet', exist_ok=True)
    
    # Create vectorized environment
    n_envs = 4  # Parallel environments for faster training
    env = DummyVecEnv([
        make_env(grid_size=20, map_type=map_type, rank=i, num_obstacles=5)
        for i in range(n_envs)
    ])
    
    # Eval environment
    eval_env = DummyVecEnv([make_env(grid_size=20, map_type=map_type, num_obstacles=5)])
    
    # Callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f'./models/pybullet_best_{map_type}/',
        log_path='./logs/pybullet/',
        eval_freq=5000,
        deterministic=True,
        render=False,
        verbose=1
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path='./checkpoints/pybullet/',
        name_prefix=f'{algorithm.lower()}_{map_type}',
        verbose=1
    )
    
    # Create model
    if algorithm == 'PPO':
        model = PPO(
            'MlpPolicy',
            env,
            verbose=1,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            tensorboard_log=None
        )
    elif algorithm == 'SAC':
        # SAC works well with continuous control
        model = SAC(
            'MlpPolicy',
            env,
            verbose=1,
            learning_rate=3e-4,
            buffer_size=100000,
            learning_starts=1000,
            batch_size=256,
            tau=0.005,
            gamma=0.99,
            tensorboard_log=None
        )
    elif algorithm == 'TD3':
        model = TD3(
            'MlpPolicy',
            env,
            verbose=1,
            learning_rate=3e-4,
            buffer_size=100000,
            learning_starts=1000,
            batch_size=256,
            tau=0.005,
            gamma=0.99,
            tensorboard_log=None
        )
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    
    # Train
    print(f"\nStarting training for {timesteps} timesteps...")
    print(f"Using {n_envs} parallel environments")
    print("This will take a while with PyBullet physics...\n")
    
    model.learn(
        total_timesteps=timesteps,
        callback=[eval_callback, checkpoint_callback],
        progress_bar=True
    )
    
    # Save
    model_path = f"models/pybullet_{algorithm}_{map_type}_final"
    model.save(model_path)
    print(f"\nModel saved: {model_path}")
    
    return model

def evaluate_pybullet(model_path, map_type='random', episodes=5):
    """Evaluate trained model"""
    print("\n" + "="*60)
    print("Evaluating PyBullet Agent")
    print("="*60 + "\n")
    
    # Load model
    try:
        model = PPO.load(model_path)
        print("Loaded PPO model")
    except:
        try:
            model = SAC.load(model_path)
            print("Loaded SAC model")
        except:
            try:
                model = TD3.load(model_path)
                print("Loaded TD3 model")
            except Exception as e:
                print(f"Error loading model: {e}")
                return
    
    # Create environment with rendering
    env = PyBulletPathfindingEnv(
        grid_size=20,
        map_type=map_type,
        render_mode='human',
        num_obstacles=5
    )
    
    successes = 0
    episode_lengths = []
    episode_rewards = []
    
    for ep in range(episodes):
        obs, info = env.reset()
        steps = 0
        episode_reward = 0
        
        print(f"\nEpisode {ep + 1}:")
        
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
        
        # Small pause between episodes
        import time
        time.sleep(1)
    
    env.close()
    
    print("\n" + "="*60)
    print("Results:")
    print(f"  Success rate: {successes}/{episodes} ({100*successes/episodes:.1f}%)")
    if episode_lengths:
        print(f"  Average steps: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")
        print(f"  Average reward: {np.mean(episode_rewards):.1f} ± {np.std(episode_rewards):.1f}")
    print("="*60)

def quick_demo():
    """Quick demo with random agent"""
    print("\n" + "="*60)
    print("PyBullet Quick Demo (Random Agent)")
    print("="*60 + "\n")
    
    env = PyBulletPathfindingEnv(
        grid_size=20,
        map_type='random',
        render_mode='human',
        num_obstacles=5
    )
    
    obs, info = env.reset()
    
    print("Watch the robot (blue sphere) move randomly in PyBullet...")
    print("It's trying to reach the green goal while avoiding gray obstacles.")
    print("The red cylinder shows which direction the robot is facing.\n")
    print("Close the PyBullet window when done.\n")
    
    for i in range(300):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated:
            print(f"\n✓ Goal reached at step {i}!")
            obs, info = env.reset()
        
        if truncated:
            print(f"\n✗ Episode timeout at step {i}")
            obs, info = env.reset()
    
    env.close()

if __name__ == "__main__":
    import sys
    
    print("\nPyBullet Pathfinding Training")
    print("="*60)
    print("\nOptions:")
    print("1. Quick Demo (random agent)")
    print("2. Train PPO on random map")
    print("3. Train PPO on maze")
    print("4. Evaluate trained model")
    
    choice = input("\nChoice (1-4): ")
    
    if choice == '1':
        quick_demo()
    
    elif choice == '2':
        print("\nTraining PPO on random obstacles...")
        print("This will take 15-20 minutes.")
        model = train_pybullet('PPO', 'random', timesteps=5000)
        print("\nTraining complete! Testing trained model...")
        evaluate_pybullet('models/pybullet_PPO_random_final', 'random', episodes=3)
    
    elif choice == '3':
        print("\nTraining PPO on maze...")
        print("This will take 20-30 minutes.")
        model = train_pybullet('PPO', 'maze', timesteps=5000)
        print("\nTraining complete! Testing trained model...")
        evaluate_pybullet('models/pybullet_PPO_maze_final', 'maze', episodes=3)
    
    elif choice == '4':
        model_path = input("\nEnter model path (e.g., models/pybullet_PPO_random_final): ")
        map_type = input("Enter map type (random/maze/grid): ")
        
        if not os.path.exists(model_path + ".zip"):
            print(f"\nError: Model not found at {model_path}")
            print("Make sure to train a model first using options 2 or 3.")
        else:
            evaluate_pybullet(model_path, map_type, episodes=5)
    
    else:
        print("Invalid choice")