"""
Improved training script with better reward shaping and curriculum learning
"""

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from enhanced_pathfinding_env import EnhancedPathfindingEnv
import numpy as np

def train_with_curriculum(map_type='maze', total_timesteps=300000):
    """
    Train with curriculum learning - start easy, get harder
    """
    print("\n" + "="*60)
    print(f"CURRICULUM TRAINING on {map_type}")
    print("="*60 + "\n")
    
    # Stage 1: Train on random (easier)
    print("\n### STAGE 1: Training on Random Map (Easy) ###")
    def make_env_random():
        env = EnhancedPathfindingEnv(grid_size=20, map_type='random', num_obstacles=3)
        return Monitor(env)
    
    env = DummyVecEnv([make_env_random])
    
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
        ent_coef=0.01,  # Encourage exploration
        tensorboard_log='./tensorboard/'
    )
    
    model.learn(total_timesteps=total_timesteps // 3)
    
    # Stage 2: Transfer to target map type
    print(f"\n### STAGE 2: Fine-tuning on {map_type.upper()} Map ###")
    def make_env_target():
        if map_type == 'maze':
            env = EnhancedPathfindingEnv(grid_size=20, map_type='maze', cell_size=3)  # Bigger cells = easier
        elif map_type == 'spiral':
            env = EnhancedPathfindingEnv(grid_size=20, map_type='spiral', num_spirals=2)
        elif map_type == 'grid':
            env = EnhancedPathfindingEnv(grid_size=20, map_type='grid', spacing=4)
        else:
            env = EnhancedPathfindingEnv(grid_size=20, map_type=map_type)
        return Monitor(env)
    
    env = DummyVecEnv([make_env_target])
    model.set_env(env)
    
    model.learn(total_timesteps=total_timesteps // 3)
    
    # Stage 3: Full difficulty
    print(f"\n### STAGE 3: Final training on {map_type.upper()} (Full Difficulty) ###")
    def make_env_hard():
        if map_type == 'maze':
            env = EnhancedPathfindingEnv(grid_size=20, map_type='maze', cell_size=2)  # Normal difficulty
        elif map_type == 'spiral':
            env = EnhancedPathfindingEnv(grid_size=20, map_type='spiral', num_spirals=3)
        elif map_type == 'grid':
            env = EnhancedPathfindingEnv(grid_size=20, map_type='grid', spacing=3)
        else:
            env = EnhancedPathfindingEnv(grid_size=20, map_type=map_type)
        return Monitor(env)
    
    env = DummyVecEnv([make_env_hard])
    model.set_env(env)
    
    model.learn(total_timesteps=total_timesteps // 3)
    
    # Save
    model.save(f"models/PPO_{map_type}_curriculum")
    print(f"\nModel saved as: models/PPO_{map_type}_curriculum")
    
    return model


def train_longer(map_type='maze', total_timesteps=500000):
    """
    Train for much longer with better hyperparameters
    """
    print("\n" + "="*60)
    print(f"EXTENDED TRAINING on {map_type}")
    print("="*60 + "\n")
    
    def make_env():
        env = EnhancedPathfindingEnv(grid_size=20, map_type=map_type)
        return Monitor(env)
    
    env = DummyVecEnv([make_env])
    
    model = PPO(
        'MlpPolicy',
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=4096,  # Increased
        batch_size=128,  # Increased
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,  # Encourage exploration
        tensorboard_log='./tensorboard/'
    )
    
    # Callbacks
    eval_env = DummyVecEnv([make_env])
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f'./models/best_{map_type}/',
        log_path='./logs/',
        eval_freq=10000,
        deterministic=True,
        render=False
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=25000,
        save_path='./checkpoints/',
        name_prefix=f'ppo_{map_type}'
    )
    
    model.learn(
        total_timesteps=total_timesteps,
        callback=[eval_callback, checkpoint_callback]
    )
    
    model.save(f"models/PPO_{map_type}_long")
    print(f"\nModel saved as: models/PPO_{map_type}_long")
    
    return model


def quick_test(model, map_type='maze', episodes=5):
    """Quick test to see if model learned anything"""
    env = EnhancedPathfindingEnv(grid_size=20, map_type=map_type, render_mode='human')
    
    successes = 0
    for ep in range(episodes):
        obs, info = env.reset()
        steps = 0
        moves = set()
        
        for _ in range(200):
            action, _ = model.predict(obs, deterministic=True)
            if isinstance(action, np.ndarray):
                action = int(action.item())
            
            # Track if robot is moving
            old_pos = tuple(env.robot_pos)
            obs, reward, terminated, truncated, info = env.step(action)
            new_pos = tuple(env.robot_pos)
            
            if old_pos != new_pos:
                moves.add(new_pos)
            
            env.render()
            steps += 1
            
            if terminated or truncated:
                if terminated:
                    successes += 1
                    print(f"Episode {ep+1}: SUCCESS in {steps} steps")
                else:
                    print(f"Episode {ep+1}: TIMEOUT - visited {len(moves)} unique positions")
                break
    
    env.close()
    
    print(f"\nSuccess rate: {successes}/{episodes}")
    return successes > 0


if __name__ == "__main__":
    import sys
    
    print("\nChoose training method:")
    print("1. Curriculum Learning (recommended for mazes)")
    print("2. Extended Training (500k steps)")
    print("3. Both")
    
    choice = input("\nChoice (1-3): ")
    
    if choice == '1':
        model = train_with_curriculum('maze', total_timesteps=300000)
        print("\nTesting trained model...")
        quick_test(model, 'maze', episodes=5)
    
    elif choice == '2':
        model = train_longer('maze', total_timesteps=500000)
        print("\nTesting trained model...")
        quick_test(model, 'maze', episodes=5)
    
    elif choice == '3':
        print("\n### First: Curriculum Learning ###")
        model1 = train_with_curriculum('maze', total_timesteps=300000)
        quick_test(model1, 'maze', episodes=3)
        
        print("\n### Second: Extended Training ###")
        model2 = train_longer('maze', total_timesteps=500000)
        quick_test(model2, 'maze', episodes=3)
    
    else:
        print("Invalid choice")