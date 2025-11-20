"""
Diagnostic script to understand why PPO isn't working
"""

from stable_baselines3 import PPO
from enhanced_pathfinding_env import EnhancedPathfindingEnv
import numpy as np
import matplotlib.pyplot as plt

def diagnose_model(model_path, map_type='maze'):
    """
    Diagnose what the model is doing
    """
    print("\n" + "="*60)
    print("MODEL DIAGNOSTICS")
    print("="*60 + "\n")
    
    # Load model
    try:
        model = PPO.load(model_path)
        print(f"✓ Model loaded: {model_path}")
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        return
    
    # Create environment
    env = EnhancedPathfindingEnv(grid_size=20, map_type=map_type)
    obs, info = env.reset()
    
    print(f"\n1. Initial State:")
    print(f"   Robot: ({obs[0]:.2f}, {obs[1]:.2f})")
    print(f"   Goal:  ({obs[2]:.2f}, {obs[3]:.2f})")
    print(f"   Distance: {np.linalg.norm(obs[:2] - obs[2:]):.2f}")
    
    # Test action distribution
    print(f"\n2. Action Predictions:")
    action_counts = {0: 0, 1: 0, 2: 0, 3: 0}
    action_names = {0: 'Up', 1: 'Right', 2: 'Down', 3: 'Left'}
    
    for _ in range(100):
        action, _ = model.predict(obs, deterministic=True)
        if isinstance(action, np.ndarray):
            action = int(action.item())
        action_counts[action] += 1
    
    print("   Action distribution (100 predictions on same state):")
    for action, count in action_counts.items():
        print(f"   {action_names[action]:6s}: {count:3d} times ({count}%)")
    
    # Check if model is deterministic
    if max(action_counts.values()) == 100:
        print("   ⚠ Model always predicts same action - might be stuck!")
    
    # Test actual movement
    print(f"\n3. Testing Movement (50 steps):")
    obs, info = env.reset()
    positions = [obs[:2].copy()]
    actions_taken = []
    rewards_received = []
    
    for step in range(50):
        action, _ = model.predict(obs, deterministic=True)
        if isinstance(action, np.ndarray):
            action = int(action.item())
        
        actions_taken.append(action)
        obs, reward, terminated, truncated, info = env.step(action)
        rewards_received.append(reward)
        positions.append(obs[:2].copy())
        
        if terminated or truncated:
            break
    
    positions = np.array(positions)
    
    # Calculate movement statistics
    total_distance_moved = 0
    for i in range(1, len(positions)):
        total_distance_moved += np.linalg.norm(positions[i] - positions[i-1])
    
    unique_positions = len(set(map(tuple, positions)))
    
    print(f"   Steps taken: {len(actions_taken)}")
    print(f"   Total distance moved: {total_distance_moved:.2f}")
    print(f"   Unique positions visited: {unique_positions}")
    print(f"   Average reward per step: {np.mean(rewards_received):.3f}")
    print(f"   Final distance to goal: {info['distance_to_goal']:.2f}")
    
    if total_distance_moved < 0.5:
        print("   ✗ Robot barely moved - MODEL NOT WORKING!")
    elif unique_positions < 5:
        print("   ⚠ Robot stuck in small area")
    else:
        print("   ✓ Robot is moving")
    
    # Action diversity
    action_diversity = len(set(actions_taken)) / 4.0
    print(f"   Action diversity: {action_diversity:.1%} (using {len(set(actions_taken))}/4 actions)")
    
    # Visualize path
    print(f"\n4. Visualizing Path...")
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Draw obstacles
    for obs_obj in env.obstacles:
        from matplotlib.patches import Circle
        circle = Circle(obs_obj['pos'], obs_obj['size'], color='gray', alpha=0.7)
        ax.add_patch(circle)
    
    # Draw goal
    goal_circle = Circle(obs[2:4], 0.5, color='green', alpha=0.6, label='Goal')
    ax.add_patch(goal_circle)
    
    # Draw path
    ax.plot(positions[:, 0], positions[:, 1], 'b-o', linewidth=2, markersize=4, label='Robot path')
    ax.plot(positions[0, 0], positions[0, 1], 'ro', markersize=10, label='Start')
    
    ax.set_xlim(0, env.grid_size)
    ax.set_ylim(0, env.grid_size)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_title(f'Model Behavior - {map_type.capitalize()} Map')
    
    plt.savefig(f'diagnostic_{map_type}.png', dpi=150, bbox_inches='tight')
    print(f"   Saved visualization: diagnostic_{map_type}.png")
    plt.show()
    
    # Overall assessment
    print(f"\n5. Overall Assessment:")
    score = 0
    
    if total_distance_moved > 5:
        print("   ✓ Robot moves significantly")
        score += 1
    else:
        print("   ✗ Robot barely moves - PROBLEM!")
    
    if action_diversity > 0.5:
        print("   ✓ Uses diverse actions")
        score += 1
    else:
        print("   ✗ Limited action variety - might be stuck")
    
    if np.mean(rewards_received) > -0.5:
        print("   ✓ Receiving reasonable rewards")
        score += 1
    else:
        print("   ✗ Very negative rewards - not learning well")
    
    print(f"\n   Score: {score}/3")
    
    if score == 0:
        print("\n   RECOMMENDATION: Model failed to learn. Try:")
        print("   - Train much longer (500k+ steps)")
        print("   - Use curriculum learning (start with easier maps)")
        print("   - Increase exploration (ent_coef=0.01)")
    elif score == 1:
        print("\n   RECOMMENDATION: Model partially learned. Try:")
        print("   - Continue training")
        print("   - Adjust reward function")
    elif score == 2:
        print("\n   RECOMMENDATION: Model is learning but needs improvement")
    else:
        print("\n   RECOMMENDATION: Model looks good!")
    
    env.close()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python diagnostic_script.py <model_path> [map_type]")
        print("\nExample:")
        print("  python diagnostic_script.py models/PPO_maze_final maze")
        sys.exit(1)
    
    model_path = sys.argv[1]
    map_type = sys.argv[2] if len(sys.argv) > 2 else 'maze'
    
    diagnose_model(model_path, map_type)