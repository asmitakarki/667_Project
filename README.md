# CS 667: RL Pathfinding Project
**Asmita Karki & Vinu Ekanayake**  
University of Kentucky

## Project Overview
This project explores reinforcement learning (RL) methods for teaching a robot to navigate from start to goal while avoiding obstacles in a bounded 3D environment. We compare RL agents (PPO, SAC, TD3) against the traditional A* baseline across different map types using PyBullet physics simulation.

## Features
✅ **Clean Grid-Based Environment**: Bounded world with visible boundaries  
✅ **Multiple Map Types**: Random, Maze, and Grid obstacle configurations  
✅ **RL Algorithms**: PPO, SAC, TD3 (via Stable-Baselines3)  
✅ **A* Baseline**: Deterministic pathfinding for comparison  
✅ **3D Physics Simulation**: Realistic robot movement with PyBullet  
✅ **Randomized Maps**: Each reset generates new obstacle layouts  


### Prerequisites
- Python 3.8+
- pip

### Setup
```bash
# If python3.11 is not installed, you might need to install it first
# For macOS with Homebrew:
brew install python@3.11

# Then create the venv
python3.11 -m venv venv

# Activate it
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Now install packages
pip install gymnasium numpy matplotlib stable-baselines3 torch scipy pandas
pip install tqdm rich


## Project Structure
```
├── grid_bounded_env.py         # Main environment (clean grid)
├── map_generators.py           # Generate different map types
├── astar_baseline.py           # A* algorithm implementation
├── main.py                     # Master script for all operations
├── train_pybullet.py          # Training utilities
└── models/                     # Saved trained models
```

## Quick Start

### 1. Visualize Different Map Types
```bash
python main.py --mode visualize
```
Shows random, maze, and grid map types in PyBullet.

### 2. Train an RL Agent
```bash
# Train PPO on random obstacles (50k timesteps)
python main.py --mode train --map-type random --algorithm PPO --timesteps 50000

# Train on maze (100k timesteps for better performance)
python main.py --mode train --map-type maze --algorithm PPO --timesteps 100000

# Train on grid
python main.py --mode train --map-type grid --algorithm PPO --timesteps 50000
```

### 3. Evaluate Trained Agent
```bash
# After training, test your model
python main.py --mode evaluate --model-path models/PPO_random_pybullet --map-type random --episodes 10
```

### 4. Test A* Baseline
```bash
# Test A* pathfinding on random map
python main.py --mode astar --map-type random --episodes 10
```

### 5. Interactive Demo
```bash
# Watch robot navigate with random actions
python main.py --mode demo
```

## Map Types

All maps are **randomly generated** each time `reset()` is called, providing varied training scenarios.

### 1. Random Obstacles
- Scattered box obstacles at random positions
- Number and size vary
- Good for basic navigation skills
```bash
python main.py --mode train --map-type random --timesteps 50000
```

### 2. Maze
- Procedurally generated maze walls
- Requires strategic pathfinding
- New maze layout every reset
```bash
python main.py --mode train --map-type maze --timesteps 100000
```

### 3. Grid
- Regular grid pattern with random gaps
- Tests systematic exploration
- Gap patterns randomized each reset
```bash
python main.py --mode train --map-type grid --timesteps 50000
```

## Environment Details

### Observation Space
Vector of 5 elements: `[robot_x, robot_y, robot_yaw, goal_x, goal_y]`

### Action Space
Discrete(4):
- 0: Move forward
- 1: Turn left
- 2: Turn right  
- 3: Move backward

### Reward Function
- **+100**: Reaching goal
- **-50**: Collision with wall or obstacle (episode ends)
- **+progress × 10**: Moving closer to goal
- **+proximity bonus**: Extra reward when near goal
- **-0.1**: Small time penalty per step

### Episode Termination
- **Success**: Robot reaches within 0.6m of goal
- **Collision**: Robot hits wall or obstacle
- **Timeout**: Max 200 steps exceeded

### Key Features
- **Bounded World**: Physical walls prevent infinite exploration
- **Visible Grid**: 1x1 meter tiles for spatial reference
- **Randomized Spawning**: Robot and goal spawn in free spaces each episode
- **Randomized Obstacles**: New layouts every reset for generalization

## Training Guide

### Recommended Training Times
- **Random obstacles**: 50,000 - 100,000 timesteps
- **Maze**: 100,000 - 200,000 timesteps
- **Grid**: 50,000 - 100,000 timesteps

### Training Example
```bash
# Train PPO with 100k timesteps
python main.py --mode train --map-type random --algorithm PPO --timesteps 100000

# This uses 4 parallel environments for faster training
# Training takes approximately 10-15 minutes
```

### Evaluation Example
```bash
# Evaluate on 20 episodes to get reliable statistics
python main.py --mode evaluate --model-path models/PPO_random_pybullet --map-type random --episodes 20
```

## Using the Environment Directly

```python
from grid_bounded_env import GridBoundedEnv

# Create environment
env = GridBoundedEnv(
    grid_size=20,
    map_type='random',  # or 'maze', 'grid'
    render_mode='human',  # or None for training
    num_obstacles=5  # for random maps
)

# Reset environment
obs, info = env.reset()
# obs = [robot_x, robot_y, robot_yaw, goal_x, goal_y]

# Take actions
for _ in range(100):
    action = env.action_space.sample()  # or use your policy
    obs, reward, terminated, truncated, info = env.step(action)
    
    if terminated:
        print("Goal reached!")
        obs, info = env.reset()
    elif truncated:
        print("Episode ended (collision or timeout)")
        obs, info = env.reset()

env.close()
```

## Training Your Own Agent

```python
from grid_bounded_env import GridBoundedEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

# Create training environment (no rendering)
def make_env():
    env = GridBoundedEnv(grid_size=20, map_type='random', render_mode=None, num_obstacles=5)
    return Monitor(env)

# Use parallel environments
n_envs = 4
env = DummyVecEnv([make_env for _ in range(n_envs)])

# Create and train model
model = PPO('MlpPolicy', env, verbose=1, learning_rate=3e-4)
model.learn(total_timesteps=100000, progress_bar=True)

# Save model
model.save('models/my_trained_agent')

# Test with visualization
test_env = GridBoundedEnv(grid_size=20, map_type='random', render_mode='human', num_obstacles=5)
obs, _ = test_env.reset()

for _ in range(500):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = test_env.step(action)
    if terminated or truncated:
        obs, _ = test_env.reset()

test_env.close()
```

## Evaluation Metrics

The project tracks:
1. **Success Rate**: Percentage of episodes reaching the goal
2. **Average Steps**: Mean path length (lower is better)
3. **Average Reward**: Cumulative reward per episode
4. **Training Time**: Wall-clock time to train

## Expected Performance

### Random Map
- **Untrained (random actions)**: ~0-5% success rate
- **After 50k timesteps**: ~30-50% success rate  
- **After 100k timesteps**: ~60-80% success rate

### Maze Map
- **Untrained**: ~0-2% success rate
- **After 100k timesteps**: ~40-60% success rate
- **After 200k timesteps**: ~70-85% success rate

### A* Baseline
- **Success rate**: ~80-95% (depends on obstacle density)
- **Computation**: Instant path planning (<0.01s)
- **Limitation**: Requires full map knowledge

## Comparison: RL vs A*

### RL Advantages
- Learns from experience (no map needed)
- Adapts to new environments
- Can handle dynamic obstacles (with retraining)

### RL Disadvantages
- Requires training time
- Success rate varies
- Not guaranteed optimal

### A* Advantages  
- Optimal paths (if path exists)
- Instant planning
- Deterministic results

### A* Disadvantages
- Needs complete map knowledge
- Cannot adapt to changes
- Grid resolution affects accuracy

## Troubleshooting

### Robot Not Moving
- Make sure you're using `grid_bounded_env.py` (not old versions)
- Check that PyBullet is installed: `pip install pybullet`

### Training Too Slow
- Reduce `--timesteps` to 25000-50000 for quick tests
- PyBullet physics makes training slower than 2D
- Use `render_mode=None` during training (much faster)

### Low Success Rate
- Train longer (try 100k-200k timesteps)
- Maze environments need more training than random
- Check that model is saving/loading correctly

### Model Not Learning
- Verify reward function is working (check episode rewards)
- Try different learning rate: `learning_rate=1e-4` or `1e-3`
- Increase entropy coefficient for more exploration: `ent_coef=0.02`

## Tips for Your Report

### Key Experiments to Run

1. **Algorithm Comparison**
   - Train PPO, SAC, TD3 on same map type
   - Compare success rates and training time
   
2. **Map Complexity Analysis**
   - Train on random, maze, and grid
   - Which is hardest? Why?

3. **Sample Efficiency**
   - Plot success rate vs timesteps
   - How much training is needed?

4. **Generalization**
   - Train on one map type
   - Test on different map type
   - Does agent generalize?

5. **RL vs A* Comparison**
   - Success rate
   - Path optimality
   - Computation time
   - Adaptability

### Suggested Analysis

- Success rate curves over training
- Average episode length over time
- Collision frequency
- Effect of obstacle density
- Generalization across map types

## Advanced Usage

### Custom Map Configuration
```python
# More obstacles for harder challenge
env = GridBoundedEnv(map_type='random', num_obstacles=10)

# Larger maze cells
env = GridBoundedEnv(map_type='maze', cell_size=3)

# Denser grid
env = GridBoundedEnv(map_type='grid', spacing=2)
```

### Hyperparameter Tuning
```python
model = PPO(
    'MlpPolicy', env,
    learning_rate=3e-4,     # Try: 1e-4, 3e-4, 1e-3
    n_steps=2048,           # Try: 1024, 2048, 4096
    batch_size=64,          # Try: 32, 64, 128
    n_epochs=10,            # Try: 5, 10, 20
    gamma=0.99,             # Try: 0.95, 0.99
    ent_coef=0.01,         # Try: 0.0, 0.01, 0.02
    verbose=1
)
```

## Files Description

- **`grid_bounded_env.py`**: Main environment with bounded grid world
- **`map_generators.py`**: Creates random, maze, and grid obstacle layouts
- **`astar_baseline.py`**: A* pathfinding implementation for comparison
- **`main.py`**: Command-line interface for all operations
- **`train_pybullet.py`**: Training utilities and helper functions

## Citation

If you use this code in your work:
```
@misc{pathfinding_rl_2024,
  title={Reinforcement Learning for Robot Navigation in Bounded Environments},
  author={Karki, Asmita and Ekanayake, Vinu},
  year={2024},
  institution={University of Kentucky},
  course={CS 667}
}
```

## License

This project is for educational purposes as part of CS 667 at the University of Kentucky.

## Acknowledgments

- PyBullet for 3D physics simulation
- Stable-Baselines3 for RL implementations
- Gymnasium for environment interface

