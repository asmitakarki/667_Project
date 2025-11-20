# CS 667: RL Pathfinding Project
**Asmita Karki & Vinu Ekanayake**  
University of Kentucky

## Project Overview
This project explores reinforcement learning (RL) methods for teaching a robot to find optimal paths from start to goal while avoiding obstacles. We compare RL agents (PPO, DQN, A2C) against the traditional A* baseline across different map types.

## Features
✅ **Multiple Map Types**: Random, Spiral, Maze, Grid, and Corridor environments  
✅ **RL Algorithms**: PPO, DQN, A2C (via Stable-Baselines3)  
✅ **A* Baseline**: Deterministic pathfinding for comparison  
✅ **Comprehensive Evaluation**: Success rate, path efficiency, training time  
✅ **Visualization**: Real-time rendering and result plots  

## Installation

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

```

## Project Structure
```
├── map_generators.py              # Generate different map types
├── enhanced_pathfinding_env.py    # Gymnasium environment
├── astar_baseline.py              # A* algorithm implementation
├── comparison_script.py           # Comprehensive comparison
├── main.py                        # Master script for all operations
└── models/                        # Saved trained models
```

## Quick Start

### 1. Visualize All Map Types
```bash
python main.py --mode visualize
```
This generates and displays all 5 map types used in the project.

### 2. Train an RL Agent
```bash
# Train PPO on a maze
python main.py --mode train --map-type maze --algorithm PPO --timesteps 100000

# Train DQN on random obstacles
python main.py --mode train --map-type random --algorithm DQN --timesteps 50000
```

### 3. Test A* Baseline
```bash
# Test A* on spiral map
python main.py --mode astar --map-type spiral --episodes 10
```

### 4. Evaluate Trained Agent
```bash
# After training, evaluate your model
python main.py --mode evaluate --model-path models/PPO_maze_final --map-type maze --episodes 10
```

### 5. Run Full Comparison
```bash
# Compare RL vs A* across all map types (takes time!)
python main.py --mode compare
```
This will:
- Train RL agents on each map type
- Evaluate both RL and A*
- Generate comparison plots
- Save results to JSON

### 6. Interactive Demo
```bash
python main.py --mode demo
```

## Map Types

### 1. Random Obstacles
- Random circular obstacles scattered across the map
- Good for testing basic navigation

### 2. Spiral
- Spiral-shaped corridors
- Tests ability to follow complex paths

### 3. Maze
- Grid-based maze with walls
- Classic pathfinding challenge

### 4. Grid
- Regular grid of obstacles with gaps
- Tests systematic exploration

### 5. Corridor
- Narrow corridors with turns
- Tests precision navigation

## Usage Examples

### Example 1: Train and Compare Multiple Algorithms
```bash
# Train different algorithms on the same map
python main.py --mode train --map-type maze --algorithm PPO --timesteps 100000
python main.py --mode train --map-type maze --algorithm DQN --timesteps 100000
python main.py --mode train --map-type maze --algorithm A2C --timesteps 100000

# Evaluate each
python main.py --mode evaluate --model-path models/PPO_maze_final --map-type maze
python main.py --mode evaluate --model-path models/DQN_maze_final --map-type maze
python main.py --mode evaluate --model-path models/A2C_maze_final --map-type maze
```

### Example 2: Progressive Training 
```python
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from enhanced_pathfinding_env import EnhancedPathfindingEnv

# Stage 1: No obstacles
env = DummyVecEnv([lambda: EnhancedPathfindingEnv(grid_size=20, map_type='random', num_obstacles=0)])
model = PPO('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=50000)

# Stage 2: Few obstacles
env = DummyVecEnv([lambda: EnhancedPathfindingEnv(grid_size=20, map_type='random', num_obstacles=3)])
model.set_env(env)
model.learn(total_timesteps=50000)

# Stage 3: Full complexity
env = DummyVecEnv([lambda: EnhancedPathfindingEnv(grid_size=20, map_type='random', num_obstacles=5)])
model.set_env(env)
model.learn(total_timesteps=50000)

model.save("models/PPO_progressive")
```

### Example 3: Custom Evaluation
```python
from enhanced_pathfinding_env import EnhancedPathfindingEnv
from stable_baselines3 import PPO

# Load trained model
model = PPO.load("models/PPO_maze_final")

# Test on different map type (generalization test)
env = EnhancedPathfindingEnv(grid_size=20, map_type='spiral', render_mode='human')

obs, info = env.reset()
for _ in range(500):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()
    
    if terminated or truncated:
        print(f"Episode finished: {'SUCCESS' if terminated else 'TIMEOUT'}")
        obs, info = env.reset()

env.close()
```

## Evaluation Metrics

The project tracks:
1. **Success Rate**: % of episodes reaching the goal
2. **Average Steps**: Mean path length (efficiency)
3. **Training Time**: Time to train RL agents
4. **Computation Time**: Planning time for A* (per episode)
5. **Episode Reward**: Cumulative reward (RL only)

## Results Format

After running comparison, you'll get:
- `comparison_results.png`: Visual comparison plots
- `comparison_results.json`: Detailed numerical results
- `map_types_overview.png`: Map visualizations

## Tips for Your Report

### Things to Analyze:
1. **Success Rate vs Map Complexity**: Which maps are harder?
2. **RL vs A* Trade-offs**: 
   - A* is optimal but requires full map knowledge
   - RL learns adaptively but needs training time
3. **Algorithm Comparison**: Which RL algorithm works best?
4. **Generalization**: Train on one map type, test on another
5. **Sample Efficiency**: How many timesteps needed for good performance?

### Experiments to Run:
- Train on simple maps, test on complex ones
- Vary number of obstacles progressively
- Compare training curves (use tensorboard)
- Test with different reward functions
- Try larger/smaller grid sizes

## Troubleshooting

### Common Issues:

**Issue**: "Model not learning"
- **Solution**: Increase training timesteps, adjust learning rate, check reward function

**Issue**: "A* fails to find path"
- **Solution**: Reduce grid_resolution in AStarPathfinder, or reduce obstacle density

**Issue**: "Training too slow"
- **Solution**: Reduce timesteps, use A2C instead of PPO, smaller grid size

**Issue**: "Out of memory"
- **Solution**: Reduce buffer size for DQN, use smaller batch sizes

## Monitoring Training

Use TensorBoard to monitor training progress:
```bash
tensorboard --logdir ./tensorboard/
```

Then open http://localhost:6006 in your browser.

## Extensions (Optional)

Consider these for bonus points:
1. **Image-based observations**: Use CNN policy
2. **Dynamic obstacles**: Moving obstacles
3. **Multi-agent**: Multiple robots
4. **Continuous actions**: Smoother movement
5. **Real-time replanning**: Handle unexpected obstacles

## Citation

If you use this code, please cite:
```
@misc{pathfinding_rl_2025,
  title={Reinforcement Learning for Autonomous Robot Navigation},
  author={Karki, Asmita and Ekanayake, Vinu},
  year={2025},
  institution={University of Kentucky}
}
```
