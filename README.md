# CS 667: RL Pathfinding Project
**Asmita Karki & Vinu Ekanayake**  
University of Kentucky

## Project Overview
This project explores reinforcement learning (RL) methods for teaching a robot to navigate from start to goal while avoiding obstacles in a bounded 3D environment. We compare RL agents (PPO, SAC, TD3) across the map using PyBullet physics simulation.


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

# Install dependencies in correct order
pip install "numpy<2.0"
pip install opencv-python==4.10.0.84
pip install torch torchvision
pip install stable-baselines3[extra] 
pip install gymnasium pybullet matplotlib 
pip install tqdm rich
```

## Project Structure
```
├── robot_pov_env.py         # robot environment with obstacles
├── demo.py                  # visualize the map
├── train.py                 # Training without CNN for now
├── test_models.py           # Compare, single, watch modes
├── plot_training.py         # script to plot PPO, SAC, and TD3 from csv files      
└── models/                  # Saved trained models
├── logs/                    # contains logs for training and csv files used to plot training curves
```

## Quick Start

### 1. Visualize Map
```bash
python demo.py
```

### 2. Train Algorithms
```bash
python train2.py --algo PPO --timesteps 2000000 --n-envs 4
python train.py --algo SAC --timesteps 2000000 --n-envs 4  
python train.py --algo TD3 --timesteps 2000000 --n-envs 4
```

### 3. Test Trained Model
```bash
# Test all algorithms and generate comparison plots
python test_models.py --mode compare --episodes 20

# Test single algorithm
python test_models.py --mode single --algo PPO --episodes 20

# Watch agent perform with visualization
python test_models.py --mode watch --algo PPO --episodes 20
```


## Output 
### Trained models are saved in:

### models/PPO/final_model.zip
### models/SAC/final_model.zip
### models/TD3/final_model.zip

### Training logs are saved in logs/ directory.
### Training curves are saved in graphs/.
### Comparison plots are saved as algorithm_comparison.png.