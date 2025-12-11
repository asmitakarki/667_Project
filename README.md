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


## Project Structure
```
├── robot_pov_env.py         # robot environment with obstacles
├── demo.py                  # visualize the map
├── train_all_without_cnn.py # Training without CNN for now
└── models/                  # Saved trained models
```

## Quick Start

### 1. Visualize Map
```bash
python demo.py
```
### 2. Train All Algorithms WITHOUT CNN (Compare PPO, SAC, TD3)
```bash
python train_all_without_cnn.py --mode compare --timesteps 200000
```

### 3. Train Single Algorithm WITHOUT CNN
```bash
# Train PPO
python train_all_without_cnn.py --mode train --algo PPO --timesteps 200000

# Train SAC
python train_all_without_cnn.py --mode train --algo SAC --timesteps 2000

# Train TD3
python train_all_without_cnn.py --mode train --algo TD3 --timesteps 200000
```

### 4. Test Trained Model
```bash
# Test PPO (shows robot's camera view)
python train.py --mode test --algo PPO --test-episodes 5

# Test SAC
python train.py --mode test --algo SAC --test-episodes 5

# Test TD3
python train_all_without_cnn.py --mode test --algo TD3 --test-episodes 5
```

## Files

### true_first_person_env.py - PyBullet environment with first-person camera view
### train_all_algorithms.py - Main training/testing script for all algorithms

## Output 
### Trained models are saved in:

### models/PPO/best/best_model.zip
### models/SAC/best/best_model.zip
### models/TD3/best/best_model.zip

### Training logs are saved in logs/ directory.
### Comparison plots are saved as algorithm_comparison.png.