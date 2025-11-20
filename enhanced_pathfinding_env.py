"""
Enhanced Pathfinding Environment that supports different map types
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from map_generators import (
    RandomObstaclesGenerator, SpiralMapGenerator, MazeMapGenerator,
    GridMapGenerator, CorridorMapGenerator
)

class EnhancedPathfindingEnv(gym.Env):
    """
    Enhanced pathfinding environment with multiple map types
    
    Map types:
    - 'random': Random obstacles
    - 'spiral': Spiral corridors
    - 'maze': Grid-based maze
    - 'grid': Regular grid obstacles
    - 'corridor': Narrow corridors with turns
    """
    
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 30}
    
    def __init__(self, grid_size=20, map_type='random', render_mode=None, **map_kwargs):
        super().__init__()
        
        self.grid_size = grid_size
        self.map_type = map_type
        self.map_kwargs = map_kwargs
        self.render_mode = render_mode
        
        # Define action space: 0=up, 1=right, 2=down, 3=left
        self.action_space = spaces.Discrete(4)
        
        # Observation: [robot_x, robot_y, goal_x, goal_y]
        self.observation_space = spaces.Box(
            low=0, 
            high=grid_size, 
            shape=(4,), 
            dtype=np.float32
        )
        
        # Initialize state
        self.robot_pos = None
        self.goal_pos = None
        self.obstacles = []
        self.step_count = 0
        self.max_steps = grid_size * 6  # More steps for complex maps
        self.map_generator = None
        
        # For visualization
        self.fig = None
        self.ax = None
        
    def _generate_map(self):
        """Generate obstacles based on map type"""
        if self.map_type == 'random':
            num_obstacles = self.map_kwargs.get('num_obstacles', 5)
            generator = RandomObstaclesGenerator(self.grid_size, num_obstacles)
        elif self.map_type == 'spiral':
            num_spirals = self.map_kwargs.get('num_spirals', 3)
            corridor_width = self.map_kwargs.get('corridor_width', 2)
            generator = SpiralMapGenerator(self.grid_size, corridor_width=corridor_width, num_spirals=num_spirals)
        elif self.map_type == 'maze':
            cell_size = self.map_kwargs.get('cell_size', 2)
            generator = MazeMapGenerator(self.grid_size, cell_size=cell_size)
        elif self.map_type == 'grid':
            spacing = self.map_kwargs.get('spacing', 3)
            generator = GridMapGenerator(self.grid_size, spacing=spacing)
        elif self.map_type == 'corridor':
            num_segments = self.map_kwargs.get('num_segments', 5)
            corridor_width = self.map_kwargs.get('corridor_width', 2.5)
            generator = CorridorMapGenerator(self.grid_size, corridor_width=corridor_width, num_segments=num_segments)
        else:
            raise ValueError(f"Unknown map type: {self.map_type}")
        
        self.map_generator = generator
        self.obstacles = generator.get_obstacles()
        
    def _find_free_position(self, min_distance_from_obstacles=1.0):
        """Find a position that's not inside an obstacle"""
        max_attempts = 100
        for _ in range(max_attempts):
            pos = np.array([
                self.np_random.uniform(1, self.grid_size - 1),
                self.np_random.uniform(1, self.grid_size - 1)
            ])
            
            # Check if position is free
            is_free = True
            for obs in self.obstacles:
                distance = np.linalg.norm(pos - obs['pos'])
                if distance < obs['size'] + min_distance_from_obstacles:
                    is_free = False
                    break
            
            if is_free:
                return pos
        
        # If no free position found, return a fallback
        return np.array([self.grid_size / 2, self.grid_size / 2])
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Reset step counter
        self.step_count = 0
        
        # Generate map
        self._generate_map()
        
        # Place robot at free position
        self.robot_pos = self._find_free_position()
        
        # Place goal far from robot in free position
        max_attempts = 50
        for _ in range(max_attempts):
            goal = self._find_free_position()
            if np.linalg.norm(self.robot_pos - goal) > self.grid_size * 0.3:
                self.goal_pos = goal
                break
        else:
            # Fallback if no good goal found
            self.goal_pos = self._find_free_position()
        
        observation = self._get_obs()
        info = self._get_info()
        
        return observation, info
    
    def step(self, action):
        self.step_count += 1
        
        # Map action to movement
        move_dict = {
            0: np.array([0, 1]),   # up
            1: np.array([1, 0]),   # right
            2: np.array([0, -1]),  # down
            3: np.array([-1, 0])   # left
        }
        
        # Move robot
        old_pos = self.robot_pos.copy()
        self.robot_pos += move_dict[action] * 0.5  # Step size
        
        # Keep within bounds
        self.robot_pos = np.clip(self.robot_pos, 0, self.grid_size)
        
        # Check collision with obstacles
        collision = self._check_collision()
        if collision:
            self.robot_pos = old_pos  # Revert movement
        
        # Calculate reward
        old_distance = np.linalg.norm(old_pos - self.goal_pos)
        new_distance = np.linalg.norm(self.robot_pos - self.goal_pos)
        reward = self._calculate_reward(collision, old_distance, new_distance)
        
        # Check if goal reached
        terminated = new_distance < 0.5
        truncated = self.step_count >= self.max_steps
        
        observation = self._get_obs()
        info = self._get_info()
        
        return observation, reward, terminated, truncated, info
    
    def _get_obs(self):
        """Return current observation"""
        return np.concatenate([self.robot_pos, self.goal_pos]).astype(np.float32)
    
    def _get_info(self):
        """Return additional info"""
        return {
            'distance_to_goal': np.linalg.norm(self.robot_pos - self.goal_pos),
            'step_count': self.step_count,
            'map_type': self.map_type
        }
    
    def _check_collision(self):
        """Check if robot collides with any obstacle"""
        for obs in self.obstacles:
            distance = np.linalg.norm(self.robot_pos - obs['pos'])
            if distance < obs['size']:
                return True
        return False
    
    def _calculate_reward(self, collision, old_distance, new_distance):
        """Calculate reward for current state with better shaping"""
        # Large positive reward for reaching goal
        if new_distance < 0.5:
            return 100.0
        
        # Negative reward for collision
        if collision:
            return -10.0  # Increased penalty
        
        # Reward for getting closer to goal (shaped reward)
        progress = old_distance - new_distance
        
        # Distance-based reward (encourages moving toward goal)
        distance_reward = progress * 5.0  # Increased from 2.0
        
        # Small time penalty (encourages efficiency but not too harsh)
        time_penalty = -0.05  # Reduced from -0.1
        
        # Bonus for being closer to goal
        proximity_bonus = 0
        if new_distance < 5.0:
            proximity_bonus = (5.0 - new_distance) * 0.5
        
        reward = time_penalty + distance_reward + proximity_bonus
        
        return reward
    
    def render(self):
        if self.render_mode == 'human':
            if self.fig is None:
                self.fig, self.ax = plt.subplots(figsize=(8, 8))
                plt.ion()
            
            self.ax.clear()
            self.ax.set_xlim(0, self.grid_size)
            self.ax.set_ylim(0, self.grid_size)
            self.ax.set_aspect('equal')
            self.ax.grid(True, alpha=0.3)
            
            # Draw obstacles
            for obs in self.obstacles:
                if 'width' in obs and 'height' in obs:
                    rect = Rectangle(
                        (obs['pos'][0] - obs['width']/2, obs['pos'][1] - obs['height']/2),
                        obs['width'], obs['height'],
                        color='gray', alpha=0.7
                    )
                    self.ax.add_patch(rect)
                else:
                    circle = Circle(obs['pos'], obs['size'], color='gray', alpha=0.7)
                    self.ax.add_patch(circle)
            
            # Draw goal
            goal_circle = Circle(self.goal_pos, 0.5, color='green', alpha=0.6)
            self.ax.add_patch(goal_circle)
            
            # Draw robot
            robot_circle = Circle(self.robot_pos, 0.3, color='blue')
            self.ax.add_patch(robot_circle)
            
            dist = np.linalg.norm(self.robot_pos - self.goal_pos)
            self.ax.set_title(f'{self.map_type.capitalize()} Map - Step: {self.step_count} - Distance: {dist:.1f}')
            plt.pause(0.01)
            plt.draw()
    
    def close(self):
        if self.fig is not None:
            plt.close(self.fig)


# Test the environment with different maps
if __name__ == "__main__":
    print("Testing different map types...\n")
    
    map_types = ['random', 'spiral', 'maze', 'grid', 'corridor']
    
    for map_type in map_types:
        print(f"Testing {map_type} map...")
        env = EnhancedPathfindingEnv(grid_size=20, map_type=map_type, render_mode='human')
        
        obs, info = env.reset()
        
        for _ in range(50):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            env.render()
            
            if terminated or truncated:
                print(f"  Episode finished. Distance to goal: {info['distance_to_goal']:.2f}")
                break
        
        env.close()
        print()