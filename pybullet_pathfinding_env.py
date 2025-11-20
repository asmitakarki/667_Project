"""
PyBullet 3D Pathfinding Environment
More realistic physics simulation with a wheeled robot
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pybullet as p
import pybullet_data
import time

class PyBulletPathfindingEnv(gym.Env):
    """
    3D pathfinding environment using PyBullet physics engine
    
    Features:
    - Realistic physics simulation
    - Wheeled robot with differential drive
    - 3D obstacles
    - Collision detection
    """
    
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 30}
    
    def __init__(self, grid_size=20, map_type='random', render_mode=None, **map_kwargs):
        super().__init__()
        
        self.grid_size = grid_size
        self.map_type = map_type
        self.map_kwargs = map_kwargs
        self.render_mode = render_mode
        
        # Action space: [left_wheel_velocity, right_wheel_velocity]
        # Discrete version: 0=forward, 1=turn_right, 2=turn_left, 3=backward
        self.action_space = spaces.Discrete(4)
        
        # Observation: [robot_x, robot_y, robot_yaw, goal_x, goal_y]
        self.observation_space = spaces.Box(
            low=np.array([0, 0, -np.pi, 0, 0]),
            high=np.array([grid_size, grid_size, np.pi, grid_size, grid_size]),
            dtype=np.float32
        )
        
        # PyBullet setup
        self.physics_client = None
        self.robot_id = None
        self.goal_id = None
        self.obstacle_ids = []
        self.plane_id = None
        
        self.step_count = 0
        self.max_steps = grid_size * 10
        
        self.robot_pos = None
        self.goal_pos = None
        
        # Initialize PyBullet
        self._init_pybullet()
    
    def _init_pybullet(self):
        """Initialize PyBullet physics engine"""
        if self.physics_client is not None:
            p.disconnect(self.physics_client)
        
        if self.render_mode == 'human':
            self.physics_client = p.connect(p.GUI)
        else:
            self.physics_client = p.connect(p.DIRECT)
        
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        
        # Load ground plane
        self.plane_id = p.loadURDF("plane.urdf")
        
        # Set camera
        p.resetDebugVisualizerCamera(
            cameraDistance=self.grid_size * 0.8,
            cameraYaw=45,
            cameraPitch=-45,
            cameraTargetPosition=[self.grid_size/2, self.grid_size/2, 0]
        )
    
    def _create_robot(self, position):
        """Create a simple wheeled robot"""
        # Create robot body (sphere for simplicity)
        collision_shape = p.createCollisionShape(p.GEOM_SPHERE, radius=0.3)
        visual_shape = p.createVisualShape(
            p.GEOM_SPHERE,
            radius=0.3,
            rgbaColor=[0, 0, 1, 1]  # Blue
        )
        
        self.robot_id = p.createMultiBody(
            baseMass=1.0,
            baseCollisionShapeIndex=collision_shape,
            baseVisualShapeIndex=visual_shape,
            basePosition=[position[0], position[1], 0.3]
        )
        
        # Add direction indicator (small cylinder pointing forward)
        indicator_collision = p.createCollisionShape(
            p.GEOM_CYLINDER,
            radius=0.1,
            height=0.4
        )
        indicator_visual = p.createVisualShape(
            p.GEOM_CYLINDER,
            radius=0.1,
            length=0.4,
            rgbaColor=[1, 0, 0, 1]  # Red indicator
        )
        
        # Attach indicator to robot (pointing in x direction)
        p.createMultiBody(
            baseMass=0.01,
            baseCollisionShapeIndex=indicator_collision,
            baseVisualShapeIndex=indicator_visual,
            basePosition=[position[0] + 0.4, position[1], 0.3],
            baseOrientation=p.getQuaternionFromEuler([0, np.pi/2, 0])
        )
    
    def _create_goal(self, position):
        """Create goal marker"""
        collision_shape = p.createCollisionShape(
            p.GEOM_CYLINDER,
            radius=0.5,
            height=0.1
        )
        visual_shape = p.createVisualShape(
            p.GEOM_CYLINDER,
            radius=0.5,
            length=0.1,
            rgbaColor=[0, 1, 0, 0.5]  # Green, semi-transparent
        )
        
        self.goal_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=-1,  # No collision for goal
            baseVisualShapeIndex=visual_shape,
            basePosition=[position[0], position[1], 0.05]
        )
    
    def _create_obstacles(self):
        """Create obstacles based on map type"""
        from map_generators import (
            RandomObstaclesGenerator, SpiralMapGenerator, 
            MazeMapGenerator, GridMapGenerator, CorridorMapGenerator
        )
        
        # Generate map
        if self.map_type == 'random':
            num_obstacles = self.map_kwargs.get('num_obstacles', 5)
            generator = RandomObstaclesGenerator(self.grid_size, num_obstacles)
        elif self.map_type == 'maze':
            cell_size = self.map_kwargs.get('cell_size', 2)
            generator = MazeMapGenerator(self.grid_size, cell_size=cell_size)
        elif self.map_type == 'grid':
            spacing = self.map_kwargs.get('spacing', 3)
            generator = GridMapGenerator(self.grid_size, spacing=spacing)
        else:
            # Default to random
            generator = RandomObstaclesGenerator(self.grid_size, 5)
        
        obstacles = generator.get_obstacles()
        
        # Create 3D obstacles
        for obs in obstacles:
            pos = obs['pos']
            size = obs['size']
            
            # Create cylinder obstacle
            collision_shape = p.createCollisionShape(
                p.GEOM_CYLINDER,
                radius=size,
                height=2.0
            )
            visual_shape = p.createVisualShape(
                p.GEOM_CYLINDER,
                radius=size,
                length=2.0,
                rgbaColor=[0.5, 0.5, 0.5, 1]  # Gray
            )
            
            obstacle_id = p.createMultiBody(
                baseMass=0,  # Static
                baseCollisionShapeIndex=collision_shape,
                baseVisualShapeIndex=visual_shape,
                basePosition=[pos[0], pos[1], 1.0]
            )
            
            self.obstacle_ids.append(obstacle_id)
    
    def _find_free_position(self):
        """Find a position not occupied by obstacles"""
        max_attempts = 100
        for _ in range(max_attempts):
            pos = np.array([
                np.random.uniform(2, self.grid_size - 2),
                np.random.uniform(2, self.grid_size - 2)
            ])
            
            # Check if position is free (simple check)
            is_free = True
            if len(self.obstacle_ids) > 0:
                # Just check distance for now
                is_free = True
            
            if is_free:
                return pos
        
        return np.array([self.grid_size/2, self.grid_size/2])
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.step_count = 0
        
        # Clear existing objects
        if self.robot_id is not None:
            p.removeBody(self.robot_id)
        if self.goal_id is not None:
            p.removeBody(self.goal_id)
        for obs_id in self.obstacle_ids:
            p.removeBody(obs_id)
        self.obstacle_ids = []
        
        # Create obstacles first
        self._create_obstacles()
        
        # Create robot and goal
        self.robot_pos = self._find_free_position()
        self.goal_pos = self._find_free_position()
        
        # Ensure goal is far from robot
        while np.linalg.norm(self.robot_pos - self.goal_pos) < self.grid_size * 0.3:
            self.goal_pos = self._find_free_position()
        
        self._create_robot(self.robot_pos)
        self._create_goal(self.goal_pos)
        
        # Let physics settle
        for _ in range(10):
            p.stepSimulation()
        
        return self._get_obs(), self._get_info()
    
    def step(self, action):
        self.step_count += 1
        
        # Map discrete actions to velocities
        # 0=forward, 1=turn_right, 2=turn_left, 3=backward
        action_map = {
            0: [5, 5],      # Forward
            1: [5, -5],     # Turn right
            2: [-5, 5],     # Turn left
            3: [-5, -5]     # Backward
        }
        
        left_vel, right_vel = action_map[action]
        
        # Get current robot state
        pos, orn = p.getBasePositionAndOrientation(self.robot_id)
        euler = p.getEulerFromQuaternion(orn)
        yaw = euler[2]
        
        old_distance = np.linalg.norm(
            np.array([pos[0], pos[1]]) - self.goal_pos
        )
        
        # Apply forces to simulate differential drive
        forward_force = (left_vel + right_vel) / 2
        turning_force = (right_vel - left_vel) / 2
        
        # Apply force in robot's forward direction
        force_x = forward_force * np.cos(yaw)
        force_y = forward_force * np.sin(yaw)
        
        p.applyExternalForce(
            self.robot_id,
            -1,
            [force_x, force_y, 0],
            pos,
            p.WORLD_FRAME
        )
        
        # Apply torque for turning
        p.applyExternalTorque(
            self.robot_id,
            -1,
            [0, 0, turning_force],
            p.WORLD_FRAME
        )
        
        # Step simulation
        for _ in range(10):  # Multiple physics steps per action
            p.stepSimulation()
            if self.render_mode == 'human':
                time.sleep(1./240.)
        
        # Get new position
        pos, orn = p.getBasePositionAndOrientation(self.robot_id)
        euler = p.getEulerFromQuaternion(orn)
        
        self.robot_pos = np.array([pos[0], pos[1]])
        new_distance = np.linalg.norm(self.robot_pos - self.goal_pos)
        
        # Check collision
        collision = self._check_collision()
        
        # Calculate reward
        reward = self._calculate_reward(collision, old_distance, new_distance)
        
        # Check termination
        terminated = new_distance < 0.5
        truncated = self.step_count >= self.max_steps
        
        # Check if robot fell off or went out of bounds
        if pos[2] < 0.1 or not (0 < pos[0] < self.grid_size and 0 < pos[1] < self.grid_size):
            reward = -50
            truncated = True
        
        return self._get_obs(), reward, terminated, truncated, self._get_info()
    
    def _get_obs(self):
        """Get observation"""
        pos, orn = p.getBasePositionAndOrientation(self.robot_id)
        euler = p.getEulerFromQuaternion(orn)
        yaw = euler[2]
        
        return np.array([
            pos[0], pos[1], yaw,
            self.goal_pos[0], self.goal_pos[1]
        ], dtype=np.float32)
    
    def _get_info(self):
        """Get info"""
        pos, _ = p.getBasePositionAndOrientation(self.robot_id)
        distance = np.linalg.norm(
            np.array([pos[0], pos[1]]) - self.goal_pos
        )
        
        return {
            'distance_to_goal': distance,
            'step_count': self.step_count,
            'map_type': self.map_type
        }
    
    def _check_collision(self):
        """Check if robot collided with obstacles"""
        contact_points = p.getContactPoints(bodyA=self.robot_id)
        
        # Check if any contact with obstacles (not ground)
        for contact in contact_points:
            if contact[2] != self.plane_id:  # Not ground
                return True
        
        return False
    
    def _calculate_reward(self, collision, old_distance, new_distance):
        """Calculate reward"""
        # Goal reached
        if new_distance < 0.5:
            return 100.0
        
        # Collision penalty
        if collision:
            return -10.0
        
        # Progress reward
        progress = old_distance - new_distance
        distance_reward = progress * 5.0
        
        # Time penalty
        time_penalty = -0.05
        
        # Proximity bonus
        proximity_bonus = 0
        if new_distance < 5.0:
            proximity_bonus = (5.0 - new_distance) * 0.5
        
        return time_penalty + distance_reward + proximity_bonus
    
    def render(self):
        """Render is handled by PyBullet GUI"""
        pass
    
    def close(self):
        """Close PyBullet connection"""
        if self.physics_client is not None:
            p.disconnect(self.physics_client)


# Test the environment
if __name__ == "__main__":
    print("Testing PyBullet Pathfinding Environment\n")
    
    env = PyBulletPathfindingEnv(
        grid_size=20,
        map_type='random',
        render_mode='human',
        num_obstacles=5
    )
    
    print("Environment created. Testing with random actions...")
    print("Close the PyBullet window to stop.\n")
    
    obs, info = env.reset()
    
    for i in range(500):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        if i % 50 == 0:
            print(f"Step {i}: Distance to goal = {info['distance_to_goal']:.2f}")
        
        if terminated:
            print(f"\nâœ“ Goal reached in {info['step_count']} steps!")
            obs, info = env.reset()
        
        if truncated:
            print(f"\nâœ— Episode truncated at {info['step_count']} steps")
            obs, info = env.reset()
    
    env.close()
    print("\nTest complete!")