"""
Clean Grid-Based Environment with Visible Boundaries
Like the Pioneer robot image - finite world with clear boundaries
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pybullet as p
import pybullet_data
import time

class GridBoundedEnv(gym.Env):
    """
    Clean grid-based environment with visible walls
    - Finite bounded world
    - Grid tiles visible
    - Walls around perimeter
    - Simple clean aesthetics
    """
    
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 30}
    
    def __init__(self, grid_size=20, map_type='random', render_mode=None, **map_kwargs):
        super().__init__()
        
        self.grid_size = grid_size
        self.map_type = map_type
        self.map_kwargs = map_kwargs
        self.render_mode = render_mode
        
        # Action space: 0=forward, 1=turn_left, 2=turn_right, 3=backward
        self.action_space = spaces.Discrete(4)
        
        # Observation: [robot_x, robot_y, robot_yaw, goal_x, goal_y]
        self.observation_space = spaces.Box(
            low=np.array([0, 0, -np.pi, 0, 0]),
            high=np.array([grid_size, grid_size, np.pi, grid_size, grid_size]),
            dtype=np.float32
        )
        
        # PyBullet IDs
        self.physics_client = None
        self.robot_id = None
        self.goal_id = None
        self.obstacle_ids = []
        self.wall_ids = []
        self.tile_ids = []
        
        self.step_count = 0
        self.max_steps = grid_size * 10
        
        self.robot_pos = None
        self.goal_pos = None
        
        # Movement parameters
        self.linear_speed = 2.5
        self.angular_speed = 2.0
        
        # Initialize PyBullet
        self._init_pybullet()
    
    def _init_pybullet(self):
        """Initialize PyBullet"""
        if self.physics_client is not None:
            p.disconnect(self.physics_client)
        
        if self.render_mode == 'human':
            self.physics_client = p.connect(p.GUI)
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
            p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)
        else:
            self.physics_client = p.connect(p.DIRECT)
        
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setRealTimeSimulation(0)
        p.setGravity(0, 0, -9.81)
        p.setPhysicsEngineParameter(fixedTimeStep=1./240.)
        
        # Create grid floor and walls
        self._create_grid_floor()
        self._create_boundary_walls()
        
        # Better camera angle
        p.resetDebugVisualizerCamera(
            cameraDistance=self.grid_size * 1.2,
            cameraYaw=45,
            cameraPitch=-50,
            cameraTargetPosition=[self.grid_size/2, self.grid_size/2, 0]
        )
    
    def _create_grid_floor(self):
        """Create visible grid tiles like in the image"""
        tile_size = 1.0  # Each tile is 1x1
        tile_height = 0.05
        
        # Two colors for checkerboard pattern (subtle)
        color1 = [0.85, 0.82, 0.78, 1]  # Light beige
        color2 = [0.78, 0.75, 0.71, 1]  # Slightly darker beige
        
        num_tiles = int(self.grid_size / tile_size)
        
        for i in range(num_tiles):
            for j in range(num_tiles):
                # Alternate colors
                if (i + j) % 2 == 0:
                    color = color1
                else:
                    color = color2
                
                # Create tile
                tile_collision = p.createCollisionShape(
                    p.GEOM_BOX,
                    halfExtents=[tile_size/2, tile_size/2, tile_height/2]
                )
                tile_visual = p.createVisualShape(
                    p.GEOM_BOX,
                    halfExtents=[tile_size/2, tile_size/2, tile_height/2],
                    rgbaColor=color
                )
                
                tile_id = p.createMultiBody(
                    baseMass=0,
                    baseCollisionShapeIndex=tile_collision,
                    baseVisualShapeIndex=tile_visual,
                    basePosition=[i * tile_size + tile_size/2, 
                                 j * tile_size + tile_size/2, 
                                 tile_height/2]
                )
                
                p.changeDynamics(tile_id, -1, lateralFriction=0.8)
                self.tile_ids.append(tile_id)
    
    def _create_boundary_walls(self):
        """Create walls around the perimeter - finite world!"""
        wall_height = 2.0
        wall_thickness = 0.2
        
        # Wall color - dark gray
        wall_color = [0.4, 0.4, 0.4, 1]
        
        # Four walls
        walls = [
            # North wall (top)
            ([self.grid_size/2, self.grid_size + wall_thickness/2, wall_height/2],
             [self.grid_size/2 + wall_thickness, wall_thickness/2, wall_height/2]),
            
            # South wall (bottom)
            ([self.grid_size/2, -wall_thickness/2, wall_height/2],
             [self.grid_size/2 + wall_thickness, wall_thickness/2, wall_height/2]),
            
            # East wall (right)
            ([self.grid_size + wall_thickness/2, self.grid_size/2, wall_height/2],
             [wall_thickness/2, self.grid_size/2 + wall_thickness, wall_height/2]),
            
            # West wall (left)
            ([-wall_thickness/2, self.grid_size/2, wall_height/2],
             [wall_thickness/2, self.grid_size/2 + wall_thickness, wall_height/2]),
        ]
        
        for position, half_extents in walls:
            collision = p.createCollisionShape(
                p.GEOM_BOX,
                halfExtents=half_extents
            )
            visual = p.createVisualShape(
                p.GEOM_BOX,
                halfExtents=half_extents,
                rgbaColor=wall_color
            )
            
            wall_id = p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=collision,
                baseVisualShapeIndex=visual,
                basePosition=position
            )
            
            p.changeDynamics(wall_id, -1, lateralFriction=1.0)
            self.wall_ids.append(wall_id)
    
    def _create_robot(self, position):
        """Create simple robot - like Pioneer mobile robot"""
        # Robot dimensions (compact mobile robot)
        robot_radius = 0.25
        robot_height = 0.2
        
        # Base
        base_collision = p.createCollisionShape(
            p.GEOM_CYLINDER,
            radius=robot_radius,
            height=robot_height
        )
        base_visual = p.createVisualShape(
            p.GEOM_CYLINDER,
            radius=robot_radius,
            length=robot_height,
            rgbaColor=[0.8, 0.3, 0.3, 1]  # Red robot
        )
        
        self.robot_id = p.createMultiBody(
            baseMass=1.0,
            baseCollisionShapeIndex=base_collision,
            baseVisualShapeIndex=base_visual,
            basePosition=[position[0], position[1], robot_height/2 + 0.05]
        )
        
        p.changeDynamics(self.robot_id, -1, lateralFriction=0.8, mass=1.0)
        
        # Add direction indicator (front sensor bar)
        indicator_visual = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[0.15, 0.05, 0.05],
            rgbaColor=[0.2, 0.2, 0.2, 1]  # Dark gray
        )
        indicator_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=-1,
            baseVisualShapeIndex=indicator_visual,
            basePosition=[position[0] + robot_radius + 0.1, position[1], robot_height/2 + 0.05]
        )
    
    def _create_goal(self, position):
        """Create goal marker - glowing circle like in image"""
        # Goal platform
        goal_radius = 0.5
        goal_height = 0.1
        
        goal_collision = p.createCollisionShape(
            p.GEOM_CYLINDER,
            radius=goal_radius,
            height=goal_height
        )
        goal_visual = p.createVisualShape(
            p.GEOM_CYLINDER,
            radius=goal_radius,
            length=goal_height,
            rgbaColor=[1.0, 0.6, 0.6, 0.7]  # Light red/pink glow
        )
        
        self.goal_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=-1,  # No collision
            baseVisualShapeIndex=goal_visual,
            basePosition=[position[0], position[1], goal_height/2 + 0.05]
        )
        
        # Add "GOAL" text visual (small cylinder on top)
        marker_visual = p.createVisualShape(
            p.GEOM_CYLINDER,
            radius=0.1,
            length=0.5,
            rgbaColor=[1.0, 0.5, 0.5, 1]
        )
        marker_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=-1,
            baseVisualShapeIndex=marker_visual,
            basePosition=[position[0], position[1], 0.5]
        )
    
    def _create_obstacles(self):
        """Create box obstacles like in the image"""
        from map_generators import (
            RandomObstaclesGenerator, MazeMapGenerator, GridMapGenerator
        )
        
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
            generator = RandomObstaclesGenerator(self.grid_size, 5)
        
        obstacles = generator.get_obstacles()
        
        # Obstacle colors - various grays like in image
        obstacle_colors = [
            [0.5, 0.5, 0.5, 1],
            [0.6, 0.6, 0.6, 1],
            [0.45, 0.45, 0.45, 1],
            [0.55, 0.55, 0.55, 1],
        ]
        
        for i, obs in enumerate(obstacles):
            pos = obs['pos']
            size = obs['size']
            
            # Random height variation
            height = np.random.uniform(1.5, 2.5)
            
            # Box obstacle
            collision_shape = p.createCollisionShape(
                p.GEOM_BOX,
                halfExtents=[size, size, height/2]
            )
            
            color = obstacle_colors[i % len(obstacle_colors)]
            visual_shape = p.createVisualShape(
                p.GEOM_BOX,
                halfExtents=[size, size, height/2],
                rgbaColor=color
            )
            
            obstacle_id = p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=collision_shape,
                baseVisualShapeIndex=visual_shape,
                basePosition=[pos[0], pos[1], height/2 + 0.05]
            )
            
            p.changeDynamics(obstacle_id, -1, lateralFriction=1.0)
            self.obstacle_ids.append(obstacle_id)
    
    def _find_free_position(self):
        """Find free position away from obstacles and walls"""
        max_attempts = 100
        min_obstacle_distance = 1.5
        margin = 1.0  # Stay away from walls
        
        for _ in range(max_attempts):
            pos = np.array([
                np.random.uniform(margin, self.grid_size - margin),
                np.random.uniform(margin, self.grid_size - margin)
            ])
            
            is_free = True
            for obs_id in self.obstacle_ids:
                obs_pos, _ = p.getBasePositionAndOrientation(obs_id)
                distance = np.linalg.norm(pos - np.array([obs_pos[0], obs_pos[1]]))
                if distance < min_obstacle_distance:
                    is_free = False
                    break
            
            if is_free:
                return pos
        
        return np.array([self.grid_size/2, self.grid_size/2])
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.step_count = 0
        
        # Clear existing
        if self.robot_id is not None:
            p.removeBody(self.robot_id)
        if self.goal_id is not None:
            p.removeBody(self.goal_id)
        for obs_id in self.obstacle_ids:
            p.removeBody(obs_id)
        self.obstacle_ids = []
        
        # Create new
        self._create_obstacles()
        self.robot_pos = self._find_free_position()
        self.goal_pos = self._find_free_position()
        
        while np.linalg.norm(self.robot_pos - self.goal_pos) < self.grid_size * 0.3:
            self.goal_pos = self._find_free_position()
        
        self._create_robot(self.robot_pos)
        self._create_goal(self.goal_pos)
        
        # Settle physics
        for _ in range(50):
            p.stepSimulation()
        
        return self._get_obs(), self._get_info()
    
    def step(self, action):
        self.step_count += 1
        
        # Get current state
        pos, orn = p.getBasePositionAndOrientation(self.robot_id)
        euler = p.getEulerFromQuaternion(orn)
        yaw = euler[2]
        
        old_distance = np.linalg.norm(
            np.array([pos[0], pos[1]]) - self.goal_pos
        )
        
        # Calculate velocities
        if action == 0:  # Forward
            linear_vel_x = self.linear_speed * np.cos(yaw)
            linear_vel_y = self.linear_speed * np.sin(yaw)
            angular_vel = 0
        elif action == 1:  # Turn left
            linear_vel_x = 0
            linear_vel_y = 0
            angular_vel = self.angular_speed
        elif action == 2:  # Turn right
            linear_vel_x = 0
            linear_vel_y = 0
            angular_vel = -self.angular_speed
        else:  # Backward
            linear_vel_x = -self.linear_speed * 0.5 * np.cos(yaw)
            linear_vel_y = -self.linear_speed * 0.5 * np.sin(yaw)
            angular_vel = 0
        
        # Apply velocities
        p.resetBaseVelocity(
            self.robot_id,
            linearVelocity=[linear_vel_x, linear_vel_y, 0],
            angularVelocity=[0, 0, angular_vel]
        )
        
        # Step simulation
        for _ in range(10):
            p.stepSimulation()
            if self.render_mode == 'human':
                time.sleep(1./240.)
        
        # Get new state
        pos, orn = p.getBasePositionAndOrientation(self.robot_id)
        euler = p.getEulerFromQuaternion(orn)
        
        self.robot_pos = np.array([pos[0], pos[1]])
        new_distance = np.linalg.norm(self.robot_pos - self.goal_pos)
        
        # Check collision with obstacles OR walls
        collision = self._check_collision()
        
        # Calculate reward
        reward = self._calculate_reward(collision, old_distance, new_distance)
        
        # Check termination
        terminated = new_distance < 0.6
        
        # Check truncation (timeout only - walls handle out of bounds)
        truncated = self.step_count >= self.max_steps
        
        if collision:
            reward = -50  # Heavy penalty for hitting walls/obstacles
            truncated = True  # End episode on collision
        
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
        """Check collision with obstacles or walls"""
        contact_points = p.getContactPoints(bodyA=self.robot_id)
        
        for contact in contact_points:
            # Check if hit obstacle or wall
            if contact[2] in self.obstacle_ids or contact[2] in self.wall_ids:
                return True
        
        return False
    
    def _calculate_reward(self, collision, old_distance, new_distance):
        """Calculate reward"""
        if new_distance < 0.6:
            return 100.0
        
        if collision:
            return -50.0
        
        progress = old_distance - new_distance
        distance_reward = progress * 10.0
        
        if new_distance < 5.0:
            proximity_bonus = (5.0 - new_distance) * 2.0
        else:
            proximity_bonus = 0
        
        time_penalty = -0.1
        
        return time_penalty + distance_reward + proximity_bonus
    
    def render(self):
        """Render handled by PyBullet"""
        pass
    
    def close(self):
        """Close PyBullet"""
        if self.physics_client is not None:
            p.disconnect(self.physics_client)
            self.physics_client = None


# Test the grid environment
if __name__ == "__main__":
    print("\n" + "="*70)
    print("Clean Grid-Based Bounded Environment")
    print("="*70)
    print("\n✨ Features:")
    print("  🔲 Visible grid tiles")
    print("  🧱 Boundary walls - finite world!")
    print("  🤖 Mobile robot (red cylinder)")
    print("  📦 Box obstacles (gray)")
    print("  🎯 Goal marker (pink glow)")
    print("\nThe world has clear boundaries - robot can't go beyond walls!\n")
    
    env = GridBoundedEnv(
        grid_size=20,
        map_type='random',
        render_mode='human',
        num_obstacles=5
    )
    
    obs, info = env.reset()
    print(f"Start: ({obs[0]:.2f}, {obs[1]:.2f})")
    print(f"Goal:  ({obs[3]:.2f}, {obs[4]:.2f})")
    print(f"Distance: {info['distance_to_goal']:.2f}m\n")
    
    episode = 1
    for i in range(500):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        if i % 30 == 0:
            action_names = ["Forward", "Turn Left", "Turn Right", "Reverse"]
            print(f"Step {i}: {action_names[action]}, Distance={info['distance_to_goal']:.2f}m")
        
        if terminated:
            print(f"\n✅ Goal reached in {info['step_count']} steps!\n")
            episode += 1
            obs, info = env.reset()
        
        if truncated:
            print(f"\n❌ Episode {episode} ended (collision or timeout)\n")
            episode += 1
            obs, info = env.reset()
    
    env.close()
    print("\n" + "="*70)
    print("Clean, bounded world - just like the Pioneer robot image!")
    print("="*70)