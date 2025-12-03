"""
PyBullet 3D Pathfinding Environment - FIXED with Velocity Control
Uses direct velocity control instead of forces for reliable movement
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pybullet as p
import pybullet_data
import time

class PyBulletPathfindingEnv(gym.Env):
    """
    3D pathfinding with VELOCITY CONTROL (much more reliable than forces)
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
        self.plane_id = None
        
        self.step_count = 0
        self.max_steps = grid_size * 10
        
        self.robot_pos = None
        self.goal_pos = None
        
        # Movement parameters
        self.linear_speed = 2.0  # meters per second
        self.angular_speed = 2.0  # radians per second
        
        # Initialize PyBullet
        self._init_pybullet()
    
    def _init_pybullet(self):
        """Initialize PyBullet"""
        if self.physics_client is not None:
            p.disconnect(self.physics_client)
        
        if self.render_mode == 'human':
            self.physics_client = p.connect(p.GUI)
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        else:
            self.physics_client = p.connect(p.DIRECT)
        
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        
        # CRITICAL: Disable real-time simulation
        p.setRealTimeSimulation(0)
        
        p.setGravity(0, 0, -9.81)
        p.setPhysicsEngineParameter(fixedTimeStep=1./240.)
        
        # Load ground
        self.plane_id = p.loadURDF("plane.urdf")
        
        # Set camera
        p.resetDebugVisualizerCamera(
            cameraDistance=self.grid_size * 0.8,
            cameraYaw=45,
            cameraPitch=-45,
            cameraTargetPosition=[self.grid_size/2, self.grid_size/2, 0]
        )
    
    def _create_robot(self, position):
        """Create robot body"""
        # Main body
        base_collision = p.createCollisionShape(p.GEOM_SPHERE, radius=0.3)
        base_visual = p.createVisualShape(
            p.GEOM_SPHERE, radius=0.3, rgbaColor=[0, 0, 1, 1]
        )
        
        # Direction indicator
        indicator_collision = p.createCollisionShape(
            p.GEOM_CYLINDER, radius=0.08, height=0.3
        )
        indicator_visual = p.createVisualShape(
            p.GEOM_CYLINDER, radius=0.08, length=0.3, rgbaColor=[1, 0, 0, 1]
        )
        
        self.robot_id = p.createMultiBody(
            baseMass=1.0,
            baseCollisionShapeIndex=base_collision,
            baseVisualShapeIndex=base_visual,
            basePosition=[position[0], position[1], 0.3],
            linkMasses=[0.1],
            linkCollisionShapeIndices=[indicator_collision],
            linkVisualShapeIndices=[indicator_visual],
            linkPositions=[[0.38, 0, 0]],
            linkOrientations=[[0, 0.7071, 0, 0.7071]],
            linkInertialFramePositions=[[0, 0, 0]],
            linkInertialFrameOrientations=[[0, 0, 0, 1]],
            linkParentIndices=[0],
            linkJointTypes=[p.JOINT_FIXED],
            linkJointAxis=[[0, 0, 1]]
        )
        
        # Set friction
        p.changeDynamics(self.robot_id, -1, lateralFriction=0.8)
    
    def _create_goal(self, position):
        """Create goal marker"""
        visual_shape = p.createVisualShape(
            p.GEOM_CYLINDER, radius=0.5, length=0.1, rgbaColor=[0, 1, 0, 0.5]
        )
        self.goal_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=-1,
            baseVisualShapeIndex=visual_shape,
            basePosition=[position[0], position[1], 0.05]
        )
    
    def _create_obstacles(self):
        """Create obstacles"""
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
        
        for obs in obstacles:
            pos = obs['pos']
            size = obs['size']
            
            collision_shape = p.createCollisionShape(
                p.GEOM_CYLINDER, radius=size, height=2.0
            )
            visual_shape = p.createVisualShape(
                p.GEOM_CYLINDER, radius=size, length=2.0, rgbaColor=[0.5, 0.5, 0.5, 1]
            )
            
            obstacle_id = p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=collision_shape,
                baseVisualShapeIndex=visual_shape,
                basePosition=[pos[0], pos[1], 1.0]
            )
            
            self.obstacle_ids.append(obstacle_id)
    
    def _find_free_position(self):
        """Find free position away from obstacles"""
        max_attempts = 100
        min_obstacle_distance = 1.5
        
        for _ in range(max_attempts):
            pos = np.array([
                np.random.uniform(2, self.grid_size - 2),
                np.random.uniform(2, self.grid_size - 2)
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
        
        # Calculate desired velocities based on action
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
        
        # Apply velocities directly (this is the key fix!)
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
        
        # Check collision
        collision = self._check_collision()
        
        # Calculate reward
        reward = self._calculate_reward(collision, old_distance, new_distance)
        
        # Check termination
        terminated = new_distance < 0.5
        
        # Check truncation
        out_of_bounds = not (0.5 < pos[0] < self.grid_size - 0.5 and 
                            0.5 < pos[1] < self.grid_size - 0.5)
        fell_through = pos[2] < 0.1
        
        truncated = (self.step_count >= self.max_steps) or out_of_bounds or fell_through
        
        if out_of_bounds or fell_through:
            reward = -50
        
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
        """Check collision with obstacles"""
        contact_points = p.getContactPoints(bodyA=self.robot_id)
        
        for contact in contact_points:
            if contact[2] in self.obstacle_ids:
                return True
        
        return False
    
    def _calculate_reward(self, collision, old_distance, new_distance):
        """Calculate reward"""
        if new_distance < 0.5:
            return 100.0
        
        if collision:
            return -10.0
        
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


if __name__ == "__main__":
    print("\n" + "="*70)
    print("Testing PyBullet Environment with VELOCITY CONTROL")
    print("="*70)
    
    env = PyBulletPathfindingEnv(
        grid_size=20,
        map_type='random',
        render_mode='human',
        num_obstacles=3
    )
    
    print("\n✓ Environment created")
    print("\nThe robot should now ACTUALLY MOVE!")
    print("Watch it navigate with random actions...\n")
    
    obs, info = env.reset()
    print(f"Start: ({obs[0]:.2f}, {obs[1]:.2f})")
    print(f"Goal:  ({obs[3]:.2f}, {obs[4]:.2f})")
    print(f"Distance: {info['distance_to_goal']:.2f}m\n")
    
    episode = 1
    for i in range(500):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        if i % 20 == 0:
            action_names = ["Forward", "Left", "Right", "Back"]
            print(f"Step {i}: Action={action_names[action]}, "
                  f"Pos=({obs[0]:.2f}, {obs[1]:.2f}), "
                  f"Distance={info['distance_to_goal']:.2f}m, "
                  f"Reward={reward:.2f}")
        
        if terminated:
            print(f"\n🎉 Episode {episode} SUCCESS in {info['step_count']} steps!\n")
            episode += 1
            obs, info = env.reset()
        
        if truncated:
            print(f"\n❌ Episode {episode} failed at {info['step_count']} steps\n")
            episode += 1
            obs, info = env.reset()
    
    env.close()
    print("\n" + "="*70)
    print("Test complete! Robot should have been moving around.")
    print("="*70)