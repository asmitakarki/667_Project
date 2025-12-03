"""
Realistic City Environment with Mountains and Scenery
Features:
- Car-like robot
- Solid ground (no checkerboard!)
- Mountains in background
- City buildings
- Better lighting and colors
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pybullet as p
import pybullet_data
import time

class RealisticPathfindingEnv(gym.Env):
    """
    Realistic city navigation with scenic environment
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
        self.scenery_ids = []
        
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
        """Initialize PyBullet with better visuals"""
        if self.physics_client is not None:
            p.disconnect(self.physics_client)
        
        if self.render_mode == 'human':
            self.physics_client = p.connect(p.GUI)
            
            # Better GUI settings
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
            p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)
            p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 0)
            p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0)
            p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0)
        else:
            self.physics_client = p.connect(p.DIRECT)
        
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setRealTimeSimulation(0)
        p.setGravity(0, 0, -9.81)
        p.setPhysicsEngineParameter(fixedTimeStep=1./240.)
        
        # Create solid ground (no checkerboard!)
        self._create_ground()
        
        # Add scenery
        self._create_scenery()
        
        # Better camera angle
        p.resetDebugVisualizerCamera(
            cameraDistance=self.grid_size * 0.9,
            cameraYaw=45,
            cameraPitch=-30,
            cameraTargetPosition=[self.grid_size/2, self.grid_size/2, 0]
        )
    
    def _create_ground(self):
        """Create solid colored ground - no checkerboard!"""
        plane_shape = p.createCollisionShape(p.GEOM_PLANE)
        
        # Nice solid green/gray ground like a park or street
        plane_visual = p.createVisualShape(
            p.GEOM_PLANE,
            rgbaColor=[0.45, 0.55, 0.45, 1]  # Muted green-gray
        )
        
        self.plane_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=plane_shape,
            baseVisualShapeIndex=plane_visual,
            basePosition=[0, 0, 0]
        )
        
        p.changeDynamics(self.plane_id, -1, lateralFriction=0.8)
        
        # Add street lines (subtle, not grid-like)
        self._create_street_lines()
    
    def _create_street_lines(self):
        """Add subtle road markings"""
        line_color = [0.6, 0.6, 0.5, 1]  # Subtle tan/gray
        line_thickness = 0.15
        
        # Just a few horizontal and vertical "roads"
        road_positions = [5, 10, 15]
        
        for pos in road_positions:
            # Horizontal road
            visual = p.createVisualShape(
                p.GEOM_BOX,
                halfExtents=[self.grid_size/2, line_thickness, 0.005],
                rgbaColor=line_color
            )
            road_id = p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=-1,
                baseVisualShapeIndex=visual,
                basePosition=[self.grid_size/2, pos, 0.005]
            )
            self.scenery_ids.append(road_id)
            
            # Vertical road
            visual = p.createVisualShape(
                p.GEOM_BOX,
                halfExtents=[line_thickness, self.grid_size/2, 0.005],
                rgbaColor=line_color
            )
            road_id = p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=-1,
                baseVisualShapeIndex=visual,
                basePosition=[pos, self.grid_size/2, 0.005]
            )
            self.scenery_ids.append(road_id)
    
    def _create_scenery(self):
        """Add mountains and background buildings"""
        # Create mountains around the perimeter
        mountain_positions = [
            # Behind (far y)
            (5, self.grid_size + 10, 15, [0.4, 0.35, 0.3]),
            (10, self.grid_size + 12, 18, [0.35, 0.3, 0.25]),
            (15, self.grid_size + 11, 20, [0.45, 0.4, 0.35]),
            
            # Left side (far negative x)
            (-10, 5, 12, [0.4, 0.35, 0.3]),
            (-12, 10, 16, [0.35, 0.3, 0.25]),
            (-11, 15, 14, [0.45, 0.4, 0.35]),
            
            # Right side (far positive x)
            (self.grid_size + 10, 5, 13, [0.4, 0.35, 0.3]),
            (self.grid_size + 12, 10, 17, [0.35, 0.3, 0.25]),
            (self.grid_size + 11, 15, 15, [0.45, 0.4, 0.35]),
        ]
        
        for x, y, height, color in mountain_positions:
            # Create cone-shaped mountain
            visual = p.createVisualShape(
                p.GEOM_CYLINDER,
                radius=4,
                length=height,
                rgbaColor=[*color, 1]
            )
            mountain_id = p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=-1,  # No collision - just visual
                baseVisualShapeIndex=visual,
                basePosition=[x, y, height/2]
            )
            self.scenery_ids.append(mountain_id)
            
            # Add snow cap
            snow_visual = p.createVisualShape(
                p.GEOM_CYLINDER,
                radius=2,
                length=height * 0.3,
                rgbaColor=[0.9, 0.9, 0.95, 1]
            )
            snow_id = p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=-1,
                baseVisualShapeIndex=snow_visual,
                basePosition=[x, y, height - height * 0.15]
            )
            self.scenery_ids.append(snow_id)
        
        # Add background city skyline
        skyline_positions = [
            (self.grid_size/2 - 5, -8, 25, [0.3, 0.3, 0.35]),
            (self.grid_size/2, -10, 30, [0.35, 0.35, 0.4]),
            (self.grid_size/2 + 5, -9, 22, [0.3, 0.3, 0.35]),
        ]
        
        for x, y, height, color in skyline_positions:
            visual = p.createVisualShape(
                p.GEOM_BOX,
                halfExtents=[2, 2, height/2],
                rgbaColor=[*color, 1]
            )
            building_id = p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=-1,
                baseVisualShapeIndex=visual,
                basePosition=[x, y, height/2]
            )
            self.scenery_ids.append(building_id)
    
    def _create_robot(self, position):
        """Create realistic car robot"""
        # Car dimensions
        body_length = 0.7
        body_width = 0.45
        body_height = 0.35
        wheel_radius = 0.15
        
        # Main car body
        body_collision = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[body_length/2, body_width/2, body_height/2]
        )
        body_visual = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[body_length/2, body_width/2, body_height/2],
            rgbaColor=[0.9, 0.2, 0.15, 1]  # Bright red sports car
        )
        
        base_height = wheel_radius + body_height/2 + 0.05
        
        self.robot_id = p.createMultiBody(
            baseMass=1.0,
            baseCollisionShapeIndex=body_collision,
            baseVisualShapeIndex=body_visual,
            basePosition=[position[0], position[1], base_height]
        )
        
        p.changeDynamics(self.robot_id, -1, lateralFriction=0.8, mass=1.0)
        
        # Add windshield
        windshield_visual = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[0.15, 0.35, 0.12],
            rgbaColor=[0.15, 0.25, 0.35, 0.6]  # Dark blue tinted glass
        )
        windshield_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=-1,
            baseVisualShapeIndex=windshield_visual,
            basePosition=[position[0] + 0.15, position[1], base_height + body_height/2 + 0.08]
        )
        
        # Add wheels (visual only)
        wheel_visual = p.createVisualShape(
            p.GEOM_CYLINDER,
            radius=wheel_radius,
            length=0.08,
            rgbaColor=[0.1, 0.1, 0.1, 1]
        )
        
        wheel_positions = [
            [body_length/2 - 0.12, body_width/2 + 0.04, wheel_radius],
            [body_length/2 - 0.12, -body_width/2 - 0.04, wheel_radius],
            [-body_length/2 + 0.12, body_width/2 + 0.04, wheel_radius],
            [-body_length/2 + 0.12, -body_width/2 - 0.04, wheel_radius],
        ]
        
        for wx, wy, wz in wheel_positions:
            wheel_id = p.createMultiBody(
                baseMass=0.01,
                baseCollisionShapeIndex=-1,
                baseVisualShapeIndex=wheel_visual,
                basePosition=[position[0] + wx, position[1] + wy, wz],
                baseOrientation=p.getQuaternionFromEuler([0, np.pi/2, 0])
            )
        
        # Add headlights
        headlight_visual = p.createVisualShape(
            p.GEOM_SPHERE,
            radius=0.08,
            rgbaColor=[1, 1, 0.9, 1]  # Bright white-yellow
        )
        
        for side in [-0.15, 0.15]:
            light_id = p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=-1,
                baseVisualShapeIndex=headlight_visual,
                basePosition=[position[0] + body_length/2, position[1] + side, base_height - 0.05]
            )
    
    def _create_goal(self, position):
        """Create goal marker"""
        # Goal platform
        platform_visual = p.createVisualShape(
            p.GEOM_CYLINDER,
            radius=0.6,
            length=0.1,
            rgbaColor=[0.2, 0.8, 0.2, 0.7]  # Green transparent
        )
        
        self.goal_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=-1,
            baseVisualShapeIndex=platform_visual,
            basePosition=[position[0], position[1], 0.05]
        )
        
        # Tall flag pole
        pole_visual = p.createVisualShape(
            p.GEOM_CYLINDER,
            radius=0.04,
            length=3.0,
            rgbaColor=[0.9, 0.9, 0.9, 1]
        )
        pole_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=-1,
            baseVisualShapeIndex=pole_visual,
            basePosition=[position[0], position[1], 1.5]
        )
        
        # Checkered flag
        flag_visual = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[0.4, 0.02, 0.3],
            rgbaColor=[0.1, 0.9, 0.1, 1]
        )
        flag_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=-1,
            baseVisualShapeIndex=flag_visual,
            basePosition=[position[0] + 0.2, position[1], 2.7]
        )
    
    def _create_obstacles(self):
        """Create city buildings as obstacles"""
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
        
        # Building color schemes
        building_styles = [
            [0.8, 0.7, 0.6, 1],   # Sandstone
            [0.6, 0.6, 0.65, 1],  # Concrete gray
            [0.7, 0.5, 0.4, 1],   # Red brick
            [0.5, 0.6, 0.7, 1],   # Blue glass
            [0.65, 0.55, 0.5, 1], # Brown brick
        ]
        
        for i, obs in enumerate(obstacles):
            pos = obs['pos']
            size = obs['size']
            
            # Vary building heights
            height = np.random.uniform(2.5, 5.0)
            
            # Create building
            collision_shape = p.createCollisionShape(
                p.GEOM_BOX,
                halfExtents=[size, size, height/2]
            )
            
            color = building_styles[i % len(building_styles)]
            visual_shape = p.createVisualShape(
                p.GEOM_BOX,
                halfExtents=[size, size, height/2],
                rgbaColor=color
            )
            
            obstacle_id = p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=collision_shape,
                baseVisualShapeIndex=visual_shape,
                basePosition=[pos[0], pos[1], height/2]
            )
            
            p.changeDynamics(obstacle_id, -1, lateralFriction=1.0)
            self.obstacle_ids.append(obstacle_id)
            
            # Add rooftop details
            self._add_rooftop_details(pos, size, height)
    
    def _add_rooftop_details(self, pos, size, height):
        """Add AC units, antennas to rooftops"""
        # Small rooftop box (AC unit)
        if np.random.random() < 0.6:
            ac_size = size * 0.2
            ac_visual = p.createVisualShape(
                p.GEOM_BOX,
                halfExtents=[ac_size, ac_size, ac_size/2],
                rgbaColor=[0.5, 0.5, 0.5, 1]
            )
            ac_id = p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=-1,
                baseVisualShapeIndex=ac_visual,
                basePosition=[pos[0] + size * 0.3, pos[1], height + ac_size/2]
            )
            self.scenery_ids.append(ac_id)
        
        # Antenna
        if np.random.random() < 0.4:
            antenna_visual = p.createVisualShape(
                p.GEOM_CYLINDER,
                radius=0.03,
                length=1.0,
                rgbaColor=[0.6, 0.6, 0.6, 1]
            )
            antenna_id = p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=-1,
                baseVisualShapeIndex=antenna_visual,
                basePosition=[pos[0], pos[1], height + 0.5]
            )
            self.scenery_ids.append(antenna_id)
    
    def _find_free_position(self):
        """Find free position away from obstacles"""
        max_attempts = 100
        min_obstacle_distance = 2.0
        
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
        
        # Check collision
        collision = self._check_collision()
        
        # Calculate reward
        reward = self._calculate_reward(collision, old_distance, new_distance)
        
        # Check termination
        terminated = new_distance < 0.8
        
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
        if new_distance < 0.8:
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


# Test the environment
if __name__ == "__main__":
    print("\n" + "="*70)
    print("🏙️  Realistic City Navigation Environment")
    print("="*70)
    print("\n✨ Features:")
    print("  🚗 Red sports car with headlights")
    print("  🏢 Colorful city buildings")
    print("  ⛰️  Mountains with snow caps in the background")
    print("  🌆 City skyline")
    print("  🟢 Solid ground (no checkerboard!)")
    print("  🚩 Green finish flag")
    print("\nNavigate the red car through the city to reach the goal!\n")
    
    env = RealisticPathfindingEnv(
        grid_size=20,
        map_type='random',
        render_mode='human',
        num_obstacles=4
    )
    
    obs, info = env.reset()
    print(f"🏁 Start: ({obs[0]:.2f}, {obs[1]:.2f})")
    print(f"🎯 Goal:  ({obs[3]:.2f}, {obs[4]:.2f})")
    print(f"📏 Distance: {info['distance_to_goal']:.2f}m\n")
    
    episode = 1
    for i in range(500):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        if i % 30 == 0:
            action_names = ["🚗 Drive", "⬅️ Left", "➡️ Right", "⬅️ Reverse"]
            print(f"Step {i}: {action_names[action]}, Distance={info['distance_to_goal']:.2f}m")
        
        if terminated:
            print(f"\n🎉 SUCCESS! Car reached goal in {info['step_count']} steps!\n")
            episode += 1
            obs, info = env.reset()
        
        if truncated:
            print(f"\n❌ Episode {episode} ended at {info['step_count']} steps\n")
            episode += 1
            obs, info = env.reset()
    
    env.close()
    print("\n" + "="*70)
    print("Demo complete! Much better than checkerboard, right? 😊")
    print("="*70)