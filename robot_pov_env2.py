"""
First-Person POV Camera Environment
The PyBullet window shows what the ROBOT SEES through its camera
Like a video game FPS view

UPDATED: Improved reward shaping for better learning
"""
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pybullet_data
import time
import math
import cv2

# Silence PyBullet logging
import os
os.environ["PYBULLET_LOGGING_LEVEL"] = "ERROR"
import pybullet as p

class RobotPOVEnv(gym.Env):
    """
    Environment where the render window shows the robot's camera view
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}
    
    def __init__(
        self,
        grid_size=20,
        map_type="city",
        render_mode=None,
        use_camera_obs=False,  # Use camera for RL observations
        camera_width=640,
        camera_height=480,
        num_obstacles=5,
    ):
        super().__init__()
        
        self.grid_size = grid_size
        self.map_type = map_type
        self.render_mode = render_mode
        self.use_camera_obs = use_camera_obs
        self.camera_width = camera_width
        self.camera_height = camera_height
        self.num_obstacles = num_obstacles
        
        # For displaying camera view
        self.camera_window_name = "Robot Camera View"
        
        # Physics
        self.physics_client = None
        self.robot_id = None
        self.goal_id = None
        self.road_tile_ids = []
        self.building_ids = []
        self.obstacle_ids = []
        self.wall_ids = []
        self.goal_extra_ids = []      # <- pole + flag
        self.lane_marker_ids = []     # <- for cleanup

        
        self.linear_speed = 3.5 # increased from 2.5
        self.angular_speed = 2.5 # increased from 2.0
        
        self.action_space = spaces.Discrete(4)
        
        if use_camera_obs:
            # Small images for RL
            self.observation_space = spaces.Box(
                low=0, high=255,
                shape=(84, 84, 3),
                dtype=np.uint8
            )
        else:
            # Position observations
            self.observation_space = spaces.Box(
                low=np.array([0, 0, -np.pi, 0, 0], dtype=np.float32),
                high=np.array([grid_size, grid_size, np.pi, grid_size, grid_size], dtype=np.float32),
            )
        
        # DISTANCE-BASED episode horizon (fair for varying start/goal distances)
        self.base_max_steps = grid_size * 10      # baseline: 200 steps
        self.steps_per_meter = 20                 # +20 steps per meter of distance
        self.min_episode_steps = grid_size * 8    # min: 160 steps
        self.max_episode_steps = grid_size * 35   # max: 700 steps
        self.max_steps = self.base_max_steps      # will be set per-episode in reset()
        self.step_count = 0
        
        self._init_pybullet()
    
    def _init_pybullet(self):
        """Initialize PyBullet in DIRECT mode (no GUI)"""
        if self.physics_client is not None:
            p.disconnect(self.physics_client)
        
        # DIRECT mode - we'll show camera view separately
        self.physics_client = p.connect(p.DIRECT)
        
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setRealTimeSimulation(0)
        p.setPhysicsEngineParameter(fixedTimeStep=1.0/240.0)
        
        self._create_ground()
        self._create_boundary_walls()
    
    def _create_ground(self):
        plane_id = p.loadURDF("plane.urdf")
        p.changeDynamics(plane_id, -1, lateralFriction=0.8)

    def _create_boundary_walls(self):
        """Create visible boundary walls"""
        wall_h = 2.0
        w_thick = 0.3
        blue = [0.2, 0.2, 0.8, 1]  # blue walls for visibility
        
        walls = [
            ([self.grid_size/2, self.grid_size + w_thick/2, wall_h/2],
             [self.grid_size/2 + w_thick, w_thick/2, wall_h/2]),
            ([self.grid_size/2, -w_thick/2, wall_h/2],
             [self.grid_size/2 + w_thick, w_thick/2, wall_h/2]),
            ([self.grid_size + w_thick/2, self.grid_size/2, wall_h/2],
             [w_thick/2, self.grid_size/2 + w_thick, wall_h/2]),
            ([-w_thick/2, self.grid_size/2, wall_h/2],
             [w_thick/2, self.grid_size/2 + w_thick, wall_h/2]),
        ]
        
        for pos, halfext in walls:
            col = p.createCollisionShape(p.GEOM_BOX, halfExtents=halfext)
            vis = p.createVisualShape(p.GEOM_BOX, halfExtents=halfext, rgbaColor=blue)
            wall = p.createMultiBody(0, col, vis, pos)
            self.wall_ids.append(wall)
    
    def _create_city_roads(self):
        """Create city with LOW buildings"""
        road_w = 2.0
        road_h = 0.05
        black = [0.1, 0.1, 0.1, 1]

        num_hroads = 3
        num_vroads = 3

        h_y = np.linspace(5, self.grid_size - 5, num_hroads)
        v_x = np.linspace(5, self.grid_size - 5, num_vroads)

        # ---- Roads ----
        for y in h_y:
            col = p.createCollisionShape(
                p.GEOM_BOX, halfExtents=[self.grid_size/2, road_w/2, road_h/2]
            )
            vis = p.createVisualShape(
                p.GEOM_BOX, halfExtents=[self.grid_size/2, road_w/2, road_h/2],
                rgbaColor=black
            )
            road_id = p.createMultiBody(0, col, vis, [self.grid_size/2, y, road_h/2])
            self.road_tile_ids.append(road_id)

        for x in v_x:
            col = p.createCollisionShape(
                p.GEOM_BOX, halfExtents=[road_w/2, self.grid_size/2, road_h/2]
            )
            vis = p.createVisualShape(
                p.GEOM_BOX, halfExtents=[road_w/2, self.grid_size/2, road_h/2],
                rgbaColor=black
            )
            road_id = p.createMultiBody(0, col, vis, [x, self.grid_size/2, road_h/2])
            self.road_tile_ids.append(road_id)

        # Add yellow lane dots after roads are created
        self._add_lane_markers(h_y, v_x, road_h)

        # ---- LOW colorful buildings ----
        colors = [
            [0.8, 0.7, 0.6, 1],
            [0.7, 0.7, 0.7, 1],
            [0.9, 0.8, 0.6, 1],
            [0.8, 0.6, 0.6, 1],
        ]

        h_bounds = [0] + list(h_y) + [self.grid_size]
        v_bounds = [0] + list(v_x) + [self.grid_size]

        for i in range(len(v_bounds) - 1):
            for j in range(len(h_bounds) - 1):
                cx0, cx1 = v_bounds[i], v_bounds[i+1]
                cy0, cy1 = h_bounds[j], h_bounds[j+1]

                bx = (cx0 + cx1) / 2
                by = (cy0 + cy1) / 2
                sx = max((cx1 - cx0)/2 - 2.0, 1.0)
                sy = max((cy1 - cy0)/2 - 2.0, 1.0)
                bh = np.random.uniform(1.0, 1.0)

                col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[sx, sy, bh/2])
                color = colors[np.random.randint(len(colors))]
                vis = p.createVisualShape(
                    p.GEOM_BOX, halfExtents=[sx, sy, bh/2], rgbaColor=color
                )
                b_id = p.createMultiBody(0, col, vis, [bx, by, bh/2])
                self.building_ids.append(b_id)
   
    def _add_lane_markers(self, h_y, v_x, road_h):
        """yellow lane dots along the center of each road."""
        lane_h = 0.03
        dot_half = 0.15
        yellow = [1.0, 1.0, 0.0, 1.0]

        z = road_h + lane_h / 2 + 1e-3   # just above road top

        # Horizontal roads: vary x, fixed y
        for y in h_y:
            for x in np.linspace(1.0, self.grid_size - 1.0, 8):
                col = p.createCollisionShape(
                    p.GEOM_BOX,
                    halfExtents=[dot_half, 0.03, lane_h / 2],
                )
                vis = p.createVisualShape(
                    p.GEOM_BOX,
                    halfExtents=[dot_half, 0.03, lane_h / 2],
                    rgbaColor=yellow,
                )
                lm_id = p.createMultiBody(
                    baseMass=0,
                    baseCollisionShapeIndex=col,
                    baseVisualShapeIndex=vis,
                    basePosition=[x, y, z],
                )
                self.lane_marker_ids.append(lm_id)

        # Vertical roads: vary y, fixed x
        for x in v_x:
            for y in np.linspace(1.0, self.grid_size - 1.0, 8):
                col = p.createCollisionShape(
                    p.GEOM_BOX,
                    halfExtents=[0.03, dot_half, lane_h / 2],
                )
                vis = p.createVisualShape(
                    p.GEOM_BOX,
                    halfExtents=[0.03, dot_half, lane_h / 2],
                    rgbaColor=yellow,
                )
                lm_id = p.createMultiBody(
                    baseMass=0,
                    baseCollisionShapeIndex=col,
                    baseVisualShapeIndex=vis,
                    basePosition=[x, y, z],
                )
                self.lane_marker_ids.append(lm_id) 

    def _create_robot(self, pos):
        """Create robot"""
        robot_h = 0.5
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.4, 0.3, robot_h/2])
        vis = p.createVisualShape(
            p.GEOM_BOX, halfExtents=[0.4, 0.3, robot_h/2],
            rgbaColor=[0.2, 0.4, 0.9, 1]  # Blue
        )
        self.robot_id = p.createMultiBody(1.0, col, vis, [pos[0], pos[1], robot_h/2])
        p.changeDynamics(self.robot_id, -1, lateralFriction=0.8)
    
    def _create_goal(self, pos):
        """Create BRIGHT visible goal"""
        # Large glowing platform
        col = p.createCollisionShape(p.GEOM_CYLINDER, radius=0.6, height=0.2)
        vis = p.createVisualShape(
            p.GEOM_CYLINDER, radius=0.6, length=0.2,
            rgbaColor=[0.0, 1.0, 0.0, 1]  # BRIGHT GREEN
        )
        self.goal_id = p.createMultiBody(0, col, vis, [pos[0], pos[1], 0.1])
        
        # Tall flag pole
        pole_vis = p.createVisualShape(
            p.GEOM_CYLINDER, radius=0.06, length=3.0,
            rgbaColor=[1, 1, 0, 1]
        )
        pole_id = p.createMultiBody(0, -1, pole_vis, [pos[0], pos[1], 1.5])
        self.goal_extra_ids.append(pole_id)

        # Flag at top
        flag_vis = p.createVisualShape(
            p.GEOM_BOX, halfExtents=[0.4, 0.02, 0.3],
            rgbaColor=[1, 0, 0, 1]
        )
        flag_id = p.createMultiBody(0, -1, flag_vis, [pos[0] + 0.2, pos[1], 2.7])
        self.goal_extra_ids.append(flag_id)
    
    def _create_cone_shape(self, radius, height, num_base_points=16):
        """
        Creates a cone by providing vertices. 
        Uses GEOM_MESH for collision and visual shapes.
        """
        vertices = []
        # 1. Generate vertices for the base circle
        for i in range(num_base_points):
            angle = 2 * np.pi * i / num_base_points
            vertices.append([radius * np.cos(angle), radius * np.sin(angle), 0])
        
        # 2. Add the apex point
        vertices.append([0, 0, height])
        
        # Create the indices for the faces (Required for GEOM_MESH visual)
        indices = []
        # Base faces (triangle fan)
        for i in range(1, num_base_points - 1):
            indices.extend([0, i, i + 1])
        # Side faces (connecting base to apex)
        apex_idx = num_base_points
        for i in range(num_base_points):
            indices.extend([i, (i + 1) % num_base_points, apex_idx])

        # Collision shape works fine with just vertices
        col = p.createCollisionShape(shapeType=p.GEOM_MESH, vertices=vertices)
        
        # Visual shape is pickier; we provide vertices AND indices
        vis = p.createVisualShape(
            shapeType=p.GEOM_MESH, 
            vertices=vertices, 
            indices=indices,
            rgbaColor=[1.0, 0.5, 0.0, 1]
        )
        
        return col, vis

    def _create_obstacles(self):
        """Create BRIGHT orange traffic cones with properly aligned white stripes"""
        
        # 1. Base Cone Geometry
        h = 0.4
        r = 0.15
        col_id, vis_id = self._create_cone_shape(r, h)
        
        # 2. Define two white stripes with different radii to match cone taper
        stripe_low_vis = p.createVisualShape(
            p.GEOM_CYLINDER, radius=r * 0.75, length=0.05, rgbaColor=[1, 1, 1, 1]
        )
        stripe_high_vis = p.createVisualShape(
            p.GEOM_CYLINDER, radius=r * 0.4, length=0.04, rgbaColor=[1, 1, 1, 1]
        )
        
        stripe_orn = [0, 0, 0, 1] 

        for _ in range(self.num_obstacles):
            road_id = np.random.choice(self.road_tile_ids)
            pos, _ = p.getBasePositionAndOrientation(road_id)
            ox = pos[0] + np.random.uniform(-1.0, 1.0)
            oy = pos[1] + np.random.uniform(-1.0, 1.0)

            # Multi-link setup
            link_masses = [0.01, 0.01]
            link_collision_indices = [-1, -1] 
            link_visual_indices = [stripe_low_vis, stripe_high_vis]
            
            link_positions = [[0, 0, 0.12], [0, 0, 0.26]] 
            link_orientations = [stripe_orn, stripe_orn]
            
            link_parent_indices = [0, 0]
            link_joint_types = [p.JOINT_FIXED, p.JOINT_FIXED]
            
            link_inertial_pos = [[0, 0, 0], [0, 0, 0]]
            link_inertial_orn = [[0, 0, 0, 1], [0, 0, 0, 1]]
            link_joint_axes = [[0, 0, 0], [0, 0, 0]]

            obs_id = p.createMultiBody(
                baseMass=0,  # CHANGED: 0 mass = static/immovable!
                baseCollisionShapeIndex=col_id,
                baseVisualShapeIndex=vis_id,
                basePosition=[ox, oy, 0.0],  # CHANGED: Place exactly on ground (was 0.05)
                linkMasses=link_masses,
                linkCollisionShapeIndices=link_collision_indices,
                linkVisualShapeIndices=link_visual_indices,
                linkPositions=link_positions,
                linkOrientations=link_orientations,
                linkInertialFramePositions=link_inertial_pos,
                linkInertialFrameOrientations=link_inertial_orn,
                linkParentIndices=link_parent_indices,
                linkJointTypes=link_joint_types,
                linkJointAxis=link_joint_axes
            )
            
            # No dynamics changes needed for static objects
            self.obstacle_ids.append(obs_id)

    def _get_camera_image(self):
        """Get first-person view from robot's camera"""
        pos, orn = p.getBasePositionAndOrientation(self.robot_id)
        rot_matrix = p.getMatrixFromQuaternion(orn)

        camera_height = 0.4
        camera_forward = 0.3

        cam_pos = [
            pos[0] + rot_matrix[0] * camera_forward,
            pos[1] + rot_matrix[3] * camera_forward,
            pos[2] + camera_height,
        ]

        forward = [rot_matrix[0], rot_matrix[3], rot_matrix[6]]
        target = [
            cam_pos[0] + forward[0] * 100,
            cam_pos[1] + forward[1] * 100,
            cam_pos[2] + forward[2] * 100,
        ]
        up = [rot_matrix[2], rot_matrix[5], rot_matrix[8]]

        view_matrix = p.computeViewMatrix(cam_pos, target, up)
        proj_matrix = p.computeProjectionMatrixFOV(
            fov=90,
            aspect=self.camera_width / self.camera_height,
            nearVal=0.1,
            farVal=100.0,
        )

        width, height, rgb, depth, seg = p.getCameraImage(
            self.camera_width,
            self.camera_height,
            view_matrix,
            proj_matrix,
            renderer=p.ER_TINY_RENDERER,
        )

        rgb_array = np.array(rgb, dtype=np.uint8).reshape((height, width, 4))[:, :, :3]
        bgr = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
        return bgr

    
    def _position_on_road(self):
        """Get position on road"""
        road_id = np.random.choice(self.road_tile_ids)
        pos, _ = p.getBasePositionAndOrientation(road_id)
        return np.array([pos[0], pos[1]])
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        
        # disconnect and reconnect PyBullet
        if self.physics_client is not None:
            try:
                p.disconnect(self.physics_client)
            except:
                pass
            self.physics_client = None
        
        # Clear all tracking lists
        self.robot_id = None
        self.goal_id = None
        self.goal_extra_ids = []
        self.road_tile_ids = []
        self.building_ids = []
        self.obstacle_ids = []
        self.lane_marker_ids = []
        self.wall_ids = []

        # Reinitialize fresh
        self._init_pybullet()
        
        # Create new environment
        self._create_city_roads()
        self._create_obstacles()
        
        robot_pos = self._position_on_road()
        goal_pos = self._position_on_road()
        
        # reasonable distance
        max_tries = 50
        min_dist = self.grid_size * 0.3
        for _ in range(max_tries):
            if np.linalg.norm(robot_pos - goal_pos) >= min_dist:
                break
            goal_pos = self._position_on_road()
        
        self._create_robot(robot_pos)
        self._create_goal(goal_pos)
        
        # DISTANCE-BASED episode horizon: longer episodes for farther goals
        initial_dist = float(np.linalg.norm(robot_pos - goal_pos))
        self.max_steps = int(
            np.clip(
                self.base_max_steps + self.steps_per_meter * initial_dist,
                self.min_episode_steps,
                self.max_episode_steps
            )
        )
        
        # Settle physics
        for _ in range(30):
            p.stepSimulation()
        
        return self._get_obs(), {}
    
    def _get_obs(self):
        """Get observation"""
        if self.use_camera_obs:
            img = self._get_camera_image()
            img_small = cv2.resize(img, (84, 84))
            return img_small
        else:
            pos, orn = p.getBasePositionAndOrientation(self.robot_id)
            yaw = p.getEulerFromQuaternion(orn)[2]
            gpos, _ = p.getBasePositionAndOrientation(self.goal_id)
            return np.array([pos[0], pos[1], yaw, gpos[0], gpos[1]], dtype=np.float32)
    
    def _draw_minimap(self, size=200):
        """Return a top-down mini-map image"""
        minimap = np.full((size, size, 3), 255, dtype=np.uint8)

        sx = size / float(self.grid_size)
        sy = size / float(self.grid_size)

        def world_to_px(x, y):
            px = int(x * sx)
            py = size - int(y * sy)
            return px, py

        cv2.rectangle(
            minimap,
            world_to_px(0, 0),
            world_to_px(self.grid_size, self.grid_size),
            (0, 0, 0),
            2,
        )

        for bid in self.building_ids:
            pos, _ = p.getBasePositionAndOrientation(bid)
            x, y = pos[0], pos[1]
            px, py = world_to_px(x, y)
            cv2.rectangle(
                minimap,
                (px - 4, py - 4),
                (px + 4, py + 4),
                (180, 180, 180),
                -1,
            )

        for oid in self.obstacle_ids:
            pos, _ = p.getBasePositionAndOrientation(oid)
            x, y = pos[0], pos[1]
            px, py = world_to_px(x, y)
            cv2.circle(minimap, (px, py), 3, (0, 140, 255), -1)

        if self.robot_id is not None:
            rpos, rorn = p.getBasePositionAndOrientation(self.robot_id)
            rx, ry = rpos[0], rpos[1]
            yaw = p.getEulerFromQuaternion(rorn)[2]

            rpx, rpy = world_to_px(rx, ry)

            tri_world_size = 0.9
            tri_px_size = int(tri_world_size * (size / self.grid_size))

            tip_x = rpx + tri_px_size * np.cos(yaw)
            tip_y = rpy - tri_px_size * np.sin(yaw)

            left_x = rpx + tri_px_size * 0.5 * np.cos(yaw + np.pi * 0.75)
            left_y = rpy - tri_px_size * 0.5 * np.sin(yaw + np.pi * 0.75)

            right_x = rpx + tri_px_size * 0.5 * np.cos(yaw - np.pi * 0.75)
            right_y = rpy - tri_px_size * 0.5 * np.sin(yaw - np.pi * 0.75)

            pts = np.array([
                [int(tip_x), int(tip_y)],
                [int(left_x), int(left_y)],
                [int(right_x), int(right_y)]
            ])

            cv2.fillConvexPoly(minimap, pts, (0, 0, 255))
            cv2.polylines(minimap, [pts], True, (0, 0, 150), 2)

        if self.goal_id is not None:
            gpos, _ = p.getBasePositionAndOrientation(self.goal_id)
            gx, gy = gpos[0], gpos[1]
            gpx, gpy = world_to_px(gx, gy)
            cv2.circle(minimap, (gpx, gpy), 6, (0, 255, 0), -1)
            cv2.circle(minimap, (gpx, gpy), 8, (0, 255, 0), 2)

        if self.robot_id is not None and self.goal_id is not None:
            cv2.line(minimap, (rpx, rpy), (gpx, gpy), (255, 0, 255), 1)

        return minimap

    def render(self):
        """Show robot's camera view in window"""
        if self.render_mode == "human":
            camera_img = self._get_camera_image()
            img_with_hud = camera_img.copy()

            dist = self._goal_dist()
            cv2.putText(img_with_hud, f"Distance to Goal: {dist:.1f}m",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 255, 0), 2)

            cv2.putText(img_with_hud, f"Step: {self.step_count}",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 255, 0), 2)

            h, w = img_with_hud.shape[:2]
            cv2.line(img_with_hud, (w//2 - 20, h//2), (w//2 + 20, h//2),
                     (0, 255, 0), 2)
            cv2.line(img_with_hud, (w//2, h//2 - 20), (w//2, h//2 + 20),
                     (0, 255, 0), 2)

            minimap = self._draw_minimap(size=180)
            mh, mw = minimap.shape[:2]
            if mh + 10 < h and mw + 10 < w:
                y0 = 10
                y1 = y0 + mh
                x1 = w - 10
                x0 = x1 - mw
                img_with_hud[y0:y1, x0:x1] = minimap

            cv2.imshow(self.camera_window_name, img_with_hud)
            cv2.waitKey(1)

        return None

    
    def step(self, action):
        self.step_count += 1
        
        pos, orn = p.getBasePositionAndOrientation(self.robot_id)
        yaw = p.getEulerFromQuaternion(orn)[2]
        old_dist = self._goal_dist()
        
        # Apply action
        if action == 0:  # forward
            vx = self.linear_speed * np.cos(yaw)
            vy = self.linear_speed * np.sin(yaw)
            wz = 0
        elif action == 1:  # turn left (ARC - move while turning)
            vx = 0.6 * self.linear_speed * np.cos(yaw)
            vy = 0.6 * self.linear_speed * np.sin(yaw)
            wz = self.angular_speed
        elif action == 2:  # turn right (ARC - move while turning)
            vx = 0.6 * self.linear_speed * np.cos(yaw)
            vy = 0.6 * self.linear_speed * np.sin(yaw)
            wz = -self.angular_speed
        else:  # reverse
            vx = -0.5 * self.linear_speed * np.cos(yaw)
            vy = -0.5 * self.linear_speed * np.sin(yaw)
            wz = 0
        
        p.resetBaseVelocity(
            self.robot_id,
            linearVelocity=[vx, vy, 0],
            angularVelocity=[0, 0, wz]
        )
        
        # Simulate
        for _ in range(5):
            p.stepSimulation()
            
            pos, orn = p.getBasePositionAndOrientation(self.robot_id)
            euler = p.getEulerFromQuaternion(orn)
            new_orn = p.getQuaternionFromEuler([0, 0, euler[2]])
            p.resetBasePositionAndOrientation(self.robot_id, pos, new_orn)

            if self.render_mode == "human":
                time.sleep(1/240)
        
        # Calculate reward
        new_dist = self._goal_dist()
        distance_reward = (old_dist - new_dist) * 15.0
        time_penalty = -0.01
        progress_bonus = 1.0 if (old_dist - new_dist) > 0.01 else 0
        reward = distance_reward + time_penalty + progress_bonus
        
        if self.render_mode == "human":
            self.render()
        
        obs = self._get_obs()
        collided = self._check_collision()

        if collided:
            return obs, reward - 50, True, False, {"success": False, "collision": True}

        if new_dist < 0.8:
            return obs, reward + 100, True, False, {"success": True, "collision": False}

        truncated = self.step_count >= self.max_steps
        return obs, reward, False, truncated, {"success": False, "collision": False}
    
    def _check_collision(self):
        contacts = p.getContactPoints(bodyA=self.robot_id)
        for c in contacts:
            bodyA = c[1]
            bodyB = c[2]
            other = bodyB if bodyA == self.robot_id else bodyA

            if other in self.building_ids or other in self.obstacle_ids or other in self.wall_ids:
                return True
        return False
        
    def _goal_dist(self):
        pos, _ = p.getBasePositionAndOrientation(self.robot_id)
        gpos, _ = p.getBasePositionAndOrientation(self.goal_id)
        return np.linalg.norm([pos[0] - gpos[0], pos[1] - gpos[1]])
    
    def set_num_obstacles(self, n: int):
        """Set number of obstacles (for curriculum learning)"""
        self.num_obstacles = int(n)
    
    def close(self):
        cv2.destroyAllWindows()
        if self.physics_client is not None:
            p.disconnect(self.physics_client)
            self.physics_client = None

class RobotPOVContinuousEnv(RobotPOVEnv):
    """
    CONTINUOUS action space for smooth control
    IMPROVED reward shaping for better learning
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # overwrite action space: [steer, throttle] in [-1, 1]
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )
    
    def step(self, action):
        """
        IMPROVED step function with COMPREHENSIVE reward shaping
        
        Key improvements:
        - Stronger distance reward (20x instead of 15x)
        - Larger time penalty to encourage efficiency
        - GRADUATED collision penalty (lenient early, harsh late)
        - Survival bonus for staying alive
        - Larger success bonus
        - Proximity bonus when getting close
        """
        action = np.clip(action, self.action_space.low, self.action_space.high)
        steer = float(action[0])     # -1 left, +1 right
        throttle = float(action[1])  # -1 reverse, +1 forward

        self.step_count += 1

        pos, orn = p.getBasePositionAndOrientation(self.robot_id)
        yaw = p.getEulerFromQuaternion(orn)[2]
        old_dist = self._goal_dist()

        # map to velocities
        max_lin = self.linear_speed
        max_ang = self.angular_speed

        vx = max_lin * throttle * np.cos(yaw)
        vy = max_lin * throttle * np.sin(yaw)
        wz = max_ang * steer

        p.resetBaseVelocity(
            self.robot_id,
            linearVelocity=[vx, vy, 0],
            angularVelocity=[0, 0, wz],
        )

        
        for _ in range(5):
            p.stepSimulation()
            
            pos, orn = p.getBasePositionAndOrientation(self.robot_id)
            euler = p.getEulerFromQuaternion(orn)
            new_orn = p.getQuaternionFromEuler([0, 0, euler[2]])
            p.resetBasePositionAndOrientation(self.robot_id, pos, new_orn)
            
            if self.render_mode == "human":
                time.sleep(1/240)

        # COMPREHENSIVE REWARD SHAPING
        new_dist = self._goal_dist()
        
        # 1. Stronger distance reward - INCREASED from 15.0 to 20.0
        distance_reward = (old_dist - new_dist) * 20.0
        
        # 2. Survival bonus - small constant reward for staying alive
        survival_bonus = 0.05
        
        # 3. Progress bonus - encourage any forward movement
        progress_bonus = 2.0 if (old_dist - new_dist) > 0.01 else 0
        
        # 4. Proximity bonus when getting close to goal
        if new_dist < 3.0:
            proximity_bonus = 5.0 / (new_dist + 0.1)
        else:
            proximity_bonus = 0
        
        # 5. Time penalty - INCREASED for distance-based horizons
        # With longer episodes (up to 700 steps), need stronger time pressure
        time_penalty = -0.02  # Was -0.01, now stronger
        
        # 6. NEW: Action penalty to discourage backward movement
        action_penalty = 0
        if throttle < -0.1:  # Moving backward
            # Soft linear + harsh quadratic = gentle reverse OK, strong reverse BAD
            action_penalty -= 0.3 * (-throttle)          # mild linear
            action_penalty -= 1.2 * ((-throttle) ** 2)   # harsher quadratic
        # NOTE: Removed forward bonus - was encouraging crashing into obstacles
        
        # Penalize excessive turning while barely moving
        if abs(throttle) < 0.2 and abs(steer) > 0.5:
            action_penalty -= 0.3  # Discourage spinning in place
        
        # NEW: Smarter standing still penalty - only punish if NOT making progress
        delta = old_dist - new_dist  # positive = got closer
        if abs(throttle) < 0.15 and abs(steer) < 0.15 and delta < 0.002:
            action_penalty -= 0.15  # Smaller penalty, only when truly stuck
        
        # Base reward
        reward = (distance_reward + survival_bonus + proximity_bonus + 
                  progress_bonus + time_penalty + action_penalty)

        if self.render_mode == "human":
            self.render()

        obs = self._get_obs()
        collided = self._check_collision()

        if collided:
            # GRADUATED collision penalty: lenient early, harsh later
            training_progress = self.step_count / self.max_steps
            collision_penalty = -20 - (80 * training_progress)  # -20 to -100
            return obs, reward + collision_penalty, True, False, {"success": False, "collision": True}

        if new_dist < 0.8:
            # INCREASED success bonus from +100 to +200
            return obs, reward + 200, True, False, {"success": True, "collision": False}
        
        truncated = self.step_count >= self.max_steps
        return obs, reward, False, truncated, {"success": False, "collision": False}