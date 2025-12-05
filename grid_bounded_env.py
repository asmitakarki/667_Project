# ============================================================
# Clean Grid/Spiral Road Environment for PyBullet
# Two map types: "spiral" and "city"
# Roads are black, buildings fill blocks, robot must stay on road.
# Medium road width. Small obstacles avoidable, big ones block.
# ============================================================

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pybullet as p
import pybullet_data
import time
import math


class GridBoundedEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        grid_size=20,
        map_type="spiral",
        render_mode=None,
        num_obstacles=5,
        **kwargs
    ):
        super().__init__()

        self.grid_size = grid_size
        self.map_type = map_type
        self.render_mode = render_mode
        self.num_obstacles = num_obstacles

        # internal bookkeeping
        self.physics_client = None
        self.robot_id = None
        self.goal_id = None
        self.road_tile_ids = []
        self.building_ids = []
        self.extra_body_ids = []
        self.obstacle_ids = []
        self.wall_ids = []

        # movement parameters
        self.linear_speed = 2.5
        self.angular_speed = 2.0

        # Discrete actions
        self.action_space = spaces.Discrete(4)
        # obs: [robot_x, robot_y, yaw, goal_x, goal_y]
        self.observation_space = spaces.Box(
            low=np.array([0, 0, -np.pi, 0, 0], dtype=np.float32),
            high=np.array([grid_size, grid_size, np.pi, grid_size, grid_size], dtype=np.float32),
        )

        self.max_steps = grid_size * 15
        self.step_count = 0

        # initialize Bullet
        self._init_pybullet()

    # ------------------------------------------------------------
    # PyBullet setup
    # ------------------------------------------------------------
    def _init_pybullet(self):
        if self.physics_client is not None:
            p.disconnect(self.physics_client)

        if self.render_mode == "human":
            self.physics_client = p.connect(p.GUI)
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        else:
            self.physics_client = p.connect(p.DIRECT)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setRealTimeSimulation(0)
        p.setPhysicsEngineParameter(fixedTimeStep=1.0 / 240.0)

        # floor + walls
        self._create_tiled_floor()
        self._create_boundary_walls()

        # camera
        p.resetDebugVisualizerCamera(
            cameraDistance=self.grid_size * 1.2,
            cameraYaw=45,
            cameraPitch=-50,
            cameraTargetPosition=[self.grid_size / 2, self.grid_size / 2, 0],
        )

    # ------------------------------------------------------------
    def _create_tiled_floor(self):
        tile_h = 0.05
        color1 = [0.88, 0.85, 0.82, 1]
        color2 = [0.82, 0.79, 0.76, 1]

        for i in range(self.grid_size):
            for j in range(self.grid_size):
                color = color1 if ((i + j) % 2 == 0) else color2
                col = p.createCollisionShape(
                    p.GEOM_BOX, halfExtents=[0.5, 0.5, tile_h / 2]
                )
                vis = p.createVisualShape(
                    p.GEOM_BOX, halfExtents=[0.5, 0.5, tile_h / 2], rgbaColor=color
                )
                p.createMultiBody(
                    baseMass=0,
                    baseCollisionShapeIndex=col,
                    baseVisualShapeIndex=vis,
                    basePosition=[i + 0.5, j + 0.5, tile_h / 2],
                )

    # ------------------------------------------------------------
    def _create_boundary_walls(self):
        wall_h = 2.0
        w_thick = 0.2
        gray = [0.4, 0.4, 0.4, 1]

        walls = [
            # NORTH
            ([self.grid_size / 2, self.grid_size + w_thick / 2, wall_h / 2],
             [self.grid_size / 2 + w_thick, w_thick / 2, wall_h / 2]),
            # SOUTH
            ([self.grid_size / 2, -w_thick / 2, wall_h / 2],
             [self.grid_size / 2 + w_thick, w_thick / 2, wall_h / 2]),
            # EAST
            ([self.grid_size + w_thick / 2, self.grid_size / 2, wall_h / 2],
             [w_thick / 2, self.grid_size / 2 + w_thick, wall_h / 2]),
            # WEST
            ([-w_thick / 2, self.grid_size / 2, wall_h / 2],
             [w_thick / 2, self.grid_size / 2 + w_thick, wall_h / 2]),
        ]

        for pos, halfext in walls:
            col = p.createCollisionShape(p.GEOM_BOX, halfExtents=halfext)
            vis = p.createVisualShape(
                p.GEOM_BOX, halfExtents=halfext, rgbaColor=gray
            )
            wall = p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=col,
                baseVisualShapeIndex=vis,
                basePosition=pos,
            )
            self.wall_ids.append(wall)

    # ------------------------------------------------------------
    # ROAD SYSTEMS
    # ------------------------------------------------------------

    # ---------------------------
    # Spiral black road
    # ---------------------------
    def _create_spiral_road(self):
        road_w = 2.0
        thickness = 0.04
        black = [0.1, 0.1, 0.1, 1]

        loops = 2.2
        b = 0.5 # controls spacing between loops, larger = more spaced out
        a = 1.0
        num_pts = 200

        theta = np.linspace(0, loops * np.pi * 2, num_pts)
        r = a + b * theta
        x = self.grid_size / 2 + r * np.cos(theta)
        y = self.grid_size / 2 + r * np.sin(theta)

        for i in range(num_pts - 1):
            p1 = np.array([x[i], y[i]])
            p2 = np.array([x[i + 1], y[i + 1]])

            mid = (p1 + p2) / 2
            seg_len = np.linalg.norm(p2 - p1)
            if seg_len < 1e-3:
                continue

            yaw = math.atan2(p2[1] - p1[1], p2[0] - p1[0])

            col = p.createCollisionShape(
                p.GEOM_BOX, halfExtents=[seg_len / 2, road_w / 2, thickness / 2]
            )
            vis = p.createVisualShape(
                p.GEOM_BOX,
                halfExtents=[seg_len / 2, road_w / 2, thickness / 2],
                rgbaColor=black,
            )

            road_id = p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=col,
                baseVisualShapeIndex=vis,
                basePosition=[mid[0], mid[1], thickness / 2 + 0.01],
                baseOrientation=p.getQuaternionFromEuler([0, 0, yaw]),
            )
            self.road_tile_ids.append(road_id)

    # ---------------------------
    # Manhattan City Road Grid
    # ---------------------------
    def _create_city_block_map(self):
        road_w = 2.0
        road_h = 0.05
        black = [0.07, 0.07, 0.07, 1]

        num_hroads = 3
        num_vroads = 3

        # evenly space roads
        h_y = np.linspace(3, self.grid_size - 3, num_hroads)
        v_x = np.linspace(3, self.grid_size - 3, num_vroads)

        # 1) Create horizontal roads
        for y in h_y:
            col = p.createCollisionShape(
                p.GEOM_BOX,
                halfExtents=[self.grid_size / 2, road_w / 2, road_h / 2],
            )
            vis = p.createVisualShape(
                p.GEOM_BOX,
                halfExtents=[self.grid_size / 2, road_w / 2, road_h / 2],
                rgbaColor=black,
            )
            road_id = p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=col,
                baseVisualShapeIndex=vis,
                basePosition=[self.grid_size / 2, y, road_h / 2 + 0.02],
            )
            self.road_tile_ids.append(road_id)

        # 2) Vertical roads
        for x in v_x:
            col = p.createCollisionShape(
                p.GEOM_BOX,
                halfExtents=[road_w / 2, self.grid_size / 2, road_h / 2],
            )
            vis = p.createVisualShape(
                p.GEOM_BOX,
                halfExtents=[road_w / 2, self.grid_size / 2, road_h / 2],
                rgbaColor=black,
            )
            road_id = p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=col,
                baseVisualShapeIndex=vis,
                basePosition=[x, self.grid_size / 2, road_h / 2 + 0.02],
            )
            self.road_tile_ids.append(road_id)

        # 3) Create buildings in the blocks
        gray = [0.6, 0.6, 0.6, 1]

        # derive intervals between roads
        h_bounds = [0] + list(h_y) + [self.grid_size]
        v_bounds = [0] + list(v_x) + [self.grid_size]

        for i in range(len(v_bounds) - 1):
            for j in range(len(h_bounds) - 1):
                cx0 = v_bounds[i]
                cx1 = v_bounds[i + 1]
                cy0 = h_bounds[j]
                cy1 = h_bounds[j + 1]

                # Skip space occupied by roads (within road_w margin)
                if (cx0 < min(v_x) + road_w and cx1 > min(v_x) - road_w):
                    pass
                if (cy0 < min(h_y) + road_w and cy1 > min(h_y) - road_w):
                    pass

                # building center & size
                bx = (cx0 + cx1) / 2
                by = (cy0 + cy1) / 2
                sx = max((cx1 - cx0) / 2 - 1.0, 0.8)
                sy = max((cy1 - cy0) / 2 - 1.0, 0.8)
                bh = np.random.uniform(1.5, 4.0)

                col = p.createCollisionShape(
                    p.GEOM_BOX, halfExtents=[sx, sy, bh / 2]
                )
                vis = p.createVisualShape(
                    p.GEOM_BOX, halfExtents=[sx, sy, bh / 2], rgbaColor=gray
                )
                b_id = p.createMultiBody(
                    baseMass=0,
                    baseCollisionShapeIndex=col,
                    baseVisualShapeIndex=vis,
                    basePosition=[bx, by, bh / 2 + 0.02],
                )
                self.building_ids.append(b_id)

    # ------------------------------------------------------------
    def _create_goal(self, pos):
        col = p.createCollisionShape(p.GEOM_CYLINDER, radius=0.3, height=0.2)
        vis = p.createVisualShape(
            p.GEOM_CYLINDER,
            radius=0.3,
            length=0.2,
            rgbaColor=[1.0, 0.6, 0.6, 0.8],
        )
        self.goal_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=col,
            baseVisualShapeIndex=vis,
            basePosition=[pos[0], pos[1], 0.15],
        )

    # ------------------------------------------------------------
    def _create_obstacles_on_roads(self):
        # spawn small cylinders along road tiles
        for _ in range(self.num_obstacles):
            road_body = np.random.choice(self.road_tile_ids)
            pos, orn = p.getBasePositionAndOrientation(road_body)
            # add slight jitter
            dx = np.random.uniform(-0.5, 0.5)
            dy = np.random.uniform(-0.5, 0.5)
            ox = pos[0] + dx
            oy = pos[1] + dy

            h = 0.3
            col = p.createCollisionShape(
                p.GEOM_CYLINDER, radius=0.3, height=h
            )
            vis = p.createVisualShape(
                p.GEOM_CYLINDER,
                radius=0.3,
                length=h,
                rgbaColor=[1.0, 0.7, 0.7, 1],
            )
            oid = p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=col,
                baseVisualShapeIndex=vis,
                basePosition=[ox, oy, h / 2],
            )
            self.obstacle_ids.append(oid)

    # ------------------------------------------------------------
    def _position_on_road(self):
        """Pick random road tile center"""
        rid = np.random.choice(self.road_tile_ids)
        pos, _ = p.getBasePositionAndOrientation(rid)
        return np.array([pos[0], pos[1]])

    # ------------------------------------------------------------
    def _on_road(self, x, y):
        """Check if robot is on road by measuring distance to nearest road tile."""
        for rid in self.road_tile_ids:
            pos, orn = p.getBasePositionAndOrientation(rid)
            rp = np.array([pos[0], pos[1]])
            if np.linalg.norm([x - rp[0], y - rp[1]]) < 1.3:  # near road center
                return True
        return False

    # ------------------------------------------------------------
    # ROBOT
    # ------------------------------------------------------------
    def _create_robot(self, pos):
        # simple box for now
        col = p.createCollisionShape(
            p.GEOM_BOX, halfExtents=[0.3, 0.2, 0.15]
        )
        vis = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[0.3, 0.2, 0.15],
            rgbaColor=[0.9, 0.2, 0.2, 1],
        )
        self.robot_id = p.createMultiBody(
            baseMass=1.0,
            baseCollisionShapeIndex=col,
            baseVisualShapeIndex=vis,
            basePosition=[pos[0], pos[1], 0.2],
        )

    # ------------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.step_count = 0

        # cleanup
        for l in [
            self.robot_id,
            self.goal_id,
            *self.road_tile_ids,
            *self.building_ids,
            *self.obstacle_ids,
        ]:
            if l:
                try:
                    p.removeBody(l)
                except:
                    pass

        self.robot_id = None
        self.goal_id = None
        self.road_tile_ids = []
        self.building_ids = []
        self.obstacle_ids = []

        # build map
        if self.map_type == "spiral":
            self._create_spiral_road()
        elif self.map_type == "city":
            self._create_city_block_map()
        else:
            raise ValueError("map_type must be 'spiral' or 'city'")

        # obstacles
        self._create_obstacles_on_roads()

        # pick start + goal
        robot_pos = self._position_on_road()
        goal_pos = self._position_on_road()
        self._create_robot(robot_pos)
        self._create_goal(goal_pos)

        for _ in range(30):
            p.stepSimulation()

        return self._get_obs(), {}

    # ------------------------------------------------------------
    def _get_obs(self):
        pos, orn = p.getBasePositionAndOrientation(self.robot_id)
        yaw = p.getEulerFromQuaternion(orn)[2]
        gpos, _ = p.getBasePositionAndOrientation(self.goal_id)

        return np.array(
            [pos[0], pos[1], yaw, gpos[0], gpos[1]], dtype=np.float32
        )

    # ------------------------------------------------------------
    def step(self, action):
        self.step_count += 1
        pos, orn = p.getBasePositionAndOrientation(self.robot_id)
        yaw = p.getEulerFromQuaternion(orn)[2]

        old_dist = self._goal_dist()

        # compute velocities
        if action == 0:
            vx = self.linear_speed * np.cos(yaw)
            vy = self.linear_speed * np.sin(yaw)
            wz = 0
        elif action == 1:
            vx = vy = 0
            wz = self.angular_speed
        elif action == 2:
            vx = vy = 0
            wz = -self.angular_speed
        else:
            vx = -0.5 * self.linear_speed * np.cos(yaw)
            vy = -0.5 * self.linear_speed * np.sin(yaw)
            wz = 0

        p.resetBaseVelocity(
            self.robot_id, linearVelocity=[vx, vy, 0], angularVelocity=[0, 0, wz]
        )

        for _ in range(10):
            p.stepSimulation()
            if self.render_mode == "human":
                time.sleep(1 / 240)

        # new state
        pos, _ = p.getBasePositionAndOrientation(self.robot_id)
        new_dist = self._goal_dist()

        # reward
        reward = (old_dist - new_dist) * 5.0 - 0.05

        # off-road penalty
        if not self._on_road(pos[0], pos[1]):
            reward -= 50
            return self._get_obs(), reward, True, False, {}

        # collision penalty
        if self._robot_collision():
            reward -= 30
            return self._get_obs(), reward, True, False, {}

        # reaching goal
        if new_dist < 0.6:
            reward += 100
            return self._get_obs(), reward, True, False, {}

        truncated = self.step_count >= self.max_steps

        return self._get_obs(), reward, False, truncated, {}

    # ------------------------------------------------------------
    def _robot_collision(self):
        contacts = p.getContactPoints(self.robot_id)
        for c in contacts:
            if c[2] in self.building_ids or c[2] in self.obstacle_ids:
                return True
        return False

    # ------------------------------------------------------------
    def _goal_dist(self):
        pos, _ = p.getBasePositionAndOrientation(self.robot_id)
        gpos, _ = p.getBasePositionAndOrientation(self.goal_id)
        return np.linalg.norm([pos[0] - gpos[0], pos[1] - gpos[1]])

    # ------------------------------------------------------------
    def close(self):
        if self.physics_client is not None:
            p.disconnect(self.physics_client)
            self.physics_client = None


# =========================================================
# Test
# =========================================================
if __name__ == "__main__":
    env = GridBoundedEnv(
        grid_size=25,
        map_type="spiral",   # try "spiral" or "city"
        render_mode="human",
        num_obstacles=5,
    )

    obs, info = env.reset()
    for _ in range(2000):
        a = env.action_space.sample()
        obs, reward, done, trunc, info = env.step(a)
        if done or trunc:
            obs, _ = env.reset()
