"""
Map Generators for Different Environment Types
Creates spiral, maze, grid, and corridor maps for testing
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from scipy.ndimage import distance_transform_edt

class MapGenerator:
    """Base class for map generation"""
    
    def __init__(self, grid_size=20):
        self.grid_size = grid_size
        self.obstacles = []
    
    def get_obstacles(self):
        """Return list of obstacles as dictionaries with 'pos' and 'size'"""
        return self.obstacles
    
    def visualize(self):
        """Visualize the generated map"""
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_xlim(0, self.grid_size)
        ax.set_ylim(0, self.grid_size)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_title(f"{self.__class__.__name__}")
        
        for obs in self.obstacles:
            if 'width' in obs and 'height' in obs:
                # Rectangle obstacle
                rect = Rectangle(
                    (obs['pos'][0] - obs['width']/2, obs['pos'][1] - obs['height']/2),
                    obs['width'], obs['height'],
                    color='gray', alpha=0.7
                )
                ax.add_patch(rect)
            else:
                # Circle obstacle
                circle = Circle(obs['pos'], obs['size'], color='gray', alpha=0.7)
                ax.add_patch(circle)
        
        plt.show()
        return fig, ax


class SpiralMapGenerator(MapGenerator):
    """Generates a spiral corridor map"""
    
    def __init__(self, grid_size=20, corridor_width=2, wall_thickness=0.5, num_spirals=3):
        super().__init__(grid_size)
        self.corridor_width = corridor_width
        self.wall_thickness = wall_thickness
        self.num_spirals = num_spirals
        self._generate_spiral()
    
    def _generate_spiral(self):
        """Create spiral-shaped walls"""
        center = np.array([self.grid_size / 2, self.grid_size / 2])
        
        # Create spiral path using parametric equations
        theta_max = 2 * np.pi * self.num_spirals
        num_points = 200
        
        for i in range(num_points - 1):
            # Inner spiral
            theta1 = (i / num_points) * theta_max
            theta2 = ((i + 1) / num_points) * theta_max
            
            r1 = 0.5 + (theta1 / theta_max) * (self.grid_size * 0.4)
            r2 = 0.5 + (theta2 / theta_max) * (self.grid_size * 0.4)
            
            x1 = center[0] + r1 * np.cos(theta1)
            y1 = center[1] + r1 * np.sin(theta1)
            x2 = center[0] + r2 * np.cos(theta2)
            y2 = center[1] + r2 * np.sin(theta2)
            
            # Create wall segments
            segment_length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
            if segment_length > 0:
                num_circles = max(1, int(segment_length / (self.wall_thickness * 0.5)))
                for j in range(num_circles):
                    t = j / num_circles
                    x = x1 + t * (x2 - x1)
                    y = y1 + t * (y2 - y1)
                    
                    if 0 < x < self.grid_size and 0 < y < self.grid_size:
                        self.obstacles.append({
                            'pos': np.array([x, y]),
                            'size': self.wall_thickness
                        })


class MazeMapGenerator(MapGenerator):
    """Generates a grid-based maze using recursive backtracking"""
    
    def __init__(self, grid_size=20, cell_size=2, wall_thickness=0.4):
        super().__init__(grid_size)
        self.cell_size = cell_size
        self.wall_thickness = wall_thickness
        self.maze_grid_size = int(grid_size / cell_size)
        self._generate_maze()
    
    def _generate_maze(self):
        """Generate maze using depth-first search"""
        # Create grid: 0 = wall, 1 = path
        maze = np.zeros((self.maze_grid_size, self.maze_grid_size), dtype=int)
        
        # Recursive backtracking to carve maze
        def carve_path(x, y):
            maze[y, x] = 1
            directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
            np.random.shuffle(directions)
            
            for dx, dy in directions:
                nx, ny = x + dx * 2, y + dy * 2
                if 0 <= nx < self.maze_grid_size and 0 <= ny < self.maze_grid_size:
                    if maze[ny, nx] == 0:
                        maze[y + dy, x + dx] = 1
                        carve_path(nx, ny)
        
        # Start carving from center
        start_x = self.maze_grid_size // 2
        start_y = self.maze_grid_size // 2
        if start_x % 2 == 0:
            start_x -= 1
        if start_y % 2 == 0:
            start_y -= 1
        
        carve_path(start_x, start_y)
        
        # Convert maze grid to obstacles
        for y in range(self.maze_grid_size):
            for x in range(self.maze_grid_size):
                if maze[y, x] == 0:
                    # Place obstacle at this cell
                    world_x = (x + 0.5) * self.cell_size
                    world_y = (y + 0.5) * self.cell_size
                    
                    self.obstacles.append({
                        'pos': np.array([world_x, world_y]),
                        'size': self.cell_size * 0.7,
                        'width': self.cell_size,
                        'height': self.cell_size
                    })


class GridMapGenerator(MapGenerator):
    """Generates a regular grid of obstacles"""
    
    def __init__(self, grid_size=20, spacing=3, obstacle_size=0.8, remove_probability=0.3):
        super().__init__(grid_size)
        self.spacing = spacing
        self.obstacle_size = obstacle_size
        self.remove_probability = remove_probability
        self._generate_grid()
    
    def _generate_grid(self):
        """Create regular grid with some random gaps"""
        num_cols = int(self.grid_size / self.spacing)
        num_rows = int(self.grid_size / self.spacing)
        
        for i in range(num_cols):
            for j in range(num_rows):
                # Randomly skip some obstacles to create paths
                if np.random.random() < self.remove_probability:
                    continue
                
                x = (i + 1) * self.spacing
                y = (j + 1) * self.spacing
                
                if x < self.grid_size and y < self.grid_size:
                    self.obstacles.append({
                        'pos': np.array([x, y]),
                        'size': self.obstacle_size
                    })


class CorridorMapGenerator(MapGenerator):
    """Generates narrow corridors with turns"""
    
    def __init__(self, grid_size=20, corridor_width=2.5, wall_thickness=0.5, num_segments=5):
        super().__init__(grid_size)
        self.corridor_width = corridor_width
        self.wall_thickness = wall_thickness
        self.num_segments = num_segments
        self._generate_corridors()
    
    def _generate_corridors(self):
        """Create corridor segments"""
        # Define corridor path
        segments = []
        current_pos = np.array([1.0, self.grid_size / 2])
        direction = 0  # 0: right, 1: up, 2: left, 3: down
        
        for _ in range(self.num_segments):
            # Create segment
            if direction == 0:  # Right
                end_pos = current_pos + np.array([self.grid_size / self.num_segments, 0])
            elif direction == 1:  # Up
                end_pos = current_pos + np.array([0, self.grid_size / self.num_segments])
            elif direction == 2:  # Left
                end_pos = current_pos + np.array([-self.grid_size / self.num_segments, 0])
            else:  # Down
                end_pos = current_pos + np.array([0, -self.grid_size / self.num_segments])
            
            segments.append((current_pos.copy(), end_pos.copy()))
            current_pos = end_pos
            
            # Change direction randomly
            if np.random.random() < 0.5:
                direction = (direction + 1) % 4
        
        # Create walls along corridors
        for start, end in segments:
            segment_vec = end - start
            segment_length = np.linalg.norm(segment_vec)
            
            if segment_length > 0:
                perpendicular = np.array([-segment_vec[1], segment_vec[0]])
                perpendicular = perpendicular / np.linalg.norm(perpendicular)
                
                # Create walls on both sides
                num_wall_points = int(segment_length / self.wall_thickness)
                for i in range(num_wall_points):
                    t = i / num_wall_points
                    point = start + t * segment_vec
                    
                    # Wall on left side
                    wall_pos_left = point + perpendicular * (self.corridor_width / 2)
                    if 0 < wall_pos_left[0] < self.grid_size and 0 < wall_pos_left[1] < self.grid_size:
                        self.obstacles.append({
                            'pos': wall_pos_left,
                            'size': self.wall_thickness
                        })
                    
                    # Wall on right side
                    wall_pos_right = point - perpendicular * (self.corridor_width / 2)
                    if 0 < wall_pos_right[0] < self.grid_size and 0 < wall_pos_right[1] < self.grid_size:
                        self.obstacles.append({
                            'pos': wall_pos_right,
                            'size': self.wall_thickness
                        })


class RandomObstaclesGenerator(MapGenerator):
    """Generates random obstacles (original version)"""
    
    def __init__(self, grid_size=20, num_obstacles=5, min_size=0.5, max_size=1.5):
        super().__init__(grid_size)
        self.num_obstacles = num_obstacles
        self.min_size = min_size
        self.max_size = max_size
        self._generate_random()
    
    def _generate_random(self):
        """Create random obstacles"""
        for _ in range(self.num_obstacles):
            self.obstacles.append({
                'pos': np.array([
                    np.random.uniform(0, self.grid_size),
                    np.random.uniform(0, self.grid_size)
                ]),
                'size': np.random.uniform(self.min_size, self.max_size)
            })


# Testing and visualization
if __name__ == "__main__":
    print("Generating different map types...\n")
    
    # Create all map types
    maps = {
        'Random': RandomObstaclesGenerator(grid_size=20, num_obstacles=8),
        'Spiral': SpiralMapGenerator(grid_size=20, num_spirals=3),
        'Maze': MazeMapGenerator(grid_size=20, cell_size=2),
        'Grid': GridMapGenerator(grid_size=20, spacing=3),
        'Corridor': CorridorMapGenerator(grid_size=20, num_segments=6)
    }
    
    # Visualize all maps
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, (name, map_gen) in enumerate(maps.items()):
        ax = axes[idx]
        ax.set_xlim(0, map_gen.grid_size)
        ax.set_ylim(0, map_gen.grid_size)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_title(name)
        
        for obs in map_gen.obstacles:
            if 'width' in obs and 'height' in obs:
                rect = Rectangle(
                    (obs['pos'][0] - obs['width']/2, obs['pos'][1] - obs['height']/2),
                    obs['width'], obs['height'],
                    color='gray', alpha=0.7
                )
                ax.add_patch(rect)
            else:
                circle = Circle(obs['pos'], obs['size'], color='gray', alpha=0.7)
                ax.add_patch(circle)
        
        print(f"{name}: {len(map_gen.obstacles)} obstacles")
    
    # Hide extra subplot
    axes[-1].axis('off')
    
    plt.tight_layout()
    plt.savefig('map_types.png', dpi=150, bbox_inches='tight')
    print("\nMap visualization saved as 'map_types.png'")
    plt.show()