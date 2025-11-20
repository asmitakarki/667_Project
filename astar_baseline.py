"""
A* Pathfinding Algorithm - Baseline for comparison with RL methods

This provides a deterministic baseline to compare against RL agents.
"""

import numpy as np
import heapq
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from matplotlib.animation import FuncAnimation
from enhanced_pathfinding_env import EnhancedPathfindingEnv
import time

class AStarPathfinder:
    """
    A* pathfinding algorithm implementation
    """
    
    def __init__(self, grid_size=20, grid_resolution=0.5):
        """
        Args:
            grid_size: Size of the environment
            grid_resolution: Resolution of the grid for pathfinding (smaller = more accurate but slower)
        """
        self.grid_size = grid_size
        self.grid_resolution = grid_resolution
        self.grid_width = int(grid_size / grid_resolution)
        self.grid_height = int(grid_size / grid_resolution)
        self.occupancy_grid = None
        
    def world_to_grid(self, world_pos):
        """Convert world coordinates to grid coordinates"""
        grid_x = int(world_pos[0] / self.grid_resolution)
        grid_y = int(world_pos[1] / self.grid_resolution)
        return np.clip(grid_x, 0, self.grid_width - 1), np.clip(grid_y, 0, self.grid_height - 1)
    
    def grid_to_world(self, grid_pos):
        """Convert grid coordinates to world coordinates"""
        world_x = (grid_pos[0] + 0.5) * self.grid_resolution
        world_y = (grid_pos[1] + 0.5) * self.grid_resolution
        return np.array([world_x, world_y])
    
    def create_occupancy_grid(self, obstacles):
        """Create a binary occupancy grid from obstacles"""
        self.occupancy_grid = np.zeros((self.grid_height, self.grid_width), dtype=bool)
        
        for obs in obstacles:
            obs_pos = obs['pos']
            obs_size = obs['size']
            
            # Mark all grid cells within obstacle radius as occupied
            for gy in range(self.grid_height):
                for gx in range(self.grid_width):
                    world_pos = self.grid_to_world((gx, gy))
                    distance = np.linalg.norm(world_pos - obs_pos)
                    if distance < obs_size + 0.2:  # Small safety margin
                        self.occupancy_grid[gy, gx] = True
    
    def heuristic(self, pos1, pos2):
        """Euclidean distance heuristic"""
        return np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
    
    def get_neighbors(self, pos):
        """Get valid neighboring grid cells (8-directional)"""
        x, y = pos
        neighbors = []
        
        # 8 directions: up, down, left, right, and diagonals
        directions = [
            (0, 1), (1, 0), (0, -1), (-1, 0),  # Cardinal
            (1, 1), (1, -1), (-1, 1), (-1, -1)  # Diagonal
        ]
        
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            
            # Check bounds
            if 0 <= nx < self.grid_width and 0 <= ny < self.grid_height:
                # Check if cell is free
                if not self.occupancy_grid[ny, nx]:
                    neighbors.append((nx, ny))
        
        return neighbors
    
    def find_path(self, start_world, goal_world, obstacles):
        """
        Find path using A* algorithm
        
        Returns:
            path: List of world coordinates, or None if no path found
            stats: Dictionary with search statistics
        """
        # Create occupancy grid
        self.create_occupancy_grid(obstacles)
        
        # Convert to grid coordinates
        start = self.world_to_grid(start_world)
        goal = self.world_to_grid(goal_world)
        
        # Check if start or goal is in obstacle
        if self.occupancy_grid[start[1], start[0]] or self.occupancy_grid[goal[1], goal[0]]:
            return None, {'success': False, 'reason': 'Start or goal in obstacle'}
        
        # A* algorithm
        start_time = time.time()
        open_set = []
        heapq.heappush(open_set, (0, start))
        
        came_from = {}
        g_score = {start: 0}
        f_score = {start: self.heuristic(start, goal)}
        
        nodes_explored = 0
        
        while open_set:
            current = heapq.heappop(open_set)[1]
            nodes_explored += 1
            
            # Goal reached
            if current == goal:
                # Reconstruct path
                path_grid = []
                while current in came_from:
                    path_grid.append(current)
                    current = came_from[current]
                path_grid.append(start)
                path_grid.reverse()
                
                # Convert to world coordinates
                path_world = [self.grid_to_world(p) for p in path_grid]
                
                stats = {
                    'success': True,
                    'path_length': len(path_world),
                    'nodes_explored': nodes_explored,
                    'computation_time': time.time() - start_time,
                    'path_cost': g_score[goal]
                }
                
                return path_world, stats
            
            # Explore neighbors
            for neighbor in self.get_neighbors(current):
                # Cost to move to neighbor (diagonal moves cost more)
                dx = abs(neighbor[0] - current[0])
                dy = abs(neighbor[1] - current[1])
                move_cost = 1.414 if (dx + dy) == 2 else 1.0
                
                tentative_g_score = g_score[current] + move_cost
                
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + self.heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
        
        # No path found
        stats = {
            'success': False,
            'reason': 'No path exists',
            'nodes_explored': nodes_explored,
            'computation_time': time.time() - start_time
        }
        return None, stats


def test_astar_on_env(env, pathfinder, num_episodes=10, render=True):
    """
    Test A* pathfinder on an environment
    
    Args:
        env: EnhancedPathfindingEnv instance
        pathfinder: AStarPathfinder instance
        num_episodes: Number of test episodes
        render: Whether to visualize
    """
    results = []
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        
        # Get start and goal positions
        start = env.robot_pos.copy()
        goal = env.goal_pos.copy()
        obstacles = env.obstacles
        
        # Find path using A*
        path, stats = pathfinder.find_path(start, goal, obstacles)
        
        if path is None:
            print(f"Episode {episode + 1}: FAILED - {stats['reason']}")
            results.append({
                'success': False,
                'reason': stats['reason']
            })
            continue
        
        # Execute path in environment
        steps = 0
        success = False
        
        if render:
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.set_xlim(0, env.grid_size)
            ax.set_ylim(0, env.grid_size)
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
            
            # Draw obstacles
            for obs in obstacles:
                circle = Circle(obs['pos'], obs['size'], color='gray', alpha=0.7)
                ax.add_patch(circle)
            
            # Draw goal
            goal_circle = Circle(goal, 0.5, color='green', alpha=0.6)
            ax.add_patch(goal_circle)
            
            # Draw path
            if len(path) > 1:
                path_array = np.array(path)
                ax.plot(path_array[:, 0], path_array[:, 1], 'r--', linewidth=2, label='A* Path')
        
        # Follow the path
        for waypoint in path[1:]:  # Skip start position
            # Move towards waypoint
            while np.linalg.norm(env.robot_pos - waypoint) > 0.3:
                direction = waypoint - env.robot_pos
                direction = direction / np.linalg.norm(direction)
                
                # Determine action based on direction
                if abs(direction[0]) > abs(direction[1]):
                    action = 1 if direction[0] > 0 else 3  # Right or Left
                else:
                    action = 0 if direction[1] > 0 else 2  # Up or Down
                
                obs, reward, terminated, truncated, info = env.step(action)
                steps += 1
                
                if render:
                    # Update robot position
                    for patch in list(ax.patches):
                        if isinstance(patch, Circle) and patch.get_facecolor()[0] > 0.5:  # Blue robot
                            patch.remove()
                    robot_circle = Circle(env.robot_pos, 0.3, color='blue')
                    ax.add_patch(robot_circle)
                    plt.pause(0.01)
                
                if terminated:
                    success = True
                    break
                
                if truncated:
                    break
            
            if terminated or truncated:
                break
        
        if render:
            ax.legend()
            plt.title(f"Episode {episode + 1}: {'SUCCESS' if success else 'FAILED'} in {steps} steps")
            plt.show()
        
        result = {
            'success': success,
            'steps': steps,
            'path_length': len(path),
            'nodes_explored': stats['nodes_explored'],
            'computation_time': stats['computation_time'],
            'distance_to_goal': info['distance_to_goal']
        }
        results.append(result)
        
        print(f"Episode {episode + 1}: {'SUCCESS' if success else 'FAILED'} - "
              f"Steps: {steps}, Path Length: {len(path)}, "
              f"Nodes Explored: {stats['nodes_explored']}, "
              f"Time: {stats['computation_time']:.4f}s")
    
    # Calculate statistics
    successful = [r for r in results if r.get('success', False)]
    success_rate = len(successful) / len(results)
    
    if successful:
        avg_steps = np.mean([r['steps'] for r in successful])
        avg_path_length = np.mean([r['path_length'] for r in successful])
        avg_computation_time = np.mean([r['computation_time'] for r in successful])
    else:
        avg_steps = avg_path_length = avg_computation_time = 0
    
    print(f"\n=== A* Results ===")
    print(f"Success rate: {len(successful)}/{len(results)} ({100*success_rate:.1f}%)")
    if successful:
        print(f"Average steps: {avg_steps:.1f}")
        print(f"Average path length: {avg_path_length:.1f}")
        print(f"Average computation time: {avg_computation_time:.4f}s")
    
    return results


if __name__ == "__main__":
    print("Testing A* pathfinding on different map types...\n")
    
    # Test on different map types
    map_configs = [
        ('random', {'num_obstacles': 5}),
        ('grid', {'spacing': 3}),
        ('maze', {'cell_size': 2}),
        ('spiral', {'num_spirals': 2}),
    ]
    
    for map_type, map_kwargs in map_configs:
        print(f"\n{'='*50}")
        print(f"Testing on {map_type} map")
        print(f"{'='*50}\n")
        
        env = EnhancedPathfindingEnv(
            grid_size=20,
            map_type=map_type,
            render_mode='human',
            **map_kwargs
        )
        
        pathfinder = AStarPathfinder(grid_size=20, grid_resolution=0.5)
        
        results = test_astar_on_env(env, pathfinder, num_episodes=5, render=True)
        
        env.close()