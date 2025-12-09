from robot_pov_env import RobotPOVEnv
import time

print("\n" + "="*70)
print("Robot POV")
print("="*70)

env = RobotPOVEnv(
    grid_size=20,
    map_type="city",
    render_mode="human",
    use_camera_obs=False,
    num_obstacles=4
)

obs, _ = env.reset()

for i in range(1000):
    action = env.action_space.sample()
    obs, reward, done, trunc, info = env.step(action)
    
    if done or trunc:
        obs, _ = env.reset()
        time.sleep(1)

env.close()