from custom_robot_env import RobotNavEnv
import time

# Initialize environment with visualization
env = RobotNavEnv(render_mode="human")
obs, _ = env.reset()

print("Starting Simulation Test...")
print("Robot is Blue. Goal is Green.")

for i in range(100):
    # Pick a random action (0=Fwd, 1=Left, 2=Right)
    action = env.action_space.sample()
    
    # Apply action
    obs, reward, terminated, truncated, _ = env.step(action)
    
    # Print what the robot 'sees' (Sensor distances normalized 0-1)
    # Format: [Left, Front, Right]
    print(f"Step {i}: Sensors={obs} | Reward={reward}")

    if terminated:
        print("--- Episode Finished (Crash or Goal) ---")
        obs, _ = env.reset()
        time.sleep(1) # Pause to see the reset

env.close()