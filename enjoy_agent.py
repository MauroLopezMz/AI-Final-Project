from stable_baselines3 import DQN
from custom_robot_env import RobotNavEnv
import time
import os

# Setup paths (Relative paths work now that you are in a safe folder!)
MODEL_PATH = "models/dqn_robot_nav"

# Check if model exists
if not os.path.exists(f"{MODEL_PATH}.zip"):
    print(f"❌ Error: Model not found at {MODEL_PATH}.zip")
    print("Did you run train_agent.py first?")
    exit()

# Load the environment with rendering ON so we can see it
env = RobotNavEnv(render_mode="human")

# Load the trained brain
print(f"Loading model from {MODEL_PATH}...")
model = DQN.load(MODEL_PATH)

obs, _ = env.reset()
print("--- Playing Trained Model ---")
print("Press Ctrl+C in the terminal to stop.")

try:
    for i in range(1000):
        # Predict the best action
        # deterministic=True means "Do exactly what you learned", no random exploring
        action, _states = model.predict(obs, deterministic=True)
        
        obs, reward, terminated, truncated, _ = env.step(action)
        
        if terminated:
            print("End of Episode (Goal or Crash)")
            obs, _ = env.reset()
            time.sleep(0.5) # Short pause to reset

except KeyboardInterrupt:
    print("Stopping...")

env.close()