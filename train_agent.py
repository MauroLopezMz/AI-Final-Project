import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback
import os

# Import your custom environment
from custom_robot_env import RobotNavEnv

# --- Hyperparameters (The "Research" Configuration) ---
# Total steps to train. 
# 100,000 is a good start for this simple grid. 
# If it's still dumb, we increase this to 500,000.
TIMESTEPS = 100000 
LOG_DIR = "./logs/"
MODEL_DIR = "./models/"

# Create directories to save results
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

def train():
    print("--- Setting up the Digital Twin ---")
    # 1. Initialize the Environment
    env = RobotNavEnv(render_mode=None) # No render during training to go faster!
    
    # 2. Wrap the environment for Logging
    # This records "Episode Length" and "Reward" for TensorBoard
    env = Monitor(env, LOG_DIR)

    # 3. Define the AI Model (DQN)
    # Policy = "MlpPolicy" (Multi-Layer Perceptron) because inputs are numbers, not images.
    # verbose = 1 (Print progress to terminal)
    model = DQN(
        "MlpPolicy", 
        env, 
        verbose=1, 
        tensorboard_log=LOG_DIR, 
        learning_rate=0.00005, 
        buffer_size=50000,
        learning_starts=1000, 
        exploration_fraction=0.4, # Explore 10% of the time, exploit 90%
        target_update_interval=1000
    )

    print("--- Starting Training (This may take a few minutes) ---")
    print(f"Target: {TIMESTEPS} timesteps")
    
    # 4. Train the Agent
    model.learn(total_timesteps=TIMESTEPS, tb_log_name="dqn_run_1")

    # 5. Save the final "Brain"
    print("--- Training Complete. Saving Model... ---")
    model.save(f"{MODEL_DIR}/dqn_robot_nav")
    print(f"Model saved to {MODEL_DIR}/dqn_robot_nav.zip")

if __name__ == "__main__":
    train()