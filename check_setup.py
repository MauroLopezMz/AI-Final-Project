import gymnasium as gym
import stable_baselines3
import torch
import tensorflow as tf
# import onnx           <-- Comment this out with #
# import onnx2tf        <-- Comment this out with #
from stable_baselines3 import DQN

print(f"✅ Python Environment Ready!")
print(f"   - PyTorch Version: {torch.__version__}")
print(f"   - SB3 Version: {stable_baselines3.__version__}")

# Test the Gym Environment
env = gym.make("CartPole-v1")
print("✅ Gymnasium works. CartPole environment loaded.")

# Test the Agent (Brain)
model = DQN("MlpPolicy", env, verbose=0)
print("✅ Stable-Baselines3 works. DQN Agent initialized.")

print("✅ Ready for Phase 1: Simulation!")