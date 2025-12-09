import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
import math

class RobotNavEnv(gym.Env):
    """
    Custom Environment that follows gym interface.
    Represents a differential drive robot with 3 ultrasonic sensors.
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, render_mode=None):
        super(RobotNavEnv, self).__init__()
        
        # --- Constants ---
        self.arena_size = 2.0  # 2 meters x 2 meters
        self.robot_radius = 0.1 # 10 cm robot radius
        self.sensor_max_range = 1.5 # Sensors see up to 1.5m
        self.goal_radius = 0.2
        
        # Sensor Angles: Left (-45), Front (0), Right (+45)
        # In radians: -pi/4, 0, pi/4
        self.sensor_angles = np.array([-math.pi/4, 0, math.pi/4])
        
        # --- Action Space ---
        # 0: Move Forward
        # 1: Turn Left
        # 2: Turn Right
        self.action_space = spaces.Discrete(3)

        # --- Observation Space ---
        # The agent receives 3 values: [Distance_Left, Distance_Front, Distance_Right]
        # Normalized between 0 and 1 for better Neural Network performance
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(3,), dtype=np.float32)

        self.render_mode = render_mode
        self.window = None
        self.clock = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize Robot Position (Center)
        self.robot_pos = np.array([1.0, 1.0])
        self.robot_angle = np.random.uniform(-math.pi, math.pi)
        
        # Initialize Goal Position (Random spot at least 0.5m away)
        while True:
            self.goal_pos = np.random.uniform(0.2, 1.8, size=2)
            if np.linalg.norm(self.goal_pos - self.robot_pos) > 0.5:
                break

        # Get initial sensor readings
        observation = self._get_obs()
        return observation, {}

    def step(self, action):
        # --- 1. Physics Update ---
        speed = 0.05      # 5 cm per step
        turn_speed = 0.2  # ~11 degrees per step

        if action == 0: # Forward
            self.robot_pos[0] += speed * math.cos(self.robot_angle)
            self.robot_pos[1] += speed * math.sin(self.robot_angle)
        elif action == 1: # Turn Left
            self.robot_angle += turn_speed
        elif action == 2: # Turn Right
            self.robot_angle -= turn_speed

        # Normalize angle to -pi to pi
        self.robot_angle = (self.robot_angle + math.pi) % (2 * math.pi) - math.pi

        # --- 2. Calculate Sensor Readings (Ray Casting) ---
        sensor_readings = self._ray_cast_sensors()
        
        # --- 3. Check Termination & Reward ---
        terminated = False
        truncated = False
        reward = -0.1 # Small penalty per step to encourage speed (Battery usage)

        # Distance to goal
        dist_to_goal = np.linalg.norm(self.robot_pos - self.goal_pos)

        # Hit Wall? (Collision)
        if (self.robot_pos[0] < self.robot_radius or self.robot_pos[0] > self.arena_size - self.robot_radius or
            self.robot_pos[1] < self.robot_radius or self.robot_pos[1] > self.arena_size - self.robot_radius):
            reward = -50 # Big penalty for crash
            terminated = True
        
        # Reached Goal?
        elif dist_to_goal < self.goal_radius:
            reward = 100 # Big reward for success
            terminated = True

        # --- 4. Return Data ---
        # Normalize readings for the Neural Network (0.0 to 1.0)
        # We need to return the observation variable we calculated!
        normalized_obs = np.array(sensor_readings, dtype=np.float32) / self.sensor_max_range
        
        if self.render_mode == "human":
            self.render()

        return normalized_obs, reward, terminated, truncated, {}

    def _get_obs(self):
         sensor_readings = self._ray_cast_sensors()
         return np.array(sensor_readings, dtype=np.float32) / self.sensor_max_range

    def _ray_cast_sensors(self):
        """
        Simulates 3 ultrasonic sensors. 
        Returns array of 3 distances in meters.
        """
        readings = []
        for angle_offset in self.sensor_angles:
            # Calculate absolute angle of the sensor
            sensor_theta = self.robot_angle + angle_offset
            
            # Find distance to all 4 walls
            # We use a very simple ray casting to the bounding box of the arena
            
            # 1. Dist to Right Wall (x = arena_size)
            if math.cos(sensor_theta) > 0:
                d1 = (self.arena_size - self.robot_pos[0]) / math.cos(sensor_theta)
            else: 
                d1 = float('inf')

            # 2. Dist to Left Wall (x = 0)
            if math.cos(sensor_theta) < 0:
                d2 = (0 - self.robot_pos[0]) / math.cos(sensor_theta)
            else:
                d2 = float('inf')

            # 3. Dist to Top Wall (y = arena_size)
            if math.sin(sensor_theta) > 0:
                d3 = (self.arena_size - self.robot_pos[1]) / math.sin(sensor_theta)
            else:
                d3 = float('inf')

            # 4. Dist to Bottom Wall (y = 0)
            if math.sin(sensor_theta) < 0:
                d4 = (0 - self.robot_pos[1]) / math.sin(sensor_theta)
            else:
                d4 = float('inf')

            # Get the smallest positive distance
            closest = min(d1, d2, d3, d4)
            
            # Clamp to max sensor range
            readings.append(min(closest, self.sensor_max_range))
            
        return readings

    def render(self):
        if self.window is None:
            plt.ion()
            self.fig, self.ax = plt.subplots()
            self.window = True
        
        self.ax.clear()
        self.ax.set_xlim(0, self.arena_size)
        self.ax.set_ylim(0, self.arena_size)
        
        # Draw Goal (Green)
        goal = plt.Circle(self.goal_pos, self.goal_radius, color='green', alpha=0.5)
        self.ax.add_artist(goal)
        
        # Draw Robot (Blue)
        robot = plt.Circle(self.robot_pos, self.robot_radius, color='blue')
        self.ax.add_artist(robot)
        
        # Draw Direction Indicator
        end_x = self.robot_pos[0] + 0.2 * math.cos(self.robot_angle)
        end_y = self.robot_pos[1] + 0.2 * math.sin(self.robot_angle)
        self.ax.plot([self.robot_pos[0], end_x], [self.robot_pos[1], end_y], 'k-')
        
        plt.pause(0.01)

    def close(self):
        if self.window:
            plt.close()