import gym
from gym import spaces
import numpy as np

class TrappedIonEnv(gym.Env):
    def __init__(self):
        super(TrappedIonEnv, self).__init__()
        
        # Define action and observation space
        # Action space: continuous adjustment to laser intensity
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        
        # Observation space: scalar measurement of the system
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32)
        
        # Initialize laser intensity
        self.laser_intensity = 0.0
        
        # Target intensity (unknown to the agent)
        self.target_intensity = 10.0
        
        # Maximum number of steps per episode
        self.max_steps = 100
        self.current_step = 0

    def reset(self):
        # Reset the environment to the initial state
        self.laser_intensity = 0.0
        self.current_step = 0
        return self._get_observation()

    def step(self, action):
        # Apply the action (adjust laser intensity)
        self.laser_intensity += action[0]
        
        # Ensure non-negative intensity
        self.laser_intensity = max(0, self.laser_intensity)
        
        # Get the new observation
        observation = self._get_observation()
        
        # Calculate reward
        reward = -abs(self.laser_intensity - self.target_intensity)
        
        # Check if the episode is done
        self.current_step += 1
        done = self.current_step >= self.max_steps
        
        # Additional info
        info = {}
        
        return observation, reward, done, info

    def _get_observation(self):
        # Simulate a measurement (add some noise)
        measurement = self.laser_intensity + np.random.normal(0, 0.1)
        return np.array([measurement], dtype=np.float32)

    def render(self, mode='human'):
        print(f"Step: {self.current_step}, Laser Intensity: {self.laser_intensity:.2f}, Target: {self.target_intensity:.2f}")