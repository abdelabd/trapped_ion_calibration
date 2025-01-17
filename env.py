import gym
from gym import spaces
import numpy as np

class TrappedIonEnv(gym.Env):
    def __init__(self, seed=None):
        super(TrappedIonEnv, self).__init__()

        self.seed(seed)

        # Target intensity (unknown to the agent)
        self.min_intensity = 0.0
        self.target_intensity = 10.0
        self.max_intensity = 25.0

        # Initialize laser intensity
        self.reset()
        
        # Define action and observation space
        # Action space: continuous adjustment to laser intensity
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        
        # Observation space: scalar measurement of the system
        self.observation_space = spaces.Box(low=self.min_intensity, high=self.max_intensity, shape=(1,), dtype=np.float32)
        
        # Maximum number of steps per episode
        self.max_steps = 100
        self.current_step = 0

        # Error threshold for early stopping
        self.relative_error_threshold = 0.01
        self.goal_bonus = 100

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def reset(self):
        # Reset the environment to the initial state
        self.initial_intensity = np.clip(self.np_random.normal(10, 4), a_min=self.min_intensity, a_max=self.max_intensity)
        self.laser_intensity = self.initial_intensity
        self.current_step = 0
        return self._get_observation()

    def step(self, action):
        # Apply the action (adjust laser intensity)
        self.laser_intensity += action[0]
        
        # Ensure non-negative intensity
        self.laser_intensity = np.clip(self.laser_intensity, a_min=self.min_intensity, a_max=self.max_intensity)
        
        # Get the new observation
        observation = self._get_observation()
        
        # Calculate reward
        reward = -abs(self.laser_intensity - self.target_intensity)
        
        # Check if the episode is done
        self.current_step += 1
        if self.current_step >= self.max_steps:
            done = True
        elif abs(self.laser_intensity-self.target_intensity)/self.target_intensity <= self.relative_error_threshold:
            done = True
            reward += self.goal_bonus
        else:
            done = False
        
        # Additional info
        info = {}
        
        return observation, reward, done, info

    def _get_observation(self):
        # Simulate a measurement (add some noise)
        measurement = self.laser_intensity 
        return np.array([measurement], dtype=np.float32)

    def render(self, mode='human'):
        print(f"Step: {self.current_step}, Laser Intensity: {self.laser_intensity:.2f}, Target: {self.target_intensity:.2f}")