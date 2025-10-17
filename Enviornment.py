from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from scipy.sparse import hstack
import gym
from gym import spaces
import shimmy
import numpy as np
import pandas as pd

class EmailEnv(gym.Env):
    def __init__(self, model, X, y):
        super(EmailEnv, self).__init__()
        self.model = model
        self.X = X
        self.y = np.array(y) if isinstance(y, pd.Series) else y  # Convert y to a numpy array
        self.current_index = 0

        # Define observation space and action space
        self.observation_space = spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
        self.action_space = spaces.Discrete(2)  # Two actions: phishing (1) or not (0)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_index = 0
        observation = self._get_obs()
        return observation, {}

    def _get_obs(self):
        prob = self.model.predict_proba(self.X[self.current_index].toarray())[0, 1]
        return np.array([prob], dtype=np.float32)

    def step(self, action):
        true_label = self.y[self.current_index]

        reward = 1 if action == true_label else -1
        #print(f"Action: {action}, True Label: {true_label}, Reward: {reward}")  # Track what's happening

        self.current_index += 1
        done = self.current_index >= len(self.y)
        obs = self._get_obs() if not done else np.array([0], dtype=np.float32)
        truncated = False
        return obs, reward, done, truncated, {}

    def provide_feedback(self, index, corrected_label):
        """This method allows the environment to update its state based on feedback."""
        self.y[index] = corrected_label
