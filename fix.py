import numpy as np
from gymnasium import spaces

class QuadcopterEnv:
    def __init__(self, config=None):
        self.config = config if config else {}
        self.state_space = spaces.Box(low=0, high=10, shape=(6,), dtype=np.float32)
        self._prev_state = None
        self._attitude_weight = self.config.get('attitude_weight', 1.0)
        self._vel_weight = self.config.get('vel_weight', 1.0)
        self._target_z = self.config.get('target_z', 1.0)

    def step(self, action):
        state = self._prev_state
        reward = 0.0

        if state is not None:
            z = state[2]
            roll = state[3]
            pitch = state[4]
            yaw = state[5]

            # Height stability reward
            reward += (z - self._target_z) * (0.5 if z > 0.5 else 1.0)

            # Attitude penalty (Roll & Pitch)
            roll_p = abs(roll) * self._attitude_weight
            pitch_p = abs(pitch) * self._attitude_weight
            reward -= (roll_p + pitch_p)

            # Angular Velocity penalty (Roll & Pitch Rate)
            # Assuming rates are in a separate part of state or scaled
            roll_rate = state[6] if len(state) > 6 else 0.0
            pitch_rate = state[7] if len(state) > 7 else 0.0
            vel_p = (abs(roll_rate) + abs(pitch_rate)) * self._vel_weight
            reward -= vel_p

        self._prev_state = state
        return state, reward, 0

    def reset(self, seed=None):
        self._prev_state = np.zeros(6, dtype=np.float32)
        return self._prev_state, 0

    def get_observation(self):
        return self._prev_state