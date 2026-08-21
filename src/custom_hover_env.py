import numpy as np

class QuadcopterHoverEnv:
    """
    Custom Quadcopter Hover Environment for RL Control.
    Implements attitude penalty, survival bonus, distance penalty, and terminal bounds.
    """
    def __init__(self, config=None):
        self.config = config if config else {}
        self._target_pos = np.array(self.config.get("target_pos", [0.0, 0.0, 1.0]), dtype=np.float32)
        self._attitude_weight = float(self.config.get("attitude_weight", 1.0))
        self._vel_weight = float(self.config.get("vel_weight", 1.0))
        self._survival_bonus = float(self.config.get("survival_bonus", 1.0))
        self._crash_penalty = float(self.config.get("crash_penalty", -10.0))
        
        # Termination thresholds
        self._min_z = float(self.config.get("min_z", 0.05))
        self._max_z = float(self.config.get("max_z", 3.0))
        self._max_xy_distance = float(self.config.get("max_xy_distance", 2.5))
        self._max_tilt_angle = float(self.config.get("max_tilt_angle", np.pi / 3))  # ~60 degrees
        
        self.state = None
        self.step_count = 0
        self.max_steps = int(self.config.get("max_steps", 500))

    def reset(self, seed=None):
        """Resets the drone state to initial position near target."""
        if seed is not None:
            np.random.seed(seed)
        self.state = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self.step_count = 0
        return self.state.copy(), {}

    def _computeTerminated(self, state):
        """
        Evaluates out-of-bounds, ground crash, or excessive tilt conditions.
        Returns: (terminated: bool, reason: str or None)
        """
        x, y, z = state[0], state[1], state[2]
        roll, pitch = state[3], state[4]
        
        # Ground crash check
        if z <= self._min_z:
            return True, "ground_crash"
            
        # Ceiling / altitude limit breach
        if z >= self._max_z:
            return True, "ceiling_breach"
            
        # Horizontal drift out-of-bounds
        xy_dist = np.sqrt(x**2 + y**2)
        if xy_dist >= self._max_xy_distance:
            return True, "out_of_bounds"
            
        # Excessive tilt / tumbling
        if abs(roll) >= self._max_tilt_angle or abs(pitch) >= self._max_tilt_angle:
            return True, "excessive_tilt"
            
        return False, None

    def _computeReward(self, state, terminated, reason):
        """Calculates step reward with distance penalty, attitude stabilization, and survival bonus."""
        if terminated and reason in ["ground_crash", "ceiling_breach", "out_of_bounds", "excessive_tilt"]:
            return self._crash_penalty

        pos = state[0:3]
        roll, pitch = state[3], state[4]
        roll_rate, pitch_rate = state[6], state[7]

        # Euclidean distance error to target
        dist = np.linalg.norm(pos - self._target_pos)
        dist_penalty = -1.0 * dist

        # Attitude and angular rate penalties
        attitude_penalty = -self._attitude_weight * (abs(roll) + abs(pitch))
        rate_penalty = -self._vel_weight * (abs(roll_rate) + abs(pitch_rate))

        # Positive constant survival bonus for maintaining flight
        total_reward = self._survival_bonus + dist_penalty + attitude_penalty + rate_penalty
        return float(total_reward)

    def step(self, action):
        """Executes a simulation step and returns (observation, reward, terminated, truncated, info)."""
        if self.state is None:
            self.reset()

        self.step_count += 1
        
        # Simple dynamics approximation for verification
        self.state[0:3] += np.clip(action[0:3] if len(action) >= 3 else [0, 0, 0], -0.1, 0.1)
        if len(action) >= 6:
            self.state[3:6] += np.clip(action[3:6], -0.1, 0.1)
            
        terminated, reason = self._computeTerminated(self.state)
        truncated = self.step_count >= self.max_steps
        reward = self._computeReward(self.state, terminated, reason)

        info = {"reason": reason, "step": self.step_count}
        return self.state.copy(), reward, terminated, truncated, info
