"""
Quadcopter RL Environment with Position-Based Distance Penalty
Implements Issue #6: Euclidean distance error + quadratic penalty in _computeReward()
"""
import numpy as np

class QuadcopterEnv:
    """Simplified quadcopter environment for RL training."""
    
    def __init__(self, target_position=None):
        self.target_position = np.array(target_position or [0.0, 0.0, 1.0])
        self.position = np.zeros(3)
        self.velocity = np.zeros(3)
        self.max_steps = 1000
        self.step_count = 0
        
    def reset(self):
        self.position = np.zeros(3)
        self.velocity = np.zeros(3)
        self.step_count = 0
        return self._get_obs()
    
    def _get_obs(self):
        return np.concatenate([self.position, self.velocity])
    
    def _compute_distance_penalty(self):
        """Calculate Euclidean distance error and base quadratic distance penalty."""
        error = self.position - self.target_position
        distance = np.linalg.norm(error)
        # Quadratic penalty: increases as drone drifts away from target
        penalty = distance ** 2
        return distance, penalty
    
    def _compute_reward(self):
        """Compute reward with position-based distance penalty."""
        distance, penalty = self._compute_distance_penalty()
        
        # Base reward is negative quadratic distance penalty
        reward = -penalty
        
        # Bonus for being very close to target (< 0.1m)
        if distance < 0.1:
            reward += 10.0
            
        return reward
    
    def step(self, action):
        """Apply action and return next state, reward, done."""
        # Simplified dynamics: action directly affects velocity
        dt = 0.01
        self.velocity += action * dt
        self.position += self.velocity * dt
        self.step_count += 1
        
        reward = self._compute_reward()
        done = self.step_count >= self.max_steps
        
        return self._get_obs(), reward, done, {}

# Verification test
if __name__ == "__main__":
    env = QuadcopterEnv(target_position=[5.0, 5.0, 2.0])
    obs = env.reset()
    
    # Test at origin (far from target)
    dist, pen = env._compute_distance_penalty()
    print(f"At origin: distance={dist:.4f}, penalty={pen:.4f}")
    
    # Move closer
    env.position = np.array([4.9, 4.9, 1.9])
    dist2, pen2 = env._compute_distance_penalty()
    print(f"Near target: distance={dist2:.4f}, penalty={pen2:.4f}")
    
    # Verify penalty decreases as we approach
    assert pen2 < pen, "Penalty should decrease when closer to target"
    print("✓ All tests passed")
