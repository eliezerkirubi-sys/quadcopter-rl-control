import unittest
import numpy as np
from src.custom_hover_env import QuadcopterHoverEnv

class TestSurvivalAndTermination(unittest.TestCase):
    def setUp(self):
        self.env = QuadcopterHoverEnv(config={
            "survival_bonus": 1.5,
            "min_z": 0.05,
            "max_z": 3.0,
            "max_tilt_angle": np.pi / 3,
            "crash_penalty": -10.0
        })

    def test_survival_bonus_awarded_when_stable(self):
        self.env.reset()
        action = np.zeros(6, dtype=np.float32)
        _, reward, terminated, _, _ = self.env.step(action)
        self.assertFalse(terminated)
        # Should be positive or include survival bonus component
        self.assertGreater(reward, 0.0)

    def test_ground_crash_termination(self):
        state = np.array([0.0, 0.0, 0.02, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        terminated, reason = self.env._computeTerminated(state)
        self.assertTrue(terminated)
        self.assertEqual(reason, "ground_crash")

    def test_excessive_tilt_termination(self):
        state = np.array([0.0, 0.0, 1.0, 1.2, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32) # > pi/3
        terminated, reason = self.env._computeTerminated(state)
        self.assertTrue(terminated)
        self.assertEqual(reason, "excessive_tilt")

    def test_crash_penalty_applied_on_termination(self):
        self.env.reset()
        state = np.array([0.0, 0.0, 0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        reward = self.env._computeReward(state, True, "ground_crash")
        self.assertEqual(reward, -10.0)

if __name__ == "__main__":
    unittest.main()
