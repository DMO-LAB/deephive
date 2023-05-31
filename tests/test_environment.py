import sys
sys.path.append('./')
import unittest
import numpy as np
from src.environment import OptimizationEnv

class TestDeepHiveEnvironment(unittest.TestCase):
    
    def setUp(self):
        self.n_agents = 10
        self.n_dim = 2
        self.optFunc = lambda x: -np.sum(x**2, axis=1)
        self.bounds = (np.array([-2, -2]), np.array([2,2]))
        self.ep_length = 20
        self.env  = OptimizationEnv(env_name="OptimizationEnv", optFunc=self.optFunc, n_agents=self.n_agents, n_dim=self.n_dim, bounds=self.bounds, ep_length=self.ep_length)

    def test_reset(self):
        obs = self.env.reset()
        self.assertEqual(obs.shape, (10, 3))
        self.assertTrue(np.all(obs >= 0) and np.all(obs <= 1))
        
    def test_step(self):
        obs = self.env.reset()
        action = np.random.uniform(low=-1, high=1, size=(10, 2))
        next_obs, reward, done, info = self.env.step(action)
        self.assertEqual(next_obs.shape, (10, 3))
        self.assertTrue(np.all(next_obs >= 0) and np.all(next_obs <= 1))
        self.assertEqual(reward.shape, (10,))
        
    def test_reward_fcn_vec(self):
        obs = self.env.reset()
        action = np.random.uniform(low=-1, high=1, size=(10, 2))
        next_obs, reward, done, info = self.env.step(action)
        self.assertEqual(reward.shape, (10,))
        # assert that the agents closest to the optimum get the highest reward
        #self.assertTrue(np.all(reward[np.argsort(reward)[-3:]] > reward[np.argsort(reward)[:-3]]))
        
    def test_get_actual_state(self):
        obs = self.env.reset()
        actual_state = self.env._get_actual_state()
        self.assertEqual(actual_state.shape, (10, 3))
        actual_value = self.optFunc(actual_state[:, :-1])
        self.assertTrue(np.all(np.round(actual_value, 2) == np.round(actual_state[:, -1], 2)))
        
    def test_generate_init_state(self):
        init_obs = self.env._generate_init_state()
        self.assertEqual(init_obs.shape, (10, 3))
        self.assertTrue(np.all(init_obs[:, :-1] >= 0) and np.all(init_obs[:, :-1] <= 1))
        
    def test_scale(self):
        d = np.array([[1, 2], [3, 4], [5, 6]])
        dmin = np.array([0, 0])
        dmax = np.array([10, 10])
        scaled_d = self.env._scale(d, dmin, dmax)
        self.assertEqual(scaled_d.shape, (3, 2))
        self.assertTrue(np.all(scaled_d >= 0) and np.all(scaled_d <= 1))
        
    def test_rescale(self):
        d = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
        dmin = np.array([0, 0])
        dmax = np.array([10, 10])
        rescaled_d = self.env._rescale(d, dmin, dmax)
        self.assertEqual(rescaled_d.shape, (3, 2))
        self.assertTrue(np.all(rescaled_d >= 0.0) and np.all(rescaled_d <= 10.0))



if __name__ == '__main__':
    unittest.main()
