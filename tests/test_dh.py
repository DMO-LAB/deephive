import unittest
from environment import OptimizationEnv
from mappo import MAPPO
from deephive import DeepHive

class TestDeepHive(unittest.TestCase):
    def setUp(self):
        self.env = OptimizationEnv()
        self.policy = MAPPO()
        self.config = {
            'train_config': {
                'n_agents': 10,
                'n_dim': 2,
                'ep_length': 100
            },
            'test_config': {
                'n_agents': 5,
                'n_dim': 2,
                'ep_length': 50,
                'n_run': 10,
                'use_gbest': 'True'
            }
        }

    def test_init(self):
        dh = DeepHive('test', self.env, self.policy, 'train', self.config)
        self.assertEqual(dh.title, 'test')
        self.assertEqual(dh.env, self.env)
        self.assertEqual(dh.policy, self.policy)
        self.assertEqual(dh.config, self.config)
        self.assertIsNone(dh.env_cache)
        self.assertEqual(dh.mode, 'train')
        self.assertEqual(dh.n_agents, 10)
        self.assertEqual(dh.n_dim, 2)
        self.assertEqual(dh.ep_length, 100)
        self.assertFalse(dh.use_gbest)

    def test_set_parameters(self):
        dh = DeepHive('test', self.env, self.policy, 'train', self.config)
        dh._set_parameters()
        self.assertEqual(dh.n_agents, 10)
        self.assertEqual(dh.n_dim, 2)
        self.assertEqual(dh.ep_length, 100)
        dh.mode = 'test'
        dh._set_parameters()
        self.assertEqual(dh.n_agents, 5)
        self.assertEqual(dh.n_dim, 2)
        self.assertEqual(dh.ep_length, 50)
        self.assertEqual(dh.n_run, 10)
        self.assertTrue(dh.use_gbest)

    def test_create_work_dir(self):
        dh = DeepHive('test', self.env, self.policy, 'train', self.config)
        exp_name, directory, plot_dir, gif_dir, checkpoint_dir = dh._create_work_dir('test', log_folder='logs')
        self.assertEqual(exp_name, 'test')
        self.assertEqual(directory, 'logs/test')
        self.assertEqual(plot_dir, 'logs/test/plots')
        self.assertEqual(gif_dir, 'logs/test/gifs')
        self.assertEqual(checkpoint_dir, 'logs/test/checkpoints')

    def test_optimize(self):
        dh = DeepHive('test', self.env, self.policy, 'test', self.config)
        dh.optimize(debug=True)
        # add more assertions here based on what you expect the output to be

if __name__ == '__main__':
    unittest.main()