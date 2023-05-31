import unittest
import torch
from src.mappo import MAPPO


class TestMAPPO(unittest.TestCase):
    def setUp(self):
        self.state_dim = 10
        self.action_dim = 2
        self.action_std = 0.5
        self.layer_size = 64
        self.std_min = 0.1
        self.std_max = 2.0
        self.std_type = "adaptive"
        self.learn_std = False
        self.initialization = None
        self.pretrained = False
        self.ckpt_folder = None
        self.lr = 0.001
        self.gamma = 0.99
        self.K_epochs = 4
        self.eps_clip = 0.2
        self.n_agents = 2
        self.split_agent = False
        self.device = torch.device("cpu")
        self.mappo = MAPPO(self.state_dim, self.action_dim, self.action_std, self.layer_size, self.std_min,
                           self.std_max, self.std_type, self.learn_std, self.initialization, self.pretrained,
                           self.ckpt_folder, self.lr, self.gamma, self.K_epochs, self.eps_clip, self.n_agents,
                           self.split_agent, self.device)

    def test_select_action(self):
        state = torch.randn(self.n_agents, self.state_dim)
        std_obs = torch.randn(self.n_agents, self.action_dim)
        action = self.mappo.select_action(state, std_obs)
        self.assertEqual(action.shape, (self.n_agents, self.action_dim))

    def test_set_action_std(self):
        new_action_std = 0.1
        self.mappo.set_action_std(new_action_std)
        self.assertEqual(self.mappo.action_std, new_action_std)

    def test_decay_action_std(self):
        action_std_decay_rate = 0.01
        min_action_std = 0.001
        learn_std = False
        self.mappo.decay_action_std(action_std_decay_rate, min_action_std, learn_std)
        self.assertGreaterEqual(self.mappo.action_std, min_action_std)

    def test__get_buffer_info(self):
        rewards = torch.tensor([1.0, 2.0, 3.0])
        states = torch.randn(3, self.state_dim)
        actions = torch.randn(3, self.action_dim)
        logprobs = torch.randn(3)
        std_obs = torch.randn(3, self.action_dim)
        self.mappo.buffer.rewards = rewards
        self.mappo.buffer.states = states
        self.mappo.buffer.actions = actions
        self.mappo.buffer.logprobs = logprobs
        self.mappo.buffer.std_obs = std_obs
        rewards_, states_, actions_, logprobs_, std_obs_ = self.mappo._MAPPO__get_buffer_info(self.mappo.buffer)
        self.assertTrue(torch.allclose(rewards_, rewards))
        self.assertTrue(torch.allclose(states_, states))
        self.assertTrue(torch.allclose(actions_, actions))
        self.assertTrue(torch.allclose(logprobs_, logprobs))
        self.assertTrue(torch.allclose(std_obs_, std_obs))

    def test_update(self):
        self.mappo.update()
        self.assertEqual(len(self.mappo.buffer.states), 0)

    def test_save_and_load(self):
        filename = "test_model"
        episode = 0
        self.mappo.save(filename, episode)
        self.mappo.load(filename + "policy-" + str(episode) + ".pth")