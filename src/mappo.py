from itertools import permutations
import numpy as np
import os
import time
from torch.distributions import Categorical
import torch.distributions as tdist
from torch.distributions import MultivariateNormal
import torch.nn as nn
import torch
import pandas as pd
import scipy.stats as stats
import math
from typing import List, Tuple, Dict, Union, Optional
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
from datetime import datetime
random_seed = 47


# Single Agent Memory Buffer
class Memory: #store old policy information for updating
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.is_terminals = []
        self.logprobs = []
        self.std_obs = []
    
    def clear_memory(self):
        del self.states[:]
        del self.actions[:]
        del self.rewards[:]
        del self.is_terminals[:]
        del self.logprobs[:]
        del self.std_obs[:]


# # Multi Agent Memory Buffer
# class MultiAgentMemory: #store old policy information for updating
#     def __init__(self, n_agents):
#         self.states = [[] for _ in range(n_agents)]
#         self.actions = [[] for _ in range(n_agents)]
#         self.rewards = [[] for _ in range(n_agents)]
#         self.is_terminals = [[] for _ in range(n_agents)]
#         self.logprobs = [[] for _ in range(n_agents)]
#         self.std_obs = [[] for _ in range(n_agents)]
    
#     def clear_memory(self):
#         for i in range(len(self.states)):
#             del self.states[i][:]
#             del self.actions[i][:]
#             del self.rewards[i][:]
#             del self.is_terminals[i][:]
#             del self.logprobs[i][:]
#             del self.std_obs[i][:]


class ActorCritic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int,  layer_size: List, 
                action_std_init : float = 0.2, std_min : float = 0.001, std_max: float = 0.2, std_type : str ='linear', learn_std: bool = True,):
        super(ActorCritic, self).__init__()

        self.actor = nn.Sequential(
            nn.Linear(state_dim, layer_size[0]),
            nn.Tanh(),
            nn.Linear(layer_size[0], layer_size[1]),
            nn.Tanh(),
            nn.Linear(layer_size[1], action_dim),
            #nn.Tanh()
        )
        
        self.critic = nn.Sequential(
            nn.Linear(state_dim, layer_size[0]),
            nn.Tanh(),
            nn.Linear(layer_size[0], layer_size[1]),
            nn.Tanh(),
            nn.Linear(layer_size[1], 1),
           #nn.Tanh()
        )

        self.action_dim = action_dim
        self.std_min = std_min
        self.learn_std=learn_std
        self.std_max = std_max
        self.std_type = std_type
        self.action_var = torch.full((action_dim,), action_std_init * action_std_init).to(device)   #action_std is a constant
          

        def direct_std(dist: np.ndarray) -> np.ndarray:
            std = dist + 0.001
            return std
        def half_direct_std(dist: np.ndarray) -> np.ndarray:
            std = dist/2 + 0.001
            std = self.std_min + (self.std_max - self.std_min) * std
            return std
        def linear_decay(dist: np.ndarray) -> np.ndarray:
            return self.std_min + (self.std_max - self.std_min) * dist
        def square_decay(dist: np.ndarray) -> np.ndarray:
            return self.std_min + (self.std_max - self.std_min) * dist**2
        def fourth_pow(dist: np.ndarray) -> np.ndarray:
            return self.std_min + (self.std_max - self.std_min) * dist**4
        def square_root_decay(dist: np.ndarray) -> np.ndarray:
            return self.std_min + (self.std_max - self.std_min) * dist**0.5
        def seventh_power_decay(dist: np.ndarray) -> np.ndarray:
            return self.std_min + (self.std_max - self.std_min) * dist**7
        def one_half_power(dist: np.ndarray) -> np.ndarray:
            return self.std_min + (self.std_max - self.std_min) * dist**1.5

        self.stds_formula = {
            'direct_decay': direct_std,
            'half_direct_decay': half_direct_std,
            'linear_decay': linear_decay,
            'square_decay': square_decay,
            'fourth_power_decay': fourth_pow,
            'square_root_decay': square_root_decay,
            'seventh_power_decay': seventh_power_decay,
            'one_half_power_decay': one_half_power
        }

    def set_action_std(self, new_action_std : float) -> torch.Tensor:
                self.action_var = torch.full(
                    (self.action_dim,), new_action_std * new_action_std).to(device)
    
    def forward(self):
        raise NotImplementedError

    def get_std(self, dist:np.ndarray) -> torch.Tensor:
        std = self.stds_formula[self.std_type](dist=dist)
        action_var = torch.from_numpy(std).to(device)
        return action_var

    def act(self, state: torch.Tensor, std_obs: np.ndarray = None) -> Tuple[torch.Tensor, torch.Tensor]:
        action_mean = self.actor(state)
    
        if self.learn_std:
            action_var = self.get_std(std_obs)
            dist = tdist.Normal(action_mean, action_var)
        else:
            cov_mat = torch.diag(self.action_var).to(device)
            dist = MultivariateNormal(action_mean, cov_mat)
        action = dist.sample()
        action_logprob = dist.log_prob(action)
        return action.detach(), action_logprob.detach()

    def evaluate(self, state: torch.Tensor, action: torch.Tensor, std_obs: np.ndarray = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        state_value = self.critic(state)
        # For Single Action Environments.
        if self.action_dim == 1:
            action = action.reshape(-1, self.action_dim)
        action_mean = self.actor(state)
        if self.learn_std:
            action_var = self.get_std(std_obs)
            dist = tdist.Normal(action_mean, action_var)
        else:
            cov_mat = torch.diag(self.action_var).to(device)
            dist = MultivariateNormal(action_mean, cov_mat)
        action_logprob = dist.log_prob(action)
        dist_entropy = dist.entropy()

        return action_logprob, torch.squeeze(state_value), dist_entropy


class PPO:
    def __init__(self, n_agents: int, n_dim: int, state_dim: int, action_dim: int, action_std: float, std_min: float,
                    std_max: float, std_type: str, learn_std: bool, layer_size: List[int],  lr: float = 0.0003, beta: float = 0.999,
                    gamma: float = 0.99, K_epochs: int = 80, eps_clip: float = 0.2, pretrained: bool = False, ckpt_folder: str = None,
                    initialization: str = None, split_agent: bool = False, **kwargs): 
       
        self.lr = lr
        self.kwargs = kwargs
        self.beta = beta
        self.gamma = gamma
        self.K_epochs = K_epochs
        self.eps_clip = eps_clip
        self.n_agents = n_agents
        self.n_dim = n_dim
        self.action_std = action_std
        self.exploitation_action_std = kwargs['exploit_init_std']
        self.exploration_action_std = kwargs['explore_init_std']

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.split_agent = split_agent
        if self.split_agent:
            self.exploration_buffer = Memory()
            self.exploitation_buffer = Memory()
        else:
            self.buffer = Memory()


        def init_weights(m):
            if type(m) == nn.Linear:
                torch.nn.init.xavier_uniform_(m.weight)

        # current policy
        if self.split_agent:
            self.exploitation_policy = ActorCritic(state_dim=kwargs['exploit_state_dim'], action_dim=action_dim,
                                                   action_std_init=kwargs['exploit_init_std'], layer_size=kwargs['exploit_hidden_dim'], 
                                                   std_min=kwargs['exploit_std_min'], std_max=kwargs['exploit_std_max'],
                                                    std_type=kwargs['exploit_std_type'], learn_std=kwargs['exploit_learn_std'],).to(self.device)
            
            self.exploration_policy = ActorCritic(state_dim=kwargs['explore_state_dim'], action_dim=action_dim,
                                                    action_std_init=kwargs['explore_init_std'], layer_size=kwargs['explore_hidden_dim'], 
                                                    std_min=kwargs['explore_std_min'], std_max=kwargs['explore_std_max'],
                                                    std_type=kwargs['explore_std_type'], learn_std=kwargs['explore_learn_std'],).to(self.device)
            
            if initialization is not None:
                self.exploitation_policy.apply(init_weights)
                self.exploration_policy.apply(init_weights)
            if pretrained:
                exploration_pretrained_model = torch.load(
                    kwargs["explore_ckpt_folder"], map_location=lambda storage, loc: storage)
                exploitation_pretrained_model = torch.load(
                    kwargs["exploit_ckpt_folder"], map_location=lambda storage, loc: storage)
                self.exploitation_policy.load_state_dict(exploitation_pretrained_model)
                self.exploration_policy.load_state_dict(exploration_pretrained_model)
                print(f"Pretrained model loaded from {kwargs['exploit_ckpt_folder']} and {kwargs['explore_ckpt_folder']}")

            self.exploitation_optimizer = torch.optim.Adam(self.exploitation_policy.parameters(), lr=lr)
            self.exploration_optimizer = torch.optim.Adam(self.exploration_policy.parameters(), lr=lr)

            # old policy
            self.old_exploitation_policy = ActorCritic(state_dim=kwargs['exploit_state_dim'], action_dim=action_dim,
                                                         action_std_init=kwargs['exploit_init_std'], layer_size=kwargs['exploit_hidden_dim'],
                                                            std_min=kwargs['exploit_std_min'], std_max=kwargs['exploit_std_max'],
                                                            std_type=kwargs['exploit_std_type'], learn_std=kwargs['exploit_learn_std'],).to(self.device)
            self.old_exploration_policy = ActorCritic(state_dim=kwargs['explore_state_dim'], action_dim=action_dim,
                                                        action_std_init=kwargs['explore_init_std'], layer_size=kwargs['explore_hidden_dim'],
                                                        std_min=kwargs['explore_std_min'], std_max=kwargs['explore_std_max'],
                                                        std_type=kwargs['explore_std_type'], learn_std=kwargs['explore_learn_std'],).to(self.device)
            self.old_exploitation_policy.load_state_dict(self.exploitation_policy.state_dict())
            self.old_exploration_policy.load_state_dict(self.exploration_policy.state_dict())

            self.MSE_loss = nn.MSELoss()
            
        else:
            self.policy = ActorCritic(state_dim, action_dim, action_std_init=action_std, layer_size=layer_size, std_min=std_min,
                                  std_max=std_max, std_type=std_type, learn_std=learn_std).to(self.device)
            
            if initialization is not None:
                self.policy.apply(init_weights)
            if pretrained:
                pretrained_model = torch.load(
                    ckpt_folder, map_location=lambda storage, loc: storage)
                self.policy.load_state_dict(pretrained_model)
                print(f"Loading pretrained model from {ckpt_folder}...")
            self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)

            # old policy
            self.old_policy = ActorCritic(
                state_dim, action_dim, action_std_init=action_std, layer_size=layer_size, std_min=std_min,
                std_max=std_max, std_type=std_type, learn_std=learn_std).to(device)
            # self.old_policy.apply(init_weights)
            self.old_policy.load_state_dict(self.policy.state_dict())

            self.MSE_loss = nn.MSELoss()

    def select_action(self, state, std_obs, agent_type='exploitation'):
        state = torch.FloatTensor(state).to(device)  # Flatten the stat
        if not self.split_agent:
            action, action_logprob = self.policy.act(state, std_obs)
            for i in range(self.n_agents):
                self.buffer.states.append(state[i])
                self.buffer.std_obs.append(std_obs[i])
                self.buffer.actions.append(action[i])
                self.buffer.logprobs.append(action_logprob[i])
        elif agent_type == 'exploitation':
            action, action_logprob = self.exploitation_policy.act(state, std_obs)
            for i in range(len(state)):
                self.exploitation_buffer.states.append(state[i])
                self.exploitation_buffer.std_obs.append(std_obs[i])
                self.exploitation_buffer.actions.append(action[i])
                self.exploitation_buffer.logprobs.append(action_logprob[i])
        elif agent_type == 'exploration':
            action, action_logprob = self.exploration_policy.act(state, std_obs)
            for i in range(len(state)):
                self.exploration_buffer.states.append(state[i])
                self.exploration_buffer.std_obs.append(std_obs[i])
                self.exploration_buffer.actions.append(action[i])
                self.exploration_buffer.logprobs.append(action_logprob[i])
        else:
            raise ValueError("agent_type must be either 'exploitation' or 'exploration'")

        return action.detach().cpu().numpy().flatten()

    def set_action_std(self, new_action_std, policy_type='exploitation'):
        if not self.split_agent:
            self.action_std = new_action_std
            self.policy.set_action_std(new_action_std)
            self.old_policy.set_action_std(new_action_std)
        elif policy_type == 'exploitation':
            self.exploitation_action_std = new_action_std
            self.exploitation_policy.set_action_std(new_action_std)
            self.old_exploitation_policy.set_action_std(new_action_std)
        elif policy_type == 'exploration':
            self.exploration_action_std = new_action_std
            self.exploration_policy.set_action_std(new_action_std)
            self.old_exploration_policy.set_action_std(new_action_std)
        # self.action_std = new_action_std
        # self.policy.set_action_std(new_action_std)
        # self.old_policy.set_action_std(new_action_std)

    def decay_action_std(self, action_std_decay_rate, min_action_std=0.001, learn_std=False, policy_type='exploitation'):
        if learn_std:
            pass
        else:
            if not self.split_agent:
                self.action_std = self.action_std - action_std_decay_rate
                self.action_std = round(self.action_std, 4)
                if (self.action_std <= min_action_std):
                    self.action_std = min_action_std
                else:
                    print(f"setting action std to {self.action_std}")
                self.set_action_std(self.action_std)
            else:
                if policy_type == 'exploitation':
                    self.exploitation_action_std = self.exploitation_action_std - action_std_decay_rate
                    self.exploitation_action_std = round(self.exploitation_action_std, 4)
                    if (self.exploitation_action_std <= min_action_std):
                        self.exploitation_action_std = min_action_std
                    else:
                        print(f"setting {policy_type} action std to {self.exploitation_action_std}")
                    self.set_action_std(self.exploitation_action_std, policy_type=policy_type)
                elif policy_type == 'exploration':
                    self.exploration_action_std = self.exploration_action_std - action_std_decay_rate
                    self.exploration_action_std = round(self.exploration_action_std, 4)
                    if (self.exploration_action_std <= min_action_std):
                        self.exploration_action_std = min_action_std
                    else:
                        print(f"setting {policy_type} action std to {self.exploration_action_std}")
                    self.set_action_std(self.exploration_action_std, policy_type=policy_type)
                else:
                    raise ValueError("policy_type must be either 'exploitation' or 'exploration'")
        

    def __get_buffer_info(self, buffer):
        # Monte Carlo estimate of state rewards:
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(buffer.rewards), reversed(buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards:
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)
        old_states = torch.stack(buffer.states).to(self.device).detach()
        old_actions = torch.stack(buffer.actions).to(self.device).detach()
        old_logprobs = torch.stack(buffer.logprobs).to(self.device).detach()
        if self.split_agent:
            old_std_obs = np.stack(buffer.std_obs)
        else:
            old_std_obs = buffer.std_obs
        return rewards, old_states, old_actions, old_logprobs, old_std_obs

    def __update_old_policy(self, policy, old_policy, optimizer, rewards, old_states, old_actions, old_logprobs, old_std_obs):
        # Optimize policy for K epochs:
        for _ in range(self.K_epochs):
            # Evaluating old actions and values :
            logprobs, state_values, dist_entropy = policy.evaluate(
                old_states, old_actions, old_std_obs)

            # Finding the ratio (pi_theta / pi_theta__old):
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss:
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip,
                                1+self.eps_clip) * advantages
            loss = -torch.min(surr1, surr2) + 0.5 * \
                self.MSE_loss(state_values, rewards) - 0.001*dist_entropy

            # take gradient step
            optimizer.zero_grad()
            loss.mean().backward()
            optimizer.step()

        # Copy new weights into old policy:
        old_policy.load_state_dict(policy.state_dict())
        return loss.mean().item()

    def update(self, policy_type='exploitation'):
        if not self.split_agent:
            # update general policy
            rewards, old_states, old_actions, old_logprobs, old_std_obs = self.__get_buffer_info(
                self.buffer)
            loss = self.__update_old_policy(self.policy, self.old_policy, self.optimizer,
                                     rewards, old_states, old_actions, old_logprobs, old_std_obs)
            self.buffer.clear_memory()
            print(f'[INFO]: policy updated with loss {loss} and buffer cleared')
            assert len(self.buffer.states) == 0
        else:
            if policy_type == 'exploration':
                # update exploration policy
                rewards, old_states, old_actions, old_logprobs, old_std_obs = self.__get_buffer_info(
                    self.exploration_buffer)
                loss = self.__update_old_policy(self.exploration_policy, self.old_exploration_policy,
                                        self.exploration_optimizer, rewards, old_states, old_actions, old_logprobs, old_std_obs)
                self.exploration_buffer.clear_memory()
                #print('[INFO]: exploration policy updated with loss {} and buffer cleared'.format(loss))
                assert len(self.exploration_buffer.states) == 0
            elif policy_type == 'exploitation':
                rewards, old_states, old_actions, old_logprobs, old_std_obs = self.__get_buffer_info(
                    self.exploitation_buffer)
                loss = self.__update_old_policy(self.exploitation_policy, self.old_exploitation_policy,
                                        self.exploitation_optimizer, rewards, old_states, old_actions, old_logprobs, old_std_obs)
                self.exploitation_buffer.clear_memory()
                #print('[INFO]: exploitation policy updated with loss {} and buffer cleared'.format(loss))
                assert len(self.exploitation_buffer.states) == 0
            else:
                raise Exception(
                    'policy_type should be either exploration or exploitation')
            

    # def update(self):
    #     # Monte Carlo estimate of rewards:
    #     rewards = []
    #     discounted_reward = 0

    #     for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
    #         if is_terminal:
    #             discounted_reward = 0
    #         discounted_reward = reward + (self.gamma * discounted_reward)
    #         rewards.insert(0, discounted_reward)

    #     # Normalizing the rewards:
    #     rewards = torch.tensor(rewards,  dtype=torch.float32).to(device)
    #     rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)
    #     #rewards = rewards.repeat_interleave(self.n_dim)

    #     # convert list to tensor
    #     old_states = torch.stack(self.buffer.states).to(device).detach()
    #     old_actions = torch.stack(self.buffer.actions).to(device).detach()
    #     old_logprobs = torch.stack(self.buffer.logprobs).to(device).detach()
    #     old_std_obs = np.stack(self.buffer.std_obs)

    #     # Optimize policy for K epochs:
    #     for _ in range(self.K_epochs):
    #         # Evaluating old actions and values :
    #         logprobs, state_values, dist_entropy = self.policy.evaluate(
    #             old_states, old_actions, old_std_obs)

    #         # Finding the ratio (pi_theta / pi_theta__old):
    #         ratios = torch.exp(logprobs - old_logprobs.detach())

    #         # Defining Advantage (A = R - V_pi_theta__old):
    #         advantages = rewards - state_values.detach()

    #         # Finding Surrogate Loss:
    #         surr_1 = ratios * advantages
    #         surr_2 = torch.clamp(ratios, 1-self.eps_clip,
    #                              1+self.eps_clip) * advantages
    #         actor_loss = -torch.min(surr_1, surr_2)

    #         # Finding the value loss:
    #         critic_loss = 0.5 * self.MSE_loss(state_values, rewards) - 0.001 * dist_entropy

    #         # Finding the total loss:
    #         loss = actor_loss + critic_loss

    #         # Updating the policy:
    #         self.optimizer.zero_grad()
    #         loss.mean().backward()
    #         self.optimizer.step()

    #     # Copy new weights into old policy:
    #     self.old_policy.load_state_dict(self.policy.state_dict())
    #     self.buffer.clear_memory()

    #save policy
    def save(self, filename, policy_type='exploitation', episode=0):
        if not self.split_agent:
            torch.save(self.policy.state_dict(), filename + "policy-" + str(episode) + ".pth")
            print("Saved policy to: ", filename)
        else:
            if policy_type == 'exploration':
                torch.save(self.exploration_policy.state_dict(),
                        filename + "exploration_policy-" + str(episode) + ".pth")
                print("Saved exploration policy to: ", filename + "exploration_policy-" + str(episode) + ".pth")
            elif policy_type == 'exploitation':
                torch.save(self.exploitation_policy.state_dict(),
                        filename + "exploitation_policy-" + str(episode) + ".pth")
                print("Saved exploitation policy to: ", filename + "exploitation_policy-" + str(episode) + ".pth")

    def load(self, filename, policy_type='exploitation'):
        if not self.split_agent:
            self.policy.load_state_dict(torch.load(filename))
            print("Loaded policy from: ", filename)
        else:    
            if policy_type == 'exploration':
                self.exploration_policy.load_state_dict(
                    torch.load(filename))
                print("Loaded exploration policy from: ", filename)
            elif policy_type == 'exploitation':
                self.exploitation_policy.load_state_dict(
                    torch.load(filename))
                print("Loaded exploitation policy from: ",
                    filename + "exploitation_policy")


