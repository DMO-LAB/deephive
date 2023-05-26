from typing import Any, Dict, List, Optional, Tuple, Union

import gym
import numpy as np
from gym import spaces


class OptimizationEnv(gym.Env):
    def __init__(self, env_name: str, optFunc: callable, n_agents: int, n_dim: int, 
                 bounds: Tuple[float, float], ep_length: int, 
                 init_state: Optional[np.ndarray] = None, opt_bound: float = 0.9, 
                 reward_type: str = "full", freeze: bool = False, opt_value: float = 0.9):
        """
        Args:
            env_name: name of the environment
            optFunc: function to optimize
            n_agents: number of agents
            n_dim: dimension of the problem
            bounds: bounds of the problem
            ep_length: length of the episode
            init_state: initial state of the agents
            opt_bound: optimal bound
            reward_type: type of reward function
            freeze: whether to freeze the best agent
            opt_value: optimal value
        """
        self.env_name = env_name
        self.optFunc = optFunc
        self.n_agents = n_agents
        self.n_dim = n_dim
        self.bounds = bounds
        self.ep_length = ep_length
        self.init_state = init_state 
        self.opt_bound = opt_bound
        self.reward_type = reward_type
        self.freeze = freeze
        self.opt_value = opt_value
        self._reset_buffer()

    def _reset_buffer(self):
        self.current_step = 0
        self.bestAgentHistory = []
        self.min_pos = self.bounds[0]
        self.max_pos = self.bounds[1]
        self.stateHistory_vec = np.zeros((self.n_agents, self.ep_length+1, self.n_dim))
        self.ValueHistory_vec = np.zeros((self.n_agents, self.ep_length+1))

        self.lower_bound_actions = np.array(
            [-np.inf for _ in range(self.n_dim)], dtype=np.float64)
        self.upper_bound_actions = np.array(
            [np.inf for _ in range(self.n_dim)], dtype=np.float64)
        self.lower_bound_obs = np.append(np.array(
            [self.min_pos[i] for i in range(self.n_dim)], dtype=np.float64), -np.inf)
        self.upper_bound_obs = np.append(np.array(
            [self.max_pos[i] for i in range(self.n_dim)], dtype=np.float64), np.inf)
        
        self.low = np.array([self.lower_bound_obs for _ in range(self.n_agents)])
        self.high = np.array([self.upper_bound_obs for _ in range(self.n_agents)])
        self.action_low = np.array([self.lower_bound_actions for _ in range(self.n_agents)])
        self.action_high = np.array([self.upper_bound_actions for _ in range(self.n_agents)])

        self.action_space = spaces.Box(
            low=self.action_low, high=self.action_high, dtype=np.float64)
        self.observation_space = spaces.Box(
            low=self.low, high=self.high, dtype=np.float64)
    
    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Run one timestep of the environment's dynamics.
        Accepts an action and returns a tuple (observation, reward, done, info).
        Args:
            actions: actions to take
        Returns:
            observation: agent's observation of the current environment
            reward: amount of reward returned after previous action
            done: whether the episode has ended, in which case further step() calls will return undefined results
            info: contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
        """
        assert self.action_space.contains(actions), f"{actions!r} ({type(actions)}) invalid"
        agents_done = []
        states = self.state.copy()
        self.bestAgent = np.argmax(self.obj_values)
        self.zprev = states[:, -1].copy()
        # update state
        states[:, :-1] += actions
        # freeze best agent
        if self.freeze:
            states[self.bestAgent, :-1] = self.state[self.bestAgent, :-1]
        # restrict state to bounds
        states = np.clip(states, np.zeros_like(states), np.ones_like(states))
        scaled_states = self._rescale(states[:, :-1], self.min_pos, self.max_pos)
        self.obj_values = self.optFunc(scaled_states)
        self.current_step += 1
        self.done = False
        if self.current_step >= self.ep_length-1:
            self.done = True
        agents_done = [self.done for _ in range(self.n_agents)]
        
        # update best objective value
        if np.max(self.obj_values) > self.best:
            self.best = np.max(self.obj_values)
            self.bestAgentChange += 1
        if np.min(self.obj_values) < self.worst:
            self.worst = np.min(self.obj_values)

        # scale objective value to [0, 1]
        states[:, -1] = self._scale(self.obj_values, self.worst, self.best).reshape(-1)
        self.state = states

        # update state history and define reward
        self.stateHistory_vec[:, self.current_step, :] = self._get_actual_state()[:, :-1]
        self.ValueHistory_vec[:, self.current_step] = self.obj_values
        self.exploiter, self.explorer = self._get_exploiting_agents()
        agents_rewards = self._reward_fcn_vec_split() if self.split_agents else self._reward_fcn_vec()
        self.bestAgentHistory.append(np.argmax(self.obj_values))
        
        return self.state, agents_rewards, agents_done, self.obj_values
    
    def _get_stuck_agents(self, threshold: int = 2) -> List[int]:
        """
        Check if agents are stuck in a local minimum
        Args:
            threshold: number of previous steps to check
        Returns:
            stuck_agents: list of agents that are stuck
        """
        prev_state = np.round(
            self.ValueHistory_vec[:, self.current_step - threshold], 4)
        if np.any(prev_state == self.obj_values):
            # get the agents that are stuck
            stuck_agents = np.where(
                np.all(self.obj_values == prev_state))[0]
            # remove the best agent
            stuck_agents = stuck_agents[stuck_agents != self.bestAgent]
            return stuck_agents
        else:
            return []
    
    def _reward_fcn_vec(self, threshold: int = 2) -> np.ndarray:
        """
        Reward function for all agents
        Args:
            threshold: number of previous steps to check
        Returns:
            reward: reward for each agent
        """
        if self.reward_type == "full":
            reward = 10*(self.state[:, -1] - self.zprev)
        elif self.reward_type == "sparse":
            reward = 0
        # check if agents are stuck
        stuck_agents = self._get_stuck_agents(threshold=threshold)
        if len(stuck_agents) > 0:
            reward[stuck_agents] = -1
        # check if agents are optimal
        if np.any(self.state[:, -1] >= self.opt_bound):
            reward[self.state[:, -1] >= self.opt_bound] += 10*self.state[self.state[:, -1] >= self.opt_bound, -1]
        if self.done:
            reward[self.state[:, -1] >= self.opt_bound] += 10
        # freeze best agent
        if self.freeze:
            reward[self.bestAgent] = 0
        return reward
    

    def _get_actual_state(self, state: np.ndarray = None) -> np.ndarray:
        """ Get the actual state of the agents by rescaling the state to the original bounds
        Args:
            state: state to rescale
        Returns:
            actual_state: actual state of the agents
        """
        if state is None:
            state = self.state.copy()
            obj_value = self.obj_values.copy()
            actual_state = self._rescale(state[:, :2], self.min_pos, self.max_pos)
            actual_state = np.append(
                    actual_state, obj_value.reshape(-1, 1), axis=1)
        else:
            actual_state = self._rescale(state[:, :2], self.min_pos, self.max_pos)
            actual_state = np.append(
                    actual_state, self.optFunc(actual_state, minimize=False).reshape(-1, 1), axis=1)
        return actual_state
            
    def _generate_init_state(self):
        """ Generate a random initial state for all agents
        Returns:
            init_state: initial state of all agents
        """
        if self.init_state is None:
            init_pos = np.round(np.random.uniform(
                low=self.low[0][:-1], high=self.high[0][:-1],), decimals=2)
        else:
            init_pos = np.array(self.init_state)
        # generate a random initial position for all agents at once
        init_pos = np.round(np.random.uniform(
            low=self.low[0][:-1], high=self.high[0][:-1], size=(self.n_agents, self.n_dim)), decimals=2)
        # get the objective value of the initial position
        self.obj_values = self.optFunc(init_pos)
        # scale the position to [0, 1]
        init_pos = self._scale(init_pos, self.min_pos, self.max_pos)
        # combine the position and objective value
        init_obs = np.append(init_pos, self.obj_values.reshape(-1, 1), axis=1)
        return init_obs
    
    def reset(self, seed: Optional[int] = None):
        """ Reset the environment
        Args:
            seed: random seed
        Returns:
            state: initial state of all agents
        """
        self.state = self._generate_init_state()
        self.stateHistory_vec[:, self.current_step, :] = self._get_actual_state()[:, :-1]
        self.ValueHistory_vec[:, self.current_step] = self.obj_values
        self.best = np.max(self.state[:, -1])
        self.worst = np.min(self.state[:, -1])
        self.state[:,-1] = self._scale(self.state[:,-1], self.worst, self.best)
        self.bestAgent = np.argmax(self.obj_values)
        self.bestAgentHistory.append(self.bestAgent)
        self.bestAgentChange = 0
        return np.array(self.state, dtype=np.float32)
    
    def _scale(self, d: np.ndarray, dmin: np.ndarray, dmax: np.ndarray):
        """ Scale the input to the range [0, 1]
        Args:
            d: input to scale
            dmin: minimum value of the input
            dmax: maximum value of the input
        Returns:
            scaled_d: scaled input
        """
        return (d-dmin)/((dmax-dmin) + 10e-9)
        
    def _rescale(self, d: np.ndarray, dmin: np.ndarray, dmax: np.ndarray):
        """ Rescale the input to the range [dmin, dmax]
        Args:
            d: input to rescale
            dmin: minimum value of the input
            dmax: maximum value of the input
        Returns:
            rescaled_d: rescaled input
        """
        return (d*(dmax-dmin)+ dmin)