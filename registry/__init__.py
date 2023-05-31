import json 
import os
from typing import List, Tuple, Union, Dict
import numpy as np
from src.environment import OptimizationEnv
from src.mappo import MAPPO
from commons.objective_functions import get_obj_func
import pickle


class Registry:
    """ 
    This class is to help register and retrieve optimization environments and objective functions.
    """
    def __init__(self):
        self.envs = {}
        self.obj_funcs = {}

    def add_env(self, env_name:str, env:OptimizationEnv):
        """ Adds an environment to the registry.
        :param env_name: The name of the environment.
        :param env: The environment.
        """
        if not isinstance(env, OptimizationEnv):
            raise TypeError('The environment must be an instance of OptimizationEnv.')
        # check if the environment is already registered
        if env_name in self.envs:
            raise ValueError(f'The environment {env_name} is already registered.')
        self.envs[env_name] = env

    def get_env(self, env_name:str) -> OptimizationEnv:
        """ Returns the environment with the given name.
        :param env_name: The name of the environment.
        :return: The environment.
        """
        if env_name not in self.envs:
            raise ValueError(f'The environment {env_name} is not registered.')
        return self.envs[env_name]
    
    def remove_env(self, env_name:str):
        """ Removes the environment with the given name.
        :param env_name: The name of the environment.
        """
        if env_name not in self.envs:
            raise ValueError(f'The environment {env_name} is not registered.')
        del self.envs[env_name]

    def add_obj_func(self, obj_func_name:str, obj_func:callable, bounds:Tuple[np.ndarray, np.ndarray]):
        """ Adds an objective function to the registry.
        :param obj_func_name: The name of the objective function.
        :param obj_func: The objective function.
        :param bounds: The bounds of the objective function.
        """
        if not callable(obj_func):
            raise TypeError('The objective function must be callable.')
        if not isinstance(bounds, tuple):
            raise TypeError('The bounds must be a tuple.')
        if not isinstance(bounds[0], np.ndarray) or not isinstance(bounds[1], np.ndarray):
            raise TypeError('The bounds must be a tuple of numpy arrays.')
        # check if the objective function is already registered
        if obj_func_name in self.obj_funcs:
            raise ValueError(f'The objective function {obj_func_name} is already registered.')
        self.obj_funcs[obj_func_name] = (obj_func, bounds)

    def get_obj_func(self, obj_func_name:str) -> Tuple[callable, Tuple[np.ndarray, np.ndarray]]:
        """ Returns the objective function with the given name.
        :param obj_func_name: The name of the objective function.
        :return: The objective function and their bounds.
        """
        if obj_func_name not in self.obj_funcs:
            raise ValueError(f'The objective function {obj_func_name} is not registered.')
        return self.obj_funcs[obj_func_name]
    
    def remove_obj_func(self, obj_func_name:str):
        """ Removes the objective function with the given name.
        :param obj_func_name: The name of the objective function.
        """
        if obj_func_name not in self.obj_funcs:
            raise ValueError(f'The objective function {obj_func_name} is not registered.')
        del self.obj_funcs[obj_func_name]

    def _save_envs(self, file_name:str):
        """ Saves the registered environments to a file.
        :param file_name: The name of the file.
        """
        save_dir = f"registry/{file_name}"
        os.makedirs(os.path.dirname(save_dir), exist_ok=True)
        # check if the file exists, load it, append the new environments and save it
        if os.path.exists(save_dir):
            with open(save_dir, 'rb') as f:
                envs = pickle.load(f)
            envs.update(self.envs)
            with open(save_dir, 'wb') as f:
                pickle.dump(envs, f)
        # if the file does not exist, save the environments
        else:
            with open(save_dir, 'wb') as f:
                pickle.dump(self.envs, f)

    def _load_envs(self, file_name:str):
        """ Loads the registered environments from a file.
        :param file_name: The name of the file.
        """
        load_dir = f"registry/{file_name}"
        if os.path.exists(load_dir):
            with open(load_dir, 'rb') as f:
                self.envs = pickle.load(f)

    def _save_obj_funcs(self, file_name:str):
        """ Saves the registered objective functions to a file.
        :param file_name: The name of the file.
        """
        save_dir = f"registry/{file_name}"
        os.makedirs(os.path.dirname(save_dir), exist_ok=True)
        with open(save_dir, 'wb') as f:
            pickle.dump(self.obj_funcs, f)

    def _load_obj_funcs(self, file_name:str):
        """ Loads the registered objective functions from a file.
        :param file_name: The name of the file.
        """
        load_dir = f"registry/{file_name}"
        if os.path.exists(load_dir):
            with open(load_dir, 'rb') as f:
                self.obj_funcs = pickle.load(f)

    def _save(self, file_name:str):
        """ Saves the registered environments and objective functions to a file.
        :param file_name: The name of the file.
        """
        self._save_envs(file_name)
        self._save_obj_funcs(file_name)

    def _load(self, file_name:str):
        """ Loads the registered environments and objective functions from a file.
        :param file_name: The name of the file.
        """
        self._load_envs(file_name)
        self._load_obj_funcs(file_name)




