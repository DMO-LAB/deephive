# imports
import numpy as np
from typing import Callable, List, Tuple, Union
from src.environment import OptimizationEnv as OptEnv
from registry import Registry
from src.mappo import MAPPO
import os
from commons.objective_functions import *
from argparse import ArgumentParser
import yaml
import torch
from decouple import config as conf
import matplotlib.pyplot as plt
import os
import imageio 
import re
from scipy.stats import sem, t

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_args():
    """
    Get the command line arguments.
    :return: The command line arguments.
    """
    # create the parser
    parser = ArgumentParser()
    # add the arguments
    parser.add_argument('-c', '--config', type=str,
                        default='config.yml', help='The path to the config file.')
    parser.add_argument('-tt', '--title', type=str,
                        default='experiment', help='The title of the experiment.')
    parser.add_argument('-m', '--mode', type=str, default='train',
                        help='The mode of the algorithm. (train, test, plot)')
    parser.add_argument('-e', '--env', type=str, default='cos_function',
                        help='The environment to train/test the algorithm on.')
    parser.add_argument('-l', '--log', type=bool,
                        default=False, help='Use neptune for logging.')
    parser.add_argument('-t', '--tags', type=str,
                        default='None', help='Tags for neptune logging.')
    parser.add_argument('-r', '--reinit', type=bool, default=False,
                        help='Whether to reinitialize the environment.')
    # parse the arguments
    args = parser.parse_args()
    # return the arguments
    return args

def get_config(path: str) -> dict:
    """
    Get the config from the yaml file.
    :param path: The path to the yaml file.
    :return: The config.
    """
    # check if the file exists
    if not os.path.exists(path):
        raise FileNotFoundError('The config file does not exist.')
    # open the file
    with open(path, 'r') as file:
        # load the yaml file
        config = yaml.safe_load(file)
    # return the config
    return config

def get_neptune_key() -> str:
    """
    Get the neptune key.
    :return: The neptune key.
    """
    return conf('neptune_secret_access_key') # type: ignore

def get_environment(config:dict, env_name: str, reinit=False) -> OptEnv:
    """
    Get the environment.
    :param config: The config.
    :param env_name: The name of the environment.
    :param reinit: Whether to reinitialize the environment.
    :return: The environment.
    """
    # create the registry
    registry = Registry()
    # load the environments from the file
    try:
        if reinit:
            raise Exception('Reinitialize environment')
        registry._load_envs(config['environment_config']['envs_cache_path'])
        env = registry.get_env(env_name)
    except:
        function_info = get_obj_func(env_name)
        if function_info is not None:
            objfunc, bounds, opt_obj_value, type = function_info
            env = OptEnv(env_name=env_name, optFunc=objfunc, n_agents=config['environment_config']['n_agents'], 
                         n_dim=config['environment_config']['n_dim'], ep_length=config['environment_config']['ep_length'], 
                         bounds=bounds, opt_value=opt_obj_value, 
                         reward_type=config['environment_config']['reward_type'], 
                         freeze=config['environment_config']['freeze'], 
                         opt_bound=config['environment_config']['opt_bound'])


            registry.add_env(env_name, env)
            registry._save_envs(config['environment_config']['envs_cache_path'])
        else:
            raise Exception('Optimization function not found. Ensure that the function is registered in commons\objective_functions.py')
    return env

def get_policy(config:dict, mode:str = "train") -> MAPPO:
    """ 
    Get the policy.
    :param config: The config.
    :param env: The environment.
    :return: The policy.
    """
    if mode == "test":
        policy_config = config['test_policy_config']
    elif mode == "train":
        policy_config = config['policy_config']
    else:
        raise ValueError("Mode must be either 'train' or 'test'")
    n_agents = config['environment_config']['n_agents']
    n_dim = config['environment_config']['n_dim']
    
    policy = MAPPO(n_agents=n_agents, n_dim=n_dim, state_dim=policy_config['state_dim'], action_dim=policy_config['action_dim'], action_std=policy_config['init_std'], std_min=policy_config['std_min'],
                    std_max=policy_config['std_max'], std_type=policy_config['std_type'], learn_std=policy_config['learn_std'], layer_size=policy_config['hidden_dim'], lr=policy_config['lr'], beta=policy_config['betas'],
                    gamma=policy_config['gamma'], K_epochs=policy_config['K_epochs'], eps_clip=policy_config['eps_clip'], pretrained=policy_config['pretrained'], ckpt_folder=policy_config['ckpt_folder'],
                    initialization=policy_config['initialization'])

    return policy
    
def plot_agents_trajectory_combined(env, plot_directory, gif_dir, title="exp.gif", fps=1.5):
    count = 0
    func = env.optFunc
    agents_pos = env.stateHistory
    markers = ['o', 'v', 's', 'p', 'P', '*', 'h', 'H', '+', 'x', 'X', 'D', 'd', '|', '_']
    while True:
        fig = plt.figure(figsize=(15, 10))
        ax = fig.add_subplot()
        fig.subplots_adjust(top=0.85)
        plt.clf()
        
        if env.n_dim == 1:
            X = np.linspace(env.min_pos, env.max_pos, 101)
            plt.plot(X, [func(x) for x in X])
            
            for i in range(len(agents_pos)):
                pos = agents_pos[i][count]
                plt.plot(pos, func(pos), marker=markers[0], markersize=10, color='blue')
        
        elif env.n_dim == 2:
            X = Y = np.linspace(env.min_pos, env.max_pos, 101)
            x, y = np.meshgrid(X, Y)
            Z = func(np.array([x.flatten(), y.flatten()]), plotable=True).reshape(x.shape)
            plt.contour(x, y, Z, 20)
            plt.colorbar()
            
            for i in range(len(agents_pos)):
                if i == env.bestAgentHistory[count]:
                    pos = agents_pos[i][count]
                    plt.plot(pos[0], pos[1], marker=markers[1], markersize=15, markerfacecolor='g')
                else:
                    pos = agents_pos[i][count]
                    plt.plot(pos[0], pos[1], marker=markers[0], markersize=15, markerfacecolor='k')
        
        elif env.n_dim > 2:
            raise ValueError("Cannot plot more than 2 dimensions")

        plt.text(1, 1, f"Step {count + 1}", style='italic', horizontalalignment='left', verticalalignment='top',
                 transform=ax.transAxes, bbox={'facecolor': 'blue', 'alpha': 0.5, 'pad': 10})
        plt.title(env.env_name)
        plt_dir = plot_directory + f"/{count}.png"
        plt.savefig(plt_dir)
        plt.close(fig)
        count += 1
        if count >= env.ep_length:
            break

    images = []
    filenames = os.listdir(plot_directory)
    filenames.sort(key=lambda f: int(re.sub('\D', '', f)))
    for filename in filenames:
        images.append(imageio.imread(f"{plot_directory}/{filename}"))
    imageio.mimsave(f"{gif_dir}/{title}", images, fps=fps)
    for filename in set(filenames):
        os.remove(f"{plot_directory}/{filename}")


def mean_confidence_interval(data, confidence=0.95):
        a = 1.0 * np.array(data)
        n = len(a)
        m, se = np.mean(a, axis = 0), sem(a) 
        h = se * t.ppf((1 + confidence) / 2., n-1)
        return m, m-h, m+h

def plot_num_function_evaluation(fopt, n_agents, save_dir, symbol_list=None, color_list=None, label_list=None, opt_value=None):
    fig = plt.figure(figsize=(6, 4), dpi=200)
    plt.rcParams["figure.figsize"] = [6, 4]
    plt.rcParams["figure.autolayout"] = True

    if symbol_list is None:
        symbol_list = ['-']
    if color_list is None:
        color_list = ['#3F7F4C']
    if label_list is None:
        label_list = ['DeepHive']

    if len(fopt) == 1:
        mf1, ml1, mh1 = mean_confidence_interval(fopt[0], 0.95)

        plt.fill_between((np.arange(len(mf1)) + 1) * n_agents, ml1, mh1, alpha=0.1, edgecolor='#3F7F4C',
                         facecolor='#7EFF99')
        plt.plot((np.arange(len(mf1)) + 1) * n_agents, mf1, linewidth=2.0, label=label_list[0], color=color_list[0])
    else:
        for i in range(len(fopt)):
            mf1, ml1, mh1 = mean_confidence_interval(fopt[i], 0.95)

            plt.fill_between((np.arange(len(mf1)) + 1) * n_agents, ml1, mh1, alpha=0.1, edgecolor='#3F7F4C',
                             facecolor=color_list[i])
            plt.plot((np.arange(len(mf1)) + 1) * n_agents, mf1, symbol_list[i], linewidth=2.0, label=label_list[i],
                     color=color_list[i])

    if opt_value is not None:
        plt.plot((np.arange(len(fopt[0])) + 1) * n_agents, np.ones(len(fopt[0])) * opt_value, linewidth=1.0, label='True OPT', 
                color='#CC4F1B') # type: ignore

    plt.xlabel('number of function evaluations', fontsize=14)
    plt.ylabel('best fitness value', fontsize=14)
    plt.legend(fontsize=8, frameon=False, loc="lower right")
    plt.xscale('log')
    plt.yticks(fontsize=14)
    plt.savefig(save_dir)
    plt.show()
