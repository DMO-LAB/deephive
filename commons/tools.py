# imports
import numpy as np
from typing import Callable, List, Tuple
from environment import OptEnv
from environment_cache import OptEnvCache
from mappo import PPO
import os
from objective_functions import *
from argparse import ArgumentParser
import yaml
import torch
import random
from decouple import config as conf
import matplotlib.pyplot as plt
import os
import imageio  # for gif
import re
import scipy
from other_algorithms.pso import pso

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
                        default='config.yaml', help='The path to the config file.')
    
    # title
    parser.add_argument('-tt', '--title', type=str,
                        default='None', help='The title of the experiment.')
    parser.add_argument('-m', '--mode', type=str, default='train',
                        help='The mode of the algorithm. (train, test, plot)')
    parser.add_argument('-e', '--env', type=str, default='cos_function',
                        help='The environment to train/test the algorithm on.')
    parser.add_argument('-l', '--log', type=str,
                        default='False', help='Use neptune for logging.')
    parser.add_argument('-t', '--tags', type=str,
                        default='None', help='Tags for neptune logging.')
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
    return conf('neptune_secret_access_key')


def prepare_environment(config: dict, load_from_file: bool = False) -> OptEnvCache:
    """
    Prepare the environment.
    :param config: The config.
    :return: The environment.
    """
    envs_cache = OptEnvCache()
    # prepare each environment as specified in the config file and cache them
    env_config = config['environment_config']
    # check of the env_config['envs_cache_path'] exists
    # if not, create it
    if not os.path.exists(env_config['envs_cache_path']):
        os.makedirs(env_config['envs_cache_path'])
    if not load_from_file:
        for i in range(len(env_config['env_list'])):
            # create the environment
            func_name = env_config['env_list'][i]
            obj_func, bounds, opt_obj_value, type = get_obj_func(
                env_config['env_list'][i])
        
            # if n_dim is not 2D, ignore the bounds and use bounds from the config file
            if env_config['n_dim'] != 2:
                bounds = env_config['bounds'][i]
            env = OptEnv(func_name, obj_func, env_config['n_agents'], env_config['n_dim'],
                         bounds, env_config['ep_length'], env_config['freeze'], 
                         opt_bound=env_config['opt_bound'], reward_type=env_config['reward_type'], split=env_config['split_agent'], opt_value=opt_obj_value)
            # cache the environment
            envs_cache.add_env(func_name, env)
        envs_cache.save_envs(
            env_config['envs_cache_path'] + env_config['envs_cache_file'])
    else:
        # load the environments from the file
        envs_cache.load_envs(
            env_config['envs_cache_path'] + env_config['envs_cache_file'])
    # return the environment
    return envs_cache

def prepare_policy(config:dict, test:bool = False) -> PPO:
    """
    Prepare the policy.
    :param config: The config.
    :param env: The environment.
    :return: The policy.
    """
    policy_config = config['policy_config']
    test_policy_config = config['test_config']
    xploit_config = config['exploiting_policy_config']
    xplore_config = config['exploring_policy_config']
    # create the policy
    n_agents = config['environment_config']['n_agents']
    n_dim = config['environment_config']['n_dim']
    split_agent = config['environment_config']['split_agent']
    if test:
        action_dim = policy_config['action_dim']
        action_std = test_policy_config['init_std']
        state_dim = policy_config['state_dim']
        std_min = test_policy_config['std_min']
        std_max = test_policy_config['std_max']
        std_type = test_policy_config['std_type']
        learn_std = test_policy_config['learn_std']
        hidden_dim = policy_config['hidden_dim']
        pretrained = test_policy_config['pretrained']
        ckpt_folder = test_policy_config['ckpt_folder']
        split_agent = test_policy_config['split_agent']
    else:
        action_dim = policy_config['action_dim']
        action_std = policy_config['init_std']
        state_dim = policy_config['state_dim']
        std_min = policy_config['std_min']
        std_max = policy_config['std_max']
        std_type = policy_config['std_type']
        learn_std = policy_config['learn_std']
        hidden_dim = policy_config['hidden_dim']
        pretrained = policy_config['pretrained']
        ckpt_folder = policy_config['ckpt_folder']

    lr = policy_config['lr']
    beta = policy_config['betas']
    gamma = policy_config['gamma']
    K_epochs = policy_config['K_epochs']
    eps_clip = policy_config['eps_clip']
    initialization = policy_config['initialization']
    kwargs = {"explore_state_dim": xplore_config['state_dim'] , "exploit_state_dim": xploit_config['state_dim'],
                "explore_action_dim": xplore_config['action_dim'], "exploit_action_dim": xploit_config['action_dim'],
                "explore_init_std": xplore_config['init_std'], "exploit_init_std": xploit_config['init_std'],
                "explore_std_min": xplore_config['std_min'], "exploit_std_min": xploit_config['std_min'],
                "explore_std_max": xplore_config['std_max'], "exploit_std_max": xploit_config['std_max'],
                "explore_std_type": xplore_config['std_type'], "exploit_std_type": xploit_config['std_type'],
                "explore_fixed_std": xplore_config['fixed_std'], "exploit_fixed_std": xploit_config['fixed_std'],
                "explore_activation": xplore_config['activation'], "exploit_activation": xploit_config['activation'],
                "explore_hidden_dim": xplore_config['hidden_dim'], "exploit_hidden_dim": xploit_config['hidden_dim'],
                "explore_learn_std": xplore_config['learn_std'], "exploit_learn_std": xploit_config['learn_std'],
                "explore_pretrained": xplore_config['pretrained'], "exploit_pretrained": xploit_config['pretrained'],
                "explore_ckpt_folder": xplore_config['ckpt_folder'], "exploit_ckpt_folder": xploit_config['ckpt_folder'],
                "explore_initialization": xplore_config['initialization'], "exploit_initialization": xploit_config['initialization'],
    }
    
    
    policy = PPO(n_agents=n_agents, n_dim=n_dim, state_dim=state_dim, action_dim=action_dim, action_std=action_std, std_min=std_min,
                 std_max=std_max, std_type=std_type, learn_std=learn_std, layer_size=hidden_dim, lr=lr, beta=beta,
                 gamma=gamma, K_epochs=K_epochs, eps_clip=eps_clip, pretrained=pretrained, ckpt_folder=ckpt_folder,
                 initialization=initialization, split_agent=split_agent, **kwargs)
    return policy

def create_work_dir(exp_name, log_folder="logs", mode="train"):
    log_dir = log_folder + "/"
    title = exp_name
    directory = log_dir + title + '/'
    if not os.path.exists(directory):
        os.makedirs(directory)
    plot_dir = directory + "plots/"
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    gif_dir = directory + "gifs/"
    if not os.path.exists(gif_dir):
        os.makedirs(gif_dir)  
    if mode == "train":  
        checkpoint_dir  = directory + "checkpoints/"
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        return exp_name, directory, plot_dir, gif_dir, checkpoint_dir
    elif mode == "test":
        return exp_name, directory, plot_dir, gif_dir   
    else:
        raise ValueError("Mode must be either train or test")
    
def scale_(d, dmin, dmax):
    # dmin = np.array(dmin)
    # dmax = np.array(dmax)
    """ Scale the input to the range [0, 1] """
    return (d-dmin)/((dmax-dmin) + 10e-9)
    
def rescale_(d, dmin, dmax):
    # dmin = np.array(dmin)
    # dmax = np.array(dmax)
    """ Rescale the input to the range [dmin, dmax] """
    return (d*(dmax-dmin)+ dmin)

def get_agent_actions(agent_actions, n_dim, n_agents, fitness=None):
    ''' Get the actions for each agent '''
    actions = [[] for _ in range(n_agents)]
    n = 0
    for i in range(0, len(agent_actions[0])):
        for dim in range(n_dim):
            action = agent_actions[dim][i]
            actions[n].append(action)
        n += 1
    return np.array(actions)

def get_agents_obs(states, n_agents, n_dim, pbest=None, gbest=None, F=False, std="euclidean", use_gbest=False):
    """ Get Observations for each Agent"""
    agent_obs = [[] for _ in range(n_dim)]
    std_obs = [[] for _ in range(n_dim)]
    nbs = []
    for agent in range(n_agents):
        agents_nbs = [i for i in range(n_agents)]
        nbs.append(agent)
        choices = list(filter(lambda ag: ag not in nbs, agents_nbs))
        if len(choices) == 0:
            choices = list(filter(lambda ag: ag != agent, agents_nbs))
        agent_nb = random.choice(choices)
        nbs.remove(agent)
        nbs.append(agent_nb)
        std = np.sqrt(np.sum((states[agent][:-1] - gbest[:-1])**2))
        
        for dim in range(n_dim):
            if use_gbest:
                obs = [(states[agent][dim]-pbest[agent][dim]),(states[agent][n_dim]-pbest[agent][n_dim]),
                    states[agent][dim]-pbest[agent_nb][dim], states[agent][n_dim]-pbest[agent_nb][n_dim],
                    states[agent][dim]-gbest[dim], states[agent][n_dim]-gbest[n_dim]]
            else:
                obs = [(states[agent][dim]-pbest[agent][dim]),(states[agent][n_dim]-pbest[agent][n_dim]),
                        states[agent][dim]-pbest[agent_nb][dim], states[agent][n_dim]-pbest[agent_nb][n_dim]]
            agent_obs[dim].append(np.array([obs]))
            if std == "euclidean":
                std_obs[dim].append(std)
            else:
                std_obs[dim].append(abs(gbest[dim]-states[agent][dim]))
        if F:
            print(f"Agent {agent} state : {states[agent]}, pbest : {pbest[agent]}, std : {std}")

    obss = [np.array(agent_obs[i]).reshape(n_agents, len(obs))
            for i in range(n_dim)]
    std_obss = [np.array(std_obs[i]).reshape(n_agents, 1)
                for i in range(n_dim)]
    return obss, std_obss

## Training utils
def get_agents_obs_v1(states, n_agents, n_dim, pbest=None, gbest=None, F=False, std="euclidean"):
    """ Get Observations for each Agent"""
    agent_obs = [[] for _ in range(n_dim)]
    std_obs = [[] for _ in range(n_dim)]
    nbs = []
    for agent in range(n_agents):
        agents_nbs = [i for i in range(n_agents)]
        nbs.append(agent)
        choices = list(filter(lambda ag: ag not in nbs, agents_nbs))
        if len(choices) == 0:
            choices = list(filter(lambda ag: ag != agent, agents_nbs))
        agent_nb = random.choice(choices)
        nbs.remove(agent)
        nbs.append(agent_nb)
        std = np.sqrt(np.sum((states[agent][:-1] - gbest[:-1])**2))
        
        for dim in range(n_dim):
            obs = [(states[agent][dim]-pbest[agent][dim]),(states[agent][n_dim]-pbest[agent][n_dim]),
                    states[agent][dim]-pbest[agent_nb][dim], states[agent][n_dim]-pbest[agent_nb][n_dim],
                    states[agent][dim]-gbest[dim], states[agent][n_dim]-gbest[n_dim]]
            agent_obs[dim].append(np.array([obs]))
            if std == "euclidean":
                std_obs[dim].append(std)
            else:
                std_obs[dim].append(abs(gbest[dim]-states[agent][dim]))
        if F:
            print(f"Agent {agent} state : {states[agent]}, pbest : {pbest[agent]}, gbest : {gbest}, std : {std}")

    obss = [np.array(agent_obs[i]).reshape(n_agents, 6)
            for i in range(n_dim)]
    std_obss = [np.array(std_obs[i]).reshape(n_agents, 1)
                for i in range(n_dim)]
    return obss, std_obss

## Plotting functions ##
def plot_agents_trajectory(env, plot_directory, opt_func_name, ep_length=20, title="exp.gif", fps=1.5):
    count = 0
    func = env.optFunc
    agents_pos = env.stateHistory_vec
    exploiter_buffer = env.exploiters_buffer
    markers = ['o','v','s','p','P','*','h','H','+','x','X','D','d','|','_']
    while True:
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot()
        fig.subplots_adjust(top=0.85)
        plt.clf()
        X = np.linspace(env.min_pos, env.max_pos, 101)
        plt.plot(X, [func(x) for x in X]) 
        for i in range(len(agents_pos)):
            pos = agents_pos[i][count]
            if i in exploiter_buffer[count]:
                plt.plot(pos, func(pos), marker=markers[1], markersize=10, color='red')
            else:                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       
                plt.plot(pos, func(pos), marker=markers[0], markersize=10, color='blue')
        plt.text(1, 1, f"Step {count+1}", style='italic',horizontalalignment='left', verticalalignment='top', transform=ax.transAxes,
            bbox={'facecolor': 'blue', 'alpha': 0.5, 'pad': 10})
        plt.title(opt_func_name)
        plt_dir = plot_directory + f"{count}.png"  
        plt.savefig(plt_dir) 
        plt.close(fig)
        count += 1
        if count >= ep_length:
            break
    images = []
    filenames = os.listdir(plot_directory)
    filenames.sort(key=lambda f: int(re.sub('\D', '', f)))
    for filename in filenames:
        images.append(imageio.imread(plot_directory + filename))
    imageio.mimsave(title, images, fps=fps)
    for filename in set(filenames):
        os.remove(plot_directory + filename)


def plot_agents_trajectory_2D(env, plot_directory, opt_func_name, ep_length=20, title="exp2D.gif", fps=0.7):
    count = 0
    func = env.optFunc
    agents_pos = env.stateHistory_vec
    exploiter_buffer = env.exploiters_buffer
    markers = ['o','v','s','p','P','*','h','H','+','x','X','D','d','|','_']
    while True:
        fig = plt.figure(figsize=(15, 10))
        ax = fig.add_subplot()
        fig.subplots_adjust(top=0.85)
        plt.clf()
        X =  Y = np.linspace(env.min_pos, env.max_pos, 101)
        x, y = np.meshgrid(X, Y)
        Z = func(np.array([x.flatten(), y.flatten()]), plotable=True ).reshape(x.shape)
        plt.contour(x, y, Z, 20)
        plt.colorbar()
        for i in range(len(agents_pos)):
            if i == env.bestAgentHistory[count]:
                pos = agents_pos[i][count]
                plt.plot(pos[0], pos[1] ,marker=markers[1], markersize=15, markerfacecolor='g')
            elif i in exploiter_buffer[count]:
                pos = agents_pos[i][count]
                plt.plot(pos[0], pos[1] ,marker=markers[0], markersize=15, markerfacecolor='r')
            else:
                pos = agents_pos[i][count]
                plt.plot(pos[0], pos[1] ,marker=markers[0], markersize=15, markerfacecolor='k')
            plt.text(env.max_pos, env.max_pos, f"Step {count+1}", style='italic',
            bbox={'facecolor': 'blue', 'alpha': 0.5, 'pad': 10})
        #plt.pause(0.5)
        plt.title(opt_func_name)
        plt_dir = plot_directory + f"{count}.png"  
        plt.savefig(plt_dir) 
        plt.close(fig)
        plt.show()
        count += 1
        if count >= ep_length:
            break
    images = []
    filenames = os.listdir(plot_directory)
    filenames.sort(key=lambda f: int(re.sub('\D', '', f)))
    for filename in filenames:
        images.append(imageio.imread(plot_directory + filename))
    imageio.mimsave(title, images, fps=fps)
    for filename in set(filenames):
        os.remove(plot_directory + filename)


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a, axis = 0), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h

def num_function_evaluation(fopt, n_agents, save_dir, opt_value):
    mf1 = np.mean(fopt, axis = 0)
    err = np.std(fopt, axis = 0)
    mf1, ml1, mh1 = mean_confidence_interval(fopt,0.95)

    fig = plt.figure(figsize=(6,4), dpi=200)
    plt.rcParams["figure.figsize"] = [6, 4]
    plt.rcParams["figure.autolayout"] = True
    plt.fill_between((np.arange(len(mf1))+1)*n_agents, ml1, mh1, alpha=0.1, edgecolor='#3F7F4C', facecolor='#7EFF99')
    plt.plot((np.arange(len(mf1))+1)*n_agents, mf1, linewidth=2.0, label = 'RL OPT', color='#3F7F4C')
    plt.plot((np.arange(len(mf1))+1)*n_agents, np.ones(len(mf1))*opt_value, linewidth=1.0, label = 'True OPT', color='#CC4F1B')

    plt.xlabel('number of function evaluations', fontsize = 14)
    plt.ylabel('best fitness value', fontsize = 14)

    plt.legend(fontsize = 14, frameon=False)
    plt.xscale('log')
    plt.yticks(fontsize = 14)
    plt.savefig(save_dir)
    plt.show()

def num_function_evaluation_mul(fopt, n_agents, save_dir, symbol_list, color_list, label_list, opt_value):
    c = color_list
    title = label_list
    fig = plt.figure(figsize=(6,4), dpi=200)
    for i in range(len(fopt)):

        mf1 = np.mean(fopt[i], axis = 0)
        err = np.std(fopt[i], axis = 0)
        mf1, ml1, mh1 = mean_confidence_interval(fopt[i],0.95)
        
        plt.rcParams["figure.figsize"] = [6, 4]
        plt.rcParams["figure.autolayout"] = True
        plt.fill_between((np.arange(len(mf1))+1)*n_agents, ml1, mh1, alpha=0.1, edgecolor='#3F7F4C', facecolor=c[i])
        plt.plot((np.arange(len(mf1))+1)*n_agents, mf1, symbol_list[i], linewidth=2.0,  label = title[i], color=c[i])
        plt.plot((np.arange(len(mf1))+1)*n_agents, np.ones(len(mf1))*opt_value, linewidth=1.0, label = 'True OPT', color='#CC4F1B')

    plt.xlabel('number of function evaluations', fontsize = 14)
    plt.ylabel('best fitness value', fontsize = 14)

    plt.legend(fontsize = 8, frameon=False,loc="lower right")
    #plt.legend()
    plt.xscale('log')
    plt.yticks(fontsize = 14)
    plt.savefig(save_dir)
    plt.show()

def get_max(series):
    series_ = series.copy()
    min_ = 10000
    for ii in range(len(series)):
        if series[ii] < min_:
            min_ = series[ii]
        series_[ii] = min_
    return series_


def test_pso(function, bounds, num_run, swarmsize, maxiter, minstep, minfunc, debug):
    lb = bounds[0]
    ub = bounds[1]
    F = []
    opt = []
    minf = 100
    def aux_function(xx, function):
        nonlocal Nfeval
        nonlocal opt 
        z = function(xx, minimize=False)
        opt += [-z]
        return -z
    aa = []
    for i in range(num_run):
        opt = []
        Nfeval = 0
        _, g = pso(aux_function, lb, ub, args=(function,), swarmsize=swarmsize, maxiter=maxiter, minstep=minstep, minfunc=minfunc, debug=debug)
        F += [g]
        aa += [-get_max(opt)]
    return aa