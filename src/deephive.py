import tools
import os
import numpy as np
from mappo import PPO
import neptune.new as neptune
from decouple import config as conf
from datetime import datetime
from distutils.util import strtobool
import parameter as param


def main(env_name, title, config, policy, env_cache):
    exp_name, directory, plot_dir, gif_dir, checkpoint_dir = tools.create_work_dir(title, log_folder="logs")
    # get necessary parameters from config
    n_agents = config['test_config']['n_agents']
    n_dim = config['test_config']['n_dim']
    ep_length = config['test_config']['ep_length']
    n_run = config['test_config']['n_run']
    env = env_cache.get_env(env_name)
    total_run_time = 0
    f = []
    bestValues = []
    split_agent = config['test_config']['split_agent']
    print("TESTING RUN SUMMARY")
    print(f"Environment: {env_name}")
    print(f"Number of agents: {n_agents}")
    print(f"Number of dimensions: {n_dim}")
    print(f"Episode length: {ep_length}")
    print(f"Number of runs: {n_run}")
    print(f"Split agent: {split_agent}")
    print(f"Title: {title}")
    if not split_agent:
        print(f"Minimum std: {param.test_min_std}")
        print(f"Inital std: {param.test_init_std}")
        print(f"Decay rate: {param.test_std_decay_rate}")
    else:
        print(f"Minimum std: {param.explore_min_std} (exploration), {param.exploit_min_std} (exploitation)")
        print(f"Inital std: {param.explore_init_std} (exploration), {param.exploit_init_std} (exploitation)")
        print(f"Decay rate: {param.explore_std_decay_rate} (exploration), {param.exploit_std_decay_rate} (exploitation)")
    
    for run in range(n_run):
        if not split_agent:
            policy.set_action_std(param.test_init_std)
        else:
            policy.set_action_std(param.explore_init_std, policy_type='exploration')
            policy.set_action_std(param.exploit_init_std, policy_type='exploitation')
        start_time = datetime.now()
        print(f"Run: {run+1}/{n_run}")
        states = env.reset()
        bestState = states.copy()
        GbestState = bestState[np.argmax(bestState[:, -1])].copy()
        stateHistory = [[] for _ in range(n_agents)] # list of lists of states
        episode_return = [0 for _ in range(n_agents)] if not split_agent else {
            'exploration_episode_return': [0 for _ in range(n_agents)],
            'exploitation_episode_return': [0 for _ in range(n_agents)]
        }
        average_return = [] if not split_agent else {
            'exploration_average_return': [],
            'exploitation_average_return': []
        }
        optAgentTrajectory = []    # trajectory of optimal agent
        bestStateValue = []
        bestStateValue.append(env.best)
        for agent in range(n_agents):
            stateHistory[agent].append(tools.rescale_(states[agent][:-1], env.min_pos, env.max_pos))
        for step in range(ep_length):
            print(f"Step: {step+1}/{ep_length}", end='\r')
            actions = np.zeros((env.n_agents, env.n_dim))
            if not split_agent:
                obs, std_obs = tools.get_agents_obs(states, n_agents, n_dim, bestState, GbestState, F=False, std="euclidean", use_gbest=param.test_use_gbest)
                #actions = [[[] for _ in range(n_dim)] for _ in range(n_agents)]
                for dim in range(n_dim):
                    agent_act = policy.select_action(obs[dim], std_obs[dim])
                    actions[:, dim] = agent_act

                #actions = tools.get_agent_actions(agent_action, n_agents=n_agents, n_dim=n_dim)
            else:
                exploit_states = states[env.exploiter]
                explore_states = states[env.explorer]
                exploit_obs, exploit_std_obs = tools.get_agents_obs(exploit_states, len(env.exploiter), n_dim, bestState, GbestState, F=False, std="euclidean", use_gbest=True)
                explore_obs, explore_std_obs = tools.get_agents_obs(explore_states, len(env.explorer), n_dim, bestState, GbestState, F=False, std="euclidean", use_gbest=False)
                for dim in range(n_dim):
                    agent_act = policy.select_action(exploit_obs[dim], exploit_std_obs[dim], agent_type='exploitation')
                    actions[env.exploiter, dim] = agent_act
                    agent_act = policy.select_action(explore_obs[dim], explore_std_obs[dim], agent_type='exploration')
                    actions[env.explorer, dim] = agent_act
                    #print(actions[env.exploiter, dim], actions[env.explorer, dim])

            next_state, rewards, done, obj_values = env.step(actions) 
            states = next_state
            optAgentTrajectory.append(max(obj_values))
            bestStateValue.append(env.best)

            if not split_agent:
                policy.decay_action_std(param.test_std_decay_rate, param.test_min_std, param.test_learn_std)
            else:
                policy.decay_action_std(param.test_std_decay_rate, param.exploit_test_min_std, param.test_learn_std, policy_type='exploitation')
                policy.decay_action_std(param.test_std_decay_rate, param.explore_test_min_std, param.test_learn_std, policy_type='exploration')


            for agent in range(n_agents):
                if policy.split_agent:
                    if agent in env.exploiter:
                        policy.exploitation_buffer.rewards += [rewards[agent]]*env.n_dim
                        policy.exploitation_buffer.is_terminals += [done[agent]]*env.n_dim
                    else:
                        policy.exploration_buffer.rewards += [rewards[agent]]*env.n_dim
                        policy.exploration_buffer.is_terminals += [done[agent]]*env.n_dim
                else:
                    policy.buffer.rewards += [rewards[agent]] * env.n_dim
                    policy.buffer.is_terminals += [done[agent]] * env.n_dim

                stateHistory[agent].append(tools.rescale_(next_state[agent][:-1], env.min_pos, env.max_pos))
                if not split_agent:
                    episode_return[agent] += rewards[agent] 
                else:
                    if agent in env.exploiter:
                        episode_return['exploitation_episode_return'][agent] += rewards[agent]
                    else:
                        episode_return['exploration_episode_return'][agent] += rewards[agent]
                # update best state
                if next_state[agent][-1] > bestState[agent][-1]: # if the agent has improved its best state
                    bestState[agent] = next_state[agent].copy() # update the best state
                    GbestState = bestState[np.argmax(bestState[:, -1])].copy()

        end_time = datetime.now()
        total_run_time += (end_time - start_time).total_seconds()
        print("Run time: ", end_time - start_time)
        print(f"Run: {run+1}/{n_run} | Best state value: {env.best} | Total run time: {total_run_time}")

        if param.plot_flag == True: #and run % (n_run/2) == 0:
            gif_title =  gif_dir + env.env_name + "_" + str(run) + ".gif"
            if n_dim == 1:
                tools.plot_agents_trajectory(env,  plot_dir, env.env_name, ep_length=ep_length, title=gif_title, fps=2)
            elif n_dim == 2:
                tools.plot_agents_trajectory_2D(env,  plot_dir, env.env_name, ep_length=ep_length, title=gif_title, fps=3)
            else:
                pass 

        f += [optAgentTrajectory]
        bestValues += [bestStateValue]

    ff = np.array(f).reshape(n_run, ep_length)
    np.save(directory+"fitness.npy", ff), np.save(directory+"best_fitness.npy", np.array(bestValues))
    tools.num_function_evaluation(ff, n_agents=n_agents, save_dir=directory+"num_function_evaluation.png", opt_value=env.opt_value)
    tools.num_function_evaluation(np.array(bestValues),n_agents=n_agents, save_dir=directory+"num_function_evaluation_best.png", opt_value=env.opt_value)

    print("Total run time: ", total_run_time)
    print("Average run time: ", total_run_time/n_run)


if __name__ == '__main__':
    #os.chdir('vec')
    # config
    args = tools.get_args()
    title = args.title
    env_name = args.env
    config_path = 'config.yml'
    config = tools.get_config(config_path)
    # prepare policy
    policy = tools.prepare_policy(config, test=True)
    # load the environment
    envs_cache = tools.prepare_environment(config, load_from_file=False)
    main(env_name, title=title,  config=config, policy=policy, env_cache=envs_cache)