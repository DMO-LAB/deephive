from src.environment import OptimizationEnv
from src.mappo import MAPPO
from distutils.util import strtobool
import os
import numpy as np
from datetime import datetime
from commons.utils import plot_agents_trajectory_combined, plot_num_function_evaluation
import logging
from logging_configuration import configure_logger

class DeepHive:
    def __init__(self, title, env, policy, mode, config, **kwargs):
        self.title = title
        self.env = env
        self.policy = policy
        self.config = config
        self.env_cache = kwargs.get('env_cache', None)
        self.log = kwargs.get('log', True)
        self.exp_name, self.directory, self.plot_dir, self.gif_dir, self.checkpoint_dir = self._create_work_dir(
            title, log_folder="logs")
        self.logger, self.run = configure_logger(title, local_only=False, level=logging.INFO)
        if self.log:
            self.logger.info(f"All directories created successfully in {self.directory}")
        self.mode = mode
        self._set_parameters()
        self._run_summary()
        
    def _set_parameters(self):
        self.n_agents = self.config['environment_config']['n_agents']
        self.n_dim = self.config['environment_config']['n_dim']
        self.ep_length = self.config['environment_config']['ep_length']
        if self.mode == 'test':
            self.n_run = self.config['run_config']['n_run']
        if self.mode == 'train':
            self.max_episodes = self.config['train_config']['max_episodes']
            self.update_timestep = self.config['train_config']['update_timestep']
            self.save_interval = self.config['train_config']['save_interval']
            self.render_interval = self.config['train_config']['render_interval']
            self.decay_timestep = self.config['train_config']['decay_timestep']
            self.log_interval = self.config['train_config']['log_interval']
            self.decay_rate = self.config['policy_config']['decay_rate']
            self.save_cutoff = self.config['train_config']['save_cutoff']
            self.learn_std = self.config['policy_config']['learn_std']
            self.std_min = self.config['policy_config']['std_min']
            self.std_max = self.config['policy_config']['std_max']       
        self.use_gbest = self.config['train_config']['use_gbest']

    def _run_summary(self):
        if self.log:
            self.logger.info(f"{self.title} : {self.mode.upper()} RUN SUMMARY")
            self.logger.info(f"Environment: {self.env.env_name}")
            self.logger.info(f"Number of agents: {self.n_agents}")
            self.logger.info(f"Number of dimensions: {self.n_dim}")
            self.logger.info(f"Episode length: {self.ep_length}")
            if self.mode == 'test':
                self.logger.info(f"Number of runs: {self.n_run}")
        else:
            print(f"{self.title} : {self.mode.upper()} RUN SUMMARY")
            print(f"Environment: {self.env.env_name}")
            print(f"Number of agents: {self.n_agents}")
            print(f"Number of dimensions: {self.n_dim}")
            print(f"Episode length: {self.ep_length}")
            if self.mode == 'test':
                print(f"Number of runs: {self.n_run}")

        
    def _create_work_dir(self, title, log_folder="logs"):
        exp_name = title
        directory = f"{log_folder}/{exp_name}"
        plot_dir = f"{directory}/plots"
        gif_dir = f"{directory}/gifs"
        checkpoint_dir = f"{directory}/checkpoints"
        os.makedirs(directory, exist_ok=True)
        os.makedirs(plot_dir, exist_ok=True)
        os.makedirs(gif_dir, exist_ok=True)
        os.makedirs(checkpoint_dir, exist_ok=True)
        # create a run_summary.txt file to store the run 
        with open(f"{directory}/run_summary.txt", 'w') as f:
            os.utime(f"{directory}/run_summary.txt", None)
            
        
        return exp_name, directory, plot_dir, gif_dir, checkpoint_dir
    
    def _write_2_summary(self, **kwargs):
        with open(f"{self.directory}/run_summary.txt", 'a') as f:
            for i, (key, value) in enumerate(kwargs.items()):
                f.write(f"{key}: {value}\n")

    def optimize(self, debug=False):
        try:
            if self.mode == 'train':
                self._train(debug=debug)
            elif self.mode == 'test':
                self._test(debug=debug)
        except Exception as e:
            if self.log:
                self.logger.exception(f"Error during {self.mode} optimization:")
            print(f"Error during {self.mode} optimization: {e}")            

    def _test(self, debug=False):
        for i in range(self.n_run):
            print(f"*************Run: {i+1}/{self.n_run}***************")
            if self.log:
                self.logger.info(f"*************Run: {i+1}/{self.n_run}***************")
            agent = np.random.randint(self.n_agents)
            print(f"Agent: {agent}")
            states = self.env.reset()
            objectives = self.env.obj_values
            personal_best = states.copy()
            global_best = personal_best[np.argmax(personal_best[:, -1])].copy()
            # write state, objectives, personal best and global best to summary
            if debug:
                self._write_2_summary(states = np.round(states[agent], 4), 
                                      objectives = np.round(objectives[agent], 4), 
                                      personal_best = np.round(personal_best[agent], 4), 
                                      global_best = np.round(global_best, 4))
            state_history = np.zeros((self.ep_length, self.n_agents, self.n_dim))
            start_time = datetime.now()
            if self.log and debug:
                self.logger.info(f"Start time: {start_time}")
            for step in range(self.ep_length): 
                if debug:
                    print(f"Step: {step+1}/{self.ep_length}", end='\r')
                    self._write_2_summary(step=step+1)
                    if self.log:
                        self.logger.info(f"Step: {step+1}/{self.ep_length}")
                # get observations 
                observations, std = self.env._generate_observations(personal_best, global_best, use_gbest=self.use_gbest)
                # get actions
                actions = np.zeros((self.n_agents, self.n_dim))
                for dim in range(self.n_dim):
                    action = self.policy.select_action(observations[dim], std[dim])
                    actions[:, dim] = action
                    
                # get next states
                states, rewards, dones, obj_values = self.env.step(actions)
                state_history[step, :, :] = self.env._rescale(states[:, :-1], self.env.min_pos, self.env.max_pos)
                if debug and self.log:
                    self.logger.info(f"Observations: {observations}")
                    self.logger.info(f"Std: {std}")
                    self.logger.info(f"Actions: {actions}")
                    self.logger.info(f"Rewards: {rewards}")
                    self._write_2_summary(observation_dim1=np.round(observations[0][agent], 4),
                                            observation_dim2=np.round(observations[1][agent], 4),
                                            action_dim1=np.round(actions[:, 0][agent], 4),
                                            action_dim2=np.round(actions[:, 1][agent], 4),
                                            reward=np.round(rewards[agent], 4))
                # update personal best
                personal_best[np.where(obj_values > objectives)[0]] = states[np.where(obj_values > objectives)[0]]
                objectives = np.maximum(objectives, obj_values)
                global_best = personal_best[np.argmax(objectives)].copy()
                if debug and self.log:
                    self.logger.info(f"Personal best: {personal_best}")
                    self.logger.info(f"Objectives: {objectives}")
                    self.logger.info(f"Global best: {global_best}")
                    self._write_2_summary(personal_best=np.round(personal_best[agent], 4),
                                            objectives=np.round(objectives[agent], 4), 
                                            global_best=np.round(global_best, 4))
                
            episode_best = self.env.bestAgentHistory[self.env.current_step]
            number_of_best_changes = self.env.bestAgentChange
            if debug and self.log:
                self.logger.info(f"Episode best: {episode_best}")
                self.logger.info(f"Number of best changes: {number_of_best_changes}")
                self._write_2_summary(episode_best=episode_best, number_of_best_changes=number_of_best_changes)

            end_time = datetime.now()
            total_runtime = (end_time - start_time).total_seconds()
            if debug:
                print(f"Episode best: {episode_best} | Number of best changes: {number_of_best_changes} | Total runtime: {total_runtime}")
            # get best agent values
            global_best_values = self.env._rescale(global_best[:-1], self.env.min_pos, self.env.max_pos)
            global_best_fitness = objectives[np.argmax(objectives)]
            print(f"Global best values: {global_best_values} | Global best fitness: {global_best_fitness}")
            if self.log:
                self.logger.info(f"Global best values: {global_best_values} | Global best fitness: {global_best_fitness}")
            # write to summary
            if debug:
                self._write_2_summary(episode_best=episode_best, number_of_best_changes=number_of_best_changes, total_runtime=total_runtime)
            # plot the trajectory of the agents
            try:
                plot_agents_trajectory_combined(self.env, self.plot_dir, self.gif_dir, title=f"run_{i+1}.gif", attention_agent=agent)
            except Exception as e:
                print(f"Error during rendering: {e}")
            

    def _train(self, debug=False):
        timestep = 0
        average_return = []
        total_time = 0
        train_summary = {}
        for episode in range(self.max_episodes):
            if episode % 20 == 0:
                print(f"Episode: {episode+1}/{self.max_episodes}")
                self.logger.info(f"Episode: {episode+1}/{self.max_episodes}")
            states = self.env.reset()
            objectives = self.env.obj_values
            personal_best = states.copy()
            global_best = personal_best[np.argmax(objectives)].copy()
            state_history = np.zeros((self.ep_length, self.n_agents, self.n_dim))
            episode_return = np.zeros(self.n_agents)
            start_time = datetime.now()
            attention_agent = np.random.randint(self.n_agents)
            for step in range(self.ep_length):
                # get observations 
                observations, std = self.env._generate_observations(personal_best, global_best, use_gbest=self.use_gbest)
                # get actions
                actions = np.zeros((self.n_agents, self.n_dim))
                for dim in range(self.n_dim):
                    action = self.policy.select_action(observations[dim], std[dim])
                    actions[:, dim] = action
                    
                # get next states
                states, rewards, dones, obj_values = self.env.step(actions)
                for agent in range(self.n_agents):
                    self.policy.buffer.rewards += [rewards[agent]] * self.n_dim
                    self.policy.buffer.is_terminals += [dones[agent]] * self.n_dim
                state_history[step, :, :] = self.env._rescale(states[:, :-1], self.env.min_pos, self.env.max_pos)
                episode_return += rewards
                # update personal best
                personal_best[np.where(obj_values > objectives)[0]] = states[np.where(obj_values > objectives)[0]]
                objectives = np.maximum(objectives, obj_values)
                # update global best
                global_best = personal_best[np.argmax(objectives)].copy()

                if step == self.ep_length - 1:
                    average_return.append(np.mean(episode_return))
                timestep += 1
                # update the policy
                if timestep % self.update_timestep == 0:
                    self.policy.update()

                # decay the std
                if timestep % self.decay_timestep == 0:
                    if timestep % (self.decay_timestep * 10) == 0:
                        self.policy.decay_action_std(self.decay_rate, self.std_min, self.learn_std, verbose=True)
                    else:
                        self.policy.decay_action_std(self.decay_rate, self.std_min, self.learn_std, verbose=False)

                if episode % 100 == 0 and self.run:
                    self.run[f"train/{episode}/best_value"].log(self.env.best)
            
            if self.run:
                self.run['Average Return'].log(np.mean(episode_return))
            # save the policy
            if episode % self.save_interval == 0:
                print(f"Average Return: {average_return[-3:]}")
                if all(a > self.save_cutoff for a in average_return[-3:]):
                    print(f"Saving policy at episode {episode}")
                    checkpoint_path = self.checkpoint_dir + "/policy-" + str(episode) + ".pth"
                    self.policy.save(self.checkpoint_dir, episode)
                    if self.run:
                        self.run[f'checkpoints/episode-{episode}-policy.pth'].upload(checkpoint_path)
                    average_return = []
            total_time += (datetime.now() - start_time).total_seconds()
            # render the environment
            if episode % self.render_interval == 0:
                try:
                    plot_agents_trajectory_combined(self.env, self.plot_dir, self.gif_dir, title=f"episode_{episode}.gif", attention_agent=attention_agent)
                    if self.run:
                        self.run[f"render/episode_{episode}.gif"].upload(self.gif_dir + f"\episode_{episode}.gif")
                except Exception as e:
                    print(f"Error during rendering: {e}")



