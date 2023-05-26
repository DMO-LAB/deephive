from decouple import config as conf
import tools


config_path = 'config.yml'
config = tools.get_config(config_path)

environment_config = config['environment_config']
policy_config = config['policy_config']
training_config = config['train_config']
test_config = config['test_config']

std_decay_rate = policy_config['std_decay_rate']
exploit_std_decay_rate = config['exploiting_policy_config']['std_decay_rate']
explore_std_decay_rate = config['exploring_policy_config']['std_decay_rate']
min_std = policy_config['std_min']
explore_min_std = config['exploring_policy_config']['std_min']
exploit_min_std = config['exploiting_policy_config']['std_min']
max_std = policy_config['std_max']
explore_max_std = config['exploring_policy_config']['std_max']
exploit_max_std = config['exploiting_policy_config']['std_max']
learn_std = policy_config['learn_std']
explore_learn_std = config['exploring_policy_config']['learn_std']
exploit_learn_std = config['exploiting_policy_config']['learn_std']
std_mode = policy_config['std_mode']
explore_std_mode = config['exploring_policy_config']['std_mode']
exploit_std_mode = config['exploiting_policy_config']['std_mode']


test_std_decay_rate = test_config['std_decay_rate']
exploit_std_decay_rate = test_config['exploit_std_decay_rate']
explore_std_decay_rate = test_config['explore_std_decay_rate']
test_min_std = test_config['std_min']
exploit_test_min_std = test_config['exploit_min_std']
explore_test_min_std = test_config['explore_min_std']
test_max_std = test_config['std_max']
test_learn_std = test_config['learn_std']
test_std_mode = test_config['std_mode']
# test_std_decay_freq = test_config['decay_interval']
test_init_std = test_config['init_std']
explore_init_std = test_config['explore_init_std']
exploit_init_std = test_config['exploit_init_std']
test_use_gbest = test_config['use_gbest']

update_timestep = training_config['update_interval']
std_decay_freq = training_config['decay_interval']
mode = training_config['mode']
transition_episode = training_config['transition_time']
log_freq = training_config['log_interval']
save_freq = training_config['save_interval']
plot_flag = training_config['plot_flag']
save_cutoff = training_config['save_cutoff']
print_freq = training_config['print_interval']
render_interval = training_config['render_interval']
use_gbest = training_config['use_gbest']

