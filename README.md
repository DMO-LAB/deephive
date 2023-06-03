# DeepHive Optimization

DeepHive is an optimization framework that combines Multi-Agent Proximal Policy Optimization (MAPPO) with a swarm intelligence-inspired algorithm to solve optimization problems in various domains. It utilizes deep reinforcement learning techniques to train a policy that guides a swarm of agents in searching for optimal solutions.

## Features

- Multi-Agent Proximal Policy Optimization (MAPPO) algorithm for swarm-based optimization.
- Support for various optimization environments and objective functions.
- Training and testing modes for optimizing and evaluating the policy.
- Logging support with Neptune.ai integration.
- Configurable parameters for environment, policy, and training settings.
- Visualization of agent trajectories and performance plots.

## Getting Started

### Prerequisites

- Python 3.7 or higher
- PyTorch
- OpenAI Gym
- Neptune.ai (optional)

### Installation

1. Clone the repository:

```
git clone https://github.com/your-username/deephive-optimization.git
```

2. Install the required dependencies:

```
pip install -r requirements.txt
```

### Usage

The DeepHive optimization framework can be used to train and test policies for solving optimization problems. The following steps outline the basic workflow:

1. Prepare the configuration file: Edit the `config.yml` file to specify the environment settings, policy configuration, and training parameters.

2. Define the objective function: If you have a custom objective function, add it to the `commons/objective_functions.py` file and update the `get_obj_func` function to include it. You can also write a different script for the objective function like the [heat exchanger optimization problem](commons/heat_exchanger.py). Ensure to import it in the [here](commons/objective_functions.py)

4. Train the policy: You can chose to train a policy on a new function or use the pretrained policy found in [models](models/model.pth). In order to train a new policy;

Run the following command to train the policy:

```
python main.py --config config.yml --title my_experiment --env my_environment --mode train
```

This will start the training process with the specified configuration, environment, and experiment title. The trained policy checkpoints will be saved in the `logs/my_experiment/checkpoints` directory.

5. Test the policy: Once the policy is trained, you can test it by running the following command:

```
python main.py --config config.yml --title my_experiment --env my_environment --mode test
```

This will evaluate the policy on the specified environment using the trained checkpoints. Make sure you update the model checkpoint in the [config](/config.yml) file under `test_policy_config`. The results of the optimization can be found in the logs folder with the experiment title. 

### NB 
There is an example [notebook](example.ipynb) 

### Customization

- Environment: To add a custom environment, create a new class in the `src/environment.py` file by extending the `OptimizationEnv` class. Implement the required methods and register the environment in the `Registry` class.

- Objective Functions: To add a custom objective function, define the function in the `commons/objective_functions.py` file and update the `get_obj_func` function to include it.

- Configuration: The `config.yml` file allows you to customize various parameters related to the environment, policy, and training settings. Modify the file according to your requirements.

- Logging: The framework supports logging experiments using Neptune.ai. To enable logging, set the `log` flag to `True` and specify the Neptune.ai API key in the `neptune_secret_access_key` configuration. You can also provide tags to categorize and filter your experiments.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

The DeepHive Optimization framework is built on top of the Multi-Agent Proximal Policy Optimization (MAPPO) algorithm and is inspired by swarm intelligence concepts. The project utilizes libraries and tools such as PyTorch, OpenAI Gym, and Neptune.ai for deep reinforcement learning, environment modeling,