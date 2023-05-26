# DEEPHIVE - A reinforcement learning framework for black-box optimization
DeepHive is a reinforcement learning framework for optimizing black-box functions. It uses a combination of exploitation and exploration agents to search for high-quality solutions in a given landscape.

### Features
- Uses multi-agent reinforcement learning to optimize black-box functions
- Combines exploitation and exploration agents to search for high-quality solutions
- Supports continuous and discrete observations and actions
- Can scale to large search spaces and complex problems
- Can scale to different number of agents
- Provides built-in plotting and visualization tools to track progress and performance

### Installation
To install DeepHive, clone the repository and install the required dependencies:

```
git clone https://github.com/elotech47/deephive.git
cd deephive
pip install -r requirements.txt
```
### Usage
To use DeepHive, define a black-box function to be optimized and create an instance of the DeepHiveOptimizer class. Then, call the optimize method to begin the optimization process:

```
from deephive import DeepHiveOptimizer

def my_function(x, y):
  return x ** 2 + y ** 2

optimizer = DeepHiveOptimizer(my_function, continuous_observations=True)
result = optimizer.optimize()

print(result)
```

For more detailed usage examples and documentation, see the DeepHive documentation.

### Contributing
We welcome contributions to DeepHive! If you have an idea for a feature or bug fix, please open an issue or pull request on GitHub.

### License
DeepHive is released under the MIT License.


## TO DO
- write tests
- extend to any optimization functions
