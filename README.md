# Automatic BS Deployment
A deep reinforcement learning (DRL)-based solution for the automatic base station (BS) deployment problem.
Algorithms are implemented by using [Ray RLlib](https://docs.ray.io/en/latest/rllib/index.html).

## Project Structure
- `agent/`: Trainable DRL agents.
- `checkpoint/`: Saved algorithm state, including all model parameters.
- `config/`: Configurations of all algorithm hyperparameters and simulation parameters.
- `data/`: Training data used for visualization.
- `env/`: Modelling of the BS deployment problem, including different versions.
- `figures/`: Visualization of training results.
- `log/`: Important interval states of environments or agents in the training/test process.
- `resource/`: Static resource used for training.
- `rl_module/`: Custom model structure of DRL algorithms (e.g. action masking).
- `runner_{algo_name}.py`: Runnable script for training and evaluating DRL agents.