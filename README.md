# SOTA (State-of-the-Art) Optimizers Repository

This repository contains the implementation of the Lion optimizer, along with other optimizers for comparison. The primary purpose is to replicate the results from the research paper on the creation of the Lion optimizer titled [Lion: A Novel Method for Stochastic Optimization](https://arxiv.org/pdf/2302.06675v4.pdf).

## Optimizers
All the optimizers can be found in the `optimizers` directory. You can explore and compare different optimizers in tasks related to neural network training or finding the minimum of a function with subsequent path visualization.

## Functions
The repository includes various functions, classified by complexity (30/40/30 split), sourced from [infinity77.net](https://infinity77.net/global_optimization/test_functions.html#test-functions-index). These functions are primarily used for the purpose of evaluating and comparing the performance of different optimizers. You can find them in the `functions` directory.

## Usage

### Example Notebook
To get started, refer to the `function_experiment.ipynb` notebook for an example of how to compare the optimizers.

### Function Visualization
If you are interested in visualizing the functions, you can refer to the `function_draw_img.ipynb` file for the corresponding code.

### Experiment Class
The main component of this repository is the `experiment` class, implemented in [experiment.py](https://github.com/Alex-Andrv/sota/blob/main/experiment/experiment.py). To conduct an experiment, follow these steps:

1. Create an instance of the `experiment` class.
2. Pass the model, training data, a list of metrics, an optimizer, a learning rate schedule, the type of model (mode), and the configuration for the wandb database.

### Torch
It is important to note that all the code in this repository is written using the Torch framework.
