![mloptimizer_banner](https://raw.githubusercontent.com/Caparrini/mloptimizer-static/main/logos/mloptimizer_banner_readme.png)

[![Documentation Status](https://readthedocs.org/projects/mloptimizer/badge/?version=master)](https://mloptimizer.readthedocs.io/en/master/?badge=master)
[![PyPI version](https://badge.fury.io/py/mloptimizer.svg)](https://badge.fury.io/py/mloptimizer)
[![PyPI pyversions](https://img.shields.io/pypi/pyversions/mloptimizer.svg)](https://pypi.python.org/pypi/mloptimizer/)
[![Tests](https://github.com/Caparrini/mloptimizer/actions/workflows/CI.yml/badge.svg)](https://github.com/Caparrini/mloptimizer/actions/workflows/CI.yml)
[![Coverage Status](http://codecov.io/github/Caparrini/mloptimizer/coverage.svg?branch=master)](https://app.codecov.io/gh/Caparrini/mloptimizer)


**mloptimizer** is a Python library for optimizing hyperparameters of machine learning algorithms using genetic algorithms. 
With mloptimizer, you can find the optimal set of hyperparameters for a given machine learning model and dataset, which can significantly improve the performance of the model. 
The library supports several popular machine learning algorithms, including decision trees, random forests, and gradient boosting classifiers. 
The genetic algorithm used in mloptimizer provides an efficient and flexible approach to search for the optimal hyperparameters in a large search space.

## Features
- Easy to use
- DEAP-based genetic algorithm ready to use with several machine learning algorithms
- Adaptable to use with any machine learning algorithm that complies with the Scikit-Learn API
- Default hyperparameter ranges
- Default score functions for evaluating the performance of the model
- Reproducibility of results

## Advanced Features
- Extensible with more machine learning algorithms that comply with the Scikit-Learn API
- Customizable hyperparameter ranges
- Customizable score functions
- Optional mlflow compatibility for tracking the optimization process

## Installation

It is recommended to create a virtual environment using the `venv` package. 
To learn more about how to use `venv`, 
check out the official Python documentation at 
https://docs.python.org/3/library/venv.html.

```bash
# Create the virtual environment
python -m venv myenv
# Activate the virtual environment
source myenv/bin/activate
```

To install `mloptimizer`, run:

```bash
pip install mloptimizer
```

You can get more information about the package installation at https://pypi.org/project/mloptimizer/.

### [!WARNING] Important Note for Windows Users

If you're using Windows and encounter an error "error: invalid path mloptimizer/aux/[file_name].py" when cloning or updating the repository, it may be due to the use of reserved file names, such as "aux", by Git.

This issue arises because Windows operating system reserves certain file names that may be legal in Linux or macOS but illegal in Windows.

To resolve this issue, you can configure Git to ignore NTFS naming restrictions by running the following command:

```bash
git config --global core.protectNTFS false
```

However, please be cautious when disabling this setting, as it may expose you to potential security risks. Depending on the file name causing the issue, turning off protectNTFS may or may not resolve the error.

Once you've completed the clone or pull operation, it's recommended to revert the protectNTFS setting to its default value to ensure the security of your system. You can do this by running:

```bash
git config --global core.protectNTFS true
```

We apologize for any inconvenience caused by this issue and thank you for your understanding. If you encounter any further difficulties or have any questions, please don't hesitate to reach out to us.

### Quickstart

Here's a simple example of how to optimize hyperparameters in a decision tree classifier using the iris dataset:

```python
from mloptimizer.core import Optimizer
from mloptimizer.hyperparams import HyperparameterSpace
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris

# 1) Load the dataset and get the features and target
X, y = load_iris(return_X_y=True)

# 2) Define the hyperparameter space (a default space is provided for some algorithms)
hyperparameter_space = HyperparameterSpace.get_default_hyperparameter_space(DecisionTreeClassifier)

# 3) Create the optimizer and optimize the classifier
opt = Optimizer(estimator_class=DecisionTreeClassifier, features=X, labels=y, hyperparam_space=hyperparameter_space)

# 4) Optimize the classifier, the optimization returns the best estimator found in the optimization process
# - 10 generations starting with a population of 10 individuals, other parameters are set to default
clf = opt.optimize_clf(population_size=10, generations=10)
```
Other algorithms can be used, such as `RandomForestClassifier` or `XGBClassifier` which have a 
default hyperparameter space defined in the library.
Even if the algorithm is not included in the default hyperparameter space, you can define your own hyperparameter space
following the documentation.

The optimization will create a directory in the current folder with a name like `YYYYMMDD_nnnnnnnnnn_SklearnOptimizer`.
This folder contains the results of the optimization, 
including the best estimator found and the log file `opt.log` informing with all the steps, 
the best estimator and the result of the optimization.

A structure like this will be created:

```
├── checkpoints
│   ├── cp_gen_0.pkl
│   └── cp_gen_1.pkl
├── graphics
│   ├── logbook.html
│   └── search_space.html
├── opt.log
├── progress
│   ├── Generation_0.csv
│   └── Generation_1.csv
└── results
    ├── logbook.csv
    └── populations.csv
```

Each item in the directory is described below:

- `checkpoints`: This directory contains the checkpoint files for each generation of the genetic optimization process. These files are used to save the state of the optimization process at each generation, allowing for the process to be resumed from a specific point if needed.
    - `cp_gen_0.pkl`, `cp_gen_1.pkl`: These are the individual checkpoint files for each generation. They are named according to the generation number and are saved in Python's pickle format.

- `graphics`: This directory contains HTML files for visualizing the optimization process.
    - `logbook.html`: This file provides a graphical representation of the logbook, which records the statistics of the optimization process over generations.
    - `search_space.html`: This file provides a graphical representation of the search space of the optimization process.

- `opt.log`: This is the log file for the optimization process. It contains detailed logs of the optimization process, including the performance of the algorithm at each generation.

- `progress`: This directory contains CSV files that record the progress of the optimization process for each generation.
    - `Generation_0.csv`, `Generation_1.csv`: These are the individual progress files for each generation. They contain detailed information about each individual in the population at each generation.

- `results`: This directory contains CSV files with the results of the optimization process.
    - `logbook.csv`: This file is a CSV representation of the logbook, which records the statistics of the optimization process over generations.
    - `populations.csv`: This file contains the final populations of the optimization process. It includes the hyperparameters and fitness values of each individual in the population.

More details in the [documentation](http://mloptimizer.readthedocs.io/).



## Examples

Examples can be found in [examples](https://mloptimizer.readthedocs.io/en/latest/auto_examples/index.html) on readthedocs.io.

## Dependencies

The following dependencies are used in `mloptimizer`:

* [Deap](https://github.com/DEAP/deap) - Genetic Algorithms
* [XGBoost](https://github.com/dmlc/xgboost) - Gradient boosting classifier
* [Scikit-Learn](https://github.com/scikit-learn/scikit-learn) - Machine learning algorithms and utilities

Optional:
* [Keras](https://keras.io) - Deep learning library
* [mlflow](https://mlflow.org) - Tracking the optimization process

## Documentation

The documentation for `mloptimizer` can be found in the project's [wiki](http://mloptimizer.readthedocs.io/)
with examples, classes and methods reference.


## Authors

* **Antonio Caparrini** - *Author* - [caparrini](https://github.com/caparrini)
* **Javier Arroyo Gallardo** - *Author* - [javiag](https://github.com/javiag)

## License

This project is licensed under the [MIT License](LICENSE).

## FAQs
- TODO
