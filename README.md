This repository contains the code associated with the numerical example of Section V of the paper "A Lyapunov Function for the Combined System-Optimizer Dynamics in Nonlinear Model Predictive Control" - Andrea Zanelli, Quoc Tran Dinh, Moritz Diehl. 

## Installation

In order to install the Python package, clone the repository on your machine 
- `git clone https://github.com/zanellia/nmpc_system_optimizer_lyapunov`

and install it using `pip`
- `pip install .`

Notice that, if you are using Python 3.7 you will not be able to install the package directly due to the fact that one of the requirements (`slycot`) cannot be installed with `pip` when using Python 3.7. However, you can install `slycot` using `conda` (this requires installing `conda`, of course) instead 
-`conda install -c conda-forge slycot`.

## Generating plots

The plots in the paper can be reproduced running 
- `python <repo_root>/nmpc_nmpc_system_optimizer_lyapunov/main.py`

The main file will first estimate the underling constants based on a sampling of the augmented state space and will then run a simulation to generated the state trajectories and evaluate the Lyapunov function. 
