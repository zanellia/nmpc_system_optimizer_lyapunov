This repository contains the code associated with the numerical example of Section V of the paper "A Lyapunov Function for the Combined System-Optimizer Dynamics in Nonlinear Model Predictive Control" - Andrea Zanelli, Quoc Tran Dinh, Moritz Diehl. 

# Installation

In order to install the Python package, clone the repository on your machine `git clone https://github.com/zanellia/nmpc_system_optimizer_lyapunov`
and install it using `pip`: `pip install .`. Notice that, if you are using Python 3.7 you will not be able to install the package directly due to the fact that one of the requirements (`slycot`) cannot be installed using `pip` when using Python 3.7. However, you can install `slycot` using `conda` instead: `conda install -c conda-forge slycot`  (this requires installing `conda`, of course).
