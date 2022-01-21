# The deep equilibrium net "continuum of agent" model

This script provides the code used to model and solve a model with a continuum of agents, and aggregate and
idiosyncratic shocks in [Appendix E](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3393482) of Azinovic, Gaegauf, & Scheidegger (2021).
For a more streamlined application, see the [analytic notebook](https://github.com/sischei/DeepEquilibriumNets/blob/master/code/jupyter-notebooks/analytic/Analytic_tf1.ipynb).

## Prerequisites / Installation

This script was programmed in TensorFlow 2 and is not TensorFlow 1 compatible. To install
Tensorflow 2, use
```shell
    $ pip install tensorflow
```
To upgrade from TensorFlow 1 to TensorFlow 2, use
```shell
    $ pip install --upgrade tensorflow
```

## Usage

There are two modes to run this code:

   1. the final network weights presented in the paper can be loaded and used to output a host of plots;
   2. the deep equilibrium net can be trained from scratch.

We have simplified the code such that the only user input is the desired running mode. To run, follow
these instructions:

In terminal:
```shell
    $ cd DeepEquilibriumNets/code/python-scripts/continuum_of_agents
```

### Mode 1: Load the trained network weights

```shell
    $ python continuum_of_agents.py
```
The results are saved to `./output/deqn_continuumagents_restart`

### Mode 2: Train from scratch

```shell
    $ python continuum_of_agents.py --train_from_scratch
```
The results are saved to `./output/deqn_continuumagents`.
