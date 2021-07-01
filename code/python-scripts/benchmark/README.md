# The Deep Equilibrium Net Benchmark Model

This script provides the code used to model and solve the benchmark model in the working paper by
[Azinovic, Gaegauf, & Scheidegger (2020)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3393482)
(see section 3). For a more streamlined application, see the [analytic notebook](https://github.com/sischei/DeepEquilibriumNets/blob/master/code/jupyter-notebooks/Analytic_tf1.ipynb).

### Prerequisites / Installation

Note that, this script was originally programmed in TensorFlow 1. The current default version of
TensorFlow is now TensorFlow 2. This script is TensorFlow 2 compatible. To install the correct
version, use
```shell
    $ pip install tensorflow
```

## Usage
There are two modes to run this code:

   1. the final network weights presented in the paper can be loaded and used to output a host of plots,
   2. the deep equilibrium net can be trained from scratch.

We have simplified the code such that the only user input is the desired running mode. To run, follow
these instructions:

In terminal:
```shell
    $ cd <PATH to the repository>/DeepEquilibriumNets/code/python-scripts/benchmark
```

### Mode 1: Load the trained network weights
```shell
    $ python benchmark.py
```
The results are saved to `./output/deqn_benchmark_restart`.

### Mode 2: Train from scratch
```shell
    $ python benchmark.py --train_from_scratch
```
The results are saved to `./output/deqn_benchmark`.

**Note**: the results presented in the paper (see, section 5) were achieved by training the neural
network on 2 training schedules. Once the first training schedule is complete (after running the
above command), uncomment lines 1621-1627 and run the previous command again
(`python benchmark.py --train_from_scratch`). The results are saved to
`./output/deqn_benchmark_2ndschedule`.
