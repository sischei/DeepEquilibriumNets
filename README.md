# Deep Equilibrium Nets

<p align="center">
<img src="screens/DEQN.png" width="600px"/>
</p>


Deep equilibrium nets (DEQN) is a generic, deep-learning-based framework to compute recursive equilibria in dynamic stochastic economic models. The method directly approximates all equilibrium functions and that are trained in an unsupervised fashion to satisfy all equilibrium conditions along simulated paths of the economy. 

This repository contains example codes in [TensorFlow](https://www.tensorflow.org/). Its goal is to make DEQNs easily accessible to the computational economics and finance community.


### Authors
* [Marlon Azinovic](https://sites.google.com/view/marlonazinovic/home) (University of Zurich, Department of Banking and Finance and Swiss Finance Institute)
* [Luca Gaegauf](https://www.bf.uzh.ch/en/persons/gaegauf-luca/team) (University of Zurich, Department of Banking and Finance)
* [Simon Scheidegger](https://sites.google.com/site/simonscheidegger/) (University of Lausanne, Department of Finance)

### The full paper can be found here
* [Deep Equilibrium Nets](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3393482)

### An illustrative example
To illustrate how DEQNs can be applied to solve economic models, we provide an example in python, which solves the model presented in Appendix [C](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3393482) of the paper.
The presented model is taken from [Krueger and Kubler (2004)](https://www.sciencedirect.com/science/article/pii/S0165188903001118) and is based on [Huffman (1987)](https://www.journals.uchicago.edu/doi/10.1086/261445). We chose this model as an illustrative example for two reasons: first, it is closely related to the models we solve in the paper and second, it has an analytical solution, so the accuracy of the solution method can easily be verified.



## Usage
We provide implementations which use python 3 as well as TensorFlow 1. We provide our implementation in two forms. A [jupyter-notebook](https://jupyter.org/) that is self-contained and also contains the model and all relevant equations:

[![Generic badge](https://img.shields.io/badge/jupyter%20nbviewer-DEQN-green)](https://nbviewer.jupyter.org/github/sischei/DeepEquilibriumNets/blob/master/code/jupyter-notebooks/Analytic_tf1.ipynb)

as well as a plain python script, which can be executed from the command line:

[![Generic badge 2](https://img.shields.io/badge/python%20script-DEQN-green)](code/python-scripts)

### Prerequisites / Installation

#### TensorFlow 1
```shell
$ pip install tensorflow==1.13.1 
```

### Running Deep Equilibrium Nets in a local installation

#### Jupyter notebook

Launch with:
```shell
    $ cd <PATH to the repository>/DeepEquilibriumNets/code/jupyter-notebooks/ 
    $ jupyter-notebook Analytic_tf1.ipynb
```

    
#### Plain Python
Launch with:
```shell
    $ cd <PATH to the repository>/DeepEquilibriumNets/code/python-scripts/ 
    $ python Analytic_tf1.py
```

## Citation

Please cite Deep Equilibrium Nets in your publications if it helps your research:

Azinovic, Marlon and Gaegauf, Luca and Scheidegger, Simon, Deep Equilibrium Nets (May 24, 2019). 
Available at SSRN: https://ssrn.com/abstract=3393482 or http://dx.doi.org/10.2139/ssrn.3393482


## Support

This work was supported the Swiss Platform for Advanced Scientific Computing (PASC) under project ID ["Computing equilibria in heterogeneous agent macro models on contemporary HPC platforms"](https://www.pasc-ch.org/projects/2017-2020/call-for-pasc-hpc-software-development-project-proposals).

