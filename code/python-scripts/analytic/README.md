# The Deep Equilibrium Net Analytic Model

To illustrate how DEQNs can be applied to solve economic models, we provide an example in python, which solves the model presented in [Appendix F](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3393482) of Azinovic, Gaegauf, & Scheidegger (2021)..
The presented model is taken from [Krueger and Kubler (2004)](https://www.sciencedirect.com/science/article/pii/S0165188903001118) and is based on [Huffman (1987)](https://www.journals.uchicago.edu/doi/10.1086/261445). We chose this model as an illustrative example for two reasons: first, it is closely related to the models we solve in the paper and second, it has an analytical solution, so the accuracy of the solution method can easily be verified.

### Prerequisites / Installation

```shell
$ pip install tensorflow==1.13.1
```

## Usage
Launch with:
```shell
    $ cd <PATH to the repository>/DeepEquilibriumNets/code/python-scripts/analytic
    $ python Analytic_tf1.py
```
