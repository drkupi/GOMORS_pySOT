# GOMORS_pySOT

GOMORS is a surrogate-assisted Multi-Objective Optimization (MO) strategy, designed for
computational expensive MO problems, e.g., expensive environmental simulation optimization
problems, hyperparameter tuning of Deep Neural Networks etc. GOMORS is implemented in the 
[pySOT lbirary](https://github.com/dme65/pySOT) and framework, and uses Radial Basis Functions
(RBFs), as surrogates. Moreover, GOMORS uses a Multi Objective Evolutionary Strategy (MOEA)
to optimize RBF surrogates in each iteration. Any MOEA methods can be connected with GOMORS
for optimizing surrogates. We currently use the [Platypus library](https://github.com/Project-Platypus/Platypus)
to link GOMORS with epsion-MOEA. GOMORS also supports modest parallelization on up to 4 cores, and hence, is 
suitable for deskptop and laptop machines. 

## Installation Instructions

The prerequisites for using the GOMORS code in this repository is installation of a python 2 environment, the pySOT
library, the platypus library and matplotlib. We recommend using a virtual environment within Anaconda. Instructions
for installation of pre-requisites is as follows:

```{python}
conda create --name mo-surrogate python=2.7
conda activate mo-surrogate
pip install pysot==0.1.36
pip install matplotlib
pip install platypus-opt
```
## Running GOMORS
An example of how to run GOMORS is provided in the file simple_experiment.py. The setup for running the algorithm
is synonymous to how optimization experiments are setup in pysot. To link GOMORS to a user-defined MO optimization problem, 
kindly look at how problems are defined in the test_problems.py python file. For further information please write to 
me at taimoor.akhtar@gmail.com.

## Citing GOMORS
If you use GOMORS, please cite our paper, [Akhtar, T., Shoemaker, C.A. Multi objective optimization of computationally expensive
multi-modal functions with RBF surrogates and multi-rule selection. J Glob Optim 64, 17–32 (2016). 
https://doi.org/10.1007/s10898-015-0270-y](https://link.springer.com/article/10.1007/s10898-015-0270-y#citeas).
 
