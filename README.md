# RL-GeneTrans
## Overview of the framework
Reinforcement Learning-based and Data-driven biological network optimization and propagation-based candidate gene prioritization for **NASH**, **NASH-HCC**, and the **NASH-to-HCC transition**. Specifically, we formulate the optimization of biological network structures as a reinforcement learning problem and design a data-driven reward function to guide the optimization process. The data-driven reward incorporates differential gene expression information between NASH-HCC tissues and adjacent non-tumor NASH tissues, thereby capturing molecular changes associated with the transition from NASH to NASH-HCC. A graph transformer–based agent is designed to integrate global and local information from the biological network and iteratively optimize its structure under the guidance of the data-driven reward.


## Installation & Dependencies

The code is written in Python 3 and was mainly tested on Python 3.9.20 and a Linux OS. The package development version is tested on Linux and Windows 10 operating systems. The developmental version of the package has been tested on the following systems:

* Linux: Ubuntu 22.04.3 LTS
* Windows: 10

RL-GeneTrans has the following dependencies:

* networkx
* torch
* torch-cluster
* torch-geometric
* torch-scatter
* torch-sparse
* torch-spline-conv
* scikit-learn
* pandas
* numpy

The details of Python dependencies used in experiments can be found in requirements.txt. 

## Running RL-GeneTrans
To clone this repository, users can use:
```
git clone https://github.com/23AIBox/RL-GeneTrans.git
```

Set up the required environment using `requirements.txt` with Python. While in the project directory, run:
```
pip install -r requirements.txt
```
It takes about 35 minutes to set up the environment. 
We also provided a conda environment file (from Linux). Users can build the environment by running:
```
conda env create -f environment.yaml
```
The source code of RL-GeneTrans is in ./src. Input data and optimized networks for **NASH**, **NASH-HCC**, and the **NASH-to-HCC transition** are in ./data. 

