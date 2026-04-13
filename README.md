# RL-GeneTrans
## Overview of the framework
RL-GeneTrans is a deep reinforcement learning-based method for identifying risk genes associated with NASH-HCC transition. To introduce transition-related information, RL-GeneTrans optimized the PPI network structure. The task of optimizing the PPI network structure is formulated as a Markov Decision Process (MDP). In this RL-GeneTrans, the state is characterized by the weighted adjacency matrix of the PPI network, and action is defined as a modification to this matrix. A data-label dual-driven reward is designed to introduce NASH-HCC transition-related information, which utilizes the expression data of NASH-HCC patients and known risk genes. 

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

We upload a trained model for NASH-HCC transition risk gene identification. To run this model, you can use the command line instructions:
```
python run.py
```
After the process is completed, the file `Gene_Ranking.csv` is output in the corresponding directory, which contains the identified gene risk ranking list.

