# Multi-agent Option Discovery based on Kronecker Product -- Complement Experiment

## How to config the environments:
- python 3.6
- pytorch 1.6
- tensorboard 2.5
- matplotlib
- pandas
- numpy
- ...

## How to run the experiments
- On Ubuntu 18.04
- To reproduce the results about the influence of collisons on the performance of MARL (Centralized Q-learning + Force) with Multi-agent Options, 
  please go to the 'MAOD_pairwise_force_influence_change' folder.  
  First, input:
  ```bash
  cd options/experiments
  ```
  and then:
  ```bash
  python rl_experiments.py
  ```
  Probably you will need a python IDE, like PyCharm, to run this file properly, after which you will get the figure in the same  folder as the python file.
- To produce the results of the MARL baselines on the 4-agent Grid Maze task, please go to the 'MARL_baselines' folder.  
  Fisrt, the parameter setup is available in 'common/arguments'.
  To run algorithm X which can be any of ['qmix', 'cwqmix', 'owqmix', 'coma', 'msac', 'maven'] with seed Y, please run:
  ```bash
  python main.py --alg=X --seed=Y
  ```
  Also, we provide file 'plot.py' for visualization.
