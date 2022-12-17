# Scalable Multi-agent Covering Option Discovery based on Kronecker Graphs

Codebase for my papers: Multi-agent Covering Option Discovery through Kronecker Product of Factor Graphs && Scalable Multi-agent Covering Option Discovery based on Kronecker Graphs

The following parts are included:
- Multi-agent maze tasks built with Mujoco as the benchmark for the continuous case.
- Multi-agent grid-maze tasks as the benchmark for the tabular case.
- Implementations of the multi-agent option discovery algorithms for tabular and deep MARL proposed in our paper. 
- Implementations of tablar MARL algorithms as baselines: Independent Q-Learning, Distributed Q-Learning, Centralized Q-Learning.
- Implementations of Deep MARL algorithms as baselines: MAPPO, MAA2C, MADDPG, QMIX, Weighted-QMIX, MAVEN, COMA, MSAC.

Please refer to the 'readme' file in each folder for further details:  
  
Folder "continuous": codes for continuous control tasks built with Mujoco  
  
Folder "tabular": codes for the discrete Grid World tasks   
  
Folder "others": codes for applying SOTA MARL on the 4-agent Grid Maze task and the performance change with the frequency of collisions (Figure 8(c))
