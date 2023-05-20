# Scalable Multi-agent Covering Option Discovery based on Kronecker Graphs

Please cite this paper:
```bash
@inproceedings{DBLP:conf/nips/ChenCLA22,
  author       = {Jiayu Chen and
                  Jingdi Chen and
                  Tian Lan and
                  Vaneet Aggarwal},
  title        = {Scalable Multi-agent Covering Option Discovery based on Kronecker
                  Graphs},
  booktitle    = {NeurIPS},
  year         = {2022},
  url          = {http://papers.nips.cc/paper\_files/paper/2022/hash/c40d1e40dd121d0e7ba8e4ab65bca81b-Abstract-Conference.html}
}
```

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
