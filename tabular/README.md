# Multi-agent Option Discovery based on Kronecker Product -- Tabular Case

## How to config the environments:
- python 3.6
- matplotlib
- tqdm
- networkx
- ...

## How to run the experiments
- On Ubuntu 18.04
- It's a pity that we don't get enough time to combine these programs together before the DDL.
- Run experiments on n-agent Maze/Room tasks using Centralized Q-Learning + Force, please go to folder 'MAOD_n_agent_force'.
- Run experiments on n-agent Maze/Room tasks using Distributed Q-Learning, please go to folder 'MAOD_n_agents'.
- Run experiments on Maze/Room tasks with subtask grouping using Centralized Q-Learning + Force, please go to folder 'MAOD_pairwise_force_group'.
- Run experiments on Maze/Room tasks with subtask grouping using Distributed Q-Learning, please go to folder 'MAOD_pairwise_group'.
- Run experiments on Maze/Room tasks with random grouping using Centralized Q-Learning + Force, please go to folder 'MAOD_pairwise_force'.
- Run experiments on Maze/Room tasks with random grouping using Distributed Q-Learning, please go to folder 'MAOD_pairwise'.
- Run experiments on Maze/Room tasks with random grouping and dynamic influence using Centralized Q-Learning + Force, please go to folder 'MAOD_pairwise_force_influence'.
- In each folder, please first input:
```bash
cd options/experiments
```
and then:
```bash
python rl_experiments.py
```
Probably you will need a python IDE, like PyCharm, to run this file properly.

- When testing on the Room tasks, please add:
```bash
--use_median=True
```
Otherwise, please add:
```bash
--use_median=False
```
- To change the test environment, please add:
```bash
--task='grid_roomX'
```
Or:
```bash
--task='grid_mazeX'
```
where 'X' needs to be replaced with a number that represents the number of agents in the test environment. Please refer to the 'tasks' subfoloder in each folder mentioned above to check the available test environments.
