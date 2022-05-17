# Multi-agent Option Discovery based on Kronecker Product -- Continuous Case

## How to config the environments:
- python 3.6
- pytorch 1.6
- tensorboard 2.5
- mujoco_py >= 1.5
- gym <= 2.0
- tianshou
- matplotlib
- pyyaml
- ...

## How to run the experiments
- On Ubuntu 18.04
- Agents with Multi-agent Options:

```bash
python main.py --mode='multiple'
```
- Agents with Single-agent Options:

```bash
python main.py --mode='single'
```

- Agents without Options -- MAPPO:

```bash
python main.py --mode='none'
```

- Agents without Options -- MADDPG:

```bash
python main.py --mode='none' --low_level_alg='maddpg'
```

- Agents without Options -- MAA2C:

```bash
python main.py --mode='none' --low_level_alg='maa2c'
```

- The default map is Mujoco Maze, to run experiments on Mujoco Room:
```bash
... --env_id='Point4Rooms-v0'
```
