# Python imports.
from __future__ import print_function

# Grab classes.

from simple_rl.tasks.grid_world.GridWorldMDPClass import GridWorldMDP
from simple_rl.tasks.grid_world.GridWorldStateClass import GridWorldState

try:
	from simple_rl.tasks.gym.GymMDPClass import GymMDP
	from simple_rl.tasks.gym.GymStateClass import GymState
except ImportError:
	print("Warning: OpenAI gym not installed.")
	pass


