REGISTRY = {}

from agents.multiple.modules.agents.rnn_agent import RNNAgent
from agents.multiple.modules.agents.maddpg_rnn_agent import MADDPGRNNAgent
REGISTRY["rnn"] = RNNAgent
REGISTRY["maddpg_rnn"] = MADDPGRNNAgent