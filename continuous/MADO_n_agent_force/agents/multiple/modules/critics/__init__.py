from agents.multiple.modules.critics.centralV import CentralVCritic
from agents.multiple.modules.critics.maddpg import MADDPGCritic
REGISTRY = {}

REGISTRY["cv_critic"] = CentralVCritic
REGISTRY["maddpg_critic"] = MADDPGCritic


