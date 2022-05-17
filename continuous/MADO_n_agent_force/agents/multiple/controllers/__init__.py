REGISTRY = {}

from agents.multiple.controllers.basic_controller import BasicMAC
from agents.multiple.controllers.maddpg_controller import MADDPGMAC

REGISTRY["basic_mac"] = BasicMAC
REGISTRY["maddpg_mac"] = MADDPGMAC