from agents.multiple.runners.episode_runner import EpisodeRunner
from agents.multiple.runners.hierarchical_episode_runner import HierarchicalEpisodeRunner

REGISTRY = {}
REGISTRY["episode"] = EpisodeRunner
REGISTRY["hierarchical_episode"] = HierarchicalEpisodeRunner

