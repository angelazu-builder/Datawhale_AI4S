"""
Extended Kesten Wealth Process & Phase Transition AI Research Package (v1.0.0)
"""

from .config import SimulationConfig
from .simulator import KestenSimulator
from .behavioral_simulator import BehavioralKestenSimulator
from .network_simulator import NetworkKestenSimulator
from .agent import BaselineAgent, RandomAgent, AdaptiveAgent
from .analysis import PhaseTransitionAnalyzer, compare_candidate_distributions
from .data_loader import EmpiricalDataLoader
from .calibration import EmpiricalCalibrator
from .policy_analysis import PolicyAnalyzer
from .logger import ExperimentLogger
from .visualizer import Visualizer

__version__ = "1.1.0"
__all__ = [
    "SimulationConfig",
    "KestenSimulator",
    "BehavioralKestenSimulator",
    "NetworkKestenSimulator",
    "BaselineAgent",
    "RandomAgent",
    "AdaptiveAgent",
    "PhaseTransitionAnalyzer",
    "compare_candidate_distributions",
    "EmpiricalDataLoader",
    "EmpiricalCalibrator",
    "PolicyAnalyzer",
    "ExperimentLogger",
    "Visualizer",
]
