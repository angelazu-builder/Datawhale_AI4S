import os
from dataclasses import dataclass, field
from typing import List, Tuple

@dataclass
class SimulationConfig:
    # Environment Core Settings
    population_size: int = 20000        # N = 20,000 agents for high-speed submission
    time_steps: int = 300               # T = 300 steps for fast equilibrium
    random_seeds: List[int] = field(default_factory=lambda: [0, 42, 123, 2024, 9999])
    
    # Initial Wealth Bounds
    w_init_min: float = 0.1
    w_init_max: float = 1.0
    
    # Parameter Search Bounds
    p_min: float = 0.001
    p_max: float = 0.1
    c_min: float = 0.01
    c_max: float = 0.3
    
    # Extended Policy Search Bounds (Taxation tau & Social Subsidy S)
    tax_min: float = 0.0
    tax_max: float = 0.20
    subsidy_min: float = 0.0
    subsidy_max: float = 0.05

    # Categorical Options
    boundary_types: List[str] = field(default_factory=lambda: ["reflect", "absorb", "soft_clamp"])
    income_distributions: List[str] = field(default_factory=lambda: ["lognormal", "exponential", "uniform"])
    
    # Budget & Agent Settings (Optimized 30 vs 30 rounds for 3-second instant submission)
    agent_budget: int = 30
    random_baseline_budget: int = 30
    initial_exploration_rounds: int = 10
    active_learning_kappa: float = 2.0  # UCB Exploration Tradeoff Parameter
    verification_seeds: List[int] = field(default_factory=lambda: [1001, 1002, 1003, 1004, 1005])
    
    # Baseline Reference Settings
    baseline_p: float = 0.01
    baseline_c: float = 0.1
    baseline_tax: float = 0.0
    baseline_subsidy: float = 0.0
    baseline_boundary: str = "reflect"
    baseline_income: str = "lognormal"
    
    # Histogram & Tail Fit Settings
    num_bins: int = 500
    poverty_threshold: float = 1.0
    tail_percentile: float = 50.0       # Lower 50% for left tail analysis
    
    # Output Directory
    output_dir: str = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")

