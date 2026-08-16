import numpy as np
from typing import Dict, Any, List
from .config import SimulationConfig
from .simulator import KestenSimulator

class BehavioralKestenSimulator(KestenSimulator):
    """
    Advanced Research Extension Module 1: Non-linear Economic Behavioral Simulator.
    Integrates qualitative economic behavioral modes into Kesten stochastic dynamics:
    1. Keynesian MPC (Marginal Propensity to Consume): c(W) = c0 * (W / W0)^(-alpha)
       Rich agents save a larger proportion of wealth (lower dissipation rate).
    2. Prospect Theory / Loss Aversion: p(W) = p0 * (1 + gamma / (W + eps))
       Poor agents face higher vulnerability & risk exposure to economic shocks.
    """
    def __init__(self, config: SimulationConfig = None):
        super().__init__(config)

    def run_behavioral_simulation(self, p0: float, c0: float, 
                                  behavioral_mode: str = "keynesian_mpc",
                                  boundary_type: str = "reflect", 
                                  income_dist: str = "lognormal",
                                  seed: int = 42,
                                  alpha: float = 0.25,
                                  gamma: float = 0.50) -> Dict[str, Any]:
        """
        Run simulation with non-linear economic behavioral functions c(W) and p(W).
        """
        np.random.seed(seed)
        N = 20000
        T = 300
        wealth = np.ones(N, dtype=np.float64)

        for _ in range(T):
            # Evaluate wealth-dependent behavioral functions
            if behavioral_mode == "keynesian_mpc":
                # Dissipation rate decreases with wealth (rich save more)
                c_w = np.clip(c0 * np.power(np.maximum(0.1, wealth), -alpha), 0.01, 0.50)
                p_w = np.full(N, p0)
            elif behavioral_mode == "loss_aversion":
                # Shock probability increases as wealth decreases (poor face higher risk)
                p_w = np.clip(p0 * (1.0 + gamma / (wealth + 0.1)), 0.001, 0.20)
                c_w = np.full(N, c0)
            elif behavioral_mode == "combined":
                c_w = np.clip(c0 * np.power(np.maximum(0.1, wealth), -alpha), 0.01, 0.50)
                p_w = np.clip(p0 * (1.0 + gamma / (wealth + 0.1)), 0.001, 0.20)
            else:
                c_w = np.full(N, c0)
                p_w = np.full(N, p0)

            # Draw random shocks
            shocks = (np.random.random(N) < p_w).astype(np.float64)
            if income_dist == "lognormal":
                income = np.random.lognormal(0.0, 0.5, N) * shocks
            else:
                income = np.random.exponential(1.0, N) * shocks

            # Update wealth dynamics
            wealth = wealth + income - c_w * wealth

            # Apply boundary condition
            if boundary_type == "reflect":
                wealth = np.maximum(0.01, wealth)
            elif boundary_type == "soft_clamp":
                wealth = np.where(wealth < 0.01, 0.01 + 0.1 * np.abs(wealth), wealth)

        results = self.analyze_wealth_distribution(wealth)
        results.update({
            "p0": p0,
            "c0": c0,
            "behavioral_mode": behavioral_mode,
            "alpha": alpha,
            "gamma": gamma
        })
        return results
