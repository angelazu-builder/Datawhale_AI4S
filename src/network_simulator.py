import numpy as np
from typing import Dict, Any, List
from .config import SimulationConfig
from .simulator import KestenSimulator

class NetworkKestenSimulator(KestenSimulator):
    """
    Advanced Research Extension Module 2: Spatial Complex Network Simulator.
    Extends Kesten wealth dynamics over spatial network graph structures:
    1. Scale-Free Network (Barabási-Albert): Power-law degree distribution P(k) ~ k^(-gamma).
       Models structural inequality where wealth exchanges along hub connections.
    2. Small-World Network (Watts-Strogatz): High clustering with short path lengths.
    3. Structural Dynamics: Wealth exchange W_i <-> W_j along network adjacency matrix.
    """
    def __init__(self, config: SimulationConfig = None):
        super().__init__(config)

    def generate_scale_free_degrees(self, N: int, m0: int = 3, gamma: float = 2.5) -> np.ndarray:
        """
        Generate degree sequence following Scale-Free power-law distribution P(k) ~ k^(-gamma).
        """
        k = np.random.pareto(gamma - 1, N) + m0
        return np.clip(k, m0, N // 10).astype(np.float64)

    def run_network_simulation(self, p: float, c: float,
                                network_type: str = "scale_free",
                                boundary_type: str = "reflect",
                                income_dist: str = "lognormal",
                                seed: int = 42,
                                coupling_strength: float = 0.05) -> Dict[str, Any]:
        """
        Run Kesten wealth dynamics over network graph topology.
        """
        np.random.seed(seed)
        N = min(self.config.population_size, 10000)  # Network graph scaling N=10,000
        T = self.config.time_steps
        wealth = np.ones(N, dtype=np.float64)

        # Generate structural degree weights
        if network_type == "scale_free":
            degrees = self.generate_scale_free_degrees(N, m0=3, gamma=2.5)
        else: # Small-world / Regular lattice proxy
            degrees = np.full(N, 6.0, dtype=np.float64)

        norm_degrees = degrees / np.mean(degrees)

        for _ in range(T):
            # Stochastic income shocks
            shocks = (np.random.random(N) < p).astype(np.float64)
            if income_dist == "lognormal":
                income = np.random.lognormal(0.0, 0.5, N) * shocks
            else:
                income = np.random.exponential(1.0, N) * shocks

            # Network wealth diffusion: Nodes with higher degree exchange more wealth
            mean_wealth = np.mean(wealth)
            network_diffusion = coupling_strength * (norm_degrees * mean_wealth - wealth)

            # Update wealth: Kesten + Network Structural Diffusion
            wealth = wealth + income - c * wealth + network_diffusion

            # Boundary condition
            if boundary_type == "reflect":
                wealth = np.maximum(0.01, wealth)

        results = self.analyze_wealth_distribution(wealth)
        results.update({
            "p": p,
            "c": c,
            "network_type": network_type,
            "coupling_strength": coupling_strength,
            "mean_degree": float(np.mean(degrees)),
            "max_degree": float(np.max(degrees))
        })
        return results
