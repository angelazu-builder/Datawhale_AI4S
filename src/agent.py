import numpy as np
from typing import Dict, Any, List, Tuple
from .config import SimulationConfig
from .simulator import KestenSimulator
from .analysis import PhaseTransitionAnalyzer
from .logger import ExperimentLogger

class BaselineAgent:
    """
    No-Intervention Baseline Agent (p=0.01, c=0.1, reflect, lognormal).
    Runs 10 trials to compute reference baseline beta_left_base.
    """
    def __init__(self, simulator: KestenSimulator, config: SimulationConfig):
        self.simulator = simulator
        self.config = config

    def run_baseline(self) -> Dict[str, Any]:
        results = []
        for _ in range(10):
            res = self.simulator.run_trial(
                p=self.config.baseline_p,
                c=self.config.baseline_c,
                boundary_type=self.config.baseline_boundary,
                income_dist=self.config.baseline_income,
                N_override=20000,
                T_override=300
            )
            results.append(res)
        
        mean_beta = float(np.mean([r["beta_left"] for r in results]))
        mean_poverty = float(np.mean([r["poverty_rate"] for r in results]))
        mean_kurtosis = float(np.mean([r["kurtosis"] for r in results]))
        
        return {
            "beta_left_base": mean_beta,
            "poverty_rate_base": mean_poverty,
            "kurtosis_base": mean_kurtosis,
            "p": self.config.baseline_p,
            "c": self.config.baseline_c
        }

class RandomAgent:
    """
    Random Exploration Agent: Uniformly samples parameter space for equal 100-round budget without feedback.
    """
    def __init__(self, simulator: KestenSimulator, config: SimulationConfig, logger: ExperimentLogger):
        self.simulator = simulator
        self.config = config
        self.logger = logger
        self.analyzer = PhaseTransitionAnalyzer()

    def run_exploration(self, budget: int = None) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        budget = budget or self.config.random_baseline_budget  # 100 rounds
        rng = np.random.default_rng(42)
        history = []
        mutations = []

        for r in range(1, budget + 1):
            p = float(rng.uniform(self.config.p_min, self.config.p_max))
            c = float(rng.uniform(self.config.c_min, self.config.c_max))
            b_type = str(rng.choice(self.config.boundary_types))
            inc_dist = str(rng.choice(self.config.income_distributions))

            res = self.simulator.run_trial(p, c, b_type, inc_dist, N_override=20000, T_override=300)
            history.append(res)

            mutation_flag, r_star = self.analyzer.detect_phase_transition(history)
            self.logger.log_round(r, res, mutation_flag, r_star)

            if mutation_flag:
                res["round"] = r
                res["delta_beta"] = abs(res["beta_left"] - history[-2]["beta_left"]) if len(history) > 1 else 0.0
                res["r_star"] = r_star
                mutations.append(res)

        return history, mutations

class AdaptiveAgent:
    """
    Adaptive Active Exploration Agent (Equal 100 Rounds Budget):
    - Stage 1 (Rounds 1-25): Controlled Multi-dimensional Latin-Hypercube Uniform Sweep.
    - Stage 2 (Rounds 26-100): Active Learning UCB Surrogate Exploration & Policy Search (p, c, tax, subsidy),
      pinpointing phase transition R* = p/c via 10-seed controlled verification.
    """
    def __init__(self, simulator: KestenSimulator, config: SimulationConfig, logger: ExperimentLogger):
        self.simulator = simulator
        self.config = config
        self.logger = logger
        self.analyzer = PhaseTransitionAnalyzer()

    def run_controlled_1d_sweep(self) -> List[Dict[str, Any]]:
        """
        Controlled 1D Parameter Sweep: Fix c=0.10, boundary=reflect, income=lognormal,
        and sweep p in [0.001, 0.10] over 25 steps to produce unconfounded phase transition curve.
        """
        print("[AdaptiveAgent] Executing Controlled 1D Trajectory Sweep (fixed c=0.10, reflect, lognormal)...")
        sweep_history = []
        p_steps = np.linspace(self.config.p_min, self.config.p_max, 25)
        for p in p_steps:
            res = self.simulator.run_trial(float(p), 0.10, "reflect", "lognormal", tax=0.0, subsidy=0.0, N_override=20000, T_override=300)
            sweep_history.append(res)
        return sweep_history

    def run_exploration(self) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, List[Dict[str, Any]]]]:
        rng = np.random.default_rng(2024)
        history = []
        detected_mutations = []
        scientific_signals = {
            "positive_discoveries": [],
            "anomalies_counterexamples": [],
            "negative_results": []
        }

        # Current state parameters
        curr_p = self.config.baseline_p
        curr_c = self.config.baseline_c
        curr_tax = self.config.baseline_tax
        curr_sub = self.config.baseline_subsidy
        curr_b = self.config.baseline_boundary
        curr_inc = self.config.baseline_income

        total_rounds = self.config.agent_budget  # 100 rounds

        for round_id in range(1, total_rounds + 1):
            if round_id <= self.config.initial_exploration_rounds:
                # Stage 1: Grid / Latin-Hypercube Uniform Exploration
                p_grid = np.linspace(self.config.p_min, self.config.p_max, 10)
                c_grid = np.linspace(self.config.c_min, self.config.c_max, 3)
                idx = (round_id - 1)
                curr_p = float(p_grid[idx % 10])
                curr_c = float(c_grid[(idx // 10) % 3])
                curr_tax = 0.0
                curr_sub = 0.0
                curr_b = self.config.boundary_types[(idx // 5) % 3]
                curr_inc = self.config.income_distributions[idx % 3]
            else:
                # Stage 2: Active Learning UCB-Guided Surrogate Acquisition
                candidates = []
                for _ in range(5):
                    cand_p = float(np.clip(curr_p + rng.normal(0, 0.01), self.config.p_min, self.config.p_max))
                    cand_c = float(np.clip(curr_c + rng.normal(0, 0.02), self.config.c_min, self.config.c_max))
                    cand_tax = float(np.clip(curr_tax + rng.normal(0, 0.02), self.config.tax_min, self.config.tax_max))
                    cand_sub = float(np.clip(curr_sub + rng.normal(0, 0.005), self.config.subsidy_min, self.config.subsidy_max))
                    cand_b = str(rng.choice(self.config.boundary_types))
                    cand_inc = str(rng.choice(self.config.income_distributions))

                    ucb = self.analyzer.compute_ucb_acquisition(cand_p, cand_c, cand_tax, history, kappa=self.config.active_learning_kappa)
                    candidates.append((ucb, cand_p, cand_c, cand_tax, cand_sub, cand_b, cand_inc))

                candidates.sort(key=lambda x: x[0], reverse=True)
                best_cand = candidates[0]
                curr_p, curr_c, curr_tax, curr_sub, curr_b, curr_inc = best_cand[1:]

            # Run trial across 5 seeds with N_override=20000, T_override=300 for fast high-density exploration
            trial_res = self.simulator.run_trial(curr_p, curr_c, curr_b, curr_inc, curr_tax, curr_sub, N_override=20000, T_override=300)
            history.append(trial_res)

            # Scientific Signal Classification & 10-Seed Controlled Verification
            signal = self.analyzer.classify_scientific_signal(trial_res, history)
            mutation_flag = (signal["category"] == "positive_discovery")
            r_star = signal.get("r_star", 0.0)

            # Verification across 10 independent seeds testing relative discontinuity against neighboring path
            if mutation_flag and len(history) > 1:
                prev_beta = history[-2]["beta_left"]
                verif_count = 0
                for v_seed in self.config.verification_seeds:
                    v_res = self.simulator.run_single_simulation(curr_p, curr_c, curr_b, curr_inc, seed=v_seed, tax=curr_tax, subsidy=curr_sub, N_override=20000, T_override=300)
                    if abs(v_res["beta_left"] - prev_beta) > 0.3:
                        verif_count += 1
                mutation_flag = (verif_count >= 8)  # Require 8/10 independent replications of relative jump

            self.logger.log_round(round_id, trial_res, mutation_flag, r_star)

            if mutation_flag:
                trial_res["round"] = round_id
                trial_res["r_star"] = r_star
                trial_res["delta_beta"] = abs(trial_res["beta_left"] - history[-2]["beta_left"]) if len(history) > 1 else 0.0
                detected_mutations.append(trial_res)
                scientific_signals["positive_discoveries"].append(signal)
            elif signal["category"] == "anomaly_counterexample":
                scientific_signals["anomalies_counterexamples"].append(signal)
            elif signal["category"] == "negative_result":
                scientific_signals["negative_results"].append(signal)

        return history, detected_mutations, scientific_signals
