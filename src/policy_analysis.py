import os
import numpy as np
from typing import Dict, Any, List, Tuple
from .config import SimulationConfig
from .simulator import KestenSimulator

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except Exception:
    HAS_MATPLOTLIB = False

class PolicyAnalyzer:
    """
    Historical evolution modeling, event impact simulation (US Reagan Tax Cuts, China Targeted Poverty Alleviation),
    and counterfactual policy deduction engine.
    """
    def __init__(self, output_dir: str = "."):
        self.output_dir = output_dir
        self.config = SimulationConfig()
        self.simulator = KestenSimulator(self.config)

    def plot_us_china_historical_comparison(self, df: Any, filename: str = "us_china_historical_comparison.png") -> str:
        """
        Generate visualization of historical Gini trends and income shares (US vs China 1980-2023).
        """
        save_path = os.path.join(self.output_dir, filename)
        if not HAS_MATPLOTLIB:
            print(f"[Notice] matplotlib is not installed. Plot skipped ({filename})")
            return save_path

        fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=300)

        # Filter by metric_name and country
        if hasattr(df, "to_dict"):
            records = df.to_dict("records")
        elif isinstance(df, list):
            records = df
        else:
            records = []

        us_gini = sorted([r for r in records if r.get("metric_name") == "gini_index" and r.get("country") == "USA"], key=lambda x: x["year"])
        cn_gini = sorted([r for r in records if r.get("metric_name") == "gini_index" and r.get("country") == "CHN"], key=lambda x: x["year"])

        axes[0].plot([r["year"] for r in us_gini], [r["value"] for r in us_gini], label="United States (USA)", color="#1f77b4", linewidth=2.5, marker="o", markersize=3)
        axes[0].plot([r["year"] for r in cn_gini], [r["value"] for r in cn_gini], label="China (CHN)", color="#d62728", linewidth=2.5, marker="s", markersize=3)
        
        # Annotate historical events
        axes[0].axvline(1981, color="#1f77b4", linestyle="--", alpha=0.6, label="US Reagan Tax Cut (1981)")
        axes[0].axvline(2001, color="#d62728", linestyle="--", alpha=0.6, label="China WTO Accession (2001)")
        axes[0].axvline(2015, color="green", linestyle=":", alpha=0.8, label="China Targeted Poverty Alleviation (2015)")

        axes[0].set_title("Empirical Gini Index Evolution (1980–2023)", fontsize=12, fontweight="bold")
        axes[0].set_xlabel("Year")
        axes[0].set_ylabel("Gini Index (0 = Perfect Equality, 100 = Max Inequality)")
        axes[0].grid(True, linestyle=":", alpha=0.6)
        axes[0].legend(fontsize=8, loc="upper left")

        # Plot 2: Bottom 20% Income Share
        us_b20 = sorted([r for r in records if r.get("metric_name") == "bottom_20_share" and r.get("country") == "USA"], key=lambda x: x["year"])
        cn_b20 = sorted([r for r in records if r.get("metric_name") == "bottom_20_share" and r.get("country") == "CHN"], key=lambda x: x["year"])

        axes[1].plot([r["year"] for r in us_b20], [r["value"] for r in us_b20], label="USA Bottom 20% Share (%)", color="#1f77b4", linewidth=2.0)
        axes[1].plot([r["year"] for r in cn_b20], [r["value"] for r in cn_b20], label="China Bottom 20% Share (%)", color="#d62728", linewidth=2.0)

        axes[1].set_title("Empirical Bottom 20% Wealth Share (%)", fontsize=12, fontweight="bold")
        axes[1].set_xlabel("Year")
        axes[1].set_ylabel("Share of Total Income/Wealth (%)")
        axes[1].grid(True, linestyle=":", alpha=0.6)
        axes[1].legend(fontsize=9)

        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        return save_path

    def run_counterfactual_experiments(self, us_params: Dict[str, Any], cn_params: Dict[str, Any],
                                        filename: str = "policy_counterfactual_simulation.png") -> Tuple[str, Dict[str, Any]]:
        """
        Perform counterfactual policy simulations:
        1. Counterfactual US: Maintain 1975 high tax rate (tau = 0.15 vs calibrated tau* = 0.02).
        2. Counterfactual China: Omit Targeted Poverty Alleviation subsidy (S = 0 vs calibrated S* = 0.05).
        """
        print("\n[PolicyAnalyzer] Executing Counterfactual Policy Experiments...")
        save_path = os.path.join(self.output_dir, filename)
        
        # Experiment 1: US Baseline vs Counterfactual High Redistribution
        us_base_res = self.simulator.run_single_simulation(us_params["p_star"], us_params["c_star"], "reflect", "lognormal", seed=42, tax=us_params["tax_star"], subsidy=us_params["subsidy_star"], N_override=20000, T_override=300)
        us_counter_res = self.simulator.run_single_simulation(us_params["p_star"], us_params["c_star"], "reflect", "lognormal", seed=42, tax=0.15, subsidy=0.02, N_override=20000, T_override=300)

        # Experiment 2: China Baseline (with Targeted Poverty Alleviation S*) vs Counterfactual (No Subsidy S=0)
        cn_base_res = self.simulator.run_single_simulation(cn_params["p_star"], cn_params["c_star"], "reflect", "lognormal", seed=42, tax=cn_params["tax_star"], subsidy=cn_params["subsidy_star"], N_override=20000, T_override=300)
        cn_counter_res = self.simulator.run_single_simulation(cn_params["p_star"], cn_params["c_star"], "reflect", "lognormal", seed=42, tax=cn_params["tax_star"], subsidy=0.00, N_override=20000, T_override=300)

        deductions = {
            "us_policy_deduction": (
                f"Increasing progressive tax tau from {us_params['tax_star']} to 0.15 reduces poverty rate by "
                f"{(us_base_res['poverty_rate'] - us_counter_res['poverty_rate'])*100:.2f} percentage points, "
                f"stiffening the left-tail exponent beta_left from {us_base_res['beta_left']:.2f} to {us_counter_res['beta_left']:.2f}."
            ),
            "china_policy_deduction": (
                f"China's Targeted Poverty Alleviation subsidy (S={cn_params['subsidy_star']:.3f}) prevents a "
                f"{(cn_counter_res['poverty_rate'] - cn_base_res['poverty_rate'])*100:.2f}% surge in poverty rate "
                f"that would occur under zero-subsidy market dynamics (S=0)."
            )
        }

        if not HAS_MATPLOTLIB:
            print(f"[Notice] matplotlib is not installed. Plot skipped ({filename})")
            return save_path, deductions

        # Plotting results
        fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=300)

        # Plot 1: US Counterfactual
        categories = ["US Actual (tau=0.02)", "US Counterfactual (tau=0.15)"]
        poverty_rates = [us_base_res["poverty_rate"] * 100, us_counter_res["poverty_rate"] * 100]
        betas = [us_base_res["beta_left"], us_counter_res["beta_left"]]

        x = np.arange(len(categories))
        width = 0.35
        axes[0].bar(x - width/2, poverty_rates, width, label="Poverty Rate (%)", color="#e74c3c")
        axes[0].bar(x + width/2, [b * 20 for b in betas], width, label="Tail Index beta (x20)", color="#3498db")
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(categories, fontsize=9, fontweight="bold")
        axes[0].set_ylabel("Metric Value")
        axes[0].set_title("US Policy Counterfactual: High Tax (tau=0.15) Effect", fontsize=11, fontweight="bold")
        axes[0].grid(True, axis="y", linestyle=":", alpha=0.6)
        axes[0].legend()

        # Plot 2: China Counterfactual
        cn_categories = ["China Actual (Subsidy S*)", "China Counterfactual (S=0)"]
        cn_poverty = [cn_base_res["poverty_rate"] * 100, cn_counter_res["poverty_rate"] * 100]
        cn_betas = [cn_base_res["beta_left"], cn_counter_res["beta_left"]]

        x_cn = np.arange(len(cn_categories))
        axes[1].bar(x_cn - width/2, cn_poverty, width, label="Poverty Rate (%)", color="#2ecc71")
        axes[1].bar(x_cn + width/2, [b * 20 for b in cn_betas], width, label="Tail Index beta (x20)", color="#9b59b6")
        axes[1].set_xticks(x_cn)
        axes[1].set_xticklabels(cn_categories, fontsize=9, fontweight="bold")
        axes[1].set_ylabel("Metric Value")
        axes[1].set_title("China Counterfactual: Omission of Poverty Subsidy (S=0)", fontsize=11, fontweight="bold")
        axes[1].grid(True, axis="y", linestyle=":", alpha=0.6)
        axes[1].legend()

        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

        return save_path, deductions
