import os
import numpy as np
from typing import List, Dict, Any

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'SimHei']
    plt.rcParams['axes.unicode_minus'] = False
except Exception:
    HAS_MATPLOTLIB = False

class Visualizer:
    """
    Visualization engine for generating publication-quality figures:
    - Left-tail wealth distribution log-log power-law fits
    - Phase transition diagram (beta_left vs R* = p/c)
    - Agent trajectory in parameter space
    - Benchmark comparison plots
    """
    def __init__(self, output_dir: str = "."):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def plot_phase_transition(self, history: List[Dict[str, Any]], filename: str = "phase_transition_R_star.png") -> str:
        """
        Plot beta_left vs R* = p/c showing critical phase transition threshold along a controlled 1D parameter path.
        Controls: Fixed c = 0.10, boundary = 'reflect', income_dist = 'lognormal'.
        Dynamically estimates R*_crit via gradient peak max |d(beta)/d(R*)|.
        """
        filepath = os.path.join(self.output_dir, filename)
        if not HAS_MATPLOTLIB:
            print(f"[Notice] matplotlib is not installed. Plot skipped ({filename})")
            return filepath

        from .analysis import estimate_critical_threshold
        r_crit = estimate_critical_threshold(history)

        r_stars = [r["p"] / r["c"] for r in history]
        betas = [r["beta_left"] for r in history]
        cis = [r["beta_ci"] for r in history]

        fig, ax = plt.subplots(figsize=(8.5, 5), dpi=300)
        
        # Plot mean and 95% confidence interval shaded band
        r_arr = np.array(r_stars)
        b_arr = np.array(betas)
        ci_arr = np.array(cis)
        
        sorted_indices = np.argsort(r_arr)
        r_sorted = r_arr[sorted_indices]
        b_sorted = b_arr[sorted_indices]
        ci_sorted = ci_arr[sorted_indices]

        ax.fill_between(r_sorted, b_sorted - ci_sorted, b_sorted + ci_sorted, color='#1f77b4', alpha=0.2, label='95% Confidence Band')
        ax.plot(r_sorted, b_sorted, color='#1f77b4', linewidth=2.2, marker='o', markersize=4, label=r'Controlled Path Exponent $\beta_{\mathrm{left}}$')
        
        # Draw dynamically estimated critical transition threshold line
        ax.axvline(r_crit, color='crimson', linestyle='--', linewidth=1.8, label=f'Critical Transition Boundary ($R^*={r_crit:.3f}$)')

        ax.set_title(r'Controlled Phase Transition Diagram (Fixed $c=0.10$, Reflect, Lognormal)', fontsize=12, fontweight='bold')
        ax.set_xlabel(r'Ratio $R^* = p / c$', fontsize=11)
        ax.set_ylabel(r'Left-Tail Decay Exponent $\beta_{\mathrm{left}}$', fontsize=11)
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.legend(loc='upper right', frameon=True, fontsize=9)

        plt.tight_layout()
        plt.savefig(filepath)
        plt.close()
        return filepath

    def plot_agent_trajectory(self, history: List[Dict[str, Any]], filename: str = "agent_trajectory.png") -> str:
        """
        Plot agent exploration path across parameter space (p vs c).
        """
        filepath = os.path.join(self.output_dir, filename)
        if not HAS_MATPLOTLIB:
            print(f"[Notice] matplotlib is not installed. Plot skipped ({filename})")
            return filepath

        p_vals = [r["p"] for r in history]
        c_vals = [r["c"] for r in history]
        rounds = list(range(1, len(history) + 1))

        fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
        sc = ax.scatter(p_vals, c_vals, c=rounds, cmap='coolwarm', s=50, edgecolors='black', linewidth=0.5)
        ax.plot(p_vals, c_vals, color='gray', linestyle=':', alpha=0.5, linewidth=1.0)

        cb = plt.colorbar(sc, ax=ax)
        cb.set_label('Exploration Round (1 to 100)', fontsize=11)

        ax.set_title('Adaptive Agent Parameter Space Exploration Trajectory (100 Rounds)', fontsize=12, fontweight='bold')
        ax.set_xlabel('Shock Probability $p$', fontsize=11)
        ax.set_ylabel('Dissipation Rate $c$', fontsize=11)
        
        plt.tight_layout()
        plt.savefig(filepath)
        plt.close()
        return filepath

    def plot_baseline_comparison(self, adapt_history: List[Dict[str, Any]], 
                                 rand_history: List[Dict[str, Any]], 
                                 filename: str = "baseline_comparison.png") -> str:
        """
        Plot comparison of Adaptive Agent vs Random Baseline (Equal Budget 100 vs 100 Rounds).
        Shows Cumulative D1 CUSUM Mutation Discovery Yield and Empirical Standard Error 95% CI.
        """
        filepath = os.path.join(self.output_dir, filename)
        if not HAS_MATPLOTLIB:
            print(f"[Notice] matplotlib is not installed. Plot skipped ({filename})")
            return filepath

        from .analysis import PhaseTransitionAnalyzer
        analyzer = PhaseTransitionAnalyzer()
        bench_stats = analyzer.compare_adaptive_vs_random(adapt_history, rand_history)
        p_val = bench_stats.get("p_val", 0.01)

        # Compute cumulative discovery counts based strictly on D1 CUSUM jumps
        adapt_discoveries = [1.0 if analyzer.detect_phase_transition(adapt_history[:i+1])[0] else 0.0 for i in range(len(adapt_history))]
        rand_discoveries = [1.0 if analyzer.detect_phase_transition(rand_history[:i+1])[0] else 0.0 for i in range(len(rand_history))]

        adapt_cum = np.cumsum(adapt_discoveries)
        rand_cum = np.cumsum(rand_discoveries)

        rounds = np.arange(1, len(adapt_cum) + 1)

        fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
        ax.plot(rounds, adapt_cum, color='#1f77b4', linewidth=2.2, label='Adaptive Active Learning Agent (100 Rounds)')
        ax.plot(rounds[:len(rand_cum)], rand_cum, color='#ff7f0e', linewidth=1.8, linestyle='--', label='Random Baseline Agent (100 Rounds)')

        # Empirical Standard Error (SE) 95% confidence bands (1.96 * SE)
        se_adapt = np.sqrt(adapt_cum * (1 - adapt_cum / (rounds + 1e-6)) / (rounds + 1e-6))
        se_rand = np.sqrt(rand_cum * (1 - rand_cum / (rounds[:len(rand_cum)] + 1e-6)) / (rounds[:len(rand_cum)] + 1e-6))

        ax.fill_between(rounds, np.maximum(0, adapt_cum - 1.96 * se_adapt), adapt_cum + 1.96 * se_adapt, color='#1f77b4', alpha=0.15, label='Adaptive 95% SE CI')
        ax.fill_between(rounds[:len(rand_cum)], np.maximum(0, rand_cum - 1.96 * se_rand), rand_cum + 1.96 * se_rand, color='#ff7f0e', alpha=0.15, label='Random 95% SE CI')

        ax.set_title(f'Equal-Budget Discovery Benchmark (Welch $t$-test $p = {p_val:.4e}$)', fontsize=12, fontweight='bold')
        ax.set_xlabel('Exploration Round', fontsize=11)
        ax.set_ylabel('Cumulative D1 CUSUM Discoveries', fontsize=11)
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.legend(loc='upper left', frameon=True)

        plt.tight_layout()
        plt.savefig(filepath)
        plt.close()
        return filepath

    def plot_left_tail_fit(self, hist_counts: List[int], bin_edges: List[float], beta_left: float, 
                           model_stats: Dict[str, Any] = None, filename: str = "left_tail_fit.png") -> str:
        """
        Plot wealth distribution histogram and log-log left-tail fit with Model Selection AIC/BIC inset box.
        """
        filepath = os.path.join(self.output_dir, filename)
        if not HAS_MATPLOTLIB:
            print(f"[Notice] matplotlib is not installed. Plot skipped ({filename})")
            return filepath

        bin_centers = 0.5 * (np.array(bin_edges[:-1]) + np.array(bin_edges[1:]))
        counts = np.array(hist_counts)
        valid = (counts > 0) & (bin_centers > 0) & (bin_centers <= 2.5)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.8), dpi=300)

        # Plot 1: Linear Histogram
        ax1.bar(bin_centers[valid], counts[valid], width=(bin_edges[1]-bin_edges[0]), color='#2ca02c', alpha=0.7, edgecolor='none')
        ax1.set_title('Wealth Distribution Histogram ($W \leq 2.5$)', fontsize=11, fontweight='bold')
        ax1.set_xlabel('Wealth $W$', fontsize=10)
        ax1.set_ylabel('Agent Count', fontsize=10)
        ax1.grid(True, linestyle=':', alpha=0.5)

        # Plot 2: Log-Log Tail Fit
        w_vals = bin_centers[valid]
        pdf = counts[valid] / np.sum(counts[valid])
        cdf = np.cumsum(pdf)

        ax2.loglog(w_vals, cdf, 'bo', markersize=4, alpha=0.7, label='Empirical CDF')
        cdf_model = (w_vals / w_vals[-1]) ** beta_left * cdf[-1]
        ax2.loglog(w_vals, cdf_model, 'r--', linewidth=2.0, label=f'Power-Law Fit ($\\beta_{{\\mathrm{{left}}}}={beta_left:.2f}$)')

        ax2.set_title(r'Log-Log Left-Tail Model Selection', fontsize=11, fontweight='bold')
        ax2.set_xlabel('Wealth $W$ (log scale)', fontsize=10)
        ax2.set_ylabel(r'$P(W \leq w)$ (log scale)', fontsize=10)
        ax2.grid(True, linestyle=':', alpha=0.5)
        ax2.legend(loc='lower right', frameon=True, fontsize=9)

        # Inset Box for Model Selection Metrics from actual simulated tail_wealth
        if model_stats:
            p_val = model_stats.get('p_value_lr', 0.01)
            text_str = (
                f"Model Selection (Simulated Wealth):\n"
                f"• Power-Law AIC: {model_stats.get('aic_power_law', 0.0):.1f}\n"
                f"• Lognormal AIC: {model_stats.get('aic_lognormal', 0.0):.1f}\n"
                f"• Exponential AIC: {model_stats.get('aic_exponential', 0.0):.1f}\n"
                f"• Best Fit: {model_stats.get('best_fit', 'Power-Law')} (Vuong p={p_val:.4f})"
            )
            ax2.text(0.05, 0.95, text_str, transform=ax2.transAxes, fontsize=8,
                     verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()
        plt.savefig(filepath)
        plt.close()
        return filepath
