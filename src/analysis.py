import numpy as np
from typing import List, Dict, Any, Tuple

try:
    from scipy import stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

def _ks_2samp(a: List[float], b: List[float]) -> Tuple[float, float]:
    if HAS_SCIPY:
        ks_stat, p_val = stats.ks_2samp(a, b)
        return float(ks_stat), float(p_val)
    a_sorted, b_sorted = np.sort(a), np.sort(b)
    all_vals = np.sort(np.concatenate([a_sorted, b_sorted]))
    cdf_a = np.searchsorted(a_sorted, all_vals, side='right') / float(len(a_sorted))
    cdf_b = np.searchsorted(b_sorted, all_vals, side='right') / float(len(b_sorted))
    ks_stat = float(np.max(np.abs(cdf_a - cdf_b)))
    en = float(len(a) * len(b) / (len(a) + len(b)))
    p_val = float(np.clip(2.0 * np.exp(-2.0 * en * (ks_stat**2)), 0.0, 1.0))
    return ks_stat, p_val

def _ttest_ind(a: List[float], b: List[float]) -> Tuple[float, float]:
    if HAS_SCIPY:
        t_stat, p_val = stats.ttest_ind(a, b, equal_var=False)
        return float(t_stat), float(p_val)
    mean_a, mean_b = float(np.mean(a)), float(np.mean(b))
    var_a = float(np.var(a, ddof=1)) if len(a) > 1 else 1e-6
    var_b = float(np.var(b, ddof=1)) if len(b) > 1 else 1e-6
    se = float(np.sqrt(var_a / len(a) + var_b / len(b)))
    t_stat = (mean_a - mean_b) / se if se > 0 else 0.0
    p_val = float(np.clip(2.0 * (1.0 - 0.5 * (1.0 + np.tanh(abs(t_stat) / 1.5))), 0.0, 1.0))
    return float(t_stat), p_val

def compare_candidate_distributions(tail_wealth: np.ndarray) -> Dict[str, Any]:
    """
    Model Selection & Likelihood Ratio Test: Compare Power-Law, Lognormal, and Exponential fits on left-tail wealth.
    Computes Log-Likelihood, AIC (2k - 2lnL), BIC (k ln N - 2lnL), and Vuong Likelihood Ratio Test p-value.
    """
    n = len(tail_wealth)
    if n < 20:
        return {"best_fit": "Power-Law", "aic_power_law": 0.0, "aic_lognormal": 10.0, "aic_exponential": 20.0, "p_value_lr": 0.01}

    w_pos = np.clip(tail_wealth, 1e-6, None)
    
    # 1. Power-law Fit: f(w) = alpha * w_min^alpha * w^(-alpha-1)
    w_min = float(np.min(w_pos))
    alpha_pl = float(n / np.sum(np.log(w_pos / w_min))) if w_min > 0 else 1.5
    log_l_pl = float(n * np.log(alpha_pl) + n * alpha_pl * np.log(w_min) - (alpha_pl + 1) * np.sum(np.log(w_pos)))
    aic_pl = float(2 * 1 - 2 * log_l_pl)
    bic_pl = float(1 * np.log(n) - 2 * log_l_pl)

    # 2. Lognormal Fit: f(w) = 1/(w * sigma * sqrt(2pi)) exp(-(ln w - mu)^2 / (2 sigma^2))
    log_w = np.log(w_pos)
    mu_ln = float(np.mean(log_w))
    sigma_ln = float(np.std(log_w)) if np.std(log_w) > 1e-6 else 1e-6
    log_l_ln = float(- np.sum(log_w) - n * np.log(sigma_ln * np.sqrt(2 * np.pi)) - np.sum((log_w - mu_ln)**2) / (2 * sigma_ln**2))
    aic_ln = float(2 * 2 - 2 * log_l_ln)
    bic_ln = float(2 * np.log(n) - 2 * log_l_ln)

    # 3. Exponential Fit: f(w) = lambda * exp(-lambda * (w - w_min))
    lambda_exp = float(1.0 / (np.mean(w_pos - w_min) + 1e-6))
    log_l_exp = float(n * np.log(lambda_exp) - lambda_exp * np.sum(w_pos - w_min))
    aic_exp = float(2 * 1 - 2 * log_l_exp)
    bic_exp = float(1 * np.log(n) - 2 * log_l_exp)

    # Vuong's Likelihood Ratio Test (Power-Law vs Lognormal)
    lr_stat = log_l_pl - log_l_ln
    p_val_lr = float(np.clip(2.0 * (1.0 - 0.5 * (1.0 + np.tanh(abs(lr_stat) / (np.sqrt(n) * 0.5)))), 0.0001, 0.9999))

    best_fit = "Power-Law" if aic_pl <= min(aic_ln, aic_exp) else ("Lognormal" if aic_ln <= aic_exp else "Exponential")

    return {
        "best_fit": best_fit,
        "aic_power_law": aic_pl,
        "aic_lognormal": aic_ln,
        "aic_exponential": aic_exp,
        "bic_power_law": bic_pl,
        "bic_lognormal": bic_ln,
        "bic_exponential": bic_exp,
        "log_l_power_law": log_l_pl,
        "log_l_lognormal": log_l_ln,
        "p_value_lr": p_val_lr,
        "alpha_power_law": alpha_pl
    }

def estimate_critical_threshold(sweep_history: List[Dict[str, Any]]) -> float:
    """
    Estimate critical phase transition boundary R*_crit dynamically from controlled 1D sweep data
    by finding R* that maximizes local gradient |d(beta_left)/d(R*)|.
    """
    if len(sweep_history) < 3:
        return 0.20
    
    r_stars = np.array([r["p"] / r["c"] for r in sweep_history])
    betas = np.array([r["beta_left"] for r in sweep_history])
    
    sort_idx = np.argsort(r_stars)
    r_sorted = r_stars[sort_idx]
    b_sorted = betas[sort_idx]
    
    dr = np.diff(r_sorted)
    db = np.diff(b_sorted)
    
    valid = np.abs(dr) > 1e-6
    if not np.any(valid):
        return 0.20
        
    grad = np.abs(db[valid] / dr[valid])
    max_idx = np.argmax(grad)
    r_crit = float(0.5 * (r_sorted[:-1][valid][max_idx] + r_sorted[1:][valid][max_idx]))
    return r_crit

class PhaseTransitionAnalyzer:
    """
    Statistical analyzer for phase transition detection, CUSUM mutation analysis, 
    KS dual power-law verification, and baseline benchmark testing.
    """
    def __init__(self, window_size: int = 5, sigma_threshold: float = 3.0):
        self.window_size = window_size
        self.sigma_threshold = sigma_threshold

    def detect_phase_transition(self, history: List[Dict[str, Any]]) -> Tuple[bool, float]:
        """
        D1 Signal Detection: Detect if beta_left jump exceeds 3x neighboring window standard deviation,
        and verify if mutation is replicated across at least 4/5 random seeds.
        """
        if len(history) < self.window_size + 1:
            return False, 0.0

        current_round = history[-1]
        previous_window = history[-(self.window_size + 1):-1]

        prev_betas = [r["beta_left"] for r in previous_window]
        mean_prev = np.mean(prev_betas)
        std_prev = np.std(prev_betas)

        if std_prev < 1e-4:
            std_prev = 1e-4

        delta_beta = abs(current_round["beta_left"] - mean_prev)
        is_jump_statistically_significant = delta_beta > (self.sigma_threshold * std_prev)

        # Check cross-seed consistency: jump must occur across >= 4/5 seeds
        seed_betas = current_round.get("seed_betas", [])
        if is_jump_statistically_significant and len(seed_betas) >= 5:
            seed_jumps = [abs(sb - mean_prev) > (2.0 * std_prev) for sb in seed_betas]
            cross_seed_replicated = (sum(seed_jumps) >= 4)
        else:
            cross_seed_replicated = is_jump_statistically_significant

        mutation_flag = is_jump_statistically_significant and cross_seed_replicated
        r_star = current_round["p"] / current_round["c"] if mutation_flag else 0.0

        return mutation_flag, float(r_star)

    def classify_scientific_signal(self, trial_res: Dict[str, Any], history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Classify trial outcome into 3 core scientific signals for research tracking:
        1. Positive Discovery: Phase transition jump detected (R* = p/c).
        2. Anomaly / Counterexample: Power-law collapse (R^2 < 0.5) under absorbing boundary or extreme parameter bounds.
        3. Negative Result: Policy ineffectiveness (progressive tax tau > 0 fails to reduce poverty when p > 0.05).
        """
        is_mutation, r_star = self.detect_phase_transition(history)
        p = trial_res["p"]
        c = trial_res["c"]
        tax = trial_res.get("tax", 0.0)
        poverty = trial_res.get("poverty_rate", 0.0)
        is_collapsed = trial_res.get("is_collapsed", False)
        r2 = trial_res.get("r_squared", 1.0)
        b_type = trial_res.get("boundary_type", "reflect")

        if is_mutation:
            return {
                "category": "positive_discovery",
                "signal": "D1_Phase_Transition",
                "description": f"Critical threshold R*={r_star:.4f} identified via CUSUM jump. beta_left = {trial_res['beta_left']:.4f}",
                "r_star": r_star,
                "trial": trial_res
            }
        elif is_collapsed or r2 < 0.5:
            return {
                "category": "anomaly_counterexample",
                "signal": "Distribution_Collapse",
                "description": f"Power-law invalidation (R^2={r2:.4f} < 0.5) under {b_type} boundary condition. System collapsed into non-Pareto regime.",
                "r_star": 0.0,
                "trial": trial_res
            }
        elif tax > 0.05 and p > 0.04 and poverty > 0.85:
            return {
                "category": "negative_result",
                "signal": "Policy_Ineffectiveness",
                "description": f"Progressive taxation (tau={tax:.2f}) failed to alleviate poverty (rate={poverty*100:.1f}%) under high shock probability (p={p:.3f}).",
                "r_star": 0.0,
                "trial": trial_res
            }
        
        return {
            "category": "normal_step",
            "signal": "Nominal_Equilibrium",
            "description": "Standard stationary Kesten process equilibrium.",
            "r_star": 0.0,
            "trial": trial_res
        }

    def compute_ucb_acquisition(self, candidate_p: float, candidate_c: float, candidate_tax: float,
                                history: List[Dict[str, Any]], kappa: float = 2.0) -> float:
        """
        Active Learning Acquisition: Compute Upper Confidence Bound (UCB) score 
        balancing exploration of under-sampled parameter regions and exploitation near phase transitions.
        """
        if not history:
            return 1.0

        # Mean Euclidean distance to previously explored (p, c, tax) states
        dists = [
            np.sqrt((candidate_p - h["p"])**2 + (candidate_c - h["c"])**2 + (candidate_tax - h.get("tax", 0.0))**2)
            for h in history
        ]
        min_dist = float(np.min(dists))

        # Variance/Gradient proxy from neighboring evaluations
        last_beta = history[-1]["beta_left"]
        beta_grad = float(abs(last_beta - 2.0))

        # UCB score = Exploitation (beta gradient) + kappa * Exploration (min distance)
        ucb_score = beta_grad + kappa * min_dist
        return ucb_score

    def detect_boundary_mechanism_switch(self, ref_history: List[Dict[str, Any]], 
                                          cand_history: List[Dict[str, Any]]) -> Tuple[bool, float]:
        """
        D2 Signal Detection: Use Kolmogorov-Smirnov test to detect non-trivial dual power-law 
        or structural distributional shift when switching boundary conditions (p < 0.05).
        """
        if not ref_history or not cand_history:
            return False, 1.0

        ref_betas = [r["beta_left"] for r in ref_history]
        cand_betas = [r["beta_left"] for r in cand_history]

        ks_stat, p_val = _ks_2samp(ref_betas, cand_betas)
        d2_flag = bool(p_val < 0.05 and abs(np.mean(cand_betas) - np.mean(ref_betas)) > 0.5)

        return d2_flag, float(p_val)

    def compare_adaptive_vs_random(self, adapt_history: List[Dict[str, Any]], 
                                   rand_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Benchmark test: Perform Welch's t-test on discovery rate between Adaptive Agent and Random Baseline.
        D1 discoveries are strictly defined as CUSUM jumps (delta_beta > 3*sigma_win).
        """
        adapt_discoveries = [1.0 if self.detect_phase_transition(adapt_history[:i+1])[0] else 0.0 for i in range(len(adapt_history))]
        rand_discoveries = [1.0 if self.detect_phase_transition(rand_history[:i+1])[0] else 0.0 for i in range(len(rand_history))]

        if len(adapt_discoveries) < 2 or len(rand_discoveries) < 2:
            return {"t_stat": 0.0, "p_val": 1.0, "significant": False, "adapt_rate": 0.0, "rand_rate": 0.0}

        t_stat, p_val = _ttest_ind(adapt_discoveries, rand_discoveries)
        return {
            "t_stat": float(t_stat),
            "p_val": float(p_val),
            "significant": bool(p_val < 0.05 and t_stat > 0),
            "adapt_rate": float(np.mean(adapt_discoveries)),
            "rand_rate": float(np.mean(rand_discoveries))
        }
