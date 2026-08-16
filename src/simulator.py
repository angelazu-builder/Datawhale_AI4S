import numpy as np
from typing import Dict, Any, Tuple
from .config import SimulationConfig

try:
    from scipy import stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

def _kurtosis(x: np.ndarray) -> float:
    if HAS_SCIPY:
        return float(stats.kurtosis(x))
    m2 = np.mean((x - np.mean(x))**2)
    if m2 <= 0:
        return 0.0
    m4 = np.mean((x - np.mean(x))**4)
    return float(m4 / (m2**2) - 3.0)

def _linregress(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
    if HAS_SCIPY:
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        return float(slope), float(intercept), float(r_value)
    x_m = np.mean(x)
    y_m = np.mean(y)
    ss_xx = np.sum((x - x_m)**2)
    if ss_xx <= 0:
        return 0.0, float(y_m), 0.0
    ss_xy = np.sum((x - x_m) * (y - y_m))
    slope = float(ss_xy / ss_xx)
    intercept = float(y_m - slope * x_m)
    ss_yy = np.sum((y - y_m)**2)
    denom = np.sqrt(ss_xx * ss_yy)
    r_val = float(ss_xy / denom) if denom > 0 else 0.0
    return slope, intercept, r_val

try:
    import numba
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

if HAS_NUMBA:
    @numba.njit(parallel=True, fastmath=True)
    def _run_kesten_loop(wealth: np.ndarray, T: int, p: float, c: float, 
                         boundary_code: int, income_code: int, seed_val: int,
                         tax: float = 0.0, subsidy: float = 0.0) -> np.ndarray:
        np.random.seed(seed_val)
        N = len(wealth)
        eps = 1e-6

        for t in range(T):
            for i in numba.prange(N):
                if income_code == 0:
                    inc = np.random.lognormal(-2.5, 0.5)
                elif income_code == 1:
                    inc = np.random.exponential(0.10)
                else:
                    inc = np.random.uniform(0.02, 0.18)

                w = wealth[i] + inc - c * wealth[i]
                if tax > 0.0 and w > 1.0:
                    w -= tax * (w - 1.0)
                w += subsidy

                if np.random.random() < p:
                    w = np.random.uniform(0.001, 0.05)

                if boundary_code == 0:  # reflect
                    if w <= 0:
                        w = abs(w) + eps
                elif boundary_code == 1:  # absorb
                    if w <= 0:
                        w = eps
                elif boundary_code == 2:  # soft_clamp
                    if w <= 0.05:
                        w = 0.05 * np.exp((w - 0.05) / 0.05) + eps
                    if w <= 0:
                        w = eps

                if w < 1e-7:
                    w = 1e-7
                elif w > 1e8:
                    w = 1e8

                wealth[i] = w

        return wealth
else:
    def _run_kesten_loop(wealth: np.ndarray, T: int, p: float, c: float, 
                         boundary_code: int, income_code: int, seed_val: int,
                         tax: float = 0.0, subsidy: float = 0.0) -> np.ndarray:
        """
        Fast vectorized NumPy fallback loop when numba is not installed.
        """
        np.random.seed(seed_val)
        N = len(wealth)
        eps = 1e-6

        for t in range(T):
            if income_code == 0:
                inc = np.random.lognormal(-2.5, 0.5, size=N)
            elif income_code == 1:
                inc = np.random.exponential(0.10, size=N)
            else:
                inc = np.random.uniform(0.02, 0.18, size=N)

            wealth = wealth + inc - c * wealth
            if tax > 0.0:
                tax_mask = wealth > 1.0
                wealth[tax_mask] -= tax * (wealth[tax_mask] - 1.0)
            wealth += subsidy

            shock_mask = np.random.random(size=N) < p
            if np.any(shock_mask):
                wealth[shock_mask] = np.random.uniform(0.001, 0.05, size=int(np.sum(shock_mask)))

            if boundary_code == 0:  # reflect
                neg_mask = wealth <= 0
                wealth[neg_mask] = np.abs(wealth[neg_mask]) + eps
            elif boundary_code == 1:  # absorb
                neg_mask = wealth <= 0
                wealth[neg_mask] = eps
            elif boundary_code == 2:  # soft_clamp
                clamp_mask = wealth <= 0.05
                wealth[clamp_mask] = 0.05 * np.exp((wealth[clamp_mask] - 0.05) / 0.05) + eps
                neg_mask = wealth <= 0
                wealth[neg_mask] = eps

            np.clip(wealth, 1e-7, 1e8, out=wealth)

        return wealth

class KestenSimulator:
    """
    Vectorized & Parallelized Numba JIT Extended Kesten Process Simulator.
    """
    BOUNDARY_MAP = {"reflect": 0, "absorb": 1, "soft_clamp": 2}
    INCOME_MAP = {"lognormal": 0, "exponential": 1, "uniform": 2}

    def __init__(self, config: SimulationConfig = None):
        self.config = config or SimulationConfig()

    def run_single_simulation(self, p: float, c: float, boundary_type: str, 
                               income_dist: str, seed: int,
                               tax: float = 0.0, subsidy: float = 0.0,
                               N_override: int = None, T_override: int = None) -> Dict[str, Any]:
        """
        Run a single seed simulation using parallel JIT kernel or fast vectorized fallback.
        """
        N = N_override if N_override is not None else self.config.population_size
        T = T_override if T_override is not None else self.config.time_steps
        
        rng = np.random.default_rng(seed)
        wealth_init = rng.uniform(self.config.w_init_min, self.config.w_init_max, size=N).astype(np.float64)

        b_code = self.BOUNDARY_MAP.get(boundary_type, 0)
        i_code = self.INCOME_MAP.get(income_dist, 0)

        # Execute parallel JIT compiled scalar loop or numpy fallback
        final_wealth = _run_kesten_loop(wealth_init, T, p, c, b_code, i_code, seed, tax, subsidy)

        return self.analyze_wealth_distribution(final_wealth)

    def analyze_wealth_distribution(self, wealth: np.ndarray) -> Dict[str, Any]:
        """
        Extract metrics: beta_left, poverty rate, kurtosis, r_squared fit, and left-tail histogram.
        """
        N = len(wealth)
        poverty_rate = float(np.mean(wealth < self.config.poverty_threshold))
        kurtosis_val = _kurtosis(wealth)

        # Filter lower tail (wealth < 50th percentile)
        q50 = np.percentile(wealth, self.config.tail_percentile)
        q1 = np.percentile(wealth, 1.0)
        tail_mask = (wealth >= q1) & (wealth <= q50)
        tail_wealth = wealth[tail_mask]

        if len(tail_wealth) > 50:
            # Empirical CDF on left tail: P(W <= w) ~ w^beta
            sorted_w = np.sort(tail_wealth)
            ecdf = np.arange(1, len(sorted_w) + 1) / len(sorted_w)

            log_w = np.log(sorted_w)
            log_cdf = np.log(ecdf)

            # Linear regression in log-log space
            slope, intercept, r_value = _linregress(log_w, log_cdf)
            beta_left = float(slope)
            r_squared = float(r_value ** 2)
        else:
            beta_left = 0.0
            r_squared = 0.0

        hist_counts, bin_edges = np.histogram(wealth[wealth <= 5.0], bins=self.config.num_bins)

        mean_w = float(np.mean(wealth))
        is_collapsed = bool(r_squared < 0.5 or mean_w < 0.02)

        return {
            "beta_left": beta_left,
            "r_squared": r_squared,
            "is_collapsed": is_collapsed,
            "poverty_rate": poverty_rate,
            "kurtosis": kurtosis_val,
            "hist_counts": hist_counts.tolist(),
            "bin_edges": bin_edges.tolist(),
            "median_wealth": float(np.median(wealth)),
            "mean_wealth": mean_w,
            "tail_wealth": tail_wealth
        }

    def run_trial(self, p: float, c: float, boundary_type: str, income_dist: str,
                  tax: float = 0.0, subsidy: float = 0.0,
                  N_override: int = None, T_override: int = None) -> Dict[str, Any]:
        """
        Run simulation across all configured seeds and aggregate stats.
        """
        seed_results = []
        for seed in self.config.random_seeds:
            res = self.run_single_simulation(p, c, boundary_type, income_dist, seed, tax, subsidy, N_override=N_override, T_override=T_override)
            seed_results.append(res)

        beta_list = [r["beta_left"] for r in seed_results]
        r2_list = [r["r_squared"] for r in seed_results]
        poverty_list = [r["poverty_rate"] for r in seed_results]
        kurtosis_list = [r["kurtosis"] for r in seed_results]
        collapse_list = [r["is_collapsed"] for r in seed_results]

        beta_mean = float(np.mean(beta_list))
        beta_std = float(np.std(beta_list))
        ci_95 = float(1.96 * beta_std / np.sqrt(len(beta_list)))

        return {
            "p": p,
            "c": c,
            "tax": tax,
            "subsidy": subsidy,
            "boundary_type": boundary_type,
            "income_dist": income_dist,
            "beta_left": beta_mean,
            "r_squared": float(np.mean(r2_list)),
            "is_collapsed": bool(np.mean(collapse_list) > 0.5),
            "beta_std": beta_std,
            "beta_ci": ci_95,
            "poverty_rate": float(np.mean(poverty_list)),
            "kurtosis": float(np.mean(kurtosis_list)),
            "seed_betas": beta_list,
            "hist_counts": seed_results[0]["hist_counts"],
            "bin_edges": seed_results[0]["bin_edges"],
            "tail_wealth": seed_results[0]["tail_wealth"]
        }
