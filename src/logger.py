import os
import csv
import json
import time
from typing import Dict, Any, List

class ExperimentLogger:
    """
    Experiment logging manager for saving state records, CSV logs, and histogram data.
    """
    def __init__(self, log_dir: str = "."):
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        self.csv_path = os.path.join(self.log_dir, "kesten_exploration_log.csv")
        self.json_path = os.path.join(self.log_dir, "kesten_exploration_data.json")
        self._init_csv()

    def _init_csv(self):
        headers = [
            "round_id", "p", "c", "boundary_type", "income_dist",
            "beta_left", "beta_std", "beta_ci", "poverty_rate", "kurtosis",
            "mutation_flag", "r_star", "timestamp"
        ]
        with open(self.csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(headers)

    def log_round(self, round_id: int, trial_data: Dict[str, Any], 
                  mutation_flag: bool, r_star: float):
        """
        Append round metadata to CSV log.
        """
        row = [
            round_id,
            f"{trial_data['p']:.5f}",
            f"{trial_data['c']:.5f}",
            trial_data["boundary_type"],
            trial_data["income_dist"],
            f"{trial_data['beta_left']:.4f}",
            f"{trial_data['beta_std']:.4f}",
            f"{trial_data['beta_ci']:.4f}",
            f"{trial_data['poverty_rate']:.4f}",
            f"{trial_data['kurtosis']:.4f}",
            1 if mutation_flag else 0,
            f"{r_star:.4f}",
            time.strftime("%Y-%m-%d %H:%M:%S")
        ]
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(row)

    class _NumpyEncoder(json.JSONEncoder):
        """Custom JSON encoder that handles numpy scalars and arrays at any nesting depth."""
        def default(self, obj):
            try:
                import numpy as np
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                if isinstance(obj, np.floating):
                    return float(obj)
                if isinstance(obj, np.integer):
                    return int(obj)
                if isinstance(obj, np.bool_):
                    return bool(obj)
            except ImportError:
                pass
            return super().default(obj)

    def _sanitize(self, obj: Any) -> Any:
        """Strip non-serializable keys (e.g. large arrays stored under 'tail_wealth')
        before passing to the JSON encoder."""
        if isinstance(obj, dict):
            return {k: self._sanitize(v) for k, v in obj.items() if k != "tail_wealth"}
        elif isinstance(obj, list):
            return [self._sanitize(v) for v in obj]
        return obj

    def _dump_json(self, data: Any, path: str):
        """Sanitize then serialize using the numpy-aware encoder."""
        clean = self._sanitize(data)
        with open(path, "w") as f:
            json.dump(clean, f, indent=2, cls=self._NumpyEncoder)

    def save_full_experiment_json(self, history: List[Dict[str, Any]], filename: str = "kesten_full_history.json"):
        """
        Save complete experiment history with histograms.
        """
        self._dump_json(history, os.path.join(self.log_dir, filename))

    def save_scientific_signals_json(self, scientific_signals: Dict[str, Any], filename: str = "scientific_signals.json"):
        """
        Save categorized scientific discoveries, anomalies/counterexamples, and negative results.
        """
        self._dump_json(scientific_signals, os.path.join(self.log_dir, filename))
