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

    def _sanitize(self, obj: Any) -> Any:
        import numpy as np
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.float32, np.float64, np.floating)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64, np.integer)):
            return int(obj)
        elif isinstance(obj, dict):
            return {k: self._sanitize(v) for k, v in obj.items() if k != "tail_wealth"}
        elif isinstance(obj, list):
            return [self._sanitize(v) for v in obj]
        return obj

    def save_full_experiment_json(self, history: List[Dict[str, Any]], filename: str = "kesten_full_history.json"):
        """
        Save complete experiment history with histograms.
        """
        target_path = os.path.join(self.log_dir, filename)
        clean_history = self._sanitize(history)
        with open(target_path, "w") as f:
            json.dump(clean_history, f, indent=2)

    def save_scientific_signals_json(self, scientific_signals: Dict[str, Any], filename: str = "scientific_signals.json"):
        """
        Save categorized scientific discoveries, anomalies/counterexamples, and negative results.
        """
        target_path = os.path.join(self.log_dir, filename)
        clean_signals = self._sanitize(scientific_signals)
        with open(target_path, "w") as f:
            json.dump(clean_signals, f, indent=2)

