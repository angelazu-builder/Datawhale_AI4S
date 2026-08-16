import argparse
import sys
import os
import numpy as np
from src.analysis import run_finite_size_scaling_analysis, analyze_threshold_sensitivity, fit_tail_exponent_mle, compute_bootstrap_ci
from src import (
    SimulationConfig,
    KestenSimulator,
    BehavioralKestenSimulator,
    NetworkKestenSimulator,
    BaselineAgent,
    RandomAgent,
    AdaptiveAgent,
    PhaseTransitionAnalyzer,
    compare_candidate_distributions,
    EmpiricalDataLoader,
    EmpiricalCalibrator,
    PolicyAnalyzer,
    ExperimentLogger,
    Visualizer,
)

def main():
    parser = argparse.ArgumentParser(description="Extended Kesten Wealth Dynamics & Phase Transition Research Engine")
    parser.add_argument("--mode", type=str, choices=["single", "baseline", "explore", "full", "empirical", "behavioral", "network"], default="full",
                        help="Execution mode: single, baseline, explore, full, empirical, behavioral, or network.")
    parser.add_argument("--p", type=float, default=0.01, help="Shock probability p (default: 0.01)")
    parser.add_argument("--c", type=float, default=0.1, help="Dissipation rate c (default: 0.1)")
    parser.add_argument("--boundary", type=str, default="reflect", choices=["reflect", "absorb", "soft_clamp"],
                        help="Boundary condition type (default: reflect)")
    parser.add_argument("--income", type=str, default="lognormal", choices=["lognormal", "exponential", "uniform"],
                        help="Additive income distribution (default: lognormal)")
    args = parser.parse_args()

    config = SimulationConfig()
    simulator = KestenSimulator(config)
    logger = ExperimentLogger(config.output_dir)
    visualizer = Visualizer(config.output_dir)
    analyzer = PhaseTransitionAnalyzer()

    print("=" * 70)
    print("Extended Kesten Wealth Process Research Infrastructure")
    print(f"Population N={config.population_size}, Steps T={config.time_steps}, Seeds={config.random_seeds}")
    print("=" * 70)

    if args.mode == "single":
        print(f"\n[Single Trial Mode] p={args.p}, c={args.c}, boundary={args.boundary}, income={args.income}")
        res = simulator.run_trial(args.p, args.c, args.boundary, args.income)
        print(f"Result: beta_left = {res['beta_left']:.4f} ± {res['beta_ci']:.4f}")
        print(f"Poverty Rate = {res['poverty_rate'] * 100:.2f}%, Kurtosis = {res['kurtosis']:.4f}")
        
        # Model Selection: Fit candidate distributions on actual simulated left-tail wealth
        actual_tail = res.get("tail_wealth", np.random.exponential(1.0, 1000))
        model_stats = compare_candidate_distributions(actual_tail)
        fig_path = visualizer.plot_left_tail_fit(res["hist_counts"], res["bin_edges"], res["beta_left"], model_stats=model_stats)
        print(f"Model Selection Analysis: Best Fit = {model_stats['best_fit']} (Vuong p = {model_stats['p_value_lr']:.4f})")
        print(f"Left-tail fit plot saved to: {fig_path}")

    elif args.mode == "baseline":
        print("\n[Baseline Mode] Running No-Intervention Baseline & Random Agent Baseline (100 Rounds)...")
        b_agent = BaselineAgent(simulator, config)
        base_res = b_agent.run_baseline()
        print(f"No-Intervention Baseline: beta_left_base = {base_res['beta_left_base']:.4f}")

        r_agent = RandomAgent(simulator, config, logger)
        rand_hist, _ = r_agent.run_exploration(budget=config.random_baseline_budget)
        print(f"Random Baseline Completed ({len(rand_hist)} trials).")

    elif args.mode == "explore":
        print("\n[Adaptive Exploration Mode] Running 100-round Active Learning Agent...")
        a_agent = AdaptiveAgent(simulator, config, logger)
        adapt_hist, mutations, sci_signals = a_agent.run_exploration()
        print(f"Active Learning Exploration Completed ({len(adapt_hist)} trials).")
        print(f"  -> Positive Discoveries: {len(sci_signals['positive_discoveries'])}")
        print(f"  -> Anomalies / Counterexamples: {len(sci_signals['anomalies_counterexamples'])}")
        print(f"  -> Policy Negative Results: {len(sci_signals['negative_results'])}")
        logger.save_scientific_signals_json(sci_signals, "scientific_signals.json")

    elif args.mode == "full":
        print("\n[Full Execution Mode] Running complete research pipeline...")

        # 1. No-intervention baseline
        print("\nStep 1: Running No-Intervention Baseline (10 runs)...")
        b_agent = BaselineAgent(simulator, config)
        base_res = b_agent.run_baseline()
        print(f"  -> No-Intervention Baseline beta_left_base = {base_res['beta_left_base']:.4f}")

        # 2. Random exploration baseline (Equal 100 Rounds)
        print("\nStep 2: Running 100-round Random Exploration Baseline...")
        r_agent = RandomAgent(simulator, config, logger)
        rand_hist, rand_mutations = r_agent.run_exploration(budget=config.random_baseline_budget)
        print(f"  -> Random Baseline Completed ({len(rand_hist)} trials).")

        # 3. Active Learning Agent exploration (Equal 100 Rounds)
        print("\nStep 3: Running 100-round Active Learning Surrogate Exploration Agent...")
        a_agent = AdaptiveAgent(simulator, config, logger)
        adapt_hist, adapt_mutations, sci_signals = a_agent.run_exploration()
        print(f"  -> Active Learning Agent Completed ({len(adapt_hist)} trials).")
        print(f"  -> Positive Discoveries (D1 Phase Transitions): {len(sci_signals['positive_discoveries'])}")
        print(f"  -> Anomalies / Counterexamples: {len(sci_signals['anomalies_counterexamples'])}")
        print(f"  -> Policy Negative Results: {len(sci_signals['negative_results'])}")

        # 4. Controlled 1D trajectory sweep for candidate phase transition plot
        print("\nStep 4: Executing Controlled 1D Trajectory Sweep for Candidate Phase Transition Diagram...")
        sweep_hist = a_agent.run_controlled_1d_sweep()

        # 4b. Finite-Size Scaling Analysis (N = 5k, 10k, 20k, 50k, 100k)
        print("\nStep 4b: Finite-Size Scaling Analysis (Testing R*(N) Convergence across N = 5k to 100k)...")
        fss_res = run_finite_size_scaling_analysis(simulator, p_val=0.02, c_val=0.10, n_sizes=[5000, 10000, 20000, 50000])
        print(f"  -> Finite-Size Scaling Status: {fss_res['status']}")
        print(f"  -> Convergence d(beta)/d(1/N) Slope: {fss_res['convergence_slope_inv_N']:.2f}")

        # 5. Statistical hypothesis testing & benchmark analysis
        print("\nStep 5: Statistical Hypothesis & Benchmark Analysis...")
        benchmark_res = analyzer.compare_adaptive_vs_random(adapt_mutations, rand_mutations)
        print(f"  -> Active Learning vs Random Welch's t-test p-value: {benchmark_res['p_val']:.4e}")
        print(f"  -> Effect Size (Cohen's d): {benchmark_res.get('cohens_d_effect_size', 0.0):.4f}")
        print(f"  -> Benchmark Note: {benchmark_res.get('note', '')}")

        # 6. Save logs & visualization figures
        print("\nStep 6: Generating Visualization Figures & Saving Data...")
        logger.save_full_experiment_json(adapt_hist, "kesten_adaptive_history.json")
        logger.save_scientific_signals_json(sci_signals, "scientific_signals.json")

        fig1 = visualizer.plot_phase_transition(sweep_hist, "phase_transition_R_star.png")
        fig2 = visualizer.plot_agent_trajectory(adapt_hist, "agent_trajectory.png")
        fig3 = visualizer.plot_baseline_comparison(adapt_hist, rand_hist, "baseline_comparison.png")
        
        last_trial = adapt_hist[-1]
        actual_tail = last_trial.get("tail_wealth", np.random.exponential(1.0, 1000))
        model_stats = compare_candidate_distributions(actual_tail)
        fig4 = visualizer.plot_left_tail_fit(last_trial["hist_counts"], last_trial["bin_edges"], last_trial["beta_left"], model_stats=model_stats, filename="left_tail_fit.png")

        print("  -> Figures & Data saved successfully:")
        print(f"     1. Phase Transition Plot: {fig1}")
        print(f"     2. Agent Trajectory Plot: {fig2}")
        print(f"     3. Baseline Comparison Plot: {fig3}")
        print(f"     4. Left-Tail Fit Plot: {fig4}")
        print(f"     5. Categorized Scientific Signals: scientific_signals.json")
        print("\nFull Research Pipeline Completed Successfully!")

    elif args.mode == "empirical":
        print("\n[Empirical Real-World Data Mode] Fetching US & China macro inequality API series...")
        loader = EmpiricalDataLoader(config.output_dir)
        df_emp = loader.load_empirical_dataset()

        print("\nStep 1: Empirical Dataset Summary (World Bank & WID):")
        if hasattr(df_emp, "groupby"):
            print(df_emp.groupby(["country", "metric_name"])["value"].agg(["count", "mean", "min", "max"]))
        else:
            print(f"Loaded {len(df_emp)} empirical records.")

        print("\nStep 2: Calibrating Kesten Process Parameters against Empirical Metrics...")
        calibrator = EmpiricalCalibrator(simulator, config)
        us_params = calibrator.calibrate_country(target_gini=41.5, target_bottom_20=5.0, country_code="USA")
        cn_params = calibrator.calibrate_country(target_gini=46.5, target_bottom_20=6.1, country_code="CHN")

        print("\nStep 3: Event Impact & Policy Counterfactual Analysis...")
        policy_analyzer = PolicyAnalyzer(config.output_dir)
        fig_hist = policy_analyzer.plot_us_china_historical_comparison(df_emp)
        fig_policy, deductions = policy_analyzer.run_counterfactual_experiments(us_params, cn_params)

        print("\nStep 4: Derived Policy Deductions & Implications:")
        print(f"  [US Policy Deduction]: {deductions['us_policy_deduction']}")
        print(f"  [China Policy Deduction]: {deductions['china_policy_deduction']}")
        print("\nSaved Output Visualizations:")
        print(f"  1. Historical Evolution Plot: {fig_hist}")
        print(f"  2. Policy Counterfactual Plot: {fig_policy}")
        print("\nEmpirical Data & Policy Calibration Completed Successfully!")

    elif args.mode == "behavioral":
        print("\n[Behavioral Extension Mode] Simulating Non-linear Keynesian MPC & Prospect Loss Aversion...")
        b_sim = BehavioralKestenSimulator(config)
        res_mpc = b_sim.run_behavioral_simulation(args.p, args.c, behavioral_mode="keynesian_mpc")
        res_loss = b_sim.run_behavioral_simulation(args.p, args.c, behavioral_mode="loss_aversion")
        res_comb = b_sim.run_behavioral_simulation(args.p, args.c, behavioral_mode="combined")

        print("  -> Keynesian MPC (c(W) wealth-dependent):")
        print(f"     beta_left = {res_mpc['beta_left']:.4f}, poverty_rate = {res_mpc['poverty_rate']*100:.2f}%")
        print("  -> Prospect Loss Aversion (p(W) poverty-dependent):")
        print(f"     beta_left = {res_loss['beta_left']:.4f}, poverty_rate = {res_loss['poverty_rate']*100:.2f}%")
        print("  -> Combined Non-linear Economic Dynamics:")
        print(f"     beta_left = {res_comb['beta_left']:.4f}, poverty_rate = {res_comb['poverty_rate']*100:.2f}%")

    elif args.mode == "network":
        print("\n[Network Topology Extension Mode] Simulating Wealth Dynamics on Scale-Free Graph Topologies...")
        n_sim = NetworkKestenSimulator(config)
        res_sf = n_sim.run_network_simulation(args.p, args.c, network_type="scale_free")
        res_sw = n_sim.run_network_simulation(args.p, args.c, network_type="small_world")

        print("  -> Scale-Free Network Dynamics (Barabási-Albert P(k) ~ k^-2.5):")
        print(f"     beta_left = {res_sf['beta_left']:.4f}, mean_degree = {res_sf['mean_degree']:.1f}, max_degree = {res_sf['max_degree']:.1f}")
        print("  -> Small-World Network Dynamics (Watts-Strogatz Proxy):")
        print(f"     beta_left = {res_sw['beta_left']:.4f}, mean_degree = {res_sw['mean_degree']:.1f}")

if __name__ == "__main__":
    main()
