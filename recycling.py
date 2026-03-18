# =========================
# 1. Imports

# =========================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import triang
import os
from SALib.sample import saltelli
from SALib.analyze import sobol
import seaborn as sns

plt.rcParams.update({
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "legend.fontsize": 8,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 300
})

# =========================
# 2. Global Settings
# =========================

np.random.seed(42)
N_MONTE_CARLO = 10000

FIGURE_DIR = "figures"
os.makedirs(FIGURE_DIR, exist_ok=True)

# =========================
# 3. Policy Scenarios (Table 3.1)
# =========================

SCENARIOS = {
    "once_through": {"recycle_fraction": 0.0},
    "partial_recycling": {"recycle_fraction": (0.30, 0.50)},
    "advanced_recycling": {"recycle_fraction": (0.70, 0.90)}
}


# =========================
# 4. Helper Functions # explain these entire section
# =========================

def triangular_sample(low, mode, high, size):
    c = (mode - low) / (high - low)
    return triang.rvs(c, loc=low, scale=high - low, size=size)  # explain this line


def get_recycle_fraction(scenario, size):
    s = SCENARIOS[scenario]  # explain this line
    if isinstance(s["recycle_fraction"], tuple):
        return np.random.uniform(s["recycle_fraction"][0], s["recycle_fraction"][1], size)
    else:
        return np.full(size, s["recycle_fraction"])


# =========================
# 5. Uncertainty Sampling (Table 3.2)
# =========================

def sample_uncertainties(n=N_MONTE_CARLO):
    return pd.DataFrame({
        "spent_fuel_mass": np.random.uniform(20, 35, n),  # tHM / yr
        "burnup": np.random.uniform(30, 60, n),  # GWd / tHM
        "U_frac": triangular_sample(0.93, 0.94, 0.96, n),
        "Pu_frac": triangular_sample(0.008, 0.0115, 0.015, n),
        "MA_frac": np.random.uniform(0.001, 0.005, n),
        "U_recovery": np.random.uniform(0.85, 0.95, n),
        "Pu_recovery": np.random.uniform(0.70, 0.95, n),
        "MA_recovery": np.random.uniform(0.40, 0.70, n),
        "reprocessing_cost": triangular_sample(1500, 3000, 6000, n),
        "disposal_cost": triangular_sample(400, 600, 1000, n),
        "U_value": triangular_sample(50, 100, 250, n),
        "Pu_value": triangular_sample(2000, 5000, 10000, n)
    })


# =========================
# 6. Scenario Evaluation
# =========================

def evaluate_scenario(samples, scenario):
    recycle_fraction = get_recycle_fraction(scenario, len(samples))
    spent_mass = samples["spent_fuel_mass"] * 1000  # tHM → kgHM

    # Actinide inventories
    U = spent_mass * samples["U_frac"]
    Pu = spent_mass * samples["Pu_frac"]
    MA = spent_mass * samples["MA_frac"]

    # Recovered masses
    U_rec = U * recycle_fraction * samples["U_recovery"]
    Pu_rec = Pu * recycle_fraction * samples["Pu_recovery"]
    MA_rec = MA * recycle_fraction * samples["MA_recovery"]

    # Economics
    revenue = U_rec * samples["U_value"] + Pu_rec * samples["Pu_value"]
    repro_cost = spent_mass * recycle_fraction * samples["reprocessing_cost"]
    disp_cost = (spent_mass * (1 - recycle_fraction) * samples["disposal_cost"]) + \
                (spent_mass * recycle_fraction * 0.15 * samples["disposal_cost"])
    net_econ = revenue - repro_cost - disp_cost

    return pd.DataFrame({
        "net_economic_outcome": net_econ,
        "recovered_actinide_fraction": (U_rec + Pu_rec + MA_rec) / (U + Pu + MA),
        "waste_mass": spent_mass * (1 - recycle_fraction),
        "revenue": revenue,
        "reprocessing_cost": repro_cost,
        "disposal_cost": disp_cost
    })


# =========================
# 7. Monte Carlo Simulation
# =========================

def run_monte_carlo():
    samples = sample_uncertainties()
    results = {s: evaluate_scenario(samples, s) for s in SCENARIOS}
    return samples, results


# =========================
# 8. Risk Metrics
# =========================

def compute_cvar(series, alpha=0.05):
    threshold = series.quantile(alpha)
    return series[series <= threshold].mean()


def expected_regret(results, metric, minimize=False):
    df_all = pd.concat([df[metric].rename(s) for s, df in results.items()], axis=1)
    if minimize:
        best = df_all.min(axis=1)
    else:
        best = df_all.max(axis=1)
    regret = df_all.apply(lambda col: best - col)
    return regret.mean()


def combined_summary_table_with_extremes(results, alpha=0.05):
    table = []
    for s, df in results.items():
        net_mean = df["net_economic_outcome"].mean()
        net_p05 = df["net_economic_outcome"].quantile(0.05)
        net_p95 = df["net_economic_outcome"].quantile(0.95)
        net_cvar = compute_cvar(df["net_economic_outcome"], alpha)
        act_frac_mean = df["recovered_actinide_fraction"].mean()
        waste_mean = df["waste_mass"].mean()
        prob_positive = (df["net_economic_outcome"] > 0).mean()
        prob_net_below_50M = (df["net_economic_outcome"] < -50e6).mean()
        prob_act_frac_below_0_3 = (df["recovered_actinide_fraction"] < 0.3).mean()
        table.append([s, net_mean, net_p95, net_p05, net_cvar, act_frac_mean, waste_mean,
                      prob_positive, prob_net_below_50M, prob_act_frac_below_0_3])
    columns = ["Scenario", "Net Econ Mean", "Net Econ P95", "Net Econ P05", "Net Econ CVaR5%",
               "Actinide Fraction Mean", "Waste Mass Mean", "Prob Net Econ>0",
               "Prob Net Econ < -50M USD", "Prob Act Fraction < 0.3"]
    return pd.DataFrame(table, columns=columns)

# =========================
# 9. Plotting Functions (UPDATED – Journal Quality)
# =========================

def plot_cdf(results, metric, threshold=None, invert=False, policy_label=None):
    plt.figure(figsize=(8, 6))

    for scenario, df in results.items():
        data = df[metric]
        if invert:
            data = -data

        sorted_vals = np.sort(data)
        cdf = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
        plt.plot(sorted_vals, cdf, linewidth=1.6, label=scenario)

    if threshold is not None:
        t_value = -threshold if invert else threshold
        plt.axvline(t_value, linestyle="--", color="black", linewidth=1.2)

        if policy_label:
            plt.text(
                t_value,
                0.05,
                f"{policy_label}\n({threshold})",
                rotation=90,
                verticalalignment="bottom",
                fontsize=8
            )

    plt.xlabel(metric.replace("_", " "))
    plt.ylabel("Cumulative Probability")
    plt.grid(alpha=0.3)

    plt.legend(
        loc="upper left",
        bbox_to_anchor=(0.02, 0.98),
        frameon=True,
        framealpha=0.9,
        borderpad=0.3,
        labelspacing=0.3,
        handlelength=1.2
    )

    plt.tight_layout(pad=1.2)
    plt.savefig(
        os.path.join(FIGURE_DIR, f"cdf_{metric}.pdf"),
        bbox_inches="tight"
    )
    plt.show()


def plot_pareto(results, x_metric, y_metric, invert_y=False,
                xlabel=None, ylabel=None, filename=None):

    plt.figure(figsize=(8, 6))

    for scenario, df in results.items():
        ydata = -df[y_metric] if invert_y else df[y_metric]
        plt.scatter(df[x_metric], ydata,
                    s=6, alpha=0.35, label=scenario)

    plt.xlabel(xlabel or x_metric.replace("_", " "))
    plt.ylabel(ylabel or y_metric.replace("_", " "))
    plt.grid(alpha=0.3)

    plt.legend(
        loc="upper left",
        bbox_to_anchor=(0.02, 0.98),
        frameon=True,
        framealpha=0.9,
        borderpad=0.3
    )

    plt.tight_layout(pad=1.2)

    if filename:
        plt.savefig(
            os.path.join(FIGURE_DIR, filename),
            bbox_inches="tight"
        )

    plt.show()


def plot_violin(results, metric):
    plt.figure(figsize=(8, 6))

    data = pd.DataFrame({s: df[metric] for s, df in results.items()})
    sns.violinplot(data=data)

    plt.ylabel(metric.replace("_", " "))
    plt.grid(alpha=0.3)

    plt.tight_layout(pad=1.2)
    plt.savefig(
        os.path.join(FIGURE_DIR, f"violin_{metric}.pdf"),
        bbox_inches="tight"
    )
    plt.show()


def plot_tornado(samples, scenario_results, scenario_name,
                 outcome="net_economic_outcome"):

    df = scenario_results.copy()
    recycle_fraction = get_recycle_fraction(scenario_name, len(samples))

    inputs = samples.copy()
    inputs["recycle_fraction"] = recycle_fraction

    non_constant_inputs = inputs.loc[:, inputs.std() > 0]
    corr = non_constant_inputs.corrwith(df[outcome]).abs().sort_values()

    plt.figure(figsize=(8, 6))
    corr.plot.barh(color="skyblue", edgecolor="black")

    plt.xlabel(f"Absolute Correlation with {outcome.replace('_',' ')}")
    plt.grid(alpha=0.3)

    plt.tight_layout(pad=1.2)
    plt.savefig(
        os.path.join(FIGURE_DIR,
                     f"tornado_{scenario_name}_{outcome}.pdf"),
        bbox_inches="tight"
    )
    plt.show()


def plot_stacked_econ(results):
    components = ["revenue", "reprocessing_cost", "disposal_cost"]

    mean_values = pd.DataFrame({
        scenario: results[scenario][components].mean()
        for scenario in SCENARIOS
    }).T

    mean_values["reprocessing_cost"] *= -1
    mean_values["disposal_cost"] *= -1

    plt.figure(figsize=(8, 6))

    mean_values.plot(
        kind="bar",
        stacked=True,
        color=["green", "red", "orange"],
        edgecolor="black"
    )

    plt.ylabel("USD")
    plt.grid(alpha=0.3)

    plt.legend(
        loc="upper left",
        bbox_to_anchor=(0.02, 0.98),
        frameon=True
    )

    plt.tight_layout(pad=1.2)
    plt.savefig(
        os.path.join(FIGURE_DIR, "stacked_econ_breakdown.pdf"),
        bbox_inches="tight"
    )
    plt.show()

# =========================
# 10. Sobol Sensitivity Analysis
# =========================

def run_sobol_analysis(scenario_name, N=1024):
    problem = {
        "num_vars": 11,
        "names": ["spent_fuel_mass", "burnup", "U_frac", "Pu_frac", "MA_frac",
                  "U_recovery", "Pu_recovery", "MA_recovery",
                  "reprocessing_cost", "disposal_cost", "recycle_fraction"],
        "bounds": [
            [20, 35], [30, 60], [0.93, 0.96], [0.008, 0.015], [0.001, 0.005],
            [0.85, 0.95], [0.70, 0.95], [0.40, 0.70], [1500, 6000],
            [400, 1000], [0.0, 0.9]
        ]
    }

    from SALib.sample import sobol as sobol_sampler

    param_values = sobol_sampler.sample(problem, N, calc_second_order=True)

    # Model function
    def model_outputs(X):
        outputs = []
        for row in X:
            sample_df = pd.DataFrame([{
                "spent_fuel_mass": row[0],
                "burnup": row[1],
                "U_frac": row[2],
                "Pu_frac": row[3],
                "MA_frac": row[4],
                "U_recovery": row[5],
                "Pu_recovery": row[6],
                "MA_recovery": row[7],
                "reprocessing_cost": row[8],
                "disposal_cost": row[9],
                "U_value": 100,
                "Pu_value": 5000,
                "recycle_fraction": row[10]
            }])
            outputs.append(evaluate_scenario(sample_df, scenario_name)["net_economic_outcome"].values[0])
        return np.array(outputs)

    Y = model_outputs(param_values)
    Si = sobol.analyze(problem, Y, print_to_console=True)

    plt.figure(figsize=(10, 6))

    x = np.arange(len(problem['names']))
    width = 0.35

    plt.bar(x - width / 2, Si['S1'], width, label='First-order (S1)')
    plt.bar(x + width / 2, Si['ST'], width, label='Total-order (ST)')

    plt.xticks(x, problem['names'], rotation=45, ha='right')
    plt.ylabel("Sobol Sensitivity Index")

    plt.legend(
        loc="upper left",
        bbox_to_anchor=(0.02, 0.98),
        frameon=True
    )

    plt.tight_layout(pad=1.2)
    plt.savefig(
        os.path.join(FIGURE_DIR, f"sobol_{scenario_name}.pdf"),
        bbox_inches="tight"
    )
    plt.show()

# =========================
# 11. Print All Results
# =========================

def print_all_results(results, summary_extremes=None, samples=None, max_rows=10):
    for scenario, df in results.items():
        print("\n" + "="*70)
        print(f"SCENARIO: {scenario.upper().replace('_',' ')}")
        print("="*70)
        if samples is not None:
            combined = pd.concat([df.reset_index(drop=True), samples.reset_index(drop=True)], axis=1)
            print("\nSampled Monte Carlo Data (first {} rows):".format(max_rows))
            print(combined.head(max_rows).round(3))
        print("\nSummary Statistics:")
        print(df.describe().round(3))
        cvar = compute_cvar(df["net_economic_outcome"])
        print(f"\nConditional Value at Risk (CVaR 5%): {cvar:,.0f} USD")
        if summary_extremes is not None:
            extreme_row = summary_extremes.loc[summary_extremes["Scenario"] == scenario]
            print("\nExtreme Outcomes & Probabilities:")
            print(extreme_row.round(3).to_string(index=False))

# =========================
# 12. MAIN
# =========================

if __name__ == "__main__":
    # Monte Carlo
    inputs, results = run_monte_carlo()

    # Summary tables
    summary_table_extremes = combined_summary_table_with_extremes(results)
    print("\n" + "="*70)
    print("COMBINED SUMMARY TABLE WITH EXTREME OUTCOMES")
    print("="*70)
    print(summary_table_extremes.round(3))

    # ---- Print all results ----
    print_all_results(results, summary_extremes=summary_table_extremes, samples=inputs, max_rows=10)

    # ---- CDF Plots ----
    plot_cdf(results, "net_economic_outcome", threshold=0, policy_label="Break-even")
    plot_cdf(results, "recovered_actinide_fraction", threshold=0.30, policy_label="Min Sustainable Recovery")
    plot_cdf(results, "waste_mass", threshold=15000, invert=True, policy_label="Max Acceptable Waste")

    # ---- Expected Regret ----
    print("\nEXPECTED REGRET (Net Economic Outcome, USD):")
    print(expected_regret(results, "net_economic_outcome"))

    # ---- Pareto Fronts ----
    plot_pareto(results, "waste_mass", "net_economic_outcome", invert_y=True, xlabel="Waste Mass (kg)", ylabel="− Net Economic Outcome (USD)", filename="pareto_waste_vs_econ.pdf")
    plot_pareto(results, "recovered_actinide_fraction", "net_economic_outcome", xlabel="Recovered Actinide Fraction", ylabel="Net Economic Outcome (USD)", filename="pareto_recovery_vs_econ.pdf")
    plot_pareto(results, "recovered_actinide_fraction", "waste_mass", xlabel="Recovered Actinide Fraction", ylabel="Waste Mass (kg)", filename="pareto_waste_vs_recovery.pdf")

    # ---- Tornado Plots ----
    for scenario in SCENARIOS:
        plot_tornado(inputs, results[scenario], scenario_name=scenario, outcome="net_economic_outcome")

    # ---- Stacked Economic Breakdown ----
    plot_stacked_econ(results)

    # ---- Sobol Sensitivity Analysis ----
    for scenario in SCENARIOS:
        run_sobol_analysis(scenario)

    # ---- Violin Plots ----
    plot_violin(results, "net_economic_outcome")
    plot_violin(results, "recovered_actinide_fraction")
    plot_violin(results, "waste_mass")