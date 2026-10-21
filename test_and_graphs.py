import os
import ast
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta
from sklearn.metrics import r2_score

# --- CONFIGURATION ---
STOCKFISH_RAW_CSV = "continuous_pilot_results_d20.csv"
LC0_RAW_CSV = "lc0_aggregated_nodes.csv"
OUTPUT_DIR = "trend_fitting_results"


# --- MATHEMATICAL MODELS ---
def exp_inc(x, A, B, k): return A - B * np.exp(-k * x)


def rat_inc(x, A, B, C): return (A * x + B) / (x + C)


def log_inc(x, A, B, k, x0): return A / (1.0 + B * np.exp(-k * (x - x0)))


def gompertz_inc(x, A, B, k): return A * np.exp(-B * np.exp(-k * x))


def exp_dec(x, A, B, k): return A + B * np.exp(-k * x)


def rat_dec(x, A, B, C): return A + B / (x + C)


def log_dec(x, A, B, k, x0): return A + B / (1.0 + np.exp(k * (x - x0)))


def pow_dec(x, A, B, k): return A + B * np.power(np.maximum(x, 1e-5), -k)


def gompertz_dec(x, A, B, C, k): return A + B * (1.0 - np.exp(-C * np.exp(-k * x)))


def weibull_dec(x, A, B, k, c): return A + B * np.exp(-k * np.power(np.maximum(x, 1e-5), c))


def second_order_rat(x, A, B, C, D, E): return (A * x ** 2 + B * x + C) / (x ** 2 + D * x + E)


def damped_sine(x, A, B, lam, omega, phi): return A + B * np.exp(-lam * x) * np.cos(omega * x + phi)


MODELS = {
    "Exp_Growth": exp_inc, "Rational_Growth": rat_inc, "Logistic_Growth": log_inc, "Gompertz_Growth": gompertz_inc,
    "Exp_Decay": exp_dec, "Rational_Decay": rat_dec, "Logistic_Decay": log_dec, "Power_Law_Decay": pow_dec,
    "Gompertz_Decay": gompertz_dec, "Weibull_Decay": weibull_dec, "Second_Order_Rational": second_order_rat,
    "Damped_Sine": damped_sine
}


def load_fit_function(filename):
    """Loads the top mathematical model and its parameters from the exhaustive fit CSV."""
    if not os.path.exists(filename):
        print(f"Warning: {filename} not found.")
        return None
    df = pd.read_csv(filename)
    top_fit = df.iloc[0]
    return {
        "model": MODELS[top_fit["Model"]],
        "x_trans": top_fit["X_Transform"],
        "y_trans": top_fit["Y_Transform"],
        "params": ast.literal_eval(top_fit["Transformed_Params"]),
        "asymptote": top_fit["Linear_Asymptote_(A)"]
    }


def predict_parameter(x_raw, fit_data):
    """Passes the x-axis through the mathematical function to get the predicted parameter."""
    x_input = np.log10(np.maximum(x_raw, 1e-5)) if fit_data["x_trans"] == "Log10" else x_raw
    y_pred_transformed = fit_data["model"](x_input, *fit_data["params"])
    return 10 ** y_pred_transformed if fit_data["y_trans"] == "Log10" else y_pred_transformed


# ==========================================
# 1. 2D SURFACE VALIDATION
# ==========================================

def test_2d_surface(raw_csv, x_col, fit_files):
    """Constructs the theoretical 2D CDF and compares it against the raw empirical data."""
    if not os.path.exists(raw_csv):
        return None, None, None, None, None

    df_raw = pd.read_csv(raw_csv)
    df_mean = df_raw[df_raw["Metric"] == "Mean"].copy()

    x_values = df_mean[x_col].values
    error_cols = [c for c in df_mean.columns if c.startswith('Rel_Error_<=')]
    e_thresholds = np.array([float(c.split('<=')[1]) for c in error_cols])

    empirical_cdf_matrix = df_mean[error_cols].values

    p0_fit = load_fit_function(fit_files["p0"])
    p1_fit = load_fit_function(fit_files["p1"])
    a_fit = load_fit_function(fit_files["alpha"])
    b_fit = load_fit_function(fit_files["beta"])

    if not all([p0_fit, p1_fit, a_fit, b_fit]):
        return None, None, None, None, None

    theoretical_cdf_matrix = np.zeros_like(empirical_cdf_matrix)

    for i, x in enumerate(x_values):
        p0 = np.clip(predict_parameter(x, p0_fit), 0, 1)
        p1 = np.clip(predict_parameter(x, p1_fit), 0, 1 - p0)
        a = predict_parameter(x, a_fit)
        b = predict_parameter(x, b_fit)

        cdf = np.zeros_like(e_thresholds)
        cdf[e_thresholds >= 0] += p0
        cdf += (1 - p0 - p1) * beta.cdf(np.clip(e_thresholds, 0, 1), a, b)
        cdf[e_thresholds >= 1] = 1.0
        theoretical_cdf_matrix[i, :] = cdf

    overall_r2 = r2_score(empirical_cdf_matrix.flatten(), theoretical_cdf_matrix.flatten())
    overall_mae = np.mean(np.abs(empirical_cdf_matrix - theoretical_cdf_matrix))

    limits = {
        "p0": np.clip(p0_fit["asymptote"], 0, 1),
        "p1": np.clip(p1_fit["asymptote"], 0, 1),
        "alpha": a_fit["asymptote"],
        "beta": b_fit["asymptote"]
    }

    return empirical_cdf_matrix, theoretical_cdf_matrix, overall_r2, overall_mae, limits


# ==========================================
# 2. INFINITE DEPTH & DIFFERENCE SIMULATION
# ==========================================

def simulate_zoib(n_samples, limits):
    """Generates a random sample array for the ZOIB distribution."""
    rand_vals = np.random.rand(n_samples)
    samples = np.zeros(n_samples)

    p0, p1, a, b = limits["p0"], limits["p1"], limits["alpha"], limits["beta"]

    p0_mask = rand_vals < p0
    p1_mask = (rand_vals >= p0) & (rand_vals < p0 + p1)
    cont_mask = rand_vals >= p0 + p1

    samples[p0_mask] = 0.0
    samples[p1_mask] = 1.0
    samples[cont_mask] = beta.rvs(a, b, size=np.sum(cont_mask))

    return samples


def plot_infinite_depth_analysis(sf_limits, lc_limits):
    """Simulates and plots the infinite depth PDFs, CDFs, and D = S - L."""
    print("\nSimulating 10,000,000 matches at Infinite Depth...")
    n_samples = 10000000

    sf_samples = simulate_zoib(n_samples, sf_limits)
    lc_samples = simulate_zoib(n_samples, lc_limits)
    D = sf_samples - lc_samples

    prob_s_greater = np.sum(D > 0) / n_samples
    prob_l_greater = np.sum(D < 0) / n_samples
    prob_equal = np.sum(D == 0) / n_samples

    print("\n--- INFINITE DEPTH ASYMPTOTES ---")
    print(
        f"Stockfish: p0={sf_limits['p0']:.4f}, p1={sf_limits['p1']:.4f}, a={sf_limits['alpha']:.4f}, b={sf_limits['beta']:.4f}")
    print(
        f"Lc0:       p0={lc_limits['p0']:.4f}, p1={lc_limits['p1']:.4f}, a={lc_limits['alpha']:.4f}, b={lc_limits['beta']:.4f}")

    print("\n--- DIFFERENCE DISTRIBUTION (D = S - L) ---")
    print(f"P(S > L) [Stockfish assigned larger error]:  {prob_s_greater * 100:.2f}%")
    print(f"P(L > S) [Lc0 assigned larger error]:        {prob_l_greater * 100:.2f}%")
    print(f"P(S == L) [Both evaluated identically]:      {prob_equal * 100:.2f}%")

    # ----- Plotting Infinite Depth Limits -----
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    x = np.linspace(1e-4, 1 - 1e-4, 500)

    # 1. Engine PDFs
    ax = axes[0, 0]
    sf_pdf = beta.pdf(x, sf_limits["alpha"], sf_limits["beta"]) * (1 - sf_limits["p0"] - sf_limits["p1"])
    lc_pdf = beta.pdf(x, lc_limits["alpha"], lc_limits["beta"]) * (1 - lc_limits["p0"] - lc_limits["p1"])
    ax.plot(x, sf_pdf, color='#1f77b4', lw=2.5, label='Stockfish Continuous Beta')
    ax.plot(x, lc_pdf, color='#ff7f0e', lw=2.5, label='Lc0 Continuous Beta')
    ax.set_title('Infinite Depth ZOIB PDFs', fontweight='bold')
    ax.set_xlabel('Relative Error (e)')
    ax.set_ylabel('Density')
    ax.set_ylim(0, 1.0)
    ax.legend()
    ax.grid(alpha=0.3)

    # 2. Engine CDFs
    ax = axes[0, 1]
    sf_cdf_x, sf_cdf_y = np.sort(sf_samples), np.arange(1, n_samples + 1) / n_samples
    lc_cdf_x, lc_cdf_y = np.sort(lc_samples), np.arange(1, n_samples + 1) / n_samples
    ax.plot(sf_cdf_x, sf_cdf_y, color='#1f77b4', lw=2.5, label='Stockfish')
    ax.plot(lc_cdf_x, lc_cdf_y, color='#ff7f0e', lw=2.5, label='Lc0')
    ax.set_title('Infinite Depth ZOIB CDFs', fontweight='bold')
    ax.set_xlabel('Relative Error (e)')
    ax.set_ylabel('Cumulative Probability')
    ax.legend()
    ax.grid(alpha=0.3)

    # 3. Difference PDF (D = S - L)
    ax = axes[1, 0]
    ax.hist(D, bins=500, density=True, color='purple', alpha=0.7)
    ax.axvline(0, color='black', linestyle='--', lw=2, label='D = 0 (Equal Error)')
    ax.set_title('Difference PDF: D = Stockfish - Lc0', fontweight='bold')
    ax.set_xlabel('Difference (D)')
    ax.set_ylabel('Density')
    ax.set_xlim(-1.05, 1.05)
    ax.legend()
    ax.grid(alpha=0.3)

    # 4. Difference CDF (D = S - L)
    ax = axes[1, 1]
    D_x, D_y = np.sort(D), np.arange(1, n_samples + 1) / n_samples
    ax.plot(D_x, D_y, color='purple', lw=3)
    ax.axvline(0, color='black', linestyle='--', lw=2)
    ax.axhline(prob_l_greater, color='gray', linestyle=':', label=f'P(D < 0) = {prob_l_greater * 100:.1f}%')
    ax.set_title('Difference CDF: D = Stockfish - Lc0', fontweight='bold')
    ax.set_xlabel('Difference (D)')
    ax.set_ylabel('Cumulative Probability')
    ax.set_xlim(-1.05, 1.05)
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("infinite_depth_difference_analysis.png", dpi=300)
    print("Saved Infinite Depth & Difference analysis to 'infinite_depth_difference_analysis.png'.")


# ==========================================
# 3. MASTER ORCHESTRATOR
# ==========================================

def run_all_diagnostics():
    sf_fits = {
        "p0": os.path.join(OUTPUT_DIR, "Stockfish_p0_(Zero_Inflation)_exhaustive_fits.csv"),
        "p1": os.path.join(OUTPUT_DIR, "Stockfish_p1_(One_Inflation)_exhaustive_fits.csv"),
        "alpha": os.path.join(OUTPUT_DIR, "Stockfish_Alpha_exhaustive_fits.csv"),
        "beta": os.path.join(OUTPUT_DIR, "Stockfish_Beta_exhaustive_fits.csv")
    }

    lc_fits = {
        "p0": os.path.join(OUTPUT_DIR, "Lc0_p0_(Zero_Inflation)_exhaustive_fits.csv"),
        "p1": os.path.join(OUTPUT_DIR, "Lc0_p1_(One_Inflation)_exhaustive_fits.csv"),
        "alpha": os.path.join(OUTPUT_DIR, "Lc0_Alpha_exhaustive_fits.csv"),
        "beta": os.path.join(OUTPUT_DIR, "Lc0_Beta_exhaustive_fits.csv")
    }

    print("Evaluating Stockfish 2D Surface...")
    sf_emp, sf_theo, sf_r2, sf_mae, sf_limits = test_2d_surface(STOCKFISH_RAW_CSV, "Depth", sf_fits)

    print("Evaluating Lc0 2D Surface...")
    lc_emp, lc_theo, lc_r2, lc_mae, lc_limits = test_2d_surface(LC0_RAW_CSV, "Target_Nodes", lc_fits)

    # Plot 2D Surface Heatmaps
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    if sf_r2 is not None:
        print(f"-> Stockfish Global R2: {sf_r2:.4f} | MAE: {sf_mae:.4f}")
        ax = axes[0, 0]
        ax.scatter(sf_emp.flatten(), sf_theo.flatten(), alpha=0.5, color='#1f77b4', s=10)
        ax.plot([0, 1], [0, 1], 'k--', lw=2)
        ax.set_title(f"Stockfish: Predicted vs Actual\nGlobal R² = {sf_r2:.4f}", fontweight='bold')
        ax.set_xlabel("Empirical CDF (Raw Data)")
        ax.set_ylabel("Theoretical CDF (Mathematical Model)")
        ax.grid(alpha=0.3)

        ax = axes[1, 0]
        cax = ax.imshow(np.abs(sf_emp - sf_theo), aspect='auto', cmap='Reds', origin='lower')
        ax.set_title("Stockfish Absolute Error Heatmap", fontweight='bold')
        ax.set_xlabel("Relative Error Thresholds (e)")
        ax.set_ylabel("Depth Iterator")
        fig.colorbar(cax, ax=ax, label="Absolute Probability Error")

    if lc_r2 is not None:
        print(f"-> Lc0 Global R2: {lc_r2:.4f} | MAE: {lc_mae:.4f}")
        ax = axes[0, 1]
        ax.scatter(lc_emp.flatten(), lc_theo.flatten(), alpha=0.5, color='#ff7f0e', s=10)
        ax.plot([0, 1], [0, 1], 'k--', lw=2)
        ax.set_title(f"Lc0: Predicted vs Actual\nGlobal R² = {lc_r2:.4f}", fontweight='bold')
        ax.set_xlabel("Empirical CDF (Raw Data)")
        ax.grid(alpha=0.3)

        ax = axes[1, 1]
        cax = ax.imshow(np.abs(lc_emp - lc_theo), aspect='auto', cmap='Reds', origin='lower')
        ax.set_title("Lc0 Absolute Error Heatmap", fontweight='bold')
        ax.set_xlabel("Relative Error Thresholds (e)")
        ax.set_ylabel("Node Iterator")
        fig.colorbar(cax, ax=ax, label="Absolute Probability Error")

    plt.tight_layout()
    plt.savefig("2d_surface_validation.png", dpi=300)
    print("Saved 2D Surface Validation to '2d_surface_validation.png'.")

    # Run Infinite Depth Analysis
    if sf_limits and lc_limits:
        plot_infinite_depth_analysis(sf_limits, lc_limits)


if __name__ == "__main__":
    run_all_diagnostics()