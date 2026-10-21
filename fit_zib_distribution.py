import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from scipy.special import betainc
from sklearn.metrics import r2_score
import warnings

# --- CONFIGURATION ---
STOCKFISH_INPUT = "continuous_pilot_results_d20.csv"
STOCKFISH_OUTPUT = "stockfish_zib_params.csv"

LC0_INPUT = "lc0_aggregated_nodes.csv"
LC0_OUTPUT = "lc0_zib_params.csv"


# The Zero-and-One-Inflated Beta CDF model
# p0 = probability mass at strictly 0 (perfect moves)
# p1 = probability mass at strictly 1 (maximum relative error)
# alpha, beta = standard shape parameters of the beta distribution
def zoib_cdf(x, p0, p1, alpha, beta):
    # Ensure x is strictly bounded [0, 1]
    x_safe = np.clip(x, 0.0, 1.0)

    # Calculate the continuous Beta CDF (betainc)
    beta_cdf = betainc(alpha, beta, x_safe)

    # Apply the piecewise ZOIB logic
    return p0 + (1.0 - p0 - p1) * beta_cdf


def extract_zoib_parameters(input_csv, output_csv, index_col_name):
    print(f"Processing {input_csv}...")
    try:
        df = pd.read_csv(input_csv)
    except FileNotFoundError:
        print(f"  -> File not found. Skipping {input_csv}.")
        return

    df_mean = df[df["Metric"] == "Mean"].copy()

    error_cols = [col for col in df.columns if col.startswith("Rel_Error")]
    e_values = np.array([float(col.split("<=")[1]) for col in error_cols])

    results = []

    for _, row in df_mean.iterrows():
        index_val = row[index_col_name]
        y_values = row[error_cols].values.astype(float)

        # Initial parameter guesses: p0=0.1, p1=0.0, alpha=1.0, beta=1.0
        p0_guess = [0.1, 0.0, 1.0, 1.0]

        # Bounds: p0 and p1 must be >= 0 and sum to <= 1.
        # curve_fit handles box bounds. We enforce p0, p1 in [0, 1].
        bounds = (
            [0.0, 0.0, 1e-3, 1e-3],
            [1.0, 1.0, 500.0, 500.0]
        )

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                popt, pcov = curve_fit(zoib_cdf, e_values, y_values, p0=p0_guess, bounds=bounds, maxfev=10000)

            p0_opt, p1_opt, alpha_opt, beta_opt = popt

            y_pred = zoib_cdf(e_values, *popt)
            r2 = r2_score(y_values, y_pred)

        except Exception as e:
            print(f"  -> Could not fit distribution for {index_col_name} {index_val}: {e}")
            p0_opt, p1_opt, alpha_opt, beta_opt, r2 = np.nan, np.nan, np.nan, np.nan, np.nan

        results.append({
            index_col_name: index_val,
            "p0_(Zero_Inflation)": p0_opt,
            "p1_(One_Inflation)": p1_opt,
            "Alpha": alpha_opt,
            "Beta": beta_opt,
            "Alpha/Beta_Ratio": alpha_opt / beta_opt if beta_opt > 0 else np.nan,
            "R_Squared": r2
        })

    df_results = pd.DataFrame(results)
    df_results.to_csv(output_csv, index=False)

    mean_r2 = df_results['R_Squared'].mean()
    print(f"  -> Completed! Average R-squared: {mean_r2:.4f}")
    print(f"  -> Saved parameter progression to {output_csv}\n")


if __name__ == "__main__":
    extract_zoib_parameters(STOCKFISH_INPUT, STOCKFISH_OUTPUT, "Depth")
    extract_zoib_parameters(LC0_INPUT, LC0_OUTPUT, "Target_Nodes")