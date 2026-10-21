import os
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
import warnings

# --- CONFIGURATION ---
STOCKFISH_INPUT = "stockfish_zib_params.csv"
LC0_INPUT = "lc0_zib_params.csv"
OUTPUT_DIR = "trend_fitting_results"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# --- BASE DECAY SHAPES (g(x)) ---
# All approach 0 as x -> infinity
def base_exp(x, k): return np.exp(-k * x)


def base_pow(x, k): return np.power(np.maximum(x, 1e-5), -k)


def base_rat(x, k): return 1.0 / (x + k)


def base_weibull(x, k, c): return np.exp(-k * np.power(np.maximum(x, 1e-5), c))


BASE_MODELS = {
    "Coupled_Exp_Decay": (base_exp, 1),  # 1 shape parameter (k)
    "Coupled_Power_Decay": (base_pow, 1),  # 1 shape parameter (k)
    "Coupled_Rational_Decay": (base_rat, 1),  # 1 shape parameter (k)
    "Coupled_Weibull_Decay": (base_weibull, 2)  # 2 shape parameters (k, c)
}


def make_joint_func(base_shape_func, y_transform):
    """Creates a unified function that handles the indicator logic and Y-transformations."""

    def joint_func(x_combined, C_a, C_b, *args):
        x = x_combined[0]
        indicator = x_combined[1]  # 0 for Alpha, 1 for Beta

        # Calculate base decay shape
        shape = base_shape_func(x, *args)

        # Apply independent scaling
        y_linear = (1 - indicator) * (C_a * shape) + indicator * (C_b * shape)

        if y_transform == "Log10":
            return np.log10(np.maximum(y_linear, 1e-15))
        return y_linear

    return joint_func


def run_coupled_exhaustive_fits(input_csv, engine_prefix, x_col):
    print(f"\n--- Running Coupled Grid Search for {engine_prefix} ---")
    try:
        df = pd.read_csv(input_csv).dropna()
    except FileNotFoundError:
        print(f"File {input_csv} not found.")
        return

    x_raw = df[x_col].values
    y_alpha_raw = df["Alpha"].values
    y_beta_raw = df["Beta"].values
    N = len(x_raw)

    # Stack Raw Data
    y_stacked_raw = np.concatenate((y_alpha_raw, y_beta_raw))
    indicator = np.concatenate((np.zeros(N), np.ones(N)))

    results = []

    # X and Y Axis Transformations
    x_transforms = [("Linear", x_raw), ("Log10", np.log10(x_raw))]
    y_transforms = [("Linear", y_stacked_raw)]
    if np.all(y_stacked_raw > 0):
        y_transforms.append(("Log10", np.log10(y_stacked_raw)))

    for x_name, x_data in x_transforms:
        x_stacked = np.concatenate((x_data, x_data))
        x_combined = np.vstack((x_stacked, indicator))

        for y_name, y_data in y_transforms:
            for model_name, (base_func, num_shape_params) in BASE_MODELS.items():

                joint_func = make_joint_func(base_func, y_name)

                # Guesses: [C_a, C_b, k, (c)]
                p0 = [np.max(y_alpha_raw), np.max(y_beta_raw)] + [0.5] * num_shape_params

                # Bounds: Everything must be strictly positive
                lower_bounds = [1e-6] * (2 + num_shape_params)
                upper_bounds = [np.inf] * (2 + num_shape_params)

                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        popt, _ = curve_fit(joint_func, x_combined, y_data, p0=p0, bounds=(lower_bounds, upper_bounds),
                                            maxfev=50000)

                    # Generate predictions in Transformed Space
                    y_pred_transformed = joint_func(x_combined, *popt)

                    # Inverse Y Transform to get True R-Squared in Linear Space
                    y_pred_linear = 10 ** y_pred_transformed if y_name == "Log10" else y_pred_transformed

                    # Overall and Individual R-Squared
                    r2_overall = r2_score(y_stacked_raw, y_pred_linear)
                    r2_alpha = r2_score(y_alpha_raw, y_pred_linear[:N])
                    r2_beta = r2_score(y_beta_raw, y_pred_linear[N:])

                    # Meaningful Bernoulli Probability limit p = Ca / (Ca + Cb)
                    C_a, C_b = popt[0], popt[1]
                    bernoulli_p = C_a / (C_a + C_b)

                    if r2_overall > -1.0 and not np.isnan(r2_overall):
                        results.append({
                            "Model": model_name,
                            "X_Transform": x_name,
                            "Y_Transform": y_name,
                            "Combined_True_R2": r2_overall,
                            "Alpha_R2": r2_alpha,
                            "Beta_R2": r2_beta,
                            "Bernoulli_Prob_(p)": bernoulli_p,
                            "Alpha_Scale_(Ca)": C_a,
                            "Beta_Scale_(Cb)": C_b,
                            "Optimal_Params": str(np.round(popt, 6).tolist())
                        })
                except Exception:
                    continue

    if results:
        df_res = pd.DataFrame(results).sort_values(by="Combined_True_R2", ascending=False)
        out_file = os.path.join(OUTPUT_DIR, f"{engine_prefix}_Coupled_Shape_exhaustive_fits.csv")
        df_res.to_csv(out_file, index=False)
        print(f"  -> Saved {len(results)} coupled fits to {out_file}")

        # Display the absolute best unified fit
        best = df_res.iloc[0]
        print(f"  -> Best Fit: {best['Model']} ({best['X_Transform']} X, {best['Y_Transform']} Y)")
        print(
            f"  -> Combined R2: {best['Combined_True_R2']:.4f} (Alpha: {best['Alpha_R2']:.4f}, Beta: {best['Beta_R2']:.4f})")
        print(f"  -> Bernoulli Probability Limit (p): {best['Bernoulli_Prob_(p)']:.4f}")


if __name__ == "__main__":
    run_coupled_exhaustive_fits(STOCKFISH_INPUT, "Stockfish", "Depth")
    run_coupled_exhaustive_fits(LC0_INPUT, "Lc0", "Target_Nodes")