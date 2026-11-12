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

# Create the output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)


# --- 1. MATHEMATICAL MODELS ---
# In all models, 'A' represents the asymptote as x -> infinity in the transformed space.

# Increasing Models
def exp_inc(x, A, B, k): return A - B * np.exp(-k * x)


def rat_inc(x, A, B, C): return (A * x + B) / (x + C)


def log_inc(x, A, B, k, x0): return A / (1.0 + B * np.exp(-k * (x - x0)))


def gompertz_inc(x, A, B, k): return A * np.exp(-B * np.exp(-k * x))


# Decreasing Models
def exp_dec(x, A, B, k): return A + B * np.exp(-k * x)


def rat_dec(x, A, B, C): return A + B / (x + C)


def log_dec(x, A, B, k, x0): return A + B / (1.0 + np.exp(k * (x - x0)))


def pow_dec(x, A, B, k): return A + B * np.power(np.maximum(x, 1e-5), -k)


def gompertz_dec(x, A, B, C, k): return A + B * (1.0 - np.exp(-C * np.exp(-k * x)))


def weibull_dec(x, A, B, k, c): return A + B * np.exp(-k * np.power(np.maximum(x, 1e-5), c))


# Complex/Non-Monotonic Asymptotic Models
def second_order_rat(x, A, B, C, D, E): return (A * x ** 2 + B * x + C) / (x ** 2 + D * x + E)


def damped_sine(x, A, B, lam, omega, phi): return A + B * np.exp(-lam * x) * np.cos(omega * x + phi)


# --- 2. MODEL DICTIONARY & GUESS GENERATORS ---
MODELS = {
    "Exp_Growth": (exp_inc, lambda y, x: [np.max(y), np.max(y) - np.min(y), 0.5]),
    "Rational_Growth": (rat_inc, lambda y, x: [np.max(y), np.min(y), 1.0]),
    "Logistic_Growth": (log_inc, lambda y, x: [np.max(y), 1.0, 1.0, np.median(x)]),
    "Gompertz_Growth": (gompertz_inc, lambda y, x: [np.max(y), 5.0, 0.5]),

    "Exp_Decay": (exp_dec, lambda y, x: [np.min(y), np.max(y) - np.min(y), 0.5]),
    "Rational_Decay": (rat_dec, lambda y, x: [np.min(y), np.max(y) - np.min(y), 1.0]),
    "Logistic_Decay": (log_dec, lambda y, x: [np.min(y), np.max(y) - np.min(y), 2.0, np.median(x)]),
    "Power_Law_Decay": (pow_dec, lambda y, x: [np.min(y), np.max(y) - np.min(y), 1.0]),
    "Gompertz_Decay": (gompertz_dec, lambda y, x: [np.min(y), np.max(y) - np.min(y), 1.0, 0.5]),
    "Weibull_Decay": (weibull_dec, lambda y, x: [np.min(y), np.max(y) - np.min(y), 0.5, 1.0]),

    "Second_Order_Rational": (second_order_rat, lambda y, x: [y[-1], 0.0, y[0], 1.0, 1.0]),
    "Damped_Sine": (damped_sine, lambda y, x: [np.mean(y[-5:]), np.max(y) - np.min(y), 0.5, 2.0, 0.0])
}


def get_dynamic_bounds(param_name, num_params, y_transform):
    """Assigns strict structural boundaries, adjusting if the Y-axis is log-transformed."""
    is_prob = "p0" in param_name or "p1" in param_name

    if y_transform == "Linear":
        A_min = 0.0
        A_max = 1.0 if is_prob else np.inf
    else:  # Log10 transform
        A_min = -np.inf
        A_max = 0.0 if is_prob else np.inf

    lower = [A_min] + [-np.inf] * (num_params - 1)
    upper = [A_max] + [np.inf] * (num_params - 1)

    # Force strict > 0 for rate/shape constants to ensure convergence
    if num_params >= 3: lower[2] = 1e-6
    if num_params >= 4: lower[3] = 1e-6

    return (lower, upper)


# --- 3. CORE FITTING ENGINE ---
def run_exhaustive_fits(input_csv, engine_prefix, x_col):
    print(f"\nEvaluating combinations for {engine_prefix}...")
    try:
        df = pd.read_csv(input_csv).dropna()
    except FileNotFoundError:
        print(f"File {input_csv} not found. Skipping.")
        return

    x_raw = df[x_col].values
    parameters = [c for c in df.columns if c in ["p0_(Zero_Inflation)", "p1_(One_Inflation)", "Alpha", "Beta"]]

    for param in parameters:
        print(f"  -> Crunching permutations for {param}...")
        y_raw = df[param].values
        results = []

        # Define Transformation Matrix
        x_transforms = [("Linear", x_raw), ("Log10", np.log10(x_raw))]

        y_transforms = [("Linear", y_raw)]
        # Only apply Log Y if all values are strictly positive to avoid -inf errors
        if np.all(y_raw > 0):
            y_transforms.append(("Log10", np.log10(y_raw)))

        for x_name, x_data in x_transforms:
            for y_name, y_data in y_transforms:

                for model_name, (func, guess_gen) in MODELS.items():
                    num_params = func.__code__.co_argcount - 1
                    p0 = guess_gen(y_data, x_data)
                    bounds = get_dynamic_bounds(param, num_params, y_name)

                    try:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            popt, _ = curve_fit(func, x_data, y_data, p0=p0, bounds=bounds, maxfev=50000)

                        # Generate predictions in the transformed space
                        y_pred_transformed = func(x_data, *popt)

                        # Inverse the Y-transform to calculate True R-Squared in linear space
                        if y_name == "Log10":
                            y_pred_linear = 10 ** y_pred_transformed
                            asymptote_linear = 10 ** popt[0]
                        else:
                            y_pred_linear = y_pred_transformed
                            asymptote_linear = popt[0]

                        # Calculate True R2
                        r2 = r2_score(y_raw, y_pred_linear)

                        # Filter out garbage fits
                        if r2 < -1.0 or np.isnan(r2):
                            continue

                        # Enforce probability constraints explicitly on the generated line
                        if ("p0" in param or "p1" in param) and (
                                np.any(y_pred_linear > 1.0) or np.any(y_pred_linear < 0.0)):
                            continue

                        results.append({
                            "X_Transform": x_name,
                            "Y_Transform": y_name,
                            "Model": model_name,
                            "True_R_Squared": r2,
                            "Linear_Asymptote_(A)": asymptote_linear,
                            "Transformed_Params": str(np.round(popt, 6).tolist())
                        })

                    except Exception:
                        pass  # Ignore fits that mathematically fail to converge

        # Save the exhaustive grid for this specific parameter
        if results:
            df_res = pd.DataFrame(results)
            # Sort by highest R-Squared
            df_res = df_res.sort_values(by="True_R_Squared", ascending=False)

            out_file = os.path.join(OUTPUT_DIR, f"{engine_prefix}_{param}_exhaustive_fits.csv")
            df_res.to_csv(out_file, index=False)
            print(f"     Saved {len(results)} successful fits to {out_file}")


if __name__ == "__main__":
    run_exhaustive_fits(STOCKFISH_INPUT, "Stockfish", "Depth")
    run_exhaustive_fits(LC0_INPUT, "Lc0", "Target_Nodes")
    print("\nAll grids complete. Check the 'trend_fitting_results' folder.")