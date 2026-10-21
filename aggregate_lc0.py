import pandas as pd
import numpy as np

# --- CONFIGURATION ---
INPUT_FILE = "lc0_erel_qvalues_nodes.csv"
OUTPUT_FILE = "lc0_aggregated_nodes.csv"


def aggregate_data():
    print(f"Loading raw data from {INPUT_FILE}...")
    df = pd.read_csv(INPUT_FILE)

    # Isolate the numeric error columns by dropping the FEN string
    error_columns = [col for col in df.columns if col.startswith("Rel_Error")]

    print("Calculating Means and Standard Deviations...")
    # Calculate Mean
    df_mean = df.groupby("Target_Nodes")[error_columns].mean().reset_index()
    df_mean.insert(1, "Metric", "Mean")

    # Calculate Standard Deviation (ddof=1 for sample std dev)
    df_std = df.groupby("Target_Nodes")[error_columns].std(ddof=1).reset_index()
    df_std.insert(1, "Metric", "Std_Dev")

    # Replace NaN with 0.0 in Std_Dev (happens if a node target only had 1 position)
    df_std = df_std.fillna(0.0)

    # Combine both DataFrames
    df_combined = pd.concat([df_mean, df_std])

    # Sort strictly by Target_Nodes (numeric ascending), then by Metric (Mean before Std_Dev)
    df_combined = df_combined.sort_values(by=["Target_Nodes", "Metric"])

    # Round all error calculations to 4 decimal places for cleaner viewing
    df_combined[error_columns] = df_combined[error_columns].round(4)

    # Save to the new CSV
    df_combined.to_csv(OUTPUT_FILE, index=False)
    print(f"Success! Aggregated data saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    aggregate_data()