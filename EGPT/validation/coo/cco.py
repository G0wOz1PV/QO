import numpy as np
import pandas as pd
import math
from numpy.random import default_rng
import os
import matplotlib.pyplot as plt
import seaborn as sns

# --- Constants ---
Rgas = 0.00198720425864083
T_default = 298.15
K_COULOMB = 332.063  # Unit conversion factor for alpha: kcal·Å/(mol·e^2)

# --- Core EGPC Function ---

def alpha_per_electron(R_A, eps_r, delta_q_A):
    """
    Calculates the EGPC barrier-lowering coefficient alpha based on geometry.
    """
    return K_COULOMB * delta_q_A / (eps_r * (R_A**2))

# --- External Benchmark: Cytochrome c Oxidase ---

def sample_alpha_distribution(R_mean=6.0, R_std=1.0, 
                              eps_range=(2.0, 4.0), dq_range=(0.3, 0.6), 
                              nsamp=200000, seed=2025):
    """
    Generates a distribution of predicted alpha values by sampling from
    physically motivated parameter distributions.
    - R is sampled from a normal distribution based on structural evidence.
    - eps_r and delta_q are sampled from uniform distributions representing
      their plausible physical ranges.
    """
    rng = default_rng(seed)
    
    # Sample R from a normal distribution
    R = rng.normal(loc=R_mean, scale=R_std, size=nsamp)
    R[R < 1.0] = 1.0  # Ensure R is physically meaningful (positive)
    
    # Sample eps_r and delta_q from uniform distributions
    eps = rng.uniform(eps_range[0], eps_range[1], nsamp)
    dq = rng.uniform(dq_range[0], dq_range[1], nsamp)
    
    # Calculate alpha for each sample
    alpha_predicted_distribution = alpha_per_electron(R, eps, dq)
    
    return alpha_predicted_distribution

def cco_benchmark(csv_in="cco_saura2022.csv", 
                  alpha_pred_out="cco_alpha_pred_dist.csv", 
                  summary_out="cco_summary.csv"):

    # --- Step 1: Infer alpha from external data ---
    try:
        df = pd.read_csv(csv_in)
    except FileNotFoundError:
        print(f"Error: Benchmark file '{csv_in}' not found.")
        print("Please create it with columns: condition_id,condition,n_e,DeltaG_barrier_kcal,T_K")
        return None

    if len(df) < 2:
        print("Error: CSV file must contain at least two rows for comparison (oxidized and reduced states).")
        return None

    # Sort by n_e to ensure consistent order (ox -> red)
    df_sorted = df.sort_values("n_e")
    
    dG_ox  = float(df_sorted.iloc[0]["DeltaG_barrier_kcal"])
    dG_red = float(df_sorted.iloc[1]["DeltaG_barrier_kcal"])
    
    alpha_inferred = dG_ox - dG_red

    # --- Step 2: Generate predicted alpha distribution ---
    nsamp = 200000
    seed = 2025
    alpha_pred_dist = sample_alpha_distribution(
        R_mean=6.0, R_std=1.0, 
        eps_range=(2.0, 4.0), 
        dq_range=(0.3, 0.6), 
        nsamp=nsamp, seed=seed
    )
    
    # Save the distribution for potential visualization
    os.makedirs("csv", exist_ok=True)
    pd.DataFrame({"alpha_pred_kcal_per_e": alpha_pred_dist}).to_csv(os.path.join("csv", alpha_pred_out), index=False)

    # --- Step 3: Compare and summarize ---
    pred_median = np.median(alpha_pred_dist)
    pred_lo = np.percentile(alpha_pred_dist, 2.5)
    pred_hi = np.percentile(alpha_pred_dist, 97.5)
    is_covered = (alpha_inferred >= pred_lo) and (alpha_inferred <= pred_hi)
    
    summary = {
        "alpha_inferred_kcal_per_e": alpha_inferred,
        "alpha_pred_median_kcal_per_e": pred_median,
        "alpha_pred_2.5_percentile_kcal_per_e": pred_lo,
        "alpha_pred_97.5_percentile_kcal_per_e": pred_hi,
        "is_inferred_alpha_in_95pct_interval": is_covered
    }
    
    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(os.path.join("csv", summary_out), index=False)
    
    return summary_df

# --- MODIFIED: Plotting Function with Serif Font ---

def plot_alpha_distribution(summary_df, alpha_pred_dist, output_filename="cco_alpha_distribution_serif.png"):
    """
    Generates and saves a violin plot of the predicted alpha distribution,
    highlighting the inferred value and the 95% prediction interval.
    The plot is rendered using a serif font for a formal, academic appearance.
    
    Args:
        summary_df (pd.DataFrame): DataFrame containing summary statistics.
        alpha_pred_dist (np.array): Array of predicted alpha values.
        output_filename (str): Name of the file to save the plot.
    """
    alpha_inferred = summary_df["alpha_inferred_kcal_per_e"].iloc[0]
    pred_lo = summary_df["alpha_pred_2.5_percentile_kcal_per_e"].iloc[0]
    pred_hi = summary_df["alpha_pred_97.5_percentile_kcal_per_e"].iloc[0]

    plt.figure(figsize=(10, 6))

    # --- FONT MODIFICATION ---
    sns.set_theme(style="whitegrid", font="serif") 
    
    plt.rcParams['mathtext.fontset'] = 'dejavuserif'
    # --- END FONT MODIFICATION ---

    # Create the violin plot
    sns.violinplot(x=alpha_pred_dist, color="skyblue", inner="quartile")

    # Add a shaded region for the 95% prediction interval
    plt.axvspan(pred_lo, pred_hi, color='gray', alpha=0.2, label=f'95% Prediction Interval [{pred_lo:.2f}, {pred_hi:.2f}]')

    # Add a vertical dashed line for the inferred value
    plt.axvline(x=alpha_inferred, color='red', linestyle='--', linewidth=2,
                label=f'Inferred α = {alpha_inferred:.2f} kcal/mol')

    # --- Plot Customization ---
    plt.xlabel(r'Barrier Lowering, $\alpha$ (kcal mol$^{-1}$)', fontsize=14)
    plt.ylabel('Probability Density', fontsize=14)
    plt.title('Predicted Distribution of α for Cytochrome c Oxidase', fontsize=16)
    plt.legend(fontsize=12)
    plt.tight_layout()
    
    # Save and show the plot
    plt.savefig(output_filename, dpi=300)
    print(f"\nPlot saved as '{output_filename}'")
    plt.show()

    # Optional: Revert to default settings to avoid affecting other plots in a larger script
    sns.set_theme()

# --- Main execution block ---

if __name__ == "__main__":
    # To make the script runnable out-of-the-box, create a dummy CSV if it doesn't exist.
    csv_filename = "cco_saura2022.csv"
    if not os.path.exists(csv_filename):
        print(f"'{csv_filename}' not found. Creating a dummy file for demonstration.")
        dummy_data = {
            "condition_id": [1, 2],
            "condition": ["oxidized", "reduced"],
            "n_e": [0, 1],
            "DeltaG_barrier_kcal": [10.0, 6.6], # 10.0 - 6.6 = 3.4
            "T_K": [298.15, 298.15]
        }
        pd.DataFrame(dummy_data).to_csv(csv_filename, index=False)

    print("--- Running External Benchmark for Cytochrome c Oxidase (CcO) ---")
    
    summary_result = cco_benchmark(csv_in=csv_filename)
    
    if summary_result is not None:
        print("\nBenchmark Summary:")
        print(summary_result.to_string(index=False))
        print(f"\nCSV outputs 'csv/cco_alpha_pred_dist.csv' and 'csv/cco_summary.csv' have been generated.")

        # Load the predicted distribution for plotting
        pred_dist_df = pd.read_csv("csv/cco_alpha_pred_dist.csv")
        alpha_distribution = pred_dist_df["alpha_pred_kcal_per_e"].values
        
        # Call the modified plotting function
        plot_alpha_distribution(summary_result, alpha_distribution)
