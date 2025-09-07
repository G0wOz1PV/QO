import os
import math
import json
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "Times", "Computer Modern Roman"],
    "mathtext.fontset": "dejavuserif",
    "font.size": 11,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "axes.linewidth": 1.0,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "figure.dpi": 300,
})

# Output directories
BASE_DIR = "si_outputs"
CSV_DIR = os.path.join(BASE_DIR, "csv")
FIG_DIR = os.path.join(BASE_DIR, "figures")
os.makedirs(CSV_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

# Random seeds for reproducibility
SEED_GLOBAL = 2025
rng = np.random.default_rng(SEED_GLOBAL)

# Physical constants and defaults (Table S1)
KBT_OVER_H = 6.21e12            # s^-1 at 298.15 K
RT_KCAL = 0.592                 # kcal/mol at 298.15 K
EV_TO_KCAL = 23.0605            # 1 eV in kcal/mol
MVpcm_TO_VA = 0.01              # 1 MV/cm -> 0.01 V/Å
ALPHA_CONST = 332.063           # kcal·mol^-1·Å per [Å^2] in α = 332.063*δq/(εr*R^2)

# Default example parameters
DG0_DEFAULT = 15.0              # kcal/mol
R_DEFAULT = 4.0                 # Å
EPSR_DEFAULT = 4.0
DQ_DEFAULT = 0.5                # Å

# Grids
R_MIN, R_MAX, R_N = 3.0, 10.0, 100
EPSR_MIN, EPSR_MAX, EPSR_N = 2.0, 20.0, 100
DQ_LIST = [0.3, 0.5, 0.8]       # Å

# Benchmark priors (Table S2)
PRIOR_R_MEAN = 6.0
PRIOR_R_SIGMA = 1.0
PRIOR_EPSR_MIN, PRIOR_EPSR_MAX = 2.0, 4.0
PRIOR_DQ_MIN, PRIOR_DQ_MAX = 0.3, 0.6
N_SAMPLES_BENCH = 200_000
SEED_BENCH = 2024

ALPHA_INFERRED_EXPT = 3.40      # kcal/mol (CcO inferred)

# Utility functions
def alpha_per_electron(R, epsr, dq):
    """Compute α = 332.063 * δq / (εr * R^2). R, dq in Å; returns kcal/mol per electron."""
    R = np.asarray(R, dtype=float)
    epsr = np.asarray(epsr, dtype=float)
    dq = np.asarray(dq, dtype=float)
    return ALPHA_CONST * dq / (epsr * R**2)

def eyring_rate(dg_kcal):
    """Eyring rate at 298.15 K given ΔG‡ in kcal/mol."""
    return KBT_OVER_H * np.exp(-dg_kcal / RT_KCAL)

def cone_mean_projection(theta0_deg):
    """Analytic mean projection for uniform directions within a cone of half-angle theta0 (degrees)."""
    theta0 = np.deg2rad(theta0_deg)
    return 0.5 * (1.0 + np.cos(theta0))

def fisher_mean_projection(kappa):
    """Analytic ⟨cosθ⟩ for von Mises-Fisher on S2 with concentration κ: coth κ − 1/κ; κ=0 -> 0."""
    kappa = np.asarray(kappa, dtype=float)
    out = np.zeros_like(kappa, dtype=float)
    mask = kappa > 1e-8
    km = kappa[mask]
    out[mask] = 1.0 / np.tanh(km) - 1.0 / km
    out[~mask] = 0.0
    return out

def sample_cos_theta_cone(theta0_deg, size, rng):
    """Sample cosθ uniformly within a cone of half-angle theta0 (uniform area on the cone cap)."""
    theta0 = np.deg2rad(theta0_deg)
    c0 = np.cos(theta0)
    # cosθ ~ Uniform[c0, 1]
    u = rng.random(size=size)
    cos_theta = c0 + u * (1.0 - c0)
    return cos_theta

def sample_cos_theta_fisher(kappa, size, rng):
    """Sample cosθ from vMF distribution on S2 with concentration κ (axis aligned)."""
    if kappa < 1e-8:
        return 2.0 * rng.random(size=size) - 1.0  # isotropic
    u = rng.random(size=size)
    # Invert CDF of z=cosθ for pdf ∝ exp(κ z) on [-1,1]:
    # F(z) = (exp(κ z) - exp(-κ)) / (exp(κ) - exp(-κ)) -> z = (1/κ) ln( exp(-κ) + u*(exp(κ)-exp(-κ)) )
    ez = np.exp(kappa)
    em = np.exp(-kappa)
    z = (1.0 / kappa) * np.log(em + u * (ez - em))
    return z

def multipolar_ratio(R, d):
    """Eaxial(d)/Eaxial(point). Two -0.5e charges separated by d perpendicular to axis.
       Ratio = [R^3] / [R^2 + (d/2)^2]^(3/2) = 1 / (1 + (d/(2R))^2)^(3/2)."""
    R = np.asarray(R, dtype=float)
    d = np.asarray(d, dtype=float)
    return 1.0 / (1.0 + (d / (2.0 * R))**2.0)**1.5

def simulate_waiting_time_once(k0, k1, tau, rng):
    """Simulate time to proton transfer event under two-state telegraph with equal stationary probabilities.
       Start state chosen equiprobably. Residence ~ Exp(1/τ). Event waiting in state i ~ Exp(ki).
    """
    t = 0.0
    state = 0 if rng.random() < 0.5 else 1
    k = [k0, k1]
    rate_switch = 1.0 / tau if tau > 0 else np.inf
    while True:
        # Draw event and switch times
        t_event = rng.exponential(1.0 / k[state])
        t_switch = rng.exponential(1.0 / rate_switch) if np.isfinite(rate_switch) else np.inf
        if t_event < t_switch:
            t += t_event
            return t
        else:
            t += t_switch
            state = 1 - state  # switch


# Table S1: Constants and defaults

def make_table_s1():
    records = []
    records.append(["Physical constant", "kBT/h", "k_B T / h", f"{KBT_OVER_H:.3e}", "s^-1", "298.15 K"])
    records.append(["Physical constant", "RT", "R T", f"{RT_KCAL:.3f}", "kcal mol^-1", "298.15 K"])
    records.append(["Conversion", "1 eV", "—", f"{EV_TO_KCAL:.4f}", "kcal mol^-1", "—"])
    records.append(["Conversion", "1 MV cm^-1", "—", f"{MVpcm_TO_VA:.3f}", "V Å^-1", "—"])
    records.append(["Model constant", "α constant", "332.063", f"{ALPHA_CONST:.3f}", "—", "α = 332.063·δq/(εr·R²)"])
    records.append(["Default parameter", "ΔG‡0", "ΔG‡0", f"{DG0_DEFAULT:.2f}", "kcal mol^-1", "Examples"])
    records.append(["Default parameter", "R", "R", f"{R_DEFAULT:.1f}", "Å", "Examples"])
    records.append(["Default parameter", "εr", "εr", f"{EPSR_DEFAULT:.1f}", "—", "Examples"])
    records.append(["Default parameter", "δq", "δq", f"{DQ_DEFAULT:.1f}", "Å", "Examples"])
    records.append(["Grid", "R grid", "R ∈ [3,10]", f"{R_N}", "points", "Heatmaps"])
    records.append(["Grid", "εr grid", "εr ∈ [2,20]", f"{EPSR_N}", "points", "Heatmaps"])
    records.append(["Grid", "δq list", "δq ∈ {0.3,0.5,0.8}", f"{len(DQ_LIST)}", "values", "Heatmaps"])
    df = pd.DataFrame(records, columns=["Category", "Name", "Symbol/Range", "Value", "Units", "Notes"])
    path = os.path.join(CSV_DIR, "table_s1_constants.csv")
    df.to_csv(path, index=False)
    # No figure for Table S1
    return path


# Figure S1: Orientation-averaged projection under alternative distributions

def make_figure_s1():
    # Panel A: cone model (analytic + MC)
    theta0_grid = np.linspace(0, 180, 181)
    analytic_cone = cone_mean_projection(theta0_grid)
    # MC sampling for a subset of θ0 to avoid heavy compute
    rng_s1 = np.random.default_rng(SEED_GLOBAL + 1)
    theta0_mc_list = np.arange(0, 181, 10)
    n_mc = 10_000
    mc_means = []
    mc_lows = []
    mc_highs = []
    for tdeg in theta0_mc_list:
        cos = sample_cos_theta_cone(tdeg, n_mc, rng_s1)
        m = cos.mean()
        s = cos.std(ddof=1)
        se = s / np.sqrt(n_mc)
        mc_means.append(m)
        mc_lows.append(m - 1.96 * se)
        mc_highs.append(m + 1.96 * se)
    df_cone_analytic = pd.DataFrame({"theta0_deg": theta0_grid, "mean_projection": analytic_cone})
    df_cone_mc = pd.DataFrame({
        "theta0_deg": theta0_mc_list,
        "mc_mean": mc_means,
        "mc_ci_low": mc_lows,
        "mc_ci_high": mc_highs
    })
    df_cone_analytic.to_csv(os.path.join(CSV_DIR, "s1_cone_analytic.csv"), index=False)
    df_cone_mc.to_csv(os.path.join(CSV_DIR, "s1_cone_mc.csv"), index=False)

    # Panel B: Fisher distribution (analytic + MC)
    kappa_grid = np.linspace(0, 10, 101)
    analytic_fisher = fisher_mean_projection(kappa_grid)
    rng_s1b = np.random.default_rng(SEED_GLOBAL + 2)
    kappa_mc_list = np.array([0.0, 0.5, 1.0, 2.0, 5.0, 10.0])
    n_mc2 = 10_000
    mc_means_b = []
    mc_lows_b = []
    mc_highs_b = []
    for kap in kappa_mc_list:
        z = sample_cos_theta_fisher(kap, n_mc2, rng_s1b)
        m = z.mean(); s = z.std(ddof=1); se = s / np.sqrt(n_mc2)
        mc_means_b.append(m)
        mc_lows_b.append(m - 1.96 * se)
        mc_highs_b.append(m + 1.96 * se)
    df_fisher_analytic = pd.DataFrame({"kappa": kappa_grid, "mean_projection": analytic_fisher})
    df_fisher_mc = pd.DataFrame({
        "kappa": kappa_mc_list,
        "mc_mean": mc_means_b,
        "mc_ci_low": mc_lows_b,
        "mc_ci_high": mc_highs_b
    })
    df_fisher_analytic.to_csv(os.path.join(CSV_DIR, "s1_fisher_analytic.csv"), index=False)
    df_fisher_mc.to_csv(os.path.join(CSV_DIR, "s1_fisher_mc.csv"), index=False)

    # Panel C: isotropic (κ=0), just a constant zero line
    df_iso = pd.DataFrame({"label": ["isotropic"], "mean_projection": [0.0]})
    df_iso.to_csv(os.path.join(CSV_DIR, "s1_isotropic.csv"), index=False)

    # Plot S1
    fig, axs = plt.subplots(1, 3, figsize=(10.5, 3.2), constrained_layout=True)
    # A
    axs[0].plot(df_cone_analytic["theta0_deg"], df_cone_analytic["mean_projection"], color="black", lw=1.5, label="Analytic")
    axs[0].errorbar(df_cone_mc["theta0_deg"], df_cone_mc["mc_mean"],
                    yerr=[df_cone_mc["mc_mean"] - df_cone_mc["mc_ci_low"],
                          df_cone_mc["mc_ci_high"] - df_cone_mc["mc_mean"]],
                    fmt="o", ms=3, color="#1f77b4", label="MC (95% CI)")
    axs[0].set_xlabel(r"Cone half-angle $\theta_0$ (deg)")
    axs[0].set_ylabel(r"$\langle \cos\theta \rangle$")
    axs[0].set_title("(A) Cone model")
    axs[0].legend(frameon=False)

    # B
    axs[1].plot(df_fisher_analytic["kappa"], df_fisher_analytic["mean_projection"], color="black", lw=1.5, label="Analytic")
    axs[1].errorbar(df_fisher_mc["kappa"], df_fisher_mc["mc_mean"],
                    yerr=[df_fisher_mc["mc_mean"] - df_fisher_mc["mc_ci_low"],
                          df_fisher_mc["mc_ci_high"] - df_fisher_mc["mc_mean"]],
                    fmt="s", ms=3, color="#ff7f0e", label="MC (95% CI)")
    axs[1].set_xlabel(r"Fisher concentration $\kappa$")
    axs[1].set_ylabel(r"$\langle \cos\theta \rangle$")
    axs[1].set_title("(B) Fisher (vMF) model")
    axs[1].legend(frameon=False)

    # C
    axs[2].axhline(0.0, color="black", lw=1.5)
    axs[2].set_xlabel("Index")
    axs[2].set_ylabel(r"$\langle \cos\theta \rangle$")
    axs[2].set_title("(C) Isotropic (κ = 0)")
    axs[2].set_xlim(0, 1)
    axs[2].set_ylim(-0.05, 0.05)

    s1_path = os.path.join(FIG_DIR, "figure_s1.png")
    fig.savefig(s1_path, dpi=300)
    plt.close(fig)
    return s1_path


# Figure S2: Heatmaps of α across (R, εr) for δq = 0.3, 0.5, 0.8 Å

def make_figure_s2():
    R_vals = np.linspace(R_MIN, R_MAX, R_N)
    epsr_vals = np.linspace(EPSR_MIN, EPSR_MAX, EPSR_N)
    RR, EE = np.meshgrid(R_vals, epsr_vals, indexing="xy")

    fig, axs = plt.subplots(1, 3, figsize=(10.5, 3.2), constrained_layout=True)
    for i, dq in enumerate(DQ_LIST):
        AA = alpha_per_electron(RR, EE, dq)
        # Save CSV as long-form
        df = pd.DataFrame({
            "R_Ang": RR.flatten(),
            "epsr": EE.flatten(),
            "alpha_kcal_per_mol_per_e": AA.flatten()
        })
        csv_path = os.path.join(CSV_DIR, f"s2_alpha_heatmap_dq_{dq:.1f}.csv")
        df.to_csv(csv_path, index=False)

        ax = axs[i]
        im = ax.pcolormesh(R_vals, epsr_vals, AA, shading="auto", cmap="viridis")
        cs = ax.contour(R_vals, epsr_vals, AA, colors="white", linewidths=0.7, levels=np.arange(0, 10.5, 1.0))
        ax.clabel(cs, inline=True, fontsize=8, fmt="%.0f")
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label(r"$\alpha$ (kcal mol$^{-1}$ per e$^-$)")
        ax.set_xlabel(r"$R$ (Å)")
        ax.set_ylabel(r"$\varepsilon_r$")
        ax.set_title(f"( {chr(65+i)} ) δq = {dq:.1f} Å")

    s2_path = os.path.join(FIG_DIR, "figure_s2.png")
    fig.savefig(s2_path, dpi=300)
    plt.close(fig)
    return s2_path


# Figure S3: Stochastic validation of orientation averaging under cone model

def make_figure_s3():
    rng_s3 = np.random.default_rng(SEED_GLOBAL + 3)
    theta0_list = np.arange(0, 181, 5)
    n_mc = 10_000
    analytic = cone_mean_projection(theta0_list)
    means = []
    lows = []
    highs = []
    for tdeg in theta0_list:
        cos = sample_cos_theta_cone(tdeg, n_mc, rng_s3)
        m = cos.mean()
        s = cos.std(ddof=1)
        se = s / np.sqrt(n_mc)
        means.append(m)
        lows.append(m - 1.96 * se)
        highs.append(m + 1.96 * se)
    df = pd.DataFrame({
        "theta0_deg": theta0_list,
        "analytic_mean": analytic,
        "mc_mean": means,
        "mc_ci_low": lows,
        "mc_ci_high": highs
    })
    df.to_csv(os.path.join(CSV_DIR, "s3_orientation_sampling.csv"), index=False)

    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    ax.plot(theta0_list, analytic, color="black", lw=1.6, label="Analytic")
    ax.fill_between(theta0_list, lows, highs, color="#1f77b4", alpha=0.25, label="MC 95% CI")
    ax.plot(theta0_list, means, color="#1f77b4", lw=1.2, label="MC mean")
    ax.set_xlabel(r"Cone half-angle $\theta_0$ (deg)")
    ax.set_ylabel(r"$\langle \cos\theta \rangle$")
    ax.set_title("Figure S3. Orientation averaging: analytic vs Monte Carlo")
    ax.legend(frameon=False)
    s3_path = os.path.join(FIG_DIR, "figure_s3.png")
    fig.savefig(s3_path, dpi=300)
    plt.close(fig)
    return s3_path


# Figure S4: Multipolar corrections (ratio lines and 2D map)

def make_figure_s4():
    R_line = np.linspace(R_MIN, R_MAX, 200)
    d_list = [0.0, 1.0, 2.0]
    data = {"R_Ang": R_line}
    for d in d_list:
        data[f"ratio_d_{d:.1f}_Ang"] = multipolar_ratio(R_line, d)
    df_lines = pd.DataFrame(data)
    df_lines.to_csv(os.path.join(CSV_DIR, "s4_ratio_lines.csv"), index=False)

    # 2D map
    R_vals = np.linspace(R_MIN, R_MAX, 200)
    d_vals = np.linspace(0.0, 2.0, 200)
    RR, DD = np.meshgrid(R_vals, d_vals, indexing="xy")
    ratio_map = multipolar_ratio(RR, DD)
    df_map = pd.DataFrame({
        "R_Ang": RR.flatten(),
        "d_Ang": DD.flatten(),
        "ratio": ratio_map.flatten()
    })
    df_map.to_csv(os.path.join(CSV_DIR, "s4_ratio_map.csv"), index=False)

    # Plot
    fig = plt.figure(figsize=(10.5, 3.2), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.2, 1.0])

    # Panel A: lines
    ax1 = fig.add_subplot(gs[0, 0])
    for d in d_list:
        ax1.plot(R_line, multipolar_ratio(R_line, d), lw=1.5, label=f"d = {d:.1f} Å")
    ax1.set_xlabel(r"$R$ (Å)")
    ax1.set_ylabel(r"$E_\mathrm{axial}(d)/E_\mathrm{axial}(\mathrm{point})$")
    ax1.set_title("(A) Ratio vs R")
    ax1.legend(frameon=False)

    # Panel B: 2D map with ≤10% deviation region
    ax2 = fig.add_subplot(gs[0, 1])
    im = ax2.pcolormesh(R_vals, d_vals, ratio_map, shading="auto", cmap="magma")
    cbar = fig.colorbar(im, ax=ax2)
    cbar.set_label(r"Ratio")
    # Overlay contour for 0.90
    cs = ax2.contour(R_vals, d_vals, ratio_map, levels=[0.90], colors="cyan", linewidths=1.2)
    ax2.clabel(cs, fmt="0.90", fontsize=8)
    ax2.set_xlabel(r"$R$ (Å)")
    ax2.set_ylabel(r"$d$ (Å)")
    ax2.set_title("(B) 2D map; cyan contour = 0.90")

    s4_path = os.path.join(FIG_DIR, "figure_s4.png")
    fig.savefig(s4_path, dpi=300)
    plt.close(fig)
    return s4_path


# Figure S5: Convergence of keff vs number of trajectories

def make_figure_s5():
    # Compute k0 and k1 from defaults
    alpha = alpha_per_electron(R_DEFAULT, EPSR_DEFAULT, DQ_DEFAULT)  # 2.594...
    dg0 = DG0_DEFAULT
    dg_n0 = dg0
    dg_n1 = dg0 - alpha
    k0 = eyring_rate(dg_n0)
    k1 = eyring_rate(dg_n1)

    # τ values (s), representative across limits
    tau_list = [1e-4, 1e-3, 1e-2]  # near-fast, crossover, near-slow
    Ntraj_list = [50, 100, 200, 500, 1000, 2000]
    n_reps = 100  # independent repeats per Ntraj to estimate RSE

    records = []
    rng_s5 = np.random.default_rng(SEED_GLOBAL + 4)
    for tau in tau_list:
        for N in Ntraj_list:
            keff_estimates = []
            # For each replicate, estimate mean waiting time from N trajectories
            for _ in range(n_reps):
                waits = [simulate_waiting_time_once(k0, k1, tau, rng_s5) for _ in range(N)]
                mean_wait = np.mean(waits)
                keff_estimates.append(1.0 / mean_wait)
            keff_arr = np.array(keff_estimates)
            mean_keff = keff_arr.mean()
            rse = keff_arr.std(ddof=1) / (np.sqrt(n_reps) * mean_keff)  # RSE of mean estimate
            records.append([tau, N, mean_keff, rse])

    df = pd.DataFrame(records, columns=["tau_s", "Ntraj", "keff_mean_s^-1", "rse_of_mean"])
    df.to_csv(os.path.join(CSV_DIR, "s5_convergence.csv"), index=False)

    # Plot: RSE vs N for each τ (log-log axes)
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    markers = ["o", "s", "^"]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    for i, tau in enumerate(tau_list):
        df_tau = df[df["tau_s"] == tau]
        ax.loglog(df_tau["Ntraj"], df_tau["rse_of_mean"], marker=markers[i], color=colors[i],
                  lw=1.5, label=fr"$\tau$ = {tau:g} s")
    ax.set_xlabel("Number of trajectories N")
    ax.set_ylabel("Relative standard error of mean $k_\\mathrm{eff}$")
    ax.set_title("Figure S5. Convergence of $k_\\mathrm{eff}$ vs $N$")
    ax.legend(frameon=False)
    s5_path = os.path.join(FIG_DIR, "figure_s5.png")
    fig.savefig(s5_path, dpi=300)
    plt.close(fig)
    return s5_path


# Table S2 and Figure S6: Priors and CDF for CcO benchmark with sensitivity

def truncated_normal(mean, sigma, size, lower):
    """Sample from N(mean, sigma^2) truncated at lower bound."""
    # Rejection sampling (sufficiently efficient with lower=1Å and mean=6, sigma~1)
    out = np.empty(size, dtype=float)
    filled = 0
    rs = np.random.default_rng(SEED_BENCH)
    while filled < size:
        n = size - filled
        samples = rs.normal(mean, sigma, size=n)
        samples = samples[samples > lower]
        m = len(samples)
        if m > 0:
            out[filled:filled + m] = samples
            filled += m
    return out

def sample_priors(n, r_sigma=PRIOR_R_SIGMA, orientation=None, seed=SEED_BENCH):
    """Sample α from priors; optionally include orientation factor sampled from a cone with half-angle orientation.theta0."""
    rs = np.random.default_rng(seed)
    # R ~ truncated normal
    R = truncated_normal(PRIOR_R_MEAN, r_sigma, n, lower=1.0)
    # epsr ~ U[2,4]
    epsr = rs.uniform(PRIOR_EPSR_MIN, PRIOR_EPSR_MAX, size=n)
    # δq ~ U[0.3,0.6]
    dq = rs.uniform(PRIOR_DQ_MIN, PRIOR_DQ_MAX, size=n)
    alpha = alpha_per_electron(R, epsr, dq)
    if orientation is not None:
        theta0 = orientation.get("theta0_deg", 60.0)
        cos = sample_cos_theta_cone(theta0, n, rs)
        alpha = alpha * cos
    return alpha

def make_table_s2_and_figure_s6():
    # Table S2 (priors)
    rows = [
        ["R (Å)", "Normal truncated", "μ = 6.0 Å; σ = 1.0 Å; lower = 1.0 Å",
         "Proximity of BNC to water/proton pathway"],
        [r"εr (—)", "Uniform", "min = 2.0; max = 4.0", "Protein interior on barrier timescale"],
        [r"δq (Å)", "Uniform", "min = 0.3 Å; max = 0.6 Å", "Hydrogen-bonded PT TS displacement"],
    ]
    df = pd.DataFrame(rows, columns=["Parameter", "Distribution", "Parameters", "Rationale"])
    df.to_csv(os.path.join(CSV_DIR, "table_s2_priors.csv"), index=False)

    # Figure S6: CDFs baseline, narrower R prior, and orientation factor
    n = N_SAMPLES_BENCH
    # Baseline
    alpha0 = sample_priors(n, r_sigma=1.0, orientation=None, seed=SEED_BENCH)
    # Narrower R prior (σ = 0.5 Å)
    alpha_narrow = sample_priors(n, r_sigma=0.5, orientation=None, seed=SEED_BENCH + 1)
    # Orientation factor: cone θ0 = 60°
    alpha_orient = sample_priors(n, r_sigma=1.0, orientation={"theta0_deg": 60.0}, seed=SEED_BENCH + 2)

    # Sort for CDF
    def sort_cdf(x):
        xs = np.sort(x)
        cdf = np.linspace(0, 1, len(xs), endpoint=False)
        return xs, cdf

    xs0, c0 = sort_cdf(alpha0)
    xsN, cN = sort_cdf(alpha_narrow)
    xsO, cO = sort_cdf(alpha_orient)

    # Quantiles for baseline
    q025, q50, q975 = np.percentile(alpha0, [2.5, 50, 97.5])
    df_cdf = pd.DataFrame({"alpha_sorted": xs0, "cdf": c0})
    df_cdf.to_csv(os.path.join(CSV_DIR, "s6_cdf_baseline.csv"), index=False)
    df_q = pd.DataFrame({"q2.5": [q025], "q50": [q50], "q97.5": [q975]})
    df_q.to_csv(os.path.join(CSV_DIR, "s6_quantiles_baseline.csv"), index=False)

    # Sensitivity series CSVs
    pd.DataFrame({"alpha_sorted": xsN, "cdf": cN}).to_csv(os.path.join(CSV_DIR, "s6_cdf_narrow_R.csv"), index=False)
    pd.DataFrame({"alpha_sorted": xsO, "cdf": cO}).to_csv(os.path.join(CSV_DIR, "s6_cdf_orientation_theta60.csv"), index=False)

    # Plot CDFs
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    ax.plot(xs0, c0, color="#1f77b4", lw=1.6, label="Baseline priors")
    ax.plot(xsN, cN, color="#ff7f0e", lw=1.6, linestyle="--", label=r"Narrow $R$ prior ($\sigma=0.5$ Å)")
    ax.plot(xsO, cO, color="#2ca02c", lw=1.6, linestyle=":", label=r"Incl. orientation ($\theta_0=60^\circ$)")
    ax.axvline(ALPHA_INFERRED_EXPT, color="red", lw=1.2, linestyle="--", label=r"$\alpha_\mathrm{inferred}=3.40$")
    ax.set_xlabel(r"$\alpha$ (kcal mol$^{-1}$ per e$^-$)")
    ax.set_ylabel("CDF")
    ax.set_title("Figure S6. CDF of predicted $\\alpha$ with sensitivities")
    ax.legend(frameon=False)
    s6_path = os.path.join(FIG_DIR, "figure_s6.png")
    fig.savefig(s6_path, dpi=300)
    plt.close(fig)

    # Also write a summary CSV for baseline stats
    summary = pd.DataFrame({
        "alpha_inferred_expt": [ALPHA_INFERRED_EXPT],
        "median": [q50],
        "p2.5": [q025],
        "p97.5": [q975],
        "covered": [q025 <= ALPHA_INFERRED_EXPT <= q975]
    })
    summary.to_csv(os.path.join(CSV_DIR, "s6_summary_baseline.csv"), index=False)
    return s6_path


# Table S3: CSV inventory

def make_table_s3_inventory():
    inventory = [
        ("table_s1_constants.csv", "Physical constants, conversions, defaults, grids", "Category,Name,Symbol/Range,Value,Units,Notes", "Table S1"),
        ("s1_cone_analytic.csv", "Analytic ⟨cosθ⟩ vs θ0 for cone model", "theta0_deg,mean_projection", "Figure S1A"),
        ("s1_cone_mc.csv", "MC ⟨cosθ⟩ vs θ0 (cone) with 95% CI", "theta0_deg,mc_mean,mc_ci_low,mc_ci_high", "Figure S1A"),
        ("s1_fisher_analytic.csv", "Analytic ⟨cosθ⟩ vs κ (Fisher vMF)", "kappa,mean_projection", "Figure S1B"),
        ("s1_fisher_mc.csv", "MC ⟨cosθ⟩ vs κ (Fisher) with 95% CI", "kappa,mc_mean,mc_ci_low,mc_ci_high", "Figure S1B"),
        ("s1_isotropic.csv", "Isotropic ⟨cosθ⟩", "label,mean_projection", "Figure S1C"),
        ("s2_alpha_heatmap_dq_0.3.csv", "α heatmap values at δq=0.3 Å", "R_Ang,epsr,alpha_kcal_per_mol_per_e", "Figure S2"),
        ("s2_alpha_heatmap_dq_0.5.csv", "α heatmap values at δq=0.5 Å", "R_Ang,epsr,alpha_kcal_per_mol_per_e", "Figure S2"),
        ("s2_alpha_heatmap_dq_0.8.csv", "α heatmap values at δq=0.8 Å", "R_Ang,epsr,alpha_kcal_per_mol_per_e", "Figure S2"),
        ("s3_orientation_sampling.csv", "Analytic and MC ⟨cosθ⟩ vs θ0 with CI", "theta0_deg,analytic_mean,mc_mean,mc_ci_low,mc_ci_high", "Figure S3"),
        ("s4_ratio_lines.csv", "Multipolar-to-point axial field ratio vs R (d=0,1,2 Å)", "R_Ang,ratio_d_0.0_Ang,ratio_d_1.0_Ang,ratio_d_2.0_Ang", "Figure S4A"),
        ("s4_ratio_map.csv", "2D map of ratio vs (R,d)", "R_Ang,d_Ang,ratio", "Figure S4B"),
        ("s5_convergence.csv", "Convergence of keff: RSE vs N for τ", "tau_s,Ntraj,keff_mean_s^-1,rse_of_mean", "Figure S5"),
        ("table_s2_priors.csv", "Prior distributions for CcO benchmark", "Parameter,Distribution,Parameters,Rationale", "Table S2"),
        ("s6_cdf_baseline.csv", "CDF data for baseline α", "alpha_sorted,cdf", "Figure S6"),
        ("s6_quantiles_baseline.csv", "Baseline α quantiles", "q2.5,q50,q97.5", "Figure S6"),
        ("s6_cdf_narrow_R.csv", "CDF for α with narrower R prior (σ=0.5 Å)", "alpha_sorted,cdf", "Figure S6"),
        ("s6_cdf_orientation_theta60.csv", "CDF for α with orientation (θ0=60°)", "alpha_sorted,cdf", "Figure S6"),
        ("s6_summary_baseline.csv", "Summary: α_inferred, median, 95% PI, coverage", "alpha_inferred_expt,median,p2.5,p97.5,covered", "Figure S6"),
    ]
    df = pd.DataFrame(inventory, columns=["filename", "description", "columns", "figure_table_refs"])
    df.to_csv(os.path.join(CSV_DIR, "table_s3_inventory.csv"), index=False)
    # No figure for Table S3
    return os.path.join(CSV_DIR, "table_s3_inventory.csv")


# Driver: generate everything

def main():
    print("Generating Table S1...")
    t1 = make_table_s1()
    print("Generating Figure S1...")
    f1 = make_figure_s1()
    print("Generating Figure S2...")
    f2 = make_figure_s2()
    print("Generating Figure S3...")
    f3 = make_figure_s3()
    print("Generating Figure S4...")
    f4 = make_figure_s4()
    print("Generating Figure S5...")
    f5 = make_figure_s5()
    print("Generating Table S2 and Figure S6...")
    f6 = make_table_s2_and_figure_s6()
    print("Generating Table S3 inventory...")
    t3 = make_table_s3_inventory()
    print("Done.")
    print(f"CSV directory: {CSV_DIR}")
    print(f"Figures directory: {FIG_DIR}")

if __name__ == "__main__":
    main()
