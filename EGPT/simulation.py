import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os


plt.rcParams["font.family"] = "serif"
plt.rcParams["figure.dpi"] = 200


K_ALPHA = 332.063
R_gas = 1.987204e-3
T = 298.15
kB = 1.380649e-23
h = 6.62607015e-34
NA = 6.02214076e23
kBT_over_h = (1.380649e-23 * T) / h
kBT_over_h *= 1.0

k_pref = kBT_over_h

def alpha_kcal_per_e(delta_q_A, epsilon_r, R_A, cos_theta=1.0):
    return K_ALPHA * delta_q_A * cos_theta / (epsilon_r * (R_A**2))

def eyring_rate(dG_kcal):
    return k_pref * np.exp(-dG_kcal / (R_gas * T))

def ensure_dirs():
    os.makedirs("csv", exist_ok=True)
    os.makedirs("png", exist_ok=True)

def fig1_alpha_vs_R(delta_q_A=0.5, epsilons=(2,4,10), R_range=(3.0, 10.0), npts=100):
    R = np.linspace(R_range[0], R_range[1], npts)
    df = pd.DataFrame({"R_A": R})
    plt.figure()
    for eps in epsilons:
        alpha_vals = alpha_kcal_per_e(delta_q_A, eps, R)
        df[f"alpha_kcal_per_e_epsilon_{eps}"] = alpha_vals
        plt.plot(R, alpha_vals, label=f"εr={eps}")
    plt.xlabel("Electron–proton distance R (Å)")
    plt.ylabel("Barrier lowering per electron α (kcal/mol)")
    plt.title(f"α vs R (δq={delta_q_A} Å)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    df.to_csv("csv/alpha_vs_R.csv", index=False)
    plt.tight_layout()
    plt.savefig("png/Fig1_alpha_vs_R.png")
    plt.close()

def fig2_dG_and_rate_vs_ne(delta_q_A=0.5, epsilon_r=4.0, R_A=4.0, G0=15.0, ne_max=3):
    ne = np.arange(0, ne_max+1)
    alpha_val = alpha_kcal_per_e(delta_q_A, epsilon_r, R_A)
    dG = G0 - alpha_val * ne
    k = eyring_rate(dG)
    df = pd.DataFrame({"n_e": ne,
                       "alpha_kcal_per_e": [alpha_val]*len(ne),
                       "dG_kcal": dG,
                       "k_s^-1": k})
    df.to_csv("csv/dG_vs_ne.csv", index=False)

    fig, ax1 = plt.subplots()
    ax1.plot(ne, dG, "o-", color="tab:blue", label="ΔG‡ (kcal/mol)")
    ax1.set_xlabel("Electron occupancy n_e")
    ax1.set_ylabel("ΔG‡ (kcal/mol)", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax2 = ax1.twinx()
    ax2.plot(ne, k, "s--", color="tab:red", label="k (s⁻¹)")
    ax2.set_ylabel("Rate constant k (s⁻¹)", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")
    ax1.grid(True, alpha=0.3)
    lines, labels = [], []
    for ax in [ax1, ax2]:
        L, Lab = ax.get_legend_handles_labels()
        lines += L; labels += Lab
    ax1.set_title(f"Barrier and rate vs n_e (R={R_A} Å, εr={epsilon_r}, δq={delta_q_A} Å)")
    fig.legend(lines, labels, loc="upper right", bbox_to_anchor=(0.85, 0.85))
    fig.tight_layout()
    fig.savefig("png/Fig2_dG_and_rate_vs_ne.png")
    plt.close()

def fig3_alpha_heatmap(delta_q_A=0.5, epsilon_r_vals=(2,4,8,12,20), R_vals=np.linspace(3,10,100)):
    records = []
    for eps in epsilon_r_vals:
        for R in R_vals:
            records.append({"delta_q_A": delta_q_A,
                            "epsilon_r": eps,
                            "R_A": R,
                            "alpha_kcal_per_e": alpha_kcal_per_e(delta_q_A, eps, R)})
    df = pd.DataFrame.from_records(records)
    df.to_csv("csv/alpha_heatmap_data.csv", index=False)

    eps_list = sorted(df["epsilon_r"].unique())
    R_list = sorted(df["R_A"].unique())
    Z = np.zeros((len(eps_list), len(R_list)))
    for i, eps in enumerate(eps_list):
        vals = df[df["epsilon_r"]==eps].sort_values("R_A")["alpha_kcal_per_e"].values
        Z[i,:] = vals

    plt.figure(figsize=(6,3))
    im = plt.imshow(Z, aspect="auto", origin="lower",
                    extent=[min(R_list), max(R_list), min(eps_list), max(eps_list)],
                    cmap="viridis")
    cbar = plt.colorbar(im)
    cbar.set_label("α (kcal/mol per e−)")
    plt.xlabel("R (Å)")
    plt.ylabel("εr")
    plt.title(f"α heatmap at δq={delta_q_A} Å")
    plt.tight_layout()
    plt.savefig("png/Fig3_alpha_heatmap.png")
    plt.close()

def fig4_orientation(cone_angles_deg=np.linspace(0,180,37), delta_q_A=0.5, epsilon_r=4.0, R_A=4.0):
    def avg_cos_theta(theta0):
        return 0.5*(1 + np.cos(theta0))
    alpha_max = alpha_kcal_per_e(delta_q_A, epsilon_r, R_A, cos_theta=1.0)
    rows = []
    y = []
    for ang in cone_angles_deg:
        th0 = np.deg2rad(ang)
        c = avg_cos_theta(th0)
        alpha_eff = alpha_max * c
        rows.append({"cone_half_angle_deg": ang, "avg_cos": c, "alpha_eff": alpha_eff, "alpha_max": alpha_max})
        y.append(alpha_eff/alpha_max)
    df = pd.DataFrame(rows)
    df.to_csv("csv/orientation_alpha.csv", index=False)

    plt.figure()
    plt.plot(cone_angles_deg, y, "o-")
    plt.xlabel("Cone half-angle θ0 (deg)")
    plt.ylabel("α_eff / α_max")
    plt.title("Orientation-averaged gating: α_eff = α_max·(1+cosθ0)/2")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("png/Fig4_orientation.png")
    plt.close()

def fig5_multipole(d_vals=(0.0, 0.5, 1.0, 2.0), R_vals=np.linspace(3,10,100), delta_q_A=0.5, epsilon_r=4.0):

    def Ex_point_charge(qe, rx, ry, rz, eps_r):

        dx = R - rx; dy = -ry; dz = -rz
        r3 = (dx*dx + dy*dy + dz*dz)**1.5
        return qe * dx / r3 / eps_r

    rows = []
    for d in d_vals:
        ratios = []
        for R in R_vals:

            Ex = Ex_point_charge(-0.5, 0.0, +d/2, 0.0, epsilon_r) + Ex_point_charge(-0.5, 0.0, -d/2, 0.0, epsilon_r)

            Ex_pc = Ex_point_charge(-1.0, 0.0, 0.0, 0.0, epsilon_r)
            ratio = Ex/Ex_pc if Ex_pc != 0 else np.nan
            ratios.append(ratio)
            rows.append({"d_A": d, "R_A": R, "ratio_multipole_to_point": ratio})

    df = pd.DataFrame(rows)
    df.to_csv("csv/multipole_vs_point.csv", index=False)

    plt.figure()
    for d in d_vals:
        sub = df[df["d_A"]==d].sort_values("R_A")
        plt.plot(sub["R_A"], sub["ratio_multipole_to_point"], label=f"d={d} Å")
    plt.xlabel("R (Å)")
    plt.ylabel("α(d) / α(point charge)")
    plt.title("Multipolar correction to gating field (two-charge model)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("png/Fig5_multipole.png")
    plt.close()

def simulate_dynamic_gating(ne_trajs=2000, tau_vals=np.logspace(-9,-3,25),
                            G0=15.0, alpha_val=2.5942421875,
                            T_K=298.15):
    k0 = eyring_rate(G0)
    k1 = eyring_rate(G0 - alpha_val)
    rows = []
    for tau in tau_vals:
        rate01 = 1.0/tau
        rate10 = 1.0/tau
        p0 = 0.5; p1 = 0.5
        k_fast = p0*k0 + p1*k1
        k_slow = 1.0 / (p0/k0 + p1/k1)

        rng = np.random.default_rng(42)
        times = []
        for _ in range(ne_trajs):
            state = 0 if rng.random() < p0 else 1
            t = 0.0
            while True:
                dwell = rng.exponential(tau)
                rate = k0 if state==0 else k1
                wait = rng.exponential(1.0/rate)
                if wait < dwell:
                    t += wait
                    times.append(t)
                    break
                else:
                    t += dwell
                    state = 1 - state
        keff_mc = 1.0 / np.mean(times)
        rows.append({"tau_s": tau, "k_eff_s^-1": keff_mc, "k_fast_limit_s^-1": k_fast, "k_slow_limit_s^-1": k_slow,
                     "k0_s^-1": k0, "k1_s^-1": k1})
    df = pd.DataFrame(rows)
    df.to_csv("csv/dynamic_gating_keff.csv", index=False)

    plt.figure()
    plt.loglog(df["tau_s"], df["k_eff_s^-1"], "o-", label="Monte Carlo keff")
    plt.loglog(df["tau_s"], df["k_fast_limit_s^-1"], "--", label="Fast-switch limit (p0 k0 + p1 k1)")
    plt.loglog(df["tau_s"], df["k_slow_limit_s^-1"], "--", label="Slow-switch limit (1/(p0/k0+p1/k1))")
    plt.axhline(df["k0_s^-1"].iloc[0], color="gray", lw=0.8, label="k0")
    plt.axhline(df["k1_s^-1"].iloc[0], color="gray", lw=0.8, ls="--", label="k1")
    plt.xlabel("Mean dwell time τ (s)")
    plt.ylabel("Effective rate k_eff (s⁻¹)")
    plt.title("Dynamic gating crossover (symmetrical telegraph process)")
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig("png/Fig6_dynamic_gating.png")
    plt.close()

if __name__ == "__main__":
    ensure_dirs()
    fig1_alpha_vs_R(delta_q_A=0.5, epsilons=(2,4,10), R_range=(3.0, 10.0), npts=100)
    fig2_dG_and_rate_vs_ne(delta_q_A=0.5, epsilon_r=4.0, R_A=4.0, G0=15.0, ne_max=3)
    fig3_alpha_heatmap(delta_q_A=0.5, epsilon_r_vals=(2,4,8,12,20), R_vals=np.linspace(3,10,100))
    # Added validation
    fig4_orientation(cone_angles_deg=np.linspace(0,180,37), delta_q_A=0.5, epsilon_r=4.0, R_A=4.0)
    fig5_multipole(d_vals=(0.0, 0.5, 1.0, 2.0), R_vals=np.linspace(3,10,100), delta_q_A=0.5, epsilon_r=4.0)
    simulate_dynamic_gating(ne_trajs=2000, tau_vals=np.logspace(-9,-3,25),
                            G0=15.0, alpha_val=2.5942421875, T_K=298.15)
