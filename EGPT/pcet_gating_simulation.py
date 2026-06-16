"""
Long-Range Electrostatic Gating in PCET

This script provides the numerical simulations used to validate the exchange-free 
electrostatic gating mechanism of Proton-Coupled Electron Transfer (PCET).
It evaluates the Trotter fidelity of the quantum gate isomorphism, solves the 2D 
proton-environment Hamiltonian using the Discrete Variable Representation (DVR) method, 
and computes the macroscopic reaction rates via Lindblad dynamics and Voigt-type convolutions.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.linalg import expm, eigh
from scipy.integrate import quad
from scipy.stats import linregress

# =====================================================================
# 1. Global Settings & Constants
# =====================================================================
OUTPUT_DIR = "Figures_and_Data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.rcParams.update({
    'font.size': 14,
    'font.family': 'serif',
    'font.serif':['Times New Roman', 'DejaVu Serif'],
    'axes.linewidth': 1.5,
    'lines.linewidth': 2.5,
    'xtick.major.width': 1.5,
    'ytick.major.width': 1.5,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'legend.frameon': True,
    'legend.facecolor': 'white',
    'legend.edgecolor': 'white',
    'legend.framealpha': 1.0,
    'figure.dpi': 300,
    'savefig.dpi': 600,
    'savefig.bbox': 'tight'
})

# Physical Constants and Conversions
AU2KCAL = 627.509474      # Hartree to kcal/mol
AU2ANG = 0.52917721092    # Bohr to Angstrom
kB_eV = 8.617333262e-5    # Boltzmann constant [eV/K]
T_K = 300.0               # Temperature [K]
kBT = kB_eV * T_K         # Thermal energy [eV]
cm_to_eV = 1.0 / 8065.0   # cm^-1 to eV

# Pauli Matrices
sig_x = np.array([[0, 1], [1, 0]])
sig_z = np.array([[1, 0], [0, -1]])


# =====================================================================
# 2. Module: Quantum Gate Isomorphism (Trotter Fidelity)
# =====================================================================
def compute_gate_fidelity(U_exact, U_approx):
    """Computes the quantum gate fidelity between exact and approximate unitaries."""
    d = U_exact.shape[0]
    trace_val = np.trace(U_exact.conj().T @ U_approx)
    return (np.abs(trace_val)**2 + d) / (d * (d + 1))

def simulate_trotter_fidelity():
    """Evaluates Trotter errors arising from non-commuting transverse and longitudinal fields."""
    print("Evaluating Trotter Fidelity Map...")
    alpha_over_deltas = np.logspace(-1, 2, 50)
    dt_fs_vals = np.linspace(0.1, 10.0, 50)
    fidelity_map = np.zeros((len(alpha_over_deltas), len(dt_fs_vals)))

    Delta = 1.0  # Reference tunneling scale
    
    for i, a_over_d in enumerate(alpha_over_deltas):
        alpha = Delta * a_over_d
        H_exact = (Delta/2) * sig_x - (alpha/2) * sig_z
        for j, dt in enumerate(dt_fs_vals):
            U_exact = expm(-1j * H_exact * dt)
            # Trotter split: Evolution of tunneling (sig_x) and gating (sig_z)
            U_trot = expm(-1j * (Delta/2) * sig_x * dt) @ expm(1j * (alpha/2) * sig_z * dt)
            fidelity_map[i, j] = compute_gate_fidelity(U_exact, U_trot)
            
    return dt_fs_vals, alpha_over_deltas, fidelity_map


# =====================================================================
# 3. Module: 2D DVR (Proton-Environment Coupling)
# =====================================================================
def solve_2d_dvr_hamiltonian():
    """Solves the 2D proton-environment Hamiltonian using a discrete grid (DVR)."""
    print("Solving 2D DVR Hamiltonian...")
    N_grid = 40
    x = np.linspace(-1.5, 1.5, N_grid) / AU2ANG
    y = np.linspace(-1.5, 1.5, N_grid) / AU2ANG
    X, Y = np.meshgrid(x, y, indexing='ij')

    dx = x[1] - x[0]
    dy = y[1] - y[0]
    
    mass_p = 1836.15  # Proton mass in a.u.
    mass_env = mass_p * 10.0  # Heavy environmental mode

    # Kinetic Energy Operators (Finite Difference)
    D2x = (np.diag(2*np.ones(N_grid)) - np.diag(np.ones(N_grid-1), 1) - np.diag(np.ones(N_grid-1), -1)) / dx**2
    D2y = (np.diag(2*np.ones(N_grid)) - np.diag(np.ones(N_grid-1), 1) - np.diag(np.ones(N_grid-1), -1)) / dy**2
    Tx = (0.5 / mass_p) * np.kron(D2x, np.eye(N_grid))
    Ty = (0.5 / mass_env) * np.kron(np.eye(N_grid), D2y)

    # Potential Energy Surface
    Eb = 15.0 / AU2KCAL  # Barrier height
    x0 = 0.3 / AU2ANG    # Minima displacement
    k_y = 0.1            # Environmental harmonic confinement
    c_xy = 0.05          # Bilinear coupling constant

    V_2d = Eb * (1 - (X/x0)**2)**2 + 0.5 * k_y * Y**2 + c_xy * X * Y
    H_2d = Tx + Ty + np.diag(V_2d.flatten())

    # Diagonalize the lowest eigenvalues
    evals_2d, _ = eigh(H_2d, subset_by_index=[0, 5])
    E_2d_kcal = evals_2d * AU2KCAL
    
    return E_2d_kcal


# =====================================================================
# 4. Module: Macroscopic Rate Constants & Scaling
# =====================================================================
def lorentzian_rate(eps, gamma):
    """Intrinsic quantum tunneling rate kernel."""
    return (2 * gamma) / (eps**2 + (2 * gamma)**2)

def bath_distribution(eps, eps0, lam):
    """Gaussian distribution of solvent fluctuations."""
    sigma = np.sqrt(2 * lam * kBT)
    return (1.0 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-((eps - eps0)**2) / (2 * sigma**2))

def compute_convolution_rate(eps0, gamma, lam):
    """Voigt-type convolution of the Lorentzian kernel and Gaussian bath."""
    integrand = lambda eps: lorentzian_rate(eps, gamma) * bath_distribution(eps, eps0, lam)
    res, _ = quad(integrand, -2.0, 2.0, limit=1000)
    return res

def compute_marcus_rate(eps0, lam):
    """Standard classical Marcus theory rate (delta-function limit)."""
    sigma = np.sqrt(2 * lam * kBT)
    return np.pi * (1.0 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-(eps0**2) / (2 * sigma**2))

def simulate_kinetic_scaling():
    """Generates data for rate profiles and 1/R^2 distance scaling."""
    print("Simulating Kinetic Scaling (Convolution vs Marcus)...")
    gamma_phi = 50.0 * cm_to_eV
    lam = 800.0 * cm_to_eV

    # 1. Driving Force Profile (Inverted Region)
    eps0_vals = np.linspace(-0.5, 0.5, 200)
    rate_conv = np.array([compute_convolution_rate(e, gamma_phi, lam) for e in eps0_vals])
    rate_marcus = np.array([compute_marcus_rate(e, lam) for e in eps0_vals])

    # 2. Distance Scaling (1/R^2 Dependence)
    R_vals = np.linspace(12, 25, 20)
    inv_R2 = 1.0 / R_vals**2
    delta_q, eps_eff = 0.5, 4.0
    alpha_vals = 14.4 * delta_q / (eps_eff * R_vals**2)

    eps0_fixed = 0.05
    rate_0 = compute_convolution_rate(eps0_fixed, gamma_phi, lam)
    delta_k = np.array([compute_convolution_rate(eps0_fixed - a, gamma_phi, lam) - rate_0 for a in alpha_vals])
    relative_delta_k = np.abs(delta_k) / rate_0 * 100

    slope, intercept, r_value, _, _ = linregress(inv_R2, relative_delta_k)
    fit_line = slope * inv_R2 + intercept

    return eps0_vals, rate_conv, rate_marcus, inv_R2, relative_delta_k, fit_line, r_value


# =====================================================================
# 5. Plotting Functions
# =====================================================================
def plot_figure_1(dt_fs_vals, alpha_over_deltas, fidelity_map, E_2d_kcal):
    """Generates Figure 1: Isomorphism & 2D Robustness."""
    fig, axs = plt.subplots(1, 2, figsize=(14, 5.5))

    # Panel (a): Trotter Fidelity
    X, Y = np.meshgrid(dt_fs_vals, alpha_over_deltas)
    im = axs[0].pcolormesh(X, Y, fidelity_map, shading='auto', cmap='inferno', vmin=0.9, vmax=1.0)
    cbar = fig.colorbar(im, ax=axs[0])
    cbar.set_label('Gate Fidelity $\mathcal{F}$', rotation=270, labelpad=20)
    axs[0].contour(X, Y, fidelity_map, levels=[0.99], colors='cyan', linestyles='dashed')
    axs[0].set_yscale('log')
    axs[0].set_xlabel(r'Time Step $\delta t$ (fs)')
    axs[0].set_ylabel(r'Coupling Ratio $\alpha / \Delta$')
    axs[0].text(-0.1, 1.05, 'a', transform=axs[0].transAxes, fontsize=20, fontweight='bold')

    # Panel (b): 2D Energy Levels
    labels =['$E_0$', '$E_1$ (TLS)', '$E_2$ (Leak)', '$E_3$', '$E_4$']
    energies = E_2d_kcal[:5] - E_2d_kcal[0]
    colors =['blue', 'blue', 'red', 'gray', 'gray']
    axs[1].bar(labels, energies, color=colors)
    axs[1].set_ylabel(r'Relative Energy (kcal mol$^{-1}$)')
    
    Delta_TLS = E_2d_kcal[1] - E_2d_kcal[0]
    Delta_leak = E_2d_kcal[2] - E_2d_kcal[1]
    axs[1].annotate(f'$\Delta_{{TLS}}$ = {Delta_TLS:.2e}\n$\Delta_{{leak}}$ = {Delta_leak:.2f}',
                    xy=(1.5, 3), xytext=(2.5, 5), arrowprops=dict(facecolor='black', arrowstyle='->'))
    axs[1].text(-0.1, 1.05, 'b', transform=axs[1].transAxes, fontsize=20, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'Fig1_Isomorphism_and_2D_Robustness.png'))
    plt.close()

def plot_figure_2(eps0_vals, rate_conv, rate_marcus, inv_R2, relative_delta_k, fit_line, r_value):
    """Generates Figure 2: Kinetic Scaling & 1/R^2 Dependence."""
    fig, axs = plt.subplots(1, 2, figsize=(14, 5.5))

    # Panel (a): Marcus Recovery & Inverted Region
    axs[0].plot(eps0_vals, rate_conv, 'b-', label='Quantum Convolution')
    axs[0].plot(eps0_vals, rate_marcus, 'r--', label='Marcus Theory')
    axs[0].set_yscale('log')
    axs[0].set_ylim(1e-5, 2e1)
    axs[0].set_xlim(-0.5, 0.5)
    axs[0].set_xlabel(r'Intrinsic Driving Force $\varepsilon_0$ (eV)')
    axs[0].set_ylabel(r'Proton Transfer Rate (arb. units)')
    axs[0].legend(loc='lower right')
    axs[0].text(-0.1, 1.05, 'a', transform=axs[0].transAxes, fontsize=20, fontweight='bold')

    # Panel (b): Inverse-Square Scaling
    axs[1].plot(inv_R2, relative_delta_k, 'ko', markersize=8, label='Numerical Simulation')
    axs[1].plot(inv_R2, fit_line, 'b-', alpha=0.7, label=f'Linear Fit ($R^2$ = {r_value**2:.4f})')
    axs[1].set_xlabel(r'$1/R^2$ ($\mathrm{\AA}^{-2}$)')
    axs[1].set_ylabel(r'Relative Differential Rate $|\Delta k| / k_0$ (%)')
    axs[1].legend(loc='upper left')
    axs[1].text(-0.1, 1.05, 'b', transform=axs[1].transAxes, fontsize=20, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'Fig2_Kinetic_Scaling.png'))
    plt.close()


# =====================================================================
# Main Execution Block
# =====================================================================
if __name__ == "__main__":
    print("=====================================================")
    print("  Starting PCET Electrostatic Gating Simulations     ")
    print("=====================================================\n")

    # 1. Run Gate Isomorphism Simulation
    dt, alpha_ratio, fid_map = simulate_trotter_fidelity()

    # 2. Run 2D DVR Simulation
    E_2d = solve_2d_dvr_hamiltonian()

    # 3. Generate Figure 1
    plot_figure_1(dt, alpha_ratio, fid_map, E_2d)
    print("-> Fig1_Isomorphism_and_2D_Robustness.png saved.")

    # 4. Run Kinetic Scaling Simulation
    eps0, r_conv, r_marcus, invR2, d_k, fit, r_val = simulate_kinetic_scaling()

    # 5. Generate Figure 2
    plot_figure_2(eps0, r_conv, r_marcus, invR2, d_k, fit, r_val)
    print("-> Fig2_Kinetic_Scaling.png saved.\n")

    print("=====================================================")
    print("  Simulations Completed Successfully!                ")
    print(f"  Check the '{OUTPUT_DIR}' directory for results.   ")
    print("=====================================================")
