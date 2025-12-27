import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.linalg import expm, norm

# ==========================================
# 1. Physical Parameters & Setup
# ==========================================
# System Parameters (Atomic Units / Scaled)
DELTA = 1.0       # Tunneling coupling (sigma_x term)
EPSILON_0 = 10.0  # Initial bias (sigma_z term) -> Localization at |L>
ALPHA_RES = 10.0  # Resonant shift amount (Alpha = Epsilon_0)

# Time Grid
DT = 0.05
T_MAX = 15.0
TIMES = np.arange(0, T_MAX, DT)

# Pauli Matrices
SIGMA_X = np.array([[0, 1], [1, 0]])
SIGMA_Z = np.array([[1, 0], [0, -1]])

# ==========================================
# 2. Core Simulation Engine
# ==========================================
def get_hamiltonian(n_val, alpha_val=ALPHA_RES):
    """
    H_EGPT = (epsilon_eff / 2) * sigma_z + Delta * sigma_x
    epsilon_eff = -Epsilon_0 + Alpha * n
    """
    eps_eff = -EPSILON_0 + alpha_val * n_val
    H = (eps_eff / 2.0) * SIGMA_Z + DELTA * SIGMA_X
    return H

def run_simulation(n_func, alpha_val=ALPHA_RES):
    """
    Runs time evolution.
    Returns:
      prob_R: Probability of |R> at each step
      norm_err: Deviation from unitarity (|psi|^2 - 1)
    """
    psi = np.array([1.0, 0.0], dtype=complex) # Start at |L>
    prob_R = []
    norm_err = []

    for t in TIMES:
        n = n_func(t)
        H = get_hamiltonian(n, alpha_val)

        # Exact Propagator for time-independent H over dt
        U = expm(-1j * H * DT)
        psi = U @ psi

        # Track Observables
        p = np.abs(psi[1])**2
        prob_R.append(p)

        # Check Norm Conservation
        current_norm = np.linalg.norm(psi)
        norm_err.append(np.abs(current_norm - 1.0))

        # Re-normalize to prevent float drift accumulation
        psi /= current_norm

    return np.array(prob_R), np.array(norm_err)

# ==========================================
# 3. Define Scenarios
# ==========================================
# Scenario Functions for n(t)
def n_off(t): return 0.0
def n_on(t): return 1.0
def n_pulse(t): return 1.0 if (2.0 < t < 8.0) else 0.0

# Run Simulations
p_off, err_off = run_simulation(n_off)
p_on, err_on = run_simulation(n_on)
p_pulse, err_pulse = run_simulation(n_pulse)

# ==========================================
# 4. Resonance Analysis (Detuning Sweep)
# ==========================================
# Sweep Alpha from 0 to 20. Peak should be at 10.
alpha_range = np.linspace(0, 20, 50)
max_probs = []

for a in alpha_range:
    # Run with static n=1
    probs, _ = run_simulation(n_on, alpha_val=a)
    max_probs.append(np.max(probs))

# ==========================================
# 5. Visualization (Publication Quality)
# ==========================================
sns.set_theme(style="whitegrid", font="serif")
fig = plt.figure(figsize=(14, 10))
gs = fig.add_gridspec(2, 2)

# --- Panel A: Time Dynamics ---
ax1 = fig.add_subplot(gs[0, :])
ax1.plot(TIMES, p_off, label="Gate OFF ($n=0$)", linestyle="--", color="gray", alpha=0.6)
ax1.plot(TIMES, p_on, label="Gate ON ($n=1$, Resonance)", color="crimson", linewidth=2)
ax1.plot(TIMES, p_pulse, label="Pulse Control ($n(t)$)", color="royalblue", linewidth=2.5)
ax1.axvspan(2.0, 8.0, color="yellow", alpha=0.1, label="Electron Presence")

ax1.set_ylabel("Transfer Probability $P_{L\\to R}$", fontsize=12)
ax1.set_xlabel("Time (a.u.)", fontsize=12)
ax1.set_title("(a) EGPT Quantum Dynamics: Resonant Gating", fontsize=14, fontweight="bold")
ax1.legend(loc="upper left")
ax1.set_ylim(-0.05, 1.05)

# --- Panel B: Resonance Condition (Detuning) ---
ax2 = fig.add_subplot(gs[1, 0])
ax2.plot(alpha_range, max_probs, color="darkgreen", linewidth=2, marker="o", markersize=4)
ax2.axvline(EPSILON_0, color="red", linestyle="--", label="Resonance ($\epsilon_0$)")
ax2.set_xlabel(r"Coupling Strength $\alpha$ (a.u.)", fontsize=12)
ax2.set_ylabel(r"Max Transfer Probability", fontsize=12)
ax2.set_title("(b) Resonance Condition Check", fontsize=14, fontweight="bold")
ax2.text(12, 0.8, "Resonance at $\\alpha = \epsilon_0$", fontsize=12, color="red")
ax2.legend()

# --- Panel C: Numerical Stability (Unitarity) ---
ax3 = fig.add_subplot(gs[1, 1])
ax3.plot(TIMES, err_pulse, color="black", linewidth=1.5)
ax3.set_xlabel("Time (a.u.)", fontsize=12)
ax3.set_ylabel("Norm Deviation $||\psi|| - 1$", fontsize=12)
ax3.set_title("(c) Numerical Stability (Unitarity Check)", fontsize=14, fontweight="bold")
ax3.set_yscale("log") # Log scale to show precision
ax3.grid(True, which="both", ls="--")

plt.tight_layout()
plt.savefig("egpt_full_analysis.png", dpi=300)
print("Plot saved: egpt_full_analysis.png")
