import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm, fractional_matrix_power
import seaborn as sns

# --- Physical Parameters ---
Delta_p = 1.0
Epsilon_p = 10.0
Alpha = 10.0        # Resonance
dt = 0.05
steps = 400
times = np.arange(0, steps * dt, dt)

# --- Operators ---
I = np.eye(2)
Sz = np.array([[1, 0], [0, -1]])
Sx = np.array([[0, 1], [1, 0]])
P0 = np.array([[1, 0], [0, 0]])
P1 = np.array([[0, 0], [0, 1]])

# Quantum Hamiltonian (Closed)
H_quant = np.kron(I, (Epsilon_p/2)*Sz + Delta_p*Sx) - (Alpha/2)*np.kron(P1, Sz)

# --- BLP Non-Markovianity Measure ---
# Breuer-Laine-Piilo (BLP) measure:
# Detect increase in trace distance between two evolved states rho1(t), rho2(t).
# Increase = Information Backflow = Non-Markovianity.

def trace_distance(rho1, rho2):
    diff = rho1 - rho2
    # Trace distance = 0.5 * Tr|rho1 - rho2| = 0.5 * sum(|eigenvalues|)
    evals = np.linalg.eigvalsh(diff)
    return 0.5 * np.sum(np.abs(evals))

def get_reduced_rho_p(psi_full):
    # Partial trace over electron (qubit 0)
    # psi = [c0L, c0R, c1L, c1R]
    # rho_p = [[ |c0L|^2+|c1L|^2, c0L*c0R* + c1L*c1R* ], ... ]
    # Easier: reshape to (2, 2) [electron, proton]
    psi_mat = psi_full.reshape(2, 2)
    # rho_p = psi_mat.T @ conj(psi_mat)
    rho_p = np.dot(psi_mat.T, np.conj(psi_mat))
    return rho_p

def simulate_quantum_trace_distance():
    # Initial Pair of Proton States (Maximal Distinguishability)
    # State 1: Proton at |L>
    # State 2: Proton at |R>
    # Electron state is same for both: (|0>+|1>)/sqrt(2)

    psi_e = np.array([1, 1]) / np.sqrt(2)
    psi_p1 = np.array([1, 0]) # |L>
    psi_p2 = np.array([0, 1]) # |R>

    psi1 = np.kron(psi_e, psi_p1)
    psi2 = np.kron(psi_e, psi_p2)

    dist_list = []

    U = expm(-1j * H_quant * dt)

    for t in times:
        rho_p1 = get_reduced_rho_p(psi1)
        rho_p2 = get_reduced_rho_p(psi2)

        d = trace_distance(rho_p1, rho_p2)
        dist_list.append(d)

        psi1 = U @ psi1
        psi2 = U @ psi2

    return np.array(dist_list)

def simulate_classical_trace_distance():
    # Classical Noise: Markovian Dephasing Limit
    # Proton moves under H_avg but suffers dephasing due to field fluctuations
    # Master Eq: d_rho/dt = -i[H_avg, rho] - gamma * (sz rho sz - rho) ...
    # Phenomenological decay

    # Simple approx: Average Dynamics dampens off-diagonal terms
    # But noise kills coherences.

    # Simulating ensemble average
    dist_list = []

    # Ensemble of trajectories
    n_traj = 500

    # Initial ensembles
    # 1: Proton L, 2: Proton R
    # Electron: Random 0 or 1 (50/50)

    psis1 = []
    psis2 = []
    for _ in range(n_traj):
        state_e = 0 if np.random.rand() > 0.5 else 1
        # Start 1: |L>, Start 2: |R>
        psis1.append((state_e, np.array([1.0, 0.0], dtype=complex)))
        psis2.append((state_e, np.array([0.0, 1.0], dtype=complex)))

    for t in times:
        # Calc ensemble Rho
        rho1 = np.zeros((2,2), dtype=complex)
        rho2 = np.zeros((2,2), dtype=complex)

        for i in range(n_traj):
            rho1 += np.outer(psis1[i][1], np.conj(psis1[i][1]))
            rho2 += np.outer(psis2[i][1], np.conj(psis2[i][1]))

        rho1 /= n_traj
        rho2 /= n_traj

        dist_list.append(trace_distance(rho1, rho2))

        # Evolve
        for i in range(n_traj):
            # Stochastic switching
            if np.random.rand() < 0.1: # Switching rate
                psis1[i] = (1 - psis1[i][0], psis1[i][1])
                psis2[i] = (1 - psis2[i][0], psis2[i][1])

            n = float(psis1[i][0]) # same for both
            H = ((Epsilon_p - Alpha * n)/2)*Sz + Delta_p*Sx
            U = expm(-1j * H * dt)

            psis1[i] = (psis1[i][0], U @ psis1[i][1])
            psis2[i] = (psis2[i][0], U @ psis2[i][1])

    return np.array(dist_list)

# --- Execution ---
td_quant = simulate_quantum_trace_distance()
td_class = simulate_classical_trace_distance()

# --- Plotting ---
sns.set_theme(style='whitegrid', font='serif')

plt.figure(figsize=(8, 5))
plt.plot(times, td_quant, label='Quantum EGPT (Non-Markovian)', color='crimson', linewidth=2)
plt.plot(times, td_class, label='Classical Noise (Markovian-like)', color='gray', linestyle='--', linewidth=2)

plt.xlabel('Time (a.u.)', fontsize=12)
plt.ylabel(r'Trace Distance $D(\rho_1(t), \rho_2(t))$', fontsize=12)

plt.title('Quantum vs. Classical Trace Distance Over Time', fontsize=14)

plt.legend(fontsize=10)

plt.ylim(0, 1.05)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("trace_distance_comparison.png", dpi=300)
print("Plot saved.")
