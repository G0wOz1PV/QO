import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.linalg import expm

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import UnitaryGate
from qiskit.quantum_info import SparsePauliOp
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import QiskitRuntimeService, Session, SamplerV2

# ==========================================
# 1. Backend and Session Setup
# ==========================================
print("Initializing QiskitRuntimeService...")
service = QiskitRuntimeService()

BACKEND_NAME = "ibm_pittsburgh"

try:
    backend = service.backend(BACKEND_NAME)
    print(f"Backend '{backend.name}' successfully loaded.")
except Exception as e:
    print(f"Failed to access '{BACKEND_NAME}'. Please check your account permissions.\nError: {e}")
    # Fallback to least busy backend if the target is unavailable
    backend = service.least_busy(simulator=False, operational=True, min_num_qubits=2)
    print(f"Falling back to '{backend.name}'.")

# ==========================================
# 2. Hamiltonian and Model Parameters
# ==========================================
Delta = 1.0
epsilon = -2.0
g = 1.0

# Qiskit qubit ordering: q1 = Bath (B), q0 = System (S)
hamiltonian = SparsePauliOp.from_list([
    ("IX", -Delta / 2),
    ("IZ", epsilon / 2),
    ("XZ", g)
])
H_matrix = hamiltonian.to_matrix()

def create_circuit(t, init_B='plus'):
    """Build a circuit for a given evolution time t and bath initial state."""
    qr = QuantumRegister(2, 'q')
    cr = ClassicalRegister(1, 'c')
    qc = QuantumCircuit(qr, cr, name=f"t_{t:.2f}_{init_B}")
    
    # System (q0) initialized in |0>

    # Bath (q1) initialization
    # |+> and |-> are used to construct the classical incoherent mixture (c=0)
    if init_B == 'plus':
        qc.h(qr[1])
    elif init_B == 'minus':
        qc.x(qr[1])
        qc.h(qr[1])
        
    # Apply time-evolution unitary e^{-iHt}
    U = expm(-1j * t * H_matrix)
    qc.append(UnitaryGate(U, label="U_t"), [qr[0], qr[1]])
    
    # Measure System (q0)
    qc.measure(qr[0], cr[0])
    
    return qc

# ==========================================
# 3. Circuit Generation and Transpilation
# ==========================================
t_list = np.linspace(0, 5.0, 41)
circuits = []

for t in t_list:
    circuits.append(create_circuit(t, init_B='plus'))
    circuits.append(create_circuit(t, init_B='minus'))

print("Transpiling circuits for the target backend topology...")
pm = generate_preset_pass_manager(backend=backend, optimization_level=3)
isa_circuits = pm.run(circuits)

# ==========================================
# 4. Job Execution via Session (hardware mode)
# ==========================================
shots = 4000
result = None

print(f"\nOpening a Session on {backend.name} and submitting jobs...")
with Session(backend=backend) as session:
    sampler = SamplerV2(mode=session)
    
    # Sampler options: enable XY4 dynamical decoupling to suppress low-frequency noise
    sampler.options.default_shots = shots
    sampler.options.dynamical_decoupling.enable = True
    sampler.options.dynamical_decoupling.sequence_type = "XY4"
    
    job = sampler.run(isa_circuits)
    print(f">>> Job ID: {job.job_id()} submitted. Waiting for completion...")
    
    result = job.result()

print("All results retrieved. Session closed.\n")

# ==========================================
# 5. Result Analysis and Population Extraction
# ==========================================
prob_plus = []
prob_minus = []

for i in range(len(t_list)):
    counts_plus = result[2*i].data.c.get_counts()
    shots_plus = sum(counts_plus.values())
    p_plus = counts_plus.get('1', 0) / shots_plus
    prob_plus.append(p_plus)
    
    counts_minus = result[2*i+1].data.c.get_counts()
    shots_minus = sum(counts_minus.values())
    p_minus = counts_minus.get('1', 0) / shots_minus
    prob_minus.append(p_minus)

prob_plus = np.array(prob_plus)
prob_minus = np.array(prob_minus)

# Coherent bath (c=1): pure |+> state
P_R_c1 = prob_plus
# Incoherent bath (c=0): equal classical mixture of |+> and |->
P_R_c0 = 0.5 * prob_plus + 0.5 * prob_minus
Delta_P = P_R_c1 - P_R_c0

# Coherence parameter scan for Fig. 3
c_list = np.linspace(0, 1.0, 21)
max_P_list = [np.max(c * P_R_c1 + (1 - c) * P_R_c0) for c in c_list]

# ==========================================
# 6. Print and Save Numerical Results
# ==========================================
os.makedirs('data', exist_ok=True)

df_time = pd.DataFrame({
    'Time (t)': np.round(t_list, 3),
    'P_R (c=0, Incoherent)': np.round(P_R_c0, 4),
    'P_R (c=1, Coherent)': np.round(P_R_c1, 4),
    'Delta P': np.round(Delta_P, 4)
})
df_time.to_csv('data/dynamics_result.csv', index=False)

print("=== Hardware Results (Fig. 1 & Fig. 2: Time Evolution) ===")
print(df_time.to_string(index=False))
print("\n" + "="*50 + "\n")

df_crossover = pd.DataFrame({
    'Coherence (c)': np.round(c_list, 2),
    'Max P_R(t)': np.round(max_P_list, 4)
})
df_crossover.to_csv('data/crossover_result.csv', index=False)

print("=== Hardware Results (Fig. 3: Coherence Scan) ===")
print(df_crossover.to_string(index=False))
print("\n" + "="*50 + "\n")

# ==========================================
# 7. Plotting and Figure Export
# ==========================================
os.makedirs('figures', exist_ok=True)
plt.rcParams.update({'font.size': 12, 'axes.linewidth': 1.2})

# ---- Fig. 1: Population Dynamics ----
plt.figure(figsize=(7, 5))
plt.plot(t_list, P_R_c0, label='Incoherent Bath ($c=0$)', linestyle='--', marker='o', markersize=4, color='blue', linewidth=2.5)
plt.plot(t_list, P_R_c1, label='Coherent Bath ($c=1$)', linestyle='-', marker='s', markersize=4, color='red', linewidth=2.5)
plt.xlabel('Evolution Time $t$', fontsize=14)
plt.ylabel('Target-State Population $P_R(t)$', fontsize=14)
plt.title(f'Population Dynamics on {backend.name}', fontsize=15)
plt.legend(loc='upper left')
plt.grid(True, alpha=0.4)
plt.tight_layout()
plt.savefig('figures/Fig1_population_dynamics_pittsburgh.png', dpi=300)
plt.show()

# ---- Fig. 2: Geometric Witness W(t) ----
plt.figure(figsize=(7, 5))
plt.plot(t_list, Delta_P, color='purple', marker='d', markersize=4, linewidth=2.5)
plt.axhline(0, color='black', linewidth=1.5, linestyle='--')
plt.fill_between(t_list, 0, Delta_P, where=(Delta_P > 0), color='purple', alpha=0.2)
plt.xlabel('Evolution Time $t$', fontsize=14)
plt.ylabel('Enhancement $\Delta P(t)$', fontsize=14)
plt.title('Coherence-Induced Enhancement Witness', fontsize=15)
plt.grid(True, alpha=0.4)
plt.tight_layout()
plt.savefig('figures/Fig2_deltaP_pittsburgh.png', dpi=300)
plt.show()

# ---- Fig. 3: Coherence Parameter Scan ----
plt.figure(figsize=(7, 5))
plt.plot(c_list, max_P_list, marker='o', color='forestgreen', linewidth=2, markersize=8)
plt.xlabel('Bath Coherence Parameter $c$', fontsize=14)
plt.ylabel('Max Target-State Population', fontsize=14)
plt.title('Crossover with Bath Coherence', fontsize=15)
plt.grid(True, alpha=0.4)
plt.tight_layout()
plt.savefig('figures/Fig3_coherence_scan_pittsburgh.png', dpi=300)
plt.show()

print("All figures and data saved. Figures -> 'figures/', Data -> 'data/'")
