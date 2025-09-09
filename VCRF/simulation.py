#!pip install --upgrade numpy pandas tqdm matplotlib
#!pip install qiskit qiskit-aer qiskit-ibm-runtime

import os
import math
import json
import time
import uuid
import copy
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional

from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import SparsePauliOp, Pauli
from qiskit.circuit import Parameter
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.primitives import Estimator as LocalEstimator
from qiskit_aer.primitives import Estimator as AerEstimator
from qiskit_aer.noise import NoiseModel

from qiskit_ibm_runtime import QiskitRuntimeService, Estimator as RuntimeEstimator, Session, Options

import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm

matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['figure.dpi'] = 140
matplotlib.rcParams['savefig.dpi'] = 300

os.makedirs("figures", exist_ok=True)
os.makedirs("data", exist_ok=True)

np.random.seed(2025)

print("Environment ready. Qiskit and dependencies loaded.")

HBAR_eVfs = 0.6582119514
KCAL_PER_MOL_PER_EV = 23.0605
KE_E2_kcalA = 332.063
R_gas_kcal = 1.987204e-3
KB_over_h_300K = 6.207e12

def beta_kcal(T=300.0):
    return 1.0/(R_gas_kcal * T)

def alpha_kcal_per_electron(delta_q_A, R_A, eps_r):
    return KE_E2_kcalA * delta_q_A / (eps_r * (R_A**2))

def timegrid(T_fs: float, N: int):
    t = np.linspace(0.0, T_fs, N, endpoint=True)
    return t

def fourier_pulse(t, omegas, amplitudes, phases):
    out = np.zeros_like(t)
    for A, w, phi in zip(amplitudes, omegas, phases):
        out += A * np.cos(w * t + phi)
    return out

def normalize_pulse_energy(t, E, E_budget=1.0):
    dt = np.gradient(t)
    energy = np.sum((E**2) * dt)
    if energy <= 1e-16:
        return E
    scale = math.sqrt(E_budget / energy)
    return E * scale

def xy_coupling_sparse_pauli(i: int, j: int, coeff: float, num_qubits=3) -> SparsePauliOp:
    assert 0 <= i < j < num_qubits
    z_string = ['I'] * num_qubits
    for k in range(i+1, j):
        z_string[k] = 'Z'
    x_string = z_string.copy()
    x_string[i] = 'X'; x_string[j] = 'X'
    y_string = z_string.copy()
    y_string[i] = 'Y'; y_string[j] = 'Y'
    ops = [''.join(reversed(x_string)), ''.join(reversed(y_string))]
    return SparsePauliOp.from_list([(ops[0], coeff/2.0), (ops[1], coeff/2.0)])

def onsite_number_op_sparse_pauli(i: int, coeff: float, num_qubits=3) -> SparsePauliOp:
    z = ['I'] * num_qubits
    z[i] = 'Z'
    I = ['I'] * num_qubits
    op_I = ''.join(reversed(I))
    op_Z = ''.join(reversed(z))
    return SparsePauliOp.from_list([(op_I, coeff/2.0), (op_Z, -coeff/2.0)])

def build_H_sparse_pauli(eps: Dict[str,float],
                         t_hop: Dict[Tuple[str,str],float],
                         drive_ij: Dict[Tuple[str,str],float],
                         onsite_mod: Dict[str,float],
                         label_map: Dict[str,int],
                         num_qubits=3) -> SparsePauliOp:

    H = SparsePauliOp.from_list([('I'*num_qubits, 0.0)])
    for site, e in eps.items():
        i = label_map[site]
        H = H + onsite_number_op_sparse_pauli(i, e, num_qubits)
    for (i_label,j_label), tval in t_hop.items():
        i = label_map[i_label]; j = label_map[j_label]
        if i > j: i, j = j, i
        H = H + xy_coupling_sparse_pauli(i, j, -tval, num_qubits)
    for (i_label,j_label), f in drive_ij.items():
        i = label_map[i_label]; j = label_map[j_label]
        if i > j: i, j = j, i
        H = H + xy_coupling_sparse_pauli(i, j, f, num_qubits)
    for site, de in onsite_mod.items():
        i = label_map[site]
        H = H + onsite_number_op_sparse_pauli(i, de, num_qubits)
    H = H.simplify()
    return H

from qiskit.synthesis import LieTrotter, SuzukiTrotter

def evolution_circuit_for_time_slices(t_grid_fs: np.ndarray,
                                      eps: Dict[str,float],
                                      t_hop: Dict[Tuple[str,str],float],
                                      E_of_t: np.ndarray,
                                      mu_pairs: Dict[Tuple[str,str],float],
                                      onsite_mod: Dict[str,np.ndarray],
                                      label_map: Dict[str,int],
                                      num_qubits=3,
                                      trotter='suzuki',
                                      order=2) -> List[QuantumCircuit]:


    if trotter == 'suzuki':
        synthesis_method = SuzukiTrotter(order=order, reps=1)
    elif trotter == 'lie':
        synthesis_method = LieTrotter(reps=1)
    else:
        synthesis_method = trotter

    circuits = []
    dt_fs = np.diff(t_grid_fs, prepend=0.0)
    times = dt_fs / HBAR_eVfs

    for j in range(len(t_grid_fs)):
        qc = QuantumCircuit(num_qubits)
        qD = label_map['D']
        qc.x(qD)
        for k in range(1, j+1):
            drive_ij = {}
            for (i_label, j_label), mu in mu_pairs.items():
                drive_ij[(i_label, j_label)] = E_of_t[k] * mu
            onsite_k = {}
            for site, arr in onsite_mod.items():
                onsite_k[site] = float(arr[k])
            Hk = build_H_sparse_pauli(eps, t_hop, drive_ij, onsite_k, label_map, num_qubits)
            tk = float(times[k])
            if abs(tk) > 1e-18 and len(Hk.paulis) > 0:
                gate = PauliEvolutionGate(Hk, time=tk, synthesis=synthesis_method)
                qc.append(gate, qc.qubits)
        qc.barrier()
        circuits.append(qc)
    return circuits

def observable_Z(i: int, num_qubits=3) -> SparsePauliOp:
    z = ['I'] * num_qubits
    z[i] = 'Z'
    return SparsePauliOp.from_list([(''.join(reversed(z)), 1.0)])

def observable_XZX(i: int, j: int, num_qubits=3) -> SparsePauliOp:
    assert 0 <= i < j < num_qubits
    z = ['I'] * num_qubits
    for k in range(i+1, j):
        z[k] = 'Z'
    x1 = z.copy()
    x1[i] = 'X'; x1[j] = 'X'
    return SparsePauliOp.from_list([(''.join(reversed(x1)), 1.0)])

def observable_YZY(i: int, j: int, num_qubits=3) -> SparsePauliOp:
    assert 0 <= i < j < num_qubits
    z = ['I'] * num_qubits
    for k in range(i+1, j):
        z[k] = 'Z'
    y1 = z.copy()
    y1[i] = 'Y'; y1[j] = 'Y'
    return SparsePauliOp.from_list([(''.join(reversed(y1)), 1.0)])

def occupancy_from_Zexp(zexp: float) -> float:
    return (1.0 - zexp) / 2.0

def compute_J_and_bounds(t_fs: np.ndarray,
                         n_gate: np.ndarray,
                         alpha_kcal: float,
                         T_kelvin: float = 300.0,
                         kappa_min: float = 0.5,
                         k0_s_inv: Optional[float] = None,
                         weight: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:

    beta = beta_kcal(T_kelvin)
    if weight is None:
        weight = np.ones_like(n_gate)

    dt_fs = np.gradient(t_fs)

    exp_term = np.exp(beta * alpha_kcal * n_gate)
    J_dimless = np.cumsum(exp_term * weight * dt_fs) 
    dt_s = dt_fs * 1e-15
    J_phys = np.cumsum(exp_term * weight * dt_s)
    if k0_s_inv is None:
        k0_s_inv = KB_over_h_300K
    P_lower = 1.0 - np.exp(-kappa_min * k0_s_inv * J_phys)
    P_upper = 1.0 - np.exp(-1.0 * k0_s_inv * J_phys)
    k_inst = k0_s_inv * exp_term
    return {
        "beta": beta,
        "exp_term": exp_term,
        "J_dimless": J_dimless,
        "J_phys": J_phys,
        "P_lower": P_lower,
        "P_upper": P_upper,
        "k_inst": k_inst
    }

def build_estimator(service: Optional[QiskitRuntimeService],
                    backend=None,
                    use_runtime: bool = True,
                    shots: int = 4000,
                    resilience_level: int = 2,
                    optimization_level: int = 1):

    if service is not None and backend is not None and use_runtime:
        options = Options()
        options.resilience_level = resilience_level
        options.optimization_level = optimization_level
        options.execution.shots = shots
        session = Session(service=service, backend=backend)
        estimator = RuntimeEstimator(session=session, options=options)
        mode = "runtime"
        print(f"Using IBM Runtime Estimator on backend {backend.name} with resilience_level={resilience_level}, shots={shots}.")
        return estimator, session, mode
    else:
        # AerEstimator (ideal or noisy)
        estimator = AerEstimator(run_options={"shots": shots})
        session = None
        mode = "aer"
        print(f"Using Aer Estimator with shots={shots}.")
        return estimator, session, mode

def estimate_observables(estimator,
                         circuits: List[QuantumCircuit],
                         observables: List[SparsePauliOp],
                         shots: int = 4000,
                         batch: int = 16) -> Tuple[np.ndarray, np.ndarray]:

    means = []
    stds = []
    for i in range(0, len(circuits), batch):
        cs = circuits[i:i+batch]
        os = observables[i:i+batch]
        job = estimator.run(cs, os, shots=shots)
        res = job.result()
        vals = res.values
        stds_batch = None
        try:
            vars_ = res.metadata.get("variance", None)
            if vars_ is None and hasattr(res, "metadata"):
                vars_ = [md.get("variance", None) for md in res.metadata]
            if vars_ is not None:
                stds_batch = [math.sqrt(max(0.0, v)/shots) for v in vars_]
        except Exception:
            pass
        means.extend(vals)
        if stds_batch is None:
            stds.extend([1.0/math.sqrt(shots)] * len(vals))
        else:
            stds.extend(stds_batch)
    return np.array(means, dtype=float), np.array(stds, dtype=float)

class FourierPulse:
    def __init__(self, omegas: np.ndarray, amplitudes: np.ndarray, phases: np.ndarray,
                 energy_budget: float = 1.0):
        self.omegas = np.array(omegas, dtype=float)
        self.A = np.array(amplitudes, dtype=float)
        self.phi = np.array(phases, dtype=float)
        self.energy_budget = float(energy_budget)

    @property
    def theta(self):
        return np.concatenate([self.A, self.phi])

    @staticmethod
    def from_theta(omegas, theta, energy_budget=1.0):
        K = len(omegas)
        A = np.array(theta[:K], dtype=float)
        phi = np.array(theta[K:2*K], dtype=float)
        return FourierPulse(omegas, A, phi, energy_budget)

    def E(self, t_grid_fs: np.ndarray):
        E_raw = fourier_pulse(t_grid_fs, self.omegas, self.A, self.phi)
        return normalize_pulse_energy(t_grid_fs, E_raw, E_budget=self.energy_budget)

def spsa_optimize(objective_fn, theta0: np.ndarray,
                  maxiter: int = 20,
                  a: float = 0.2,
                  c: float = 0.1,
                  alpha: float = 0.602,
                  gamma: float = 0.101,
                  verbose: bool = True):

    theta = theta0.copy()
    best_theta = theta.copy()
    best_val = -np.inf
    hist = []
    
    for k in range(1, maxiter+1):
        ak = a / (k**alpha)
        ck = c / (k**gamma)
        delta = 2*np.random.binomial(1, 0.5, size=theta.shape) - 1
        thetap = theta + ck * delta
        thetam = theta - ck * delta
        
        fp = objective_fn(thetap)
        fm = objective_fn(thetam)
        ghat = (fp - fm) / (2.0 * ck * delta)
        theta = theta + ak * ghat
        val = objective_fn(theta)
        hist.append((k, val))
        
        if val > best_val:
            best_val = val
            best_theta = theta.copy()
        
        if verbose and (k % max(1, maxiter//5) == 0):
            print(f"Iter {k:3d} | J_dimless ~ {val:.6f} (best {best_val:.6f})")
    
    return best_theta, hist

def get_ibm_service(verbose=False):
    """Helper function to get IBM Quantum service."""
    try:
        token = os.environ.get("IBM_QUANTUM_TOKEN")
        if not token:
            if verbose:
                print("No IBM_QUANTUM_TOKEN found. Using simulator mode.")
            return None
        service = QiskitRuntimeService(token=token)
        if verbose:
            print("IBM Quantum service initialized successfully.")
        return service
    except Exception as e:
        if verbose:
            print(f"Failed to initialize IBM service: {e}")
        return None

def choose_backend(service, n_qubits=3, simulator_ok=True):
    """Helper function to choose an appropriate backend."""
    if service is None:
        return None
    
    try:
        # Try to get available backends
        backends = service.backends()
        
        # Filter backends that have enough qubits and are operational
        suitable_backends = []
        for backend in backends:
            if hasattr(backend, 'num_qubits') and backend.num_qubits >= n_qubits:
                if hasattr(backend, 'status') and backend.status().operational:
                    suitable_backends.append(backend)
        
        if suitable_backends:
            # Choose the backend with least pending jobs
            backend = min(suitable_backends, key=lambda b: b.status().pending_jobs)
            print(f"Selected backend: {backend.name}")
            return backend
        elif simulator_ok:
            # Fallback to simulator if available
            try:
                simulator = service.backend('ibmq_qasm_simulator')
                print("Using IBMQ QASM Simulator")
                return simulator
            except:
                pass
        
        print("No suitable backend found")
        return None
        
    except Exception as e:
        print(f"Error selecting backend: {e}")
        return None

def run_vcrf_dba_experiment(run_on_ibm: bool = True,
                            ibm_shots: int = 4000,
                            runtime_resilience: int = 2,
                            total_time_fs: float = 100.0,
                            N_time: int = 16,
                            trotter_order: int = 2,
                            T_kelvin: float = 300.0,
                            dq_A: float = 0.30,
                            R_A: float = 6.0,
                            eps_r: float = 6.0,
                            kappa_min: float = 0.5,
                            k0_s_inv: Optional[float] = None,
                            do_optimization: bool = True,
                            opt_iters: int = 20):
    
    print("Starting VCRF-DBA experiment...")
    
    # System setup
    label_map = {'D':0, 'B':1, 'A':2}
    num_qubits = 3
    eps = {'D': 0.00, 'B': 0.10, 'A': -0.05}
    t_hop = {('D','B'): 0.05, ('B','A'): 0.05, ('D','A'): 0.005}
    mu_pairs = {('D','B'): 0.03, ('B','A'): 0.03, ('D','A'): 0.01}
    onsite_mod = {'D': np.zeros(N_time), 'B': np.zeros(N_time), 'A': np.zeros(N_time)}
    
    # Time grid and initial pulse
    t_grid_fs = timegrid(total_time_fs, N_time)
    omegas = np.array([0.05, 0.08]) * (2*np.pi)
    A0 = np.array([0.8, 0.5])
    phi0 = np.array([0.0, np.pi/4])
    pulse = FourierPulse(omegas, A0, phi0, energy_budget=1.0)
    E_t = pulse.E(t_grid_fs)
    
    # Backend setup
    service = get_ibm_service(verbose=True) if run_on_ibm else None
    backend = choose_backend(service, n_qubits=num_qubits, simulator_ok=True) if service is not None else None
    estimator, session, mode = build_estimator(
        service, backend, 
        use_runtime=(service is not None and backend is not None and run_on_ibm),
        shots=ibm_shots, resilience_level=runtime_resilience, optimization_level=1
    )
    
    # Generate quantum circuits
    circuits = evolution_circuit_for_time_slices(
        t_grid_fs, eps, t_hop, E_t, mu_pairs, onsite_mod, label_map,
        num_qubits=num_qubits, trotter='suzuki', order=trotter_order
    )
    
    # Measure observables
    ZA = observable_Z(label_map['A'], num_qubits=num_qubits)
    observables = [ZA] * len(circuits)
    means_Z, std_Z = estimate_observables(estimator, circuits, observables, shots=ibm_shots, batch=8)
    
    # Process results
    nA = occupancy_from_Zexp(means_Z)
    nA_err = 0.5 * std_Z
    alpha = alpha_kcal_per_electron(dq_A, R_A, eps_r)
    egpt = compute_J_and_bounds(t_grid_fs, nA, alpha, T_kelvin, kappa_min, k0_s_inv, weight=None)
    
    # Save baseline data
    df = pd.DataFrame({
        "time_fs": t_grid_fs,
        "E_t": E_t,
        "Z_A": means_Z,
        "n_A": nA,
        "n_A_err": nA_err,
        "exp_beta_alpha_n": egpt["exp_term"],
        "J_dimless": egpt["J_dimless"],
        "J_phys_s": egpt["J_phys"],
        "P_lower": egpt["P_lower"],
        "P_upper": egpt["P_upper"],
        "k_inst_s-1": egpt["k_inst"]
    })
    
    baseline_csv = f"data/DBA_baseline_{mode}.csv"
    df.to_csv(baseline_csv, index=False)
    print(f"Saved baseline data to {baseline_csv}")
    
    # Generate plots
    plt.figure(figsize=(6,3.2))
    plt.plot(t_grid_fs, E_t, 'k-', lw=2)
    plt.xlabel("Time (fs)")
    plt.ylabel("Field amplitude (arb.)")
    plt.title("Virtual optical pulse (Fourier-shaped)")
    plt.grid(True, alpha=0.3)
    fig1 = "figures/fig1_pulse.png"
    plt.tight_layout(); plt.savefig(fig1)
    print(f"Saved {fig1}")
    
    plt.figure(figsize=(6,3.2))
    plt.errorbar(t_grid_fs, nA, yerr=nA_err, fmt='o', color='tab:blue', capsize=3, label=f"{mode} data")
    plt.xlabel("Time (fs)")
    plt.ylabel("Occupancy n_A(t)")
    plt.title("Real-time gate occupancy at the acceptor site")
    plt.grid(True, alpha=0.3)
    plt.legend()
    fig2 = "figures/fig2_nA.png"
    plt.tight_layout(); plt.savefig(fig2)
    print(f"Saved {fig2}")
    
    fig, ax = plt.subplots(1,2, figsize=(10,3.2), sharex=False)
    ax[0].plot(t_grid_fs, egpt["exp_term"], 'r-', lw=2)
    ax[0].set_xlabel("Time (fs)")
    ax[0].set_ylabel("exp(β α n_A(t))")
    ax[0].set_title("Exponential gain (EGPT mapping)")
    ax[0].grid(True, alpha=0.3)
    ax[1].plot(t_grid_fs, egpt["J_dimless"], 'g-', lw=2)
    ax[1].set_xlabel("Time (fs)")
    ax[1].set_ylabel("Cumulative J (dimensionless)")
    ax[1].set_title("Monotone objective for reaction probability")
    ax[1].grid(True, alpha=0.3)
    fig3 = "figures/fig3_expJ.png"
    plt.tight_layout(); plt.savefig(fig3)
    print(f"Saved {fig3}")
    
    plt.figure(figsize=(6,3.2))
    plt.fill_between(t_grid_fs, egpt["P_lower"], egpt["P_upper"], color='tab:orange', alpha=0.4, label="Bounds")
    plt.plot(t_grid_fs, egpt["P_lower"], 'tab:orange', lw=2, label="Lower")
    plt.plot(t_grid_fs, egpt["P_upper"], 'tab:orange', lw=2, linestyle='--', label="Upper")
    plt.xlabel("Time (fs)")
    plt.ylabel("Reaction probability bound")
    plt.title("EGPT-based bounds on reaction probability")
    plt.grid(True, alpha=0.3); plt.legend()
    fig4 = "figures/fig4_prob_bounds.png"
    plt.tight_layout(); plt.savefig(fig4)
    print(f"Saved {fig4}")
    
    # Store results
    results = {
        "mode": mode,
        "t_grid_fs": t_grid_fs,
        "E_t": E_t,
        "nA": nA, "nA_err": nA_err,
        "alpha_kcal": alpha,
        "egpt": egpt,
        "baseline_csv": baseline_csv,
        "figs": [fig1, fig2, fig3, fig4],
    }
    
    # Optimization phase
    if do_optimization:
        print("\nStarting SPSA pulse optimization to maximize J_dimless(T) (simulator mode is faster).")
        K = len(omegas)
        theta0 = np.concatenate([A0, phi0])
        
        def objective(theta):
            pulse_opt = FourierPulse.from_theta(omegas, theta, energy_budget=1.0)
            Eopt = pulse_opt.E(t_grid_fs)
            circs = evolution_circuit_for_time_slices(
                t_grid_fs, eps, t_hop, Eopt, mu_pairs, onsite_mod, label_map,
                num_qubits=num_qubits, trotter='suzuki', order=trotter_order
            )
            means_Z_, std_Z_ = estimate_observables(estimator, circs, [ZA]*len(circs), shots=ibm_shots, batch=8)
            nA_ = occupancy_from_Zexp(means_Z_)
            egpt_ = compute_J_and_bounds(t_grid_fs, nA_, alpha, T_kelvin, kappa_min, k0_s_inv)
            return egpt_["J_dimless"][-1]
        
        best_theta, hist = spsa_optimize(objective, theta0, maxiter=opt_iters, a=0.3, c=0.2, alpha=0.602, gamma=0.101, verbose=True)
        
        # Evaluate optimized pulse
        pulse_best = FourierPulse.from_theta(omegas, best_theta, energy_budget=1.0)
        E_best = pulse_best.E(t_grid_fs)
        circs_best = evolution_circuit_for_time_slices(
            t_grid_fs, eps, t_hop, E_best, mu_pairs, onsite_mod, label_map,
            num_qubits=num_qubits, trotter='suzuki', order=trotter_order
        )
        means_Zb, std_Zb = estimate_observables(estimator, circs_best, [ZA]*len(circs_best), shots=ibm_shots, batch=8)
        nA_best = occupancy_from_Zexp(means_Zb)
        egpt_best = compute_J_and_bounds(t_grid_fs, nA_best, alpha, T_kelvin, kappa_min, k0_s_inv)
        
        # Save optimization data
        df_opt = pd.DataFrame({"iter": [k for k,_ in hist], "J_dimless": [v for _,v in hist]})
        out_csv = f"data/DBA_optimization_{mode}.csv"
        df_opt.to_csv(out_csv, index=False)
        print(f"Saved optimization trace to {out_csv}")
        
        # Optimization plots
        plt.figure(figsize=(6,3.2))
        plt.plot(t_grid_fs, E_t, 'k-', lw=2, label="Initial")
        plt.plot(t_grid_fs, E_best, 'r--', lw=2, label="Optimized")
        plt.xlabel("Time (fs)"); plt.ylabel("Field amplitude (arb.)")
        plt.title("Pulse shaping for electron-gated proton transfer")
        plt.grid(True, alpha=0.3); plt.legend()
        fig5 = "figures/fig5_pulse_optimization.png"
        plt.tight_layout(); plt.savefig(fig5); print(f"Saved {fig5}")
        
        plt.figure(figsize=(6,3.2))
        plt.plot(t_grid_fs, nA, 'b-', lw=2, label="Initial")
        plt.plot(t_grid_fs, nA_best, 'r--', lw=2, label="Optimized")
        plt.xlabel("Time (fs)"); plt.ylabel("Occupancy n_A(t)")
        plt.title("Acceptor occupancy: initial vs optimized")
        plt.grid(True, alpha=0.3); plt.legend()
        fig6 = "figures/fig6_nA_optimization.png"
        plt.tight_layout(); plt.savefig(fig6); print(f"Saved {fig6}")
        
        plt.figure(figsize=(6,3.2))
        plt.plot(t_grid_fs, egpt["J_dimless"], 'g-', lw=2, label="Initial")
        plt.plot(t_grid_fs, egpt_best["J_dimless"], 'm--', lw=2, label="Optimized")
        plt.xlabel("Time (fs)"); plt.ylabel("Cumulative J (dimensionless)")
        plt.title("Objective growth under optimized control")
        plt.grid(True, alpha=0.3); plt.legend()
        fig7 = "figures/fig7_J_optimization.png"
        plt.tight_layout(); plt.savefig(fig7); print(f"Saved {fig7}")
        
        results.update({
            "opt": {
                "theta_best": best_theta,
                "hist": hist,
                "E_best": E_best,
                "nA_best": nA_best,
                "egpt_best": egpt_best,
                "opt_csv": out_csv,
                "figs": [fig5, fig6, fig7]
            }
        })
    
    # Clean up
    if session is not None:
        session.close()
    
    print("\nNarrative: Virtual coherent Raman field-driven DBA experiment completed successfully.")
    return results

def fold_circuit(circ: QuantumCircuit, level: int = 1) -> QuantumCircuit:
    """Apply circuit folding for noise mitigation."""
    qc = circ.copy()
    for _ in range(level):
        qc = qc.compose(circ.inverse())
        qc = qc.compose(circ)
    return qc

# Main execution
print("Running VCRF-DBA experiment...")

RUN_ON_IBM = bool(os.environ.get("IBM_QUANTUM_TOKEN", ""))

res = run_vcrf_dba_experiment(
    run_on_ibm=RUN_ON_IBM,
    ibm_shots=4000 if RUN_ON_IBM else 2000,
    runtime_resilience=2,
    total_time_fs=100.0,
    N_time=12,            # Increase for higher time resolution
    trotter_order=2,
    T_kelvin=300.0,
    dq_A=0.30, R_A=6.0, eps_r=6.0,
    kappa_min=0.5,
    k0_s_inv=None,        # use k_B T / h scale
    do_optimization=True,
    opt_iters=10 if RUN_ON_IBM else 20  # fewer iters on real hw
)

print("\nDone. Inspect 'figures/*.png' and 'data/*.csv'.")
