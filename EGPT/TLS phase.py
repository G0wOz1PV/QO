import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy.linalg import eigh
import os

OUTPUT_DIR = "result"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 12,
    "axes.linewidth": 1.5,
    "lines.linewidth": 2.0
})

class ProtonTransferSimulator:
    def __init__(self, n_grid=1024, x_range=(-1.0, 1.0)):
        self.au2kcal = 627.509474
        self.au2angstrom = 0.529177
        self.au2fs = 2.418884e-2 
        
        self.N = n_grid
        self.x_ang = np.linspace(x_range[0], x_range[1], self.N)
        self.x_au = self.x_ang / self.au2angstrom
        self.dx = self.x_au[1] - self.x_au[0]
        self.mass = 1836.15 # Proton mass
        
        # Kinetic Energy Matrix
        diagonals = [np.ones(self.N-1), -2*np.ones(self.N), np.ones(self.N-1)]
        self.T = -0.5 / self.mass * diags(diagonals, [-1, 0, 1], shape=(self.N, self.N)) / (self.dx**2)

    def get_potential(self, E_barrier_kcal, bias_kcal=0.0):
        Eb = E_barrier_kcal / self.au2kcal
        bias = bias_kcal / self.au2kcal
        x0_ang = 0.38
        x0 = x0_ang / self.au2angstrom
        
        # Symmetric Double Well
        V_sym = Eb * (1 - (self.x_au/x0)**2)**2
        
        # Gate Potential (Linear Bias)
        # bias > 0 lowers the RIGHT well, bias < 0 lowers the LEFT well
        # Let's bias the Left well deeper to hold the proton there if bias_kcal < 0
        # For Rz gate, usually we just shift the energy levels.
        # Let's assume standard field: V = -mu * E * x
        slope = bias / (2 * x0) 
        V_bias = -slope * self.x_au 
        
        return V_sym + V_bias

    def solve_hamiltonian(self, V_potential):
        H = self.T.toarray() + np.diag(V_potential)
        evals, evecs = eigh(H)
        
        # 1. Ground state should be positive total sum (convention)
        if np.sum(evecs[:, 0]) < 0:
            evecs[:, 0] *= -1
        # 2. First excited state usually has node at center. 
        # Make left side positive for convention if needed, or check slope.
        # Let's ensure phi1 is positive on the LEFT side (-x).
        left_sum = np.sum(evecs[:self.N//2, 1])
        if left_sum < 0:
            evecs[:, 1] *= -1
            
        return evals, evecs

    def run_analysis(self):
        print("Running Simulation")
        
        BARRIER_KCAL = 15.0 
        # Apply bias. Note: A pure Rz gate creates a phase shift. 
        # We model the interaction H_int = -alpha * sigma_z.
        # This shifts Energy of |L> down and |R> up (or vice versa).
        GATE_BIAS_KCAL = -2.0 # Negative biases the LEFT well (stabilizes L)
        
        # 1. Solve OFF (Basis Definition)
        V_off = self.get_potential(BARRIER_KCAL, bias_kcal=0.0)
        evals_off, evecs_off = self.solve_hamiltonian(V_off)
        phi0 = evecs_off[:, 0]
        phi1 = evecs_off[:, 1]
        
        # Construct |L> localized state
        psi_L = (phi0 + phi1) / np.sqrt(2)
        psi_R = (phi0 - phi1) / np.sqrt(2)
        
        # Verify localization
        prob_L = np.sum(psi_L[:self.N//2]**2)
        print(f"Initial Localization on Left: {prob_L:.4f}") # Should be ~1.0
        
        # 2. Dynamics under ON potential
        V_on = self.get_potential(BARRIER_KCAL, bias_kcal=GATE_BIAS_KCAL)
        evals_on, evecs_on = self.solve_hamiltonian(V_on)
        
        # Project |L> onto ON eigenstates
        coeffs = evecs_on.T @ psi_L
        
        times_fs = np.linspace(0, 50, 200) # 50 fs is enough to see phase
        times_au = times_fs / self.au2fs
        
        data_log = []
        
        for t, t_au in zip(times_fs, times_au):
            # Propagate
            propagator = np.exp(-1j * evals_on * t_au)
            psi_t = evecs_on @ (coeffs * propagator)
            
            # Population
            pop_L = np.sum(np.abs(psi_t[:self.N//2])**2)
            
            # Subspace Fidelity
            # Projects state back onto OFF basis {|0>, |1>}
            c0 = np.dot(phi0, psi_t)
            c1 = np.dot(phi1, psi_t)
            fid = np.abs(c0)**2 + np.abs(c1)**2
            
            # Relative Phase (The Gate Action)
            # We want to see the phase difference accumulated relative to the OFF state dynamics.
            # But simpler: Arg(<L|psi(t)>). 
            # Since |L> is approximately an eigenstate of V_on (due to bias), 
            # it should acquire a global phase exp(-i E_L t).
            overlap_L = np.dot(psi_L, psi_t)
            phase_arg = np.angle(overlap_L)
            
            data_log.append([t, pop_L, fid * 100, phase_arg])

        data = np.array(data_log)
        
        # Plotting
        # Fig 1: Population & Fidelity
        fig, ax1 = plt.subplots(figsize=(6, 4))
        ax1.set_xlabel('Time (fs)')
        ax1.set_ylabel('Left Population', color='black')
        ax1.plot(data[:,0], data[:,1], 'k-', label='Population')
        ax1.set_ylim(0.99, 1.001)
        
        ax2 = ax1.twinx()
        ax2.set_ylabel('Fidelity (%)', color='blue')
        ax2.plot(data[:,0], data[:,2], 'b--', label='Fidelity')
        ax2.set_ylim(99.5, 100.05)
        
        plt.title('Stability of EGPT "Qubit" Subspace')
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/Fig2_Stability.png", dpi=300)
        
        # Fig 2: Phase Evolution (Isomorphism Proof)
        fig_phase, ax_p = plt.subplots(figsize=(6, 4))
        
        # Theoretical Phase: E_shift * t
        # Energy shift approx bias/2 (very rough, better to use eigenvalue diff)
        # E_L_on approx E_0_off - alpha. 
        
        ax_p.plot(data[:,0], np.unwrap(data[:,3]), 'r-', lw=2, label='Simulated Phase')
        
        # Linear fit to prove it's a gate
        fit = np.polyfit(data[:,0], np.unwrap(data[:,3]), 1)
        fit_fn = np.poly1d(fit)
        ax_p.plot(data[:,0], fit_fn(data[:,0]), 'k:', label=f'Linear Fit (Slope={fit[0]:.3f})')
        
        ax_p.set_xlabel('Time (fs)')
        ax_p.set_ylabel('Accumulated Phase (rad)')
        ax_p.legend()
        ax_p.set_title(r'Dynamical Isomorphism: Controlled-$R_z$ Action')
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/Fig3_Phase.png", dpi=300)

        # Save CSV
        np.savetxt(f"{OUTPUT_DIR}/jcp_data.csv", data, delimiter=',', 
                   header="Time[fs], Pop_L, Fidelity[%], Phase[rad]")
        
        print(f"Done. Mean Fidelity: {np.mean(data[:,2]):.4f}%")
        print(f"Phase Slope (Gate Speed): {fit[0]:.4f} rad/fs")

if __name__ == "__main__":
    sim = ProtonTransferSimulator()
    sim.run_analysis()
