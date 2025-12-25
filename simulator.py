import numpy as np
import numba
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.sparse.linalg import spsolve

PHYSICAL_CONSTANTS = {
    'T_kelvin': 310.15, 'R': 8.31446, 'F': 96485.33,
    'mu_Na': 5.19e-8 * 1e6, 'mu_K': 7.62e-8 * 1e6, 'mu_Cl': 7.91e-8 * 1e6,
    'RT_over_F': (8.31446 * 310.15 / 96485.33) * 1e3,
    'z_Na': 1, 'z_K': 1, 'z_Cl': -1,
    'epsilon_0': 8.854e-12, 'epsilon_r_water': 81.0,
    'epsilon_r_membrane': 2.0, 'D_membrane_factor': 1e-6,
    'c0_Na_in': 14.0, 'c0_K_in': 140.0, 'c0_Cl_in': 154.0,
    'c0_Na_ext': 160.0, 'c0_K_ext': 4.0, 'c0_Cl_ext': 164.0,
    'V_rest': -66.5, 'C_m': 1.0,
    'dx': 0.5,
    'dt': 1e-4,
}

PHYSICAL_CONSTANTS['epsilon_water'] = PHYSICAL_CONSTANTS['epsilon_0'] * PHYSICAL_CONSTANTS['epsilon_r_water'] * 1e-6
PHYSICAL_CONSTANTS['epsilon_membrane'] = PHYSICAL_CONSTANTS['epsilon_0'] * PHYSICAL_CONSTANTS['epsilon_r_membrane'] * 1e-6
PHYSICAL_CONSTANTS['D_Na'] = PHYSICAL_CONSTANTS['mu_Na'] * PHYSICAL_CONSTANTS['RT_over_F']
PHYSICAL_CONSTANTS['D_K'] = PHYSICAL_CONSTANTS['mu_K'] * PHYSICAL_CONSTANTS['RT_over_F']
PHYSICAL_CONSTANTS['D_Cl'] = PHYSICAL_CONSTANTS['mu_Cl'] * PHYSICAL_CONSTANTS['RT_over_F']

class FastPoissonSolver:
    def __init__(self, nx, ny, dx, epsilon_map, inside_mask, outside_mask):
        self.nx, self.ny, self.dx = nx, ny, dx
        self.inside_mask, self.outside_mask = inside_mask.flatten(), outside_mask.flatten()
        self.fixed_nodes = np.logical_or(self.inside_mask, self.outside_mask)
        self.free_nodes = ~self.fixed_nodes

        N = nx * ny
        diagonals = [-4.0*np.ones(N), np.ones(N-1), np.ones(N-1), np.ones(N-ny), np.ones(N-ny)]
        diagonals[1][np.arange(1,N)%ny==0]=0; diagonals[2][np.arange(1,N)%ny==0]=0
        offsets = [0, 1, -1, ny, -ny]
        self.A = sp.diags(diagonals, offsets, shape=(N,N), format='csr')
        self.A_free = self.A[self.free_nodes][:, self.free_nodes]
        self.solver = splinalg.factorized(self.A_free)

    def solve(self, rho, epsilon_map, Vm):
        rhs = -rho.flatten() * (self.dx**2) / epsilon_map.flatten()
        phi_full = np.zeros(self.nx*self.ny)
        phi_full[self.inside_mask] = Vm
        b = rhs - self.A.dot(phi_full)
        phi_full[self.free_nodes] = self.solver(b[self.free_nodes])
        return phi_full.reshape(self.nx, self.ny)


@numba.jit(nopython=True)
def _update_concentration_numba(c, flux_x, flux_y, dt, dx, nx, ny):
    """농도 업데이트 (NaN 방지)"""
    padded_flux_x = np.zeros((nx + 1, ny))
    padded_flux_x[1:-1, :] = flux_x[:-1, :]
    padded_flux_y = np.zeros((nx, ny + 1))
    padded_flux_y[:, 1:-1] = flux_y[:, :-1]

    div_flux = (
        (padded_flux_x[1:, :] - padded_flux_x[:-1, :]) / dx +
        (padded_flux_y[:, 1:] - padded_flux_y[:, :-1]) / dx
    )

    c_new = c - dt * div_flux

    # Safety Clamp
    for i in range(nx):
        for j in range(ny):
            val = c_new[i, j]
            if np.isnan(val): c_new[i, j] = 1e-9
            elif val < 1e-9: c_new[i, j] = 1e-9
            elif val > 10000.0: c_new[i, j] = 10000.0
    return c_new

@numba.jit(nopython=True)
def _calculate_flux_sg(nx, ny, dx, D_map, z, c, phi, RT_over_F):
    """Scharfetter-Gummel Flux (Exp 폭발 방지)"""
    flux_x = np.zeros((nx, ny)); flux_y = np.zeros((nx, ny))
    MAX_EXP = 50.0

    for i in numba.prange(nx): # Y-flux
        dV = z * (phi[i, 1:] - phi[i, :-1]) / RT_over_F
        dV = np.maximum(np.minimum(dV, MAX_EXP), -MAX_EXP)
        edV = np.exp(dV)
        B_p = np.empty_like(dV); B_m = np.empty_like(dV)
        for k in range(len(dV)):
            if abs(dV[k]) < 1e-6: B_p[k] = 1 - 0.5 * dV[k]; B_m[k] = 1 + 0.5 * dV[k]
            else: B_p[k] = dV[k] / (edV[k] - 1); B_m[k] = dV[k] * edV[k] / (edV[k] - 1)
        D_int = 2 * D_map[i, 1:] * D_map[i, :-1] / (D_map[i, 1:] + D_map[i, :-1] + 1e-9)
        flux_y[i, :-1] = -(D_int / dx) * (c[i, 1:] * B_p - c[i, :-1] * B_m)

    for j in numba.prange(ny): # X-flux
        dV = z * (phi[1:, j] - phi[:-1, j]) / RT_over_F
        dV = np.maximum(np.minimum(dV, MAX_EXP), -MAX_EXP)
        edV = np.exp(dV)
        B_p = np.empty_like(dV); B_m = np.empty_like(dV)
        for k in range(len(dV)):
            if abs(dV[k]) < 1e-6: B_p[k] = 1 - 0.5 * dV[k]; B_m[k] = 1 + 0.5 * dV[k]
            else: B_p[k] = dV[k] / (edV[k] - 1); B_m[k] = dV[k] * edV[k] / (edV[k] - 1)
        D_int = 2 * D_map[1:, j] * D_map[:-1, j] / (D_map[1:, j] + D_map[:-1, j] + 1e-9)
        flux_x[:-1, j] = -(D_int / dx) * (c[1:, j] * B_p - c[:-1, j] * B_m)

    return flux_x, flux_y

class PNPSimulatorAdaptiveLight:
    def __init__(self, nx=32, ny=32, membrane_thickness_dx=2):
        self.nx, self.ny = nx, ny
        self.constants = PHYSICAL_CONSTANTS.copy()
        self.dx = self.constants['dx']

        self.ions = ['Na', 'K', 'Cl']
        self.D_bulk = {ion: self.constants[f'D_{ion}'] for ion in self.ions}
        self.z = {ion: self.constants[f'z_{ion}'] for ion in self.ions}
        self.F = self.constants['F']
        self.RT_over_F = self.constants['RT_over_F']
        self.charge_conv = self.F * 1e-18
        self.flux_conv = 10.0 / self.F

        self.mem_thick_dx = membrane_thickness_dx
        self.mem_start_y = ny // 2 - membrane_thickness_dx // 2
        self.mem_end_y = ny // 2 + membrane_thickness_dx // 2

        self.Vm = self.constants['V_rest']
        self._init_fields()

        self.fast_solver = FastPoissonSolver(nx, ny, self.dx, self.epsilon_map, self.inside_mask, self.outside_mask)

        self.m, self.h, self.n = self._get_gates(self.Vm)
        self.g_bar_Na, self.g_bar_K = 120.0, 20.0

        g_leak_total = 0.2
        self.g_leak_K = g_leak_total * 1.0
        self.g_leak_Na = g_leak_total * 0.05
        self.g_leak_Cl = g_leak_total * 0.45

        self.I_max_pump = 4.0
        self.K_m_Na = 15.0; self.K_m_K = 1.5

    def _init_fields(self):
        self.c = {}; self.flux_x = {}; self.flux_y = {}
        for ion in self.ions:
            self.c[ion] = np.zeros((self.nx, self.ny))
            self.flux_x[ion] = np.zeros((self.nx, self.ny))
            self.flux_y[ion] = np.zeros((self.nx, self.ny))

        self.inside_mask = np.zeros((self.nx, self.ny), dtype=bool)
        self.outside_mask = np.zeros((self.nx, self.ny), dtype=bool)
        self.inside_mask[:, :self.mem_start_y] = True
        self.outside_mask[:, self.mem_end_y:] = True

        self.membrane_mask = np.zeros((self.nx, self.ny), dtype=bool)
        self.membrane_mask[:, self.mem_start_y:self.mem_end_y] = True

        self.epsilon_map = np.full((self.nx, self.ny), self.constants['epsilon_water'])
        self.epsilon_map[self.membrane_mask] = self.constants['epsilon_membrane']

        self.D_map = {}
        for ion in self.ions:
            self.D_map[ion] = np.full((self.nx, self.ny), self.D_bulk[ion])
            self.D_map[ion][self.membrane_mask] = 1e-12

        for ion in self.ions:
            c_in = self.constants[f'c0_{ion}_in']
            c_ext = self.constants[f'c0_{ion}_ext']
            self.c[ion][self.inside_mask] = c_in
            self.c[ion][self.outside_mask] = c_ext
            for j in range(self.mem_start_y, self.mem_end_y):
                r = (j - self.mem_start_y) / self.mem_thick_dx
                self.c[ion][:, j] = c_in*(1-r) + c_ext*r

        self.phi = np.zeros((self.nx, self.ny))
        self._apply_bc(self.Vm)

    def _apply_bc(self, Vm):
        self.phi[self.inside_mask] = Vm
        self.phi[self.outside_mask] = 0.0

    def _get_gates(self, V):
        an = 0.01*(V+55)/(1-np.exp(-(V+55)/10)) if abs(V+55)>1e-5 else 0.1
        bn = 0.125*np.exp(-(V+65)/80)
        am = 0.1*(V+40)/(1-np.exp(-(V+40)/10)) if abs(V+40)>1e-5 else 1.0
        bm = 4.0*np.exp(-(V+65)/18)
        ah = 0.07*np.exp(-(V+65)/20)
        bh = 1.0/(1+np.exp(-(V+35)/10))
        return an/(an+bn), am/(am+bm), ah/(ah+bh)

    def step(self, dt, ext_current=0.0):
        # 1. HH Dynamics
        if not hasattr(self, 'E'): self.E = {}
        for ion in ['Na', 'K', 'Cl']:
            cin = np.mean(self.c[ion][self.inside_mask])
            cout = np.mean(self.c[ion][self.outside_mask])
            cin = max(cin, 0.1); cout = max(cout, 0.1)
            self.E[ion] = self.RT_over_F/self.z[ion] * np.log(cout/cin)

        V = np.clip(self.Vm, -200, 200)

        def get_ab(v, mode):
            if mode=='n':
                a = 0.01*(v+55)/(1-np.exp(-(v+55)/10)) if abs(v+55)>1e-5 else 0.1
                b = 0.125*np.exp(-(v+65)/80)
                return a, b
            elif mode=='m':
                a = 0.1*(v+40)/(1-np.exp(-(v+40)/10)) if abs(v+40)>1e-5 else 1.0
                b = 4.0*np.exp(-(v+65)/18)
                return a, b
            elif mode=='h':
                a = 0.07*np.exp(-(v+65)/20)
                b = 1.0/(1+np.exp(-(v+35)/10))
                return a, b

        for gate, mode in [('n', 'n'), ('m', 'm'), ('h', 'h')]:
            a, b = get_ab(V, mode)
            tau = 1.0 / (a + b)
            inf = a / (a + b)
            curr_val = getattr(self, gate)
            setattr(self, gate, inf + (curr_val - inf) * np.exp(-dt / tau))

        I_Na = self.g_bar_Na * (self.m**3) * self.h * (self.Vm - self.E['Na'])
        I_K  = self.g_bar_K  * (self.n**4) * (self.Vm - self.E['K'])
        I_L  = self.g_leak_Na*(self.Vm-self.E['Na']) + \
               self.g_leak_K *(self.Vm-self.E['K']) + \
               self.g_leak_Cl*(self.Vm-self.E['Cl'])

        c_Na_in = max(np.mean(self.c['Na'][self.inside_mask]), 0.5)
        c_K_ext = max(np.mean(self.c['K'][self.outside_mask]), 0.5)
        pNa = 1/(1+(self.K_m_Na/c_Na_in)**2)
        pK = 1/(1+(self.K_m_K/c_K_ext)**2)
        I_pump_Na = self.I_max_pump * pNa * pK
        I_pump_K = -(2/3)*I_pump_Na

        I_stim = ext_current
        I_total = I_Na + I_K + I_L + I_pump_Na + I_pump_K

        self.Vm += (I_stim - I_total) / self.constants['C_m'] * dt

        # 2. PNP Update
        J_total = {
            'Na': I_Na + self.g_leak_Na*(self.Vm-self.E['Na']) + I_pump_Na,
            'K':  I_K  + self.g_leak_K *(self.Vm-self.E['K']) + I_pump_K,
            'Cl': I_L  + self.g_leak_Cl*(self.Vm-self.E['Cl'])
        }
        inv_vol_factor = 5e-4

        for ion in self.ions:
            # (1) Flux 계산 및 이동
            self.flux_x[ion], self.flux_y[ion] = _calculate_flux_sg(
                self.nx, self.ny, self.dx, self.D_map[ion], self.z[ion], self.c[ion], self.phi, self.RT_over_F)
            self.flux_y[ion][:, self.mem_start_y-1:self.mem_end_y] = 0.0
            self.c[ion] = _update_concentration_numba(
                self.c[ion], self.flux_x[ion], self.flux_y[ion], dt, self.dx, self.nx, self.ny)

            # (2) HH 전류에 의한 농도 변화 (Source/Sink)
            dC = (J_total[ion] / self.z[ion]) * self.flux_conv * inv_vol_factor * dt
            self.c[ion][self.inside_mask] -= dC
            self.c[ion][self.outside_mask] += dC
            self.c[ion][:, -1] = self.constants[f'c0_{ion}_ext']
            self.c[ion][:, 0] = self.constants[f'c0_{ion}_in']
            self.c[ion] = np.maximum(self.c[ion], 0.001)

        # 3. Poisson Solve
        self._apply_bc(self.Vm)
        rho_neutral = np.zeros((self.nx, self.ny))
        self.phi = self.fast_solver.solve(rho_neutral, self.epsilon_map, self.Vm)


    def run_hybrid_stimulation(self, target_time=100.0):
        print("Running Hybrid Stimulation (with Ion Tracking)...")

        rheobase_estimate = 2.7
        dc_bias = rheobase_estimate * 0.95
        noise_sigma = 5.0
        tau_noise = 20.0

        dt_min = 1e-4; dt_max = 0.05
        curr_t = 0.0; dt = dt_max

        self.hist = {'t':[], 'V':[], 'I_input':[], 'm':[], 'h':[], 'n':[],
                     'K_in':[], 'Na_in':[]}

        save_timer = 0.0; I_fluct = 0.0
        pbar = tqdm(total=target_time)

        while curr_t < target_time:
            v_prev = self.Vm
            eta = np.random.normal(0, 1)
            dI = -(I_fluct / tau_noise) * dt + noise_sigma * np.sqrt(2.0/tau_noise) * np.sqrt(dt) * eta
            I_fluct += dI

            # 5ms 이후 자극 시작
            if curr_t > 5.0:
                current_input = dc_bias + I_fluct
            else:
                current_input = 0.0

            try:
                self.step(dt, ext_current=current_input)
            except FloatingPointError:
                dt *= 0.5; self.Vm = v_prev; continue

            dv = abs(self.Vm - v_prev)
            if dv > 0.5: dt = max(dt * 0.5, dt_min)
            elif dv < 0.05: dt = min(dt * 1.2, dt_max)

            save_timer += dt
            if save_timer >= 0.1:
                self.hist['t'].append(curr_t)
                self.hist['V'].append(self.Vm)
                self.hist['I_input'].append(current_input)
                self.hist['m'].append(self.m)
                self.hist['h'].append(self.h)
                self.hist['n'].append(self.n)

                k_val = np.mean(self.c['K'][:, self.mem_start_y-1])
                na_val = np.mean(self.c['Na'][:, self.mem_start_y-1])
                self.hist['K_in'].append(k_val)
                self.hist['Na_in'].append(na_val)

                save_timer = 0.0
            curr_t += dt
            pbar.update(dt)
        pbar.close()
