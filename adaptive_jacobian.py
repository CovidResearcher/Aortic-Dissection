import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 1. Parameters as pandas DataFrame
# ============================================================
param_data = {
    'symbol': ['k_T', 'k_L', 'delta_T', 'delta_L', 'delta_R', 'delta_H', 'delta_M',
               'alpha_T', 'alpha_L', 'beta', 'H_max', 'gamma_H', 'gamma_R', 'p', 'mu',
               'alpha_tau', 'alpha_sigma', 'kappa_cav'],
    'value': [0.05, 0.03, 0.02, 0.015, 0.1, 0.05, 0.08,
              0.2, 0.15, 0.3, 1.0, 0.25, 0.1, 0.01, 0.05,
              0.05, 0.03, 0.1],
    'description': [
        'TMAO production rate from dysbiosis',
        'LPS production rate from dysbiosis',
        'TMAO clearance',
        'LPS clearance',
        'ROS clearance',
        'HIF clearance',
        'MMP clearance',
        'ROS from TMAO',
        'ROS from LPS',
        'HIF stabilization by ROS',
        'Max HIF level',
        'MMP from HIF',
        'MMP from ROS',
        'Basal ECM synthesis',
        'MMP-catalyzed ECM degradation',
        'ROS from wall shear stress',
        'ROS from intramural stress',
        'Cavitation damage coefficient'
    ]
}
params_df = pd.DataFrame(param_data)
params = dict(zip(params_df['symbol'], params_df['value']))
print("=== Parameters DataFrame ===")
print(params_df.to_string(index=False))
print()

# ============================================================
# 2. Anisotropic multi-layer wall properties as DataFrame
# ============================================================
layer_data = {
    'layer': ['Intima', 'Media', 'Adventitia'],
    'mu0': [0.05e6, 0.15e6, 0.08e6],          # Pa, base isotropic modulus
    'k1_0': [0.1e6, 0.5e6, 0.3e6],            # Pa, base collagen stiffness
    'k2': [5.0, 8.0, 10.0],                   # dimensionless
    'kappa': [0.1, 0.15, 0.25],               # dispersion
    'fiber_angle_deg': [0.0, 30.0, 45.0],     # approximate mean fiber angle
    'thickness_frac': [0.1, 0.6, 0.3]
}
layers_df = pd.DataFrame(layer_data)
print("=== Anisotropic Multi-Layer Properties DataFrame ===")
print(layers_df.to_string(index=False))
print()

# ============================================================
# 3. Biochemical ODE right-hand side and Analytic Jacobian
# ============================================================
def bio_ode(t, y, D, tau_bar, sigma_bar, cav_ind, pdict):
    T, L, R, H, M, E = y
    dT = pdict['k_T'] * D - pdict['delta_T'] * T
    dL = pdict['k_L'] * D - pdict['delta_L'] * L
    dR = (pdict['alpha_T'] * T + pdict['alpha_L'] * L
          + pdict['alpha_tau'] * tau_bar
          + pdict['alpha_sigma'] * sigma_bar
          - pdict['delta_R'] * R)
    dH = pdict['beta'] * R * (pdict['H_max'] - H) - pdict['delta_H'] * H
    dM = pdict['gamma_H'] * H + pdict['gamma_R'] * R - pdict['delta_M'] * M
    dE = pdict['p'] - pdict['mu'] * M * E - pdict['kappa_cav'] * cav_ind * E
    return [dT, dL, dR, dH, dM, dE]

def bio_jacobian(t, y, D, tau_bar, sigma_bar, cav_ind, pdict):
    T, L, R, H, M, E = y
    J = np.zeros((6, 6))
    # dT/dt
    J[0, 0] = -pdict['delta_T']
    # dL/dt
    J[1, 1] = -pdict['delta_L']
    # dR/dt
    J[2, 0] = pdict['alpha_T']
    J[2, 1] = pdict['alpha_L']
    J[2, 2] = -pdict['delta_R']
    # dH/dt
    J[3, 2] = pdict['beta'] * (pdict['H_max'] - H)
    J[3, 3] = -pdict['beta'] * R - pdict['delta_H']
    # dM/dt
    J[4, 2] = pdict['gamma_R']
    J[4, 3] = pdict['gamma_H']
    J[4, 4] = -pdict['delta_M']
    # dE/dt
    J[5, 4] = -pdict['mu'] * E
    J[5, 5] = -pdict['mu'] * M - pdict['kappa_cav'] * cav_ind
    return J

# ============================================================
# 4. Weak-coupling simulation with Runge-Kutta (solve_ivp RK45 + jac)
# ============================================================
# Initial state
y0 = np.array([0.1, 0.05, 0.02, 0.1, 0.05, 1.0])  # T, L, R, H, M, E
state_names = ['TMAO', 'LPS', 'ROS', 'HIF1a', 'MMP', 'ECM_E']

# Mock mechanical stimuli schedule (from "FSI")
n_outer = 20
bio_dt = 24.0  # hours
D_drive = 1.0  # constant dysbiosis

# Storage lists
history = []
layer_moduli_history = []

current_y = y0.copy()
t_bio = 0.0

for step in range(n_outer):
    # --- Mock FSI-derived stimuli (replace with real FSI output) ---
    E_curr = current_y[5]
    tau_bar = 2.0 + 3.0 * (1.0 - E_curr)          # normalized
    sigma_bar = 50.0 + 80.0 * (1.0 - E_curr)      # normalized
    cav_ind = max(0.0, 0.3 * (1.0 - E_curr) - 0.1)

    # --- Runge-Kutta advance of biochemistry (RK45 + analytic Jacobian) ---
    sol = solve_ivp(
        fun=lambda t, y: bio_ode(t, y, D_drive, tau_bar, sigma_bar, cav_ind, params),
        t_span=(0, bio_dt),
        y0=current_y,
        method='RK45',
        jac=lambda t, y: bio_jacobian(t, y, D_drive, tau_bar, sigma_bar, cav_ind, params),
        rtol=1e-6, atol=1e-8
    )
    current_y = sol.y[:, -1]
    t_bio += bio_dt

    # Store biochemical state
    row = {'time_h': t_bio, 'tau_bar': tau_bar, 'sigma_bar': sigma_bar, 'cav_ind': cav_ind}
    for i, name in enumerate(state_names):
        row[name] = current_y[i]
    history.append(row)

    # Update multi-layer moduli with current E
    E = current_y[5]
    moduli_row = {'time_h': t_bio, 'E': E}
    for _, layer in layers_df.iterrows():
        mu_scaled = layer['mu0'] * E
        k1_scaled = layer['k1_0'] * E
        moduli_row[f"{layer['layer']}_mu"] = mu_scaled
        moduli_row[f"{layer['layer']}_k1"] = k1_scaled
    layer_moduli_history.append(moduli_row)

# ============================================================
# 5. Results as pandas DataFrames
# ============================================================
results_df = pd.DataFrame(history)
moduli_df = pd.DataFrame(layer_moduli_history)

print("=== Biochemical State History (DataFrame) ===")
print(results_df.to_string(index=False))
print()
print("=== Anisotropic Multi-Layer Moduli Evolution (DataFrame) ===")
print(moduli_df.to_string(index=False))
print()
print("=== Summary statistics of ECM integrity E ===")
print(results_df['ECM_E'].describe())
