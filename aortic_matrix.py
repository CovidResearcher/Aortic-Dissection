import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

# ODE right-hand side
def aortic_matrix_odes(t, y, params):
    T, L, R, H, M, E = y
    (D, k_T, delta_T, k_L, delta_L,
     alpha_T, alpha_L, delta_R,
     beta, H_max, delta_H,
     gamma_H, gamma_R, delta_M,
     p, mu) = params

    dT = k_T * D - delta_T * T
    dL = k_L * D - delta_L * L
    dR = alpha_T * T + alpha_L * L - delta_R * R
    dH = beta * R * (H_max - H) - delta_H * H
    dM = gamma_H * H + gamma_R * R - delta_M * M
    dE = p - mu * M * E
    return [dT, dL, dR, dH, dM, dE]

# Illustrative parameters (order-of-magnitude; calibrate to data as needed)
params = [
    1.0,   # D  (dysbiosis drive)
    0.5,   # k_T
    0.1,   # δ_T
    0.3,   # k_L
    0.08,  # δ_L
    0.4,   # α_T
    0.6,   # α_L
    0.2,   # δ_R
    0.15,  # β
    1.0,   # H_max
    0.1,   # δ_H
    0.3,   # γ_H
    0.2,   # γ_R
    0.15,  # δ_M
    0.05,  # p   (basal ECM synthesis)
    0.25   # μ   (MMP-dependent degradation)
]

y0 = [0.1, 0.1, 0.05, 0.1, 0.05, 1.0]   # [T, L, R, H, M, E]
t_span = (0, 50)
t_eval = np.linspace(0, 50, 501)

# Adaptive Runge–Kutta (RK45 = Dormand–Prince 5(4))
sol = solve_ivp(
    fun=lambda t, y: aortic_matrix_odes(t, y, params),
    t_span=t_span,
    y0=y0,
    t_eval=t_eval,
    method='RK45',
    rtol=1e-6,
    atol=1e-8
)

# Store trajectory in pandas DataFrame
df = pd.DataFrame({
    'time'  : sol.t,
    'TMAO'  : sol.y[0],
    'LPS'   : sol.y[1],
    'ROS'   : sol.y[2],
    'HIF1a' : sol.y[3],
    'MMP'   : sol.y[4],
    'ECM_E' : sol.y[5]
})

print(df.head(10))
print(df.tail(5))
print(df.describe())
print(f"Final ECM integrity E(t=50) = {df['ECM_E'].iloc[-1]:.4f}")
