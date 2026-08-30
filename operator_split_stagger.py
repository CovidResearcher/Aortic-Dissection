"""
Pure-Python implementation of the operator-split and staggered schemes
for the biochemical cascade (T, L, R, H, M, E) described in the
coupled FSI–biochemical model.

"""

from math import exp
from typing import List, Callable, Tuple

# ---------------------------------------------------------------------------
# State vector ordering
# y = [T, L, R, H, M, E]
# ---------------------------------------------------------------------------

# Nominal clearance rates (1/time) – replace with calibrated values
DELTA = {
    "T": 0.1,
    "L": 0.05,
    "R": 0.2,
    "H": 0.15,
    "M": 0.08,
    "E": 0.0,          # E has no linear clearance; degradation is nonlinear
}

# Production / interaction coefficients (placeholders)
K_T, K_L          = 1.0, 1.0
ALPHA_T, ALPHA_L  = 0.5, 0.3
BETA              = 0.4
H_MAX             = 1.0
GAMMA_H, GAMMA_R  = 0.2, 0.1
MU                = 0.05          # MMP-driven degradation coefficient
P_PROD            = 0.01          # basal ECM production
KAPPA             = 0.0           # cavitation damage coefficient (set >0 when active)

# ---------------------------------------------------------------------------
# Mechanical stimulus functions (user-supplied)
# ---------------------------------------------------------------------------
def g_tau(tau_w: float) -> float:
    """Mechanotransduction from wall shear stress."""
    return max(0.0, tau_w - 1.5)          # example threshold

def h_sigma(sigma: float) -> float:
    """Mechanotransduction from intramural stress."""
    return max(0.0, sigma - 50.0)

def phi_eps(eps: float) -> float:
    """Strain-driven MMP upregulation."""
    return max(0.0, eps - 0.1)

def q_cav(cav_indicator: float) -> float:
    """Cavitation damage contribution."""
    return cav_indicator

# ---------------------------------------------------------------------------
# Linear clearance matrix A (diagonal)
# ---------------------------------------------------------------------------
def apply_linear_clearance(y: List[float], dt: float) -> List[float]:
    """
    Exact solution of dy/dt = A y  for a diagonal clearance matrix.
    y* = exp(dt * A) @ y
    """
    T, L, R, H, M, E = y
    return [
        T * exp(-DELTA["T"] * dt),
        L * exp(-DELTA["L"] * dt),
        R * exp(-DELTA["R"] * dt),
        H * exp(-DELTA["H"] * dt),
        M * exp(-DELTA["M"] * dt),
        E,                               # no linear term for E
    ]

# ---------------------------------------------------------------------------
# Nonlinear + mechanical sources N(y) + m(mechanics)
# ---------------------------------------------------------------------------
def nonlinear_and_mech_sources(
    y: List[float],
    tau_w: float,
    sigma: float,
    eps: float,
    cav: float,
    D: float = 1.0,                     # dysbiosis drive (external)
) -> List[float]:
    T, L, R, H, M, E = y

    dT = K_T * D
    dL = K_L * D
    dR = (ALPHA_T * T + ALPHA_L * L
          + 0.3 * g_tau(tau_w) + 0.2 * h_sigma(sigma))
    dH = BETA * R * (H_MAX - H)
    dM = GAMMA_H * H + GAMMA_R * R + 0.15 * phi_eps(eps)
    dE = P_PROD - MU * M * E - KAPPA * q_cav(cav)

    return [dT, dL, dR, dH, dM, dE]

# ---------------------------------------------------------------------------
# 1. Lie–Trotter (first-order) operator split
# ---------------------------------------------------------------------------
def lie_trotter_step(
    y: List[float],
    dt: float,
    tau_w: float,
    sigma: float,
    eps: float,
    cav: float,
    D: float = 1.0,
) -> List[float]:
    """
    First-order Lie–Trotter splitting:
        y*   = exp(dt A) y^n
        y^{n+1} = y* + dt (N(y*) + m)
    """
    y_star = apply_linear_clearance(y, dt)
    sources = nonlinear_and_mech_sources(y_star, tau_w, sigma, eps, cav, D)
    return [ys + dt * s for ys, s in zip(y_star, sources)]

# ---------------------------------------------------------------------------
# 2. Strang (second-order) operator split
# ---------------------------------------------------------------------------
def strang_step(
    y: List[float],
    dt: float,
    tau_w: float,
    sigma: float,
    eps: float,
    cav: float,
    D: float = 1.0,
) -> List[float]:
    """
    Second-order Strang splitting:
        half linear → full nonlinear → half linear
    """
    half = 0.5 * dt
    y1 = apply_linear_clearance(y, half)
    sources = nonlinear_and_mech_sources(y1, tau_w, sigma, eps, cav, D)
    y2 = [a + dt * s for a, s in zip(y1, sources)]
    return apply_linear_clearance(y2, half)

# ---------------------------------------------------------------------------
# 3. Explicit staggered / quasi-static biochemical update
#    (the scheme most often used in fluid–solid–growth literature)
# ---------------------------------------------------------------------------
def staggered_biochemical_update(
    y: List[float],
    dt_bio: float,
    averaged_stimuli: Tuple[float, float, float, float],
    D: float = 1.0,
    n_substeps: int = 1,
) -> List[float]:
    """
    Advance the biochemical system over a long time step dt_bio
    using cycle-averaged mechanical stimuli obtained from an FSI solve
    performed at fixed E.

    Parameters
    ----------
    y : current biochemical state
    dt_bio : long biochemical time step (hours–days)
    averaged_stimuli : (tau_w_bar, sigma_bar, eps_bar, cav_bar)
    n_substeps : optional sub-stepping for stability
    """
    tau_bar, sigma_bar, eps_bar, cav_bar = averaged_stimuli
    dt = dt_bio / n_substeps
    y_new = y[:]
    for _ in range(n_substeps):
        # any of the splitters can be used here; Lie–Trotter is cheapest
        y_new = lie_trotter_step(y_new, dt, tau_bar, sigma_bar, eps_bar, cav_bar, D)
    return y_new

# ---------------------------------------------------------------------------
# 4. Forward-Euler monolithic step (for reference / verification)
# ---------------------------------------------------------------------------
def forward_euler_step(
    y: List[float],
    dt: float,
    tau_w: float,
    sigma: float,
    eps: float,
    cav: float,
    D: float = 1.0,
) -> List[float]:
    sources = nonlinear_and_mech_sources(y, tau_w, sigma, eps, cav, D)
    clear = [
        -DELTA["T"] * y[0],
        -DELTA["L"] * y[1],
        -DELTA["R"] * y[2],
        -DELTA["H"] * y[3],
        -DELTA["M"] * y[4],
        0.0,
    ]
    dydt = [c + s for c, s in zip(clear, sources)]
    return [yi + dt * di for yi, di in zip(y, dydt)]

# ---------------------------------------------------------------------------
# Demo / test harness
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Initial state (arbitrary but positive)
    y0 = [0.1, 0.1, 0.05, 0.2, 0.1, 0.9]   # T, L, R, H, M, E

    # Example mechanical stimuli (would come from an FSI solver)
    tau_w, sigma, eps, cav = 2.0, 80.0, 0.15, 0.0

    dt = 0.05          # short step for operator-split demo
    dt_bio = 24.0      # long biochemical step (e.g. one day)

    print("Initial state:", ["{:.4f}".format(v) for v in y0])

    # --- Lie–Trotter ---
    y_lt = lie_trotter_step(y0, dt, tau_w, sigma, eps, cav)
    print("After Lie–Trotter step:", ["{:.4f}".format(v) for v in y_lt])

    # --- Strang ---
    y_st = strang_step(y0, dt, tau_w, sigma, eps, cav)
    print("After Strang step:    ", ["{:.4f}".format(v) for v in y_st])

    # --- Staggered long-step update (using the same stimuli as “averages”) ---
    y_stag = staggered_biochemical_update(
        y0, dt_bio, (tau_w, sigma, eps, cav), n_substeps=20
    )
    print("After staggered bio update (dt_bio={}):".format(dt_bio),
          ["{:.4f}".format(v) for v in y_stag])

    # --- Reference forward Euler (many small steps) ---
    y_fe = y0[:]
    for _ in range(20):
        y_fe = forward_euler_step(y_fe, dt, tau_w, sigma, eps, cav)
    print("Forward-Euler reference:", ["{:.4f}".format(v) for v in y_fe])
