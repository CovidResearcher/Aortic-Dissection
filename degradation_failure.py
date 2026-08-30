"""
Aortic Wall Degradation-to-Failure Limit Model
Pure Python implementation of the biomechanical formula:

UTS_degraded(t) = UTS_0 * Φ_matrix(t) * Φ_fatigue(t) * Φ_geometry(t)

Failure occurs when σ_local(t) ≥ UTS_degraded(t)
"""

import math

# ------------------------------------------------------------
# Baseline parameters (representative literature values)
# ------------------------------------------------------------
UTS_0 = 2.0          # Healthy circumferential UTS [MPa]
P_mean = 0.0133      # Mean arterial pressure ≈ 100 mmHg [MPa]
r_h_ratio = 10.0     # Typical radius / wall-thickness ratio
eta = 1.5            # Geometry amplification factor

# Rate constants for biochemical degradation
kappa_MMP = 0.8
kappa_ROS = 0.6

# Fatigue parameters
N_f = 1.5e9          # Approximate cycles to failure under moderate stress
m = 0.7              # Fatigue exponent


# ------------------------------------------------------------
# Component functions
# ------------------------------------------------------------
def phi_matrix(c_mmp: float, c_ros: float) -> float:
    """
    Biochemical matrix-degradation factor.
    c_mmp, c_ros : normalized cumulative MMP and ROS activity (0 → healthy, 1 → severe)
    """
    return math.exp(-kappa_MMP * c_mmp - kappa_ROS * c_ros)


def phi_fatigue(n_cycles: float) -> float:
    """
    Cyclic-fatigue factor (balloon-expansion + jet loading).
    n_cycles : total cardiac / stress cycles experienced
    """
    ratio = min(n_cycles / N_f, 0.999)          # avoid exact 1.0
    return (1.0 - ratio) ** m


def phi_geometry(delta_E_over_E0: float) -> float:
    """
    Geometric / stiffness factor.
    delta_E_over_E0 : relative increase in local elastic modulus
    """
    return 1.0 / (1.0 + eta * delta_E_over_E0)


def uts_degraded(c_mmp: float, c_ros: float,
                 n_cycles: float, delta_E: float) -> float:
    """
    Remaining ultimate tensile strength after degradation [MPa]
    """
    return (UTS_0
            * phi_matrix(c_mmp, c_ros)
            * phi_fatigue(n_cycles)
            * phi_geometry(delta_E))


def sigma_local(pressure: float = P_mean,
                r_h: float = r_h_ratio,
                tau_jet: float = 0.0,
                sigma_balloon: float = 0.0) -> float:
    """
    Local wall stress [MPa]
    Laplace term + jet shear + balloon-expansion concentration
    """
    return (pressure * r_h) + tau_jet + sigma_balloon


def safety_factor(uts: float, sigma: float) -> float:
    """Instantaneous safety factor"""
    if sigma <= 0:
        return float('inf')
    return uts / sigma


# ------------------------------------------------------------
# Example simulation: progressive degradation over “years”
# ------------------------------------------------------------
def simulate_degradation(years: int = 30, steps_per_year: int = 12):
    """
    Simple time-stepping example.
    Returns lists of time, UTS, sigma, and SF for inspection.
    """
    dt = 1.0 / steps_per_year          # years per step
    heart_rate = 70 * 525600           # approximate cycles per year

    times, uts_list, sigma_list, sf_list = [], [], [], []

    c_mmp = 0.0
    c_ros = 0.0
    n_cycles = 0.0
    delta_E = 0.0

    for step in range(years * steps_per_year + 1):
        t = step * dt

        # Progressive biochemical load (example linear ramps)
        c_mmp = min(0.9, 0.025 * t)           # reaches severe after ~36 y
        c_ros = min(0.85, 0.022 * t)
        n_cycles = heart_rate * t
        delta_E = min(2.0, 0.06 * t)          # stiffness increase

        # Occasional pressure / jet spikes (every ~5 years)
        pressure = P_mean
        tau_jet = 0.0
        sigma_balloon = 0.0
        if step % (5 * steps_per_year) == 0 and step > 0:
            pressure = 0.0266                 # ~200 mmHg spike
            tau_jet = 0.15
            sigma_balloon = 0.25

        uts = uts_degraded(c_mmp, c_ros, n_cycles, delta_E)
        sigma = sigma_local(pressure, r_h_ratio, tau_jet, sigma_balloon)
        sf = safety_factor(uts, sigma)

        times.append(t)
        uts_list.append(uts)
        sigma_list.append(sigma)
        sf_list.append(sf)

        # Optional early-exit on failure
        if sf <= 1.0:
            print(f"*** FAILURE predicted at t ≈ {t:.1f} years "
                  f"(UTS={uts:.3f} MPa, σ={sigma:.3f} MPa, SF={sf:.2f})")
            break

    return times, uts_list, sigma_list, sf_list


# ------------------------------------------------------------
# Demonstration
# ------------------------------------------------------------
if __name__ == "__main__":
    print("Aortic Degradation-to-Failure Model (pure Python)\n")

    # Snapshot at a representative degraded state
    c_mmp, c_ros = 0.6, 0.55
    n_cyc = 1.2e9
    dE = 1.2

    uts = uts_degraded(c_mmp, c_ros, n_cyc, dE)
    sigma = sigma_local(pressure=0.018, tau_jet=0.08, sigma_balloon=0.12)
    sf = safety_factor(uts, sigma)

    print(f"Example degraded state:")
    print(f"  Φ_matrix   = {phi_matrix(c_mmp, c_ros):.3f}")
    print(f"  Φ_fatigue  = {phi_fatigue(n_cyc):.3f}")
    print(f"  Φ_geometry = {phi_geometry(dE):.3f}")
    print(f"  UTS_degraded = {uts:.3f} MPa")
    print(f"  σ_local      = {sigma:.3f} MPa")
    print(f"  Safety factor = {sf:.2f}")
    print()

    # Run a 30-year progressive simulation
    print("Running 30-year progressive degradation simulation...")
    t, uts_hist, sig_hist, sf_hist = simulate_degradation(years=30)

    print(f"\nFinal state at t = {t[-1]:.1f} years:")
    print(f"  UTS = {uts_hist[-1]:.3f} MPa")
    print(f"  σ   = {sig_hist[-1]:.3f} MPa")
    print(f"  SF  = {sf_hist[-1]:.2f}")
