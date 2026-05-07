"""
Lifetime_WT_model — MxlPy conversion
=====================================
Original MATLAB code by Rebecca Lee & Tsung-Yen Lee (v1.03, 2025-05-29).
Python / MxlPy conversion.

Model state vector (13 dynamic variables + 1 derived):
  V, A, Z          – xanthophyll-cycle pigments (violaxanthin, antheraxanthin, zeaxanthin)
  PV, PA, PZ       – pigment–protein complexes
  QV, QA, QZ       – quenching-competent complexes
  PL, QL           – Lhcsr-associated states
  PSIId            – PSII damage fraction
  E                – VDE enzyme activation (relative to light-eq)

Derived (not a state):
  P_free  = Ptot - Xtot + V + A + Z   (free antenna protein)

Light function I(t):
  Returns 1 (dark) or 2 (light) as integer index into rate-constant pairs,
  mirroring the MATLAB indexing convention [dark=1, light=2] → Python [0,1].

Usage
-----
See ``build_model()`` for the MxlPy Model object and ``run_simulation()``
for a convenience wrapper that mirrors the MATLAB function signature.

Install:
    pip install mxlpy scipy numpy
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import CubicSpline
from mxlpy import Model

# ---------------------------------------------------------------------------
# 1.  Light-intensity helper (returns 0=dark, 1=light)
# ---------------------------------------------------------------------------

def make_light_function(DeltaT):
    """
    Return I(t) -> int {0, 1}  (0 = dark, 1 = light).

    Parameters
    ----------
    DeltaT : float or array-like
        If scalar  : half-period of a symmetric L/D square wave (period = 2*DeltaT).
        If sequence: cumulative switch times, e.g. [0, 5, 5] for L5-D5.
    """
    DeltaT = np.asarray(DeltaT, dtype=float)
    if DeltaT.ndim == 0 or DeltaT.size == 1:
        period = float(DeltaT.flat[0])
        def I(t):
            # light on in first half of each period (matches MATLAB sign convention)
            phase = t % period
            return 1 if phase < 0.5 * period else 0
    else:
        cum = np.cumsum(DeltaT)
        def I(t):
            n_passed = int(np.sum(t >= cum))
            return n_passed % 2   # even = light (1st segment), odd = dark
    return I


# ---------------------------------------------------------------------------
# 2.  Initial-equilibrium calculation  (port of MATLAB initialEQ)
# ---------------------------------------------------------------------------

def compute_initial_equilibrium(V0, Ptot, PL0, k, light_on=False):
    """
    Compute equilibrium concentrations from initial conditions.

    Parameters
    ----------
    V0      : float   – initial violaxanthin concentration (par[0])
    Ptot    : float   – total protein concentration (par[1])
    PL0     : float   – initial PL concentration (par[2])
    k       : dict    – rate-constant dictionary (see ``unpack_parameters``)
    light_on: bool    – True for light-equilibrium, False (default) for dark

    Returns
    -------
    M0      : np.ndarray, shape (13,) – initial state vector
              [V, A, Z, PV, PA, PZ, QV, QA, QZ, PL, QL, PSIId, E]
    Xtot    : float   – total xanthophyll-cycle pool
    P_free0 : float   – free protein at t=0
    Keq     : dict    – equilibrium constants
    """
    Lind = 1 if light_on else 0   # index into [dark, light] pairs

    Keq = {
        "pz": k["pzf"] / k["pzb"],
        "pv": k["pvf"] / k["pvb"],
        "pa": k["paf"] / k["pab"],
        "a" : k["va"]  / k["av"],           # [dark, light]
        "z" : k["az"]  / k["za"],           # [dark, light]
        "qv": k["qvf"] / k["qvb"],          # [dark, light]
        "qa": k["qaf"] / k["qab"],          # [dark, light]
        "qz": k["qzf"] / k["qzb"],          # [dark, light]
    }

    Ka  = Keq["a"][Lind]
    Kz  = Keq["z"][Lind]
    Kqv = Keq["qv"][Lind]
    Kqa = Keq["qa"][Lind]
    Kqz = Keq["qz"][Lind]

    denom = (1 + V0 * (Keq["pv"] + Keq["pa"] * Ka + Keq["pz"] * Kz * Ka
                       + Kqv * Keq["pv"] + Kqa * Keq["pa"] * Ka
                       + Kqz * Keq["pz"] * Kz * Ka))
    P0  = Ptot / denom
    A0  = Ka  * V0
    Z0  = Kz  * A0
    PV0 = Keq["pv"] * P0 * V0
    PA0 = Keq["pa"] * Ka * V0 * P0
    PZ0 = Keq["pz"] * Kz * Ka * P0 * V0
    QV0 = Kqv * Keq["pv"] * P0 * V0
    QA0 = Keq["pa"] * Ka * Kqa * P0 * V0
    QZ0 = Kqz * Keq["pz"] * Kz * Ka * V0 * P0

    Xtot = V0 + Z0 + A0 + PV0 + PZ0 + PA0 + QA0 + QV0 + QZ0
    QL0  = 0.0
    E0   = k["va"][0] / k["va"][1]   # dark-equilibrium VDE activation
    PSIId0 = 0.0

    M0 = np.array([V0, A0, Z0, PV0, PA0, PZ0, QV0, QA0, QZ0,
                   PL0, QL0, PSIId0, E0])
    return M0, Xtot, P0, Keq


# ---------------------------------------------------------------------------
# 3.  Unpack the 32-element parameter vector
# ---------------------------------------------------------------------------

def unpack_parameters(par):
    """Convert the flat par vector to a structured dict of rate constants."""
    k = {}
    k["va"]     = np.array([par[3],  par[4]])   # [dark, light]
    k["av"]     = par[5]
    k["az"]     = np.array([0.0,     par[6]])    # dark calculated later
    k["za"]     = par[7]
    k["pvf"]    = par[8]
    k["pvb"]    = par[9]
    k["paf"]    = par[10]
    k["pab"]    = par[11]
    k["pzf"]    = par[12]
    k["pzb"]    = par[13]
    k["qvf"]    = np.array([0.0,     par[14]])
    k["qvb"]    = par[15]
    k["qaf"]    = np.array([0.0,     par[16]])
    k["qab"]    = par[17]
    k["qzf"]    = np.array([0.0,     par[18]])
    k["qzb"]    = par[19]
    k["vde"]    = np.array([par[20], par[21]])
    k["qlf"]    = np.array([0.0,     par[22]])
    k["qlb"]    = par[23]
    k["damage"] = np.array([par[29], par[30]])

    # Fix dark k.az using equilibrium ratio
    gamma       = k["va"][0] / k["va"][1]
    k["az"][0]  = k["az"][1] * gamma

    kappa = {
        "QV": par[24],
        "QA": par[25],
        "QZ": par[26],
        "QL": par[27],
        "qZ": par[28],
    }
    k_qI = par[31]

    return k, kappa, k_qI, gamma


# ---------------------------------------------------------------------------
# 4.  Build the MxlPy Model
# ---------------------------------------------------------------------------

def build_model(par, DeltaT):
    """
    Construct an MxlPy Model for the WT lifetime kinetic system.

    Parameters
    ----------
    par    : array-like, length 32 – parameter vector
    DeltaT : float or array-like    – light/dark schedule (see make_light_function)

    Returns
    -------
    model  : mxlpy.Model
    M0     : np.ndarray (13,) – initial concentrations
    I      : callable         – light function I(t) -> 0 (dark) or 1 (light)
    aux    : dict             – auxiliary quantities (Xtot, Ptot, delta, kappas, k_qI)
    """
    par  = np.asarray(par, dtype=float)
    k, kappa, k_qI, gamma = unpack_parameters(par)

    V0   = par[0]
    Ptot = par[1]
    PL0  = par[2]

    M0, Xtot, P0_eq, _ = compute_initial_equilibrium(V0, Ptot, PL0, k, light_on=False)

    # delta: basal 1/tau term derived from first data lifetime point
    # NOTE: delta depends on data; pass it as a parameter to the model
    # (will be set externally via update_parameters when fitting)
    # We expose it as a named model parameter "delta".

    I = make_light_function(DeltaT)

    aux = {
        "Xtot"    : Xtot,
        "Ptot"    : Ptot,
        "kappa_QV": kappa["QV"],
        "kappa_QA": kappa["QA"],
        "kappa_QZ": kappa["QZ"],
        "kappa_QL": kappa["QL"],
        "kappa_qZ": kappa["qZ"],
        "k_qI"    : k_qI,
        "k"       : k,
    }

    # ------------------------------------------------------------------
    # Rate-law functions (plain Python, as required by MxlPy)
    # Each function signature: (variables..., parameters...) -> float
    # MxlPy passes variables and parameters by NAME.
    # We use a closure over `k`, `kappa`, `k_qI`, `I`, `Xtot`, `Ptot`.
    # ------------------------------------------------------------------

    # Helper: light index from time parameter `t`
    # MxlPy passes simulation time `t` as a keyword to rate functions
    # when the reaction is added with `args=["t", ...]`.

    def P_free(V, A, Z):
        return Ptot - Xtot + V + A + Z

    # --- xanthophyll-cycle reactions ---

    def v_VA(t, V, A, E):
        """V -> A  (forward, light-dependent VDE activation)"""
        li = I(t)
        return k["va"][li] * E * V

    def v_AV(A):
        """A -> V  (reverse, ZEP)"""
        return k["av"] * A

    def v_AZ(t, A, E):
        """A -> Z  (light-only deepoxidation)"""
        li = I(t)
        return k["az"][li] * A * E

    def v_ZA(Z):
        """Z -> A  (reverse)"""
        return k["za"] * Z

    # --- pigment–protein binding reactions ---

    def v_PVf(V, A, Z):
        """V + P -> PV"""
        return k["pvf"] * V * P_free(V, A, Z)

    def v_PVb(PV):
        """PV -> V + P"""
        return k["pvb"] * PV

    def v_PAf(A, V, Z):
        """A + P -> PA"""
        return k["paf"] * A * P_free(V, A, Z)

    def v_PAb(PA):
        """PA -> A + P"""
        return k["pab"] * PA

    def v_PZf(Z, V, A):
        """Z + P -> PZ"""
        return k["pzf"] * Z * P_free(V, A, Z)

    def v_PZb(PZ):
        """PZ -> Z + P"""
        return k["pzb"] * PZ

    # --- quenching-complex formation ---

    def v_QVf(t, PV):
        li = I(t)
        return k["qvf"][li] * PV

    def v_QVb(QV):
        return k["qvb"] * QV

    def v_QAf(t, PA):
        li = I(t)
        return k["qaf"][li] * PA

    def v_QAb(QA):
        return k["qab"] * QA

    def v_QZf(t, PZ):
        li = I(t)
        return k["qzf"][li] * PZ

    def v_QZb(QZ):
        return k["qzb"] * QZ

    # --- Lhcsr (PL / QL) ---

    def v_QLf(t, PL):
        li = I(t)
        return k["qlf"][li] * PL

    def v_QLb(QL):
        return k["qlb"] * QL

    # --- PSII damage ---
    # dPSIId/dt = k_damage(I) * tau * (1 - PSIId)
    # tau is a derived variable; we implement this as a derived-variable
    # reaction where the rate depends on the current lifetime.

    def v_damage(t, V, A, Z, QV, QA, QZ, QL, PSIId, delta):
        li = I(t)
        tau = 1.0 / (delta
                     + kappa["QV"] * QV
                     + kappa["QA"] * QA
                     + kappa["QZ"] * QZ
                     + kappa["QL"] * QL
                     + kappa["qZ"] * Z
                     + k_qI * PSIId)
        return k["damage"][li] * tau * (1.0 - PSIId)

    # --- VDE enzyme activation ---

    def v_VDE(t, E):
        li = I(t)
        E_eq = [gamma, 1.0][li]
        return k["vde"][li] * (E_eq - E)

    # ------------------------------------------------------------------
    # Assemble the MxlPy model
    # ------------------------------------------------------------------
    m = Model()

    # Parameters
    m.add_parameters({
        "delta": 0.0,   # placeholder; set from data before simulating
    })

    # Variables and initial conditions
    var_names = ["V", "A", "Z", "PV", "PA", "PZ", "QV", "QA", "QZ",
                 "PL", "QL", "PSIId", "E"]
    m.add_variables(dict(zip(var_names, M0)))

    # Reactions  (name, rate_fn, stoichiometry_dict)
    # Stoichiometry: {variable: coefficient}
    # Positive = production, negative = consumption.

    m.add_reaction("VA",    v_VA,    {"V": -1, "A":  1}, args=["t", "V", "A", "E"])
    m.add_reaction("AV",    v_AV,    {"A": -1, "V":  1}, args=["A"])
    m.add_reaction("AZ",    v_AZ,    {"A": -1, "Z":  1}, args=["t", "A", "E"])
    m.add_reaction("ZA",    v_ZA,    {"Z": -1, "A":  1}, args=["Z"])

    m.add_reaction("PVf",   v_PVf,   {"V": -1, "PV":  1}, args=["V", "A", "Z"])
    m.add_reaction("PVb",   v_PVb,   {"PV": -1, "V":  1}, args=["PV"])
    m.add_reaction("PAf",   v_PAf,   {"A": -1, "PA":  1}, args=["A", "V", "Z"])
    m.add_reaction("PAb",   v_PAb,   {"PA": -1, "A":  1}, args=["PA"])
    m.add_reaction("PZf",   v_PZf,   {"Z": -1, "PZ":  1}, args=["Z", "V", "A"])
    m.add_reaction("PZb",   v_PZb,   {"PZ": -1, "Z":  1}, args=["PZ"])

    m.add_reaction("QVf",   v_QVf,   {"PV": -1, "QV":  1}, args=["t", "PV"])
    m.add_reaction("QVb",   v_QVb,   {"QV": -1, "PV":  1}, args=["QV"])
    m.add_reaction("QAf",   v_QAf,   {"PA": -1, "QA":  1}, args=["t", "PA"])
    m.add_reaction("QAb",   v_QAb,   {"QA": -1, "PA":  1}, args=["QA"])
    m.add_reaction("QZf",   v_QZf,   {"PZ": -1, "QZ":  1}, args=["t", "PZ"])
    m.add_reaction("QZb",   v_QZb,   {"QZ": -1, "PZ":  1}, args=["QZ"])

    m.add_reaction("QLf",   v_QLf,   {"PL": -1, "QL":  1}, args=["t", "PL"])
    m.add_reaction("QLb",   v_QLb,   {"QL": -1, "PL":  1}, args=["QL"])

    m.add_reaction("damage", v_damage, {"PSIId": 1},
                   args=["t", "V", "A", "Z", "QV", "QA", "QZ", "QL", "PSIId", "delta"])

    m.add_reaction("VDE",   v_VDE,   {"E": 1}, args=["t", "E"])

    return m, M0, I, aux


# ---------------------------------------------------------------------------
# 5.  Simulation and residual computation  (mirrors MATLAB function output)
# ---------------------------------------------------------------------------

def lifetime_from_conc(AllConc, kappa, k_qI, delta, col):
    """
    Compute fluorescence lifetime from concentration time series.

    AllConc columns: [V,A,Z,PV,PA,PZ,QV,QA,QZ,PL,QL,PSIId,E]
    col index map:    0 1 2  3  4  5  6  7  8  9 10  11   12
    """
    QV   = AllConc[:, 6]
    QA   = AllConc[:, 7]
    QZ   = AllConc[:, 8]
    QL   = AllConc[:, 10]
    Z    = AllConc[:, 2]
    PSIId= AllConc[:, 11]
    tau  = 1.0 / (delta
                  + kappa["QV"] * QV
                  + kappa["QA"] * QA
                  + kappa["QZ"] * QZ
                  + kappa["QL"] * QL
                  + kappa["qZ"] * Z
                  + k_qI * PSIId)
    return tau


def run_simulation(data_in, DeltaT_in, par_in):
    """
    Python equivalent of the MATLAB ``Lifetime_WT_model`` function.

    Parameters
    ----------
    data_in   : array-like, shape (N, 2) – [time, lifetime]
    DeltaT_in : float or array-like      – light/dark schedule
    par_in    : array-like, length 32    – model parameters

    Returns
    -------
    dict with keys:
        'time'            – simulation time grid
        'AllConc'         – concentration matrix (T x 13)
        'P_free'          – free-protein time series
        'lifetime_tot'    – lifetime on dense grid
        'lifetime_scaled' – lifetime interpolated to data time points
        'fit_qual_scaled' – scalar fit quality (sum-of-squares on 1/tau)
    """
    par      = np.asarray(par_in,  dtype=float)
    data_in  = np.asarray(data_in, dtype=float)
    DeltaT   = np.asarray(DeltaT_in, dtype=float)

    data_time     = data_in[:, 0]
    data_lifetime = data_in[:, 1]

    total_time = float(np.sum(DeltaT)) if DeltaT.ndim > 0 else float(DeltaT)
    time_grid  = np.arange(0.0, total_time + 0.1, 0.1)

    # delta from the first data point and initial Z
    k, kappa, k_qI, gamma = unpack_parameters(par)
    V0, Ptot, PL0 = par[0], par[1], par[2]
    M0, Xtot, _, _ = compute_initial_equilibrium(V0, Ptot, PL0, k, light_on=False)
    Z0    = M0[2]
    delta = 1.0 / data_in[0, 1] - kappa["qZ"] * Z0

    model, _, I, aux = build_model(par, DeltaT_in)
    model.update_parameters({"delta": delta})

    try:
        result = model.simulate(time_grid)
        AllConc = result.variables.values  # shape (T, 13)

        # Derived: free protein
        V_t  = AllConc[:, 0]
        A_t  = AllConc[:, 1]
        Z_t  = AllConc[:, 2]
        P_free_t = Ptot - Xtot + V_t + A_t + Z_t

        lifetime_tot = lifetime_from_conc(AllConc, kappa, k_qI, delta, col=None)

        # Interpolate to data time points (mimics MATLAB spline)
        cs = CubicSpline(time_grid, lifetime_tot)
        lifetime_scaled = cs(data_time)

        # Fit quality: sum-of-squares on 1/tau residuals
        resids = 1.0 / data_lifetime - 1.0 / lifetime_scaled
        fit_qual_scaled = float(np.sum(resids**2))

        return {
            "time"            : time_grid,
            "AllConc"         : AllConc,
            "P_free"          : P_free_t,
            "lifetime_tot"    : lifetime_tot,
            "lifetime_scaled" : lifetime_scaled,
            "fit_qual_scaled" : fit_qual_scaled,
        }

    except Exception as exc:
        print(f"[run_simulation] ODE integration failed: {exc}")
        return {
            "time"            : time_grid,
            "AllConc"         : None,
            "P_free"          : None,
            "lifetime_tot"    : None,
            "lifetime_scaled" : None,
            "fit_qual_scaled" : 1e5,
        }


# ---------------------------------------------------------------------------
# 6.  Quick self-test / example
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import numpy as np

    # Example parameter vector (32 values; replace with your real values)
    par_example = np.array([
        # par[0]  V0
        10.0,
        # par[1]  Ptot
        50.0,
        # par[2]  PL0
        5.0,
        # par[3..4]  k.va  [dark, light]
        0.1, 1.0,
        # par[5]   k.av
        0.2,
        # par[6]   k.az[light]
        0.5,
        # par[7]   k.za
        0.1,
        # par[8..9]  k.pvf, k.pvb
        0.3, 0.05,
        # par[10..11] k.paf, k.pab
        0.3, 0.05,
        # par[12..13] k.pzf, k.pzb
        0.3, 0.05,
        # par[14..15] k.qvf[light], k.qvb
        0.4, 0.1,
        # par[16..17] k.qaf[light], k.qab
        0.4, 0.1,
        # par[18..19] k.qzf[light], k.qzb
        0.4, 0.1,
        # par[20..21] k.vde [dark, light]
        0.05, 0.5,
        # par[22..23] k.qlf[light], k.qlb
        0.2, 0.1,
        # par[24..28] kappa_QV, kappa_QA, kappa_QZ, kappa_QL, kappa_qZ
        0.1, 0.2, 0.3, 0.15, 0.05,
        # par[29..30] k.damage [dark, light]
        0.001, 0.01,
        # par[31] k_qI
        0.05,
    ])

    # Synthetic data: L5-D5 protocol
    DeltaT_in = np.array([5.0, 5.0])
    t_data    = np.linspace(0.1, 9.9, 30)
    tau_data  = 2.0 + 0.5 * np.sin(t_data)   # fake lifetime data (ns)
    data_in   = np.column_stack([t_data, tau_data])

    out = run_simulation(data_in, DeltaT_in, par_example)

    if out["AllConc"] is not None:
        print(f"Fit quality (SS residuals on 1/tau): {out['fit_qual_scaled']:.4f}")
        print(f"Simulated {out['AllConc'].shape[0]} time points, "
              f"{out['AllConc'].shape[1]} variables.")
    else:
        print("Integration failed.")