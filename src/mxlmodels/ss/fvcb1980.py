r"""Farquhar, von Caemmerer & Berry (1980) C3 photosynthesis model.

Steady-state leaf CO2 assimilation as the minimum of Rubisco-limited and
electron-transport-limited rates. `model_version="2025"` switches to the modern
min-W formulation with triose-phosphate utilisation limitation.

|             |                                                                                |
| ----------- | ------------------------------------------------------------------------------ |
| doi         | 10.1007/BF00386231                                                             |
| main author | Graham D. Farquhar                                                             |
| paper title | A biochemical model of photosynthetic CO2 assimilation in leaves of C3 species |
| published   | June 1980                                                                      |
| journal     | Planta                                                                         |
| organism    | C3 leaf                                                                        |
"""

import math

from mxlpy import SteadyStateModelBuilder
from mxlpy.units import (
    bar,
    dimensionless,
    joule,
    kelvin,
    micro,
    milli,
    mol,
    ppfd,
    second,
    sqm,
)

ubar = micro * bar
mbar = milli * bar
rate = micro * mol / sqm / second
uEq_rate = micro * mol / sqm / second


def min_solve(a: float, b: float, c: float) -> float:
    r"""Return the smaller root of a quadratic.

    The units of `a`, `b`, and `c` depend on the equation being solved; the
    returned root has the units of `x`.
    """
    discriminant = max(0.0, (b**2) - (4 * a * c))
    root1 = (-b + math.sqrt(discriminant)) / (2 * a)
    root2 = (-b - math.sqrt(discriminant)) / (2 * a)
    return min(root1, root2)


def arrhenius(param_298: float, Ea: float, T: float) -> float:
    r"""Scale a parameter at 298 K to temperature `T`.

    `Ea` is in `\mathrm{J\,mol^{-1}}` and `T` is in `\mathrm{K}`. The returned
    value has the same units as `param_298`.
    """
    return param_298 * math.exp(Ea * (T - 298) / (298 * 8.314 * T))


def jmax_tempscaling(
    jmax_298: float, Ea_jmax: float, S: float, H: float, T: float
) -> float:
    r"""Scale `Jmax` to `T`.

    `jmax_298` is in `\mathrm{\mu Eq\,m^{-2}\,s^{-1}}`, `Ea_jmax` and `H` are in
    `\mathrm{J\,mol^{-1}}`, `S` is in `\mathrm{J\,K^{-1}\,mol^{-1}}`, and `T` is in
    `\mathrm{K}`.
    """
    R = 8.314
    den_298 = 1 + math.exp((S * 298 - H) / (R * 298))
    den_T = 1 + math.exp((S * T - H) / (R * T))
    return arrhenius(jmax_298, Ea_jmax, T) * (den_298 / den_T)


def _derived_ko(kc):
    r"""Calculate oxygenase turnover, in `\mathrm{s^{-1}}`."""
    return 0.21 * kc


def _derived_phi(ko, kc, O2, km_o2, Ci, km_co2):
    r"""Calculate the dimensionless oxygenation/carboxylation ratio."""
    return (ko / kc) * ((O2 / km_o2) / (Ci / km_co2))


def _derived_Wc(Ci, vcmax, km_co2, km_o2, O2):
    r"""Calculate Rubisco-limited carboxylation in `\mathrm{\mu mol\,m^{-2}\,s^{-1}}`."""
    return (Ci * vcmax) / (Ci + km_co2 * (1 + (O2) / (km_o2)))


def _derived_J(f, I, j_max, z):
    r"""Calculate potential electron transport in `\mathrm{\mu Eq\,m^{-2}\,s^{-1}}`."""
    return min_solve(
        a=1.0, b=-(0.5 * (1 - f) * I + j_max + z), c=0.5 * (1 - f) * I * j_max
    )


def _derived_Wj(J, phi, Wc, vc_max):
    r"""Calculate RuBP-regeneration-limited carboxylation in `\mathrm{\mu mol\,m^{-2}\,s^{-1}}`."""
    J_prime = J / (2 * (2 + 2 * phi))
    P = Wc * (300.0 / 87.0)
    M = 2.0 * vc_max
    M_prime = M / (2.0 + 1.5 * phi)
    B = J_prime + P + (J_prime * P) / M_prime
    C_quad = J_prime * P
    return min_solve(a=1.0, b=-B, c=C_quad)


def _derived_gammastar(km_co2, O2, ko, km_o2, kc):
    r"""Calculate `Gamma*` in `\mathrm{\mu bar}`."""
    return km_co2 * O2 * ko / (2 * km_o2 * kc)


def _derived_Vc(Wc, Wj):
    r"""Calculate carboxylation in `\mathrm{\mu mol\,m^{-2}\,s^{-1}}`."""
    return min(Wc, Wj)


def _derived_A(vc, gammastar, Ci, r_light):
    r"""Calculate net assimilation in `\mathrm{\mu mol\,m^{-2}\,s^{-1}}`."""
    return vc * (1 - gammastar / Ci) - r_light


def _derived_Vo(vc, phi):
    r"""Calculate oxygenation in `\mathrm{\mu mol\,m^{-2}\,s^{-1}}`."""
    return vc * phi


def get_fvcb1980() -> SteadyStateModelBuilder:
    m = SteadyStateModelBuilder()
    m.add_parameter("Ci", 230, unit=ubar)
    m.add_parameter("vcmax_25", 98, unit=rate)
    m.add_parameter("km_co2_25", 460, unit=ubar)
    m.add_parameter("km_o2_25", 330, unit=mbar)
    m.add_parameter("O2", 210, unit=mbar)
    m.add_parameter("kc_25", 2.5, unit=1 / second)
    m.add_parameter("f", 0.23, unit=dimensionless)
    m.add_parameter("I", 1000, unit=ppfd)
    m.add_parameter("j_max_25", 210, unit=uEq_rate)
    m.add_parameter("z", 20, unit=uEq_rate)
    m.add_parameter("r_light_25", 1.1, unit=rate)
    m.add_parameter("T", 298, unit=kelvin)

    ### Activation energies
    m.add_parameter("Ea_km_co2", 59356, unit=joule / mol)
    m.add_parameter("Ea_km_o2", 35948, unit=joule / mol)
    m.add_parameter("Ea_vcmax", 58520, unit=joule / mol)
    m.add_parameter("Ea_kc", 58520, unit=joule / mol)
    m.add_parameter("Ea_r_light", 66405, unit=joule / mol)

    m.add_parameter("Ea_jmax", 37000, unit=joule / mol)
    m.add_parameter("S", 710, unit=joule / kelvin / mol)
    m.add_parameter("H", 220000, unit=joule / mol)

    ### Temp scaling
    m.add_derived("km_co2", arrhenius, args=["km_co2_25", "Ea_km_co2", "T"], unit=ubar)
    m.add_derived("km_o2", arrhenius, args=["km_o2_25", "Ea_km_o2", "T"], unit=mbar)
    m.add_derived("vcmax", arrhenius, args=["vcmax_25", "Ea_vcmax", "T"], unit=rate)
    m.add_derived("kc", arrhenius, args=["kc_25", "Ea_kc", "T"], unit=1 / second)
    m.add_derived(
        "r_light", arrhenius, args=["r_light_25", "Ea_r_light", "T"], unit=rate
    )

    m.add_derived(
        "j_max",
        jmax_tempscaling,
        args=["j_max_25", "Ea_jmax", "S", "H", "T"],
        unit=uEq_rate,
    )

    m.add_derived("ko", _derived_ko, args=["kc"], unit=1 / second)
    m.add_derived(
        "phi",
        _derived_phi,
        args=["ko", "kc", "O2", "km_o2", "Ci", "km_co2"],
        unit=dimensionless,
    )
    m.add_derived(
        "Wc", _derived_Wc, args=["Ci", "vcmax", "km_co2", "km_o2", "O2"], unit=rate
    )
    m.add_derived("J", _derived_J, args=["f", "I", "j_max", "z"], unit=uEq_rate)
    m.add_derived("Wj", _derived_Wj, args=["J", "phi", "Wc", "vcmax"], unit=rate)
    m.add_derived(
        "gammastar",
        _derived_gammastar,
        args=["km_co2", "O2", "ko", "km_o2", "kc"],
        unit=ubar,
    )
    m.add_derived("Vc", _derived_Vc, args=["Wc", "Wj"], unit=rate)
    m.add_derived("A", _derived_A, args=["Vc", "gammastar", "Ci", "r_light"], unit=rate)
    m.add_derived("Vo", _derived_Vo, args=["Vc", "phi"], unit=rate)

    return m
