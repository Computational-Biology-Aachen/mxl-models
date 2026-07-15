from __future__ import annotations

from typing import Literal

EMPTY: Literal[""] = ""

###############################################################################
# Parameter fns
###############################################################################


def loc(name: str, compartment: str, tissue: str) -> str:
    r"""Localise a component to a compartment and tissue."""
    return f"{name}{compartment}{tissue}"


def e0(enzyme: str) -> str:
    return f"E0_{enzyme}"


def e(enzyme: str) -> str:
    return f"E_{enzyme}"


def kcat(enzyme: str) -> str:
    return f"kcat_{enzyme}"


def vmax(enzyme: str) -> str:
    return f"vmax_{enzyme}"


def keq(enzyme: str) -> str:
    return f"keq_{enzyme}"


def kre(enzyme: str) -> str:
    return f"kre_{enzyme}"


def kf(enzyme: str) -> str:
    return f"kf_{enzyme}"


def kh(enzyme: str) -> str:
    r"""Hill constant"""
    return f"kh_{enzyme}"


def ksat(enzyme: str) -> str:
    return f"ksat_{enzyme}"


def km(enzyme: str, substrate: str | None = None) -> str:
    if substrate is None:
        return f"km_{enzyme}"
    return f"km_{enzyme}_{substrate}"


def kms(enzyme: str) -> str:
    return km(enzyme, "s")


def kmp(enzyme: str) -> str:
    return km(enzyme, "p")


def ki(enzyme: str, substrate: str | None = None) -> str:
    if substrate is None:
        return f"ki_{enzyme}"
    return f"ki_{enzyme}_{substrate}"


def ka(enzyme: str, substrate: str | None = None) -> str:
    if substrate is None:
        return f"ki_{enzyme}"
    return f"ki_{enzyme}_{substrate}"


def k(n: str) -> str:
    return f"k_{n}"


# def ph(n: str) -> str:
#     return f"ph_{n}"


def proton_fn(n: str) -> str:
    return f"h_{n}"


def act_energy(n: str) -> str:
    return f"activation_energy_{n}"


def q10(n: str) -> str:
    return f"q10_{n}"


###############################################################################
# Parameters / Variables
###############################################################################


def a0(compartment: str = EMPTY, tissue: str = EMPTY) -> str:
    r"""Photosystem II reaction center 0"""
    return loc("A0", compartment, tissue)


def p700_fa(compartment: str = EMPTY, tissue: str = EMPTY) -> str:
    return loc("P700FA", compartment, tissue)


def p700_plus_fa_minus(compartment: str = EMPTY, tissue: str = EMPTY) -> str:
    return loc("P700+FA-", compartment, tissue)


def p700_fa_minus(compartment: str = EMPTY, tissue: str = EMPTY) -> str:
    return loc("P700FA-", compartment, tissue)


def p700_plus_fa(compartment: str = EMPTY, tissue: str = EMPTY) -> str:
    return loc("P700+FA", compartment, tissue)


def a1(compartment: str = EMPTY, tissue: str = EMPTY) -> str:
    r"""Photosystem II reaction center 1"""
    return loc("A1", compartment, tissue)


def a2(compartment: str = EMPTY, tissue: str = EMPTY) -> str:
    r"""Photosystem II reaction center 2"""
    return loc("A2", compartment, tissue)


def b0(compartment: str = EMPTY, tissue: str = EMPTY) -> str:
    return loc("B0", compartment, tissue)


def b1(compartment: str = EMPTY, tissue: str = EMPTY) -> str:
    return loc("B1", compartment, tissue)


def b2(compartment: str = EMPTY, tissue: str = EMPTY) -> str:
    return loc("B2", compartment, tissue)


def b3(compartment: str = EMPTY, tissue: str = EMPTY) -> str:
    return loc("B3", compartment, tissue)


def fd_ox(compartment: str = EMPTY, tissue: str = EMPTY) -> str:
    return loc("Ferredoxine (oxidised)", compartment, tissue)


def fd_red(compartment: str = EMPTY, tissue: str = EMPTY) -> str:
    return loc("Ferredoxine (reduced)", compartment, tissue)


def pc_ox(compartment: str = EMPTY, tissue: str = EMPTY) -> str:
    return loc("Plastocyanine (oxidised)", compartment, tissue)


def pc_red(compartment: str = EMPTY, tissue: str = EMPTY) -> str:
    return loc("Plastocyanine (reduced)", compartment, tissue)


def pq_ox(compartment: str = EMPTY, tissue: str = EMPTY) -> str:
    return loc("Plastoquinone (oxidised)", compartment, tissue)


def pq_red(compartment: str = EMPTY, tissue: str = EMPTY) -> str:
    return loc("Plastoquinone (reduced)", compartment, tissue)


def pfd(compartment: str = EMPTY, tissue: str = EMPTY) -> str:
    r"""Photosynthetic Photon Flux Density"""
    return loc("PPFD", compartment, tissue)


def h2o2(compartment: str = EMPTY, tissue: str = EMPTY) -> str:
    return loc("H2O2", compartment, tissue)


def ps2cs(compartment: str = EMPTY, tissue: str = EMPTY) -> str:
    return loc("PSII_cross_section", compartment, tissue)


def o2(compartment: str = EMPTY, tissue: str = EMPTY) -> str:
    return loc("O2 (dissolved)", compartment, tissue)


def h(compartment: str = EMPTY, tissue: str = EMPTY) -> str:
    return loc("H", compartment, tissue)


atp = "ATP"
adp = "ADP"
pq = "PQ"
pqh2 = "PQH2"
proton = "H"

quencher = "Quencher act."
zx = "Zx"
vx = "Vx"
psbs = "PsbS"
psbsp = "PsbSP"

pc = "PC"  # oxidised plastocyan
fd = "Fd"  # oxidised ferrodoxin
nadph = "NADPH"  # stromal concentration of NADPH
lhc = "LHC"

atpact = "ATPactivity"

fluo = "Fluo"
light = "PPFD"
fr_light = "pfd_fr"

temp_c = "temperature(deg.C)"
temp_k = "T"  # modified for now

k_lumen = "K_lumen"
k_stroma = "K_stroma"

h_lumen = "H_lumen"
h_stroma = "H_stroma"

ph_lumen = "pH_lumen"
ph_stroma = "pH_stroma"

pmf = "pmf(V)"
delta_psi = "delta_psi"  # r'$\Delta \Psi$'
delta_ph = "delta_pH"  # r'$\Delta pH$'
delta_mu_proton = "delta_mu_proton"  # r'$\Delta \tilde{\mu}_{H^+}$'
