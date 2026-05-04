"""Pfennig 2024 cyanobacteria model.

Reference: Tobias Pfennig, Elena Kullmann, Tomáš Zavřel, Andreas Nakielski, \
Oliver Ebenhöh, Jan Červený, Gábor Bernát, Anna Barbara Matuszyńska
"Shedding light on blue-green photosynthesis: A wavelength-dependent mathematical model\
of photosynthesis in Synechocystis sp. PCC 6803"
PLOS Computational Biology 20.9 (2024): e1012445.
doi: 10.1371/journal.pcbi.1012445

Terms
-----
cbb: Calvin-Benson-Bassham cycle
ocp: orange carotenoid protein
pbs: phycobilisome
psi: photosystem I
psii: photosystem II
lcf: light conversion factor
pq: plastoquinone
pc: plastocyanine
fd: ferredoxin
Ho: cytoplasmic protons
Hi: lumenal protons
CCM: carbon-concentrating mechanism
FNR: Ferredoxin-NADP+ Reductase
2PG: 2-phosphoglycolate
3PGA: 3-phosphoglycerate
ADP: Adenosine diphosphate
ATP: Adenosine triphosphate
ATPsynth: ATP synthase
COX: Cytochrome c oxidase
Cyd: Cytochrome bd quinol oxidase
Cyt b6f: Cytochrome b6f complex
Flv 1/3: Flavodiiron protein dimer 1/3
NADP+: Nicotinamide adenine dinucleotide phosphate
NADPH: reduced Nicotinamide adenine dinucleotide phosphate
NAD+: Nicotinamide adenine dinucleotide
NADH: reduced Nicotinamide adenine dinucleotide
NDH-1: NAD(P)H Dehydrogenase-like complex 1
NDH-2: NAD(P)H Dehydrogenase complex 2
Oxy: RuBisCO oxygenation
PR: Photorespiration
SDH: Succinate dehydrogenase
"""

from typing import Literal, cast

import numpy as np
import pandas as pd
from mxlpy import Derived, Model, fns
from mxlpy.surrogates import qss
from scipy.integrate import simpson

parameters: dict[str, float] = {
    "PSIItot": 0.830,  # [mmol mol(Chl)^-1] total concentration of photosystem II complexes (Zavrel2023)
    "PSItot": 3.270,  # [mmol mol(Chl)^-1] total concentration of photosystem I complexes (Zavrel2023)
    "PQ_tot": 13.000,  # [mmol mol(Chl)^-1] total PHOTOACTIVE PQ concentration (Khorobrykh2020)
    "PC_tot": 1.571,  # [mmol mol(Chl)^-1] total concentration of plastocyanin (PC_ox + PC_red) (Zavrel2019)
    "Fd_tot": 3.597,  # [mmol mol(Chl)^-1] total concentration of ferredoxin (Fd_ox + Fd_red) (Moal2012)
    "NADP_tot": 26.805,  # [mmol mol(Chl)^-1] total concentration of NADP species (NADP + NADPH) (Kauny2014)
    "NAD_tot": 11.169,  # [mmol mol(Chl)^-1] total concentration of NAD species (NAD + NADH) (Tanaka2021)
    "AP_tot": 430.143,  # [mmol mol(Chl)^-1] total concentration of adenosine species (ADP + ATP) (Doello2018)
    "O2ext": 55.402,  # [mmol mol(Chl)^-1] concentration of oxygen in the surrounding medium (Kihara2014)
    "CO2ext": 3.103,  # [mmol mol(Chl)^-1] saturated concentration of CO2 in 25 °C water with ~10 ‰ Cl^- ions (Li1971)
    "F": 96.485,  # [C mmol^-1] Faraday's constant (Richardson2019)
    "R": 8.300e-03,  # [J K^-1 mmol^-1] ideal gas constant (Richardson2019)
    "T": 298.150,  # [K] temperature (set)
    "bHi": 100.000,  # [unitless] buffering constant of the thylakoid lumen (estimated)
    "bHo": 1.111e03,  # [unitless] buffering constant of the cytoplasm, assumed to be 1/f_V_lumen times larger (estimated)
    "cf_lumen": 4.613e-05,  # [mol(Chl) ml^-1] conversion factor for [mmol mol(Chl)^-1] -> [mol l^-1] for the thylakoid lumen (derived)
    "cf_cytoplasm": 4.562e-06,  # [mol(Chl) ml^-1] conversion factor for [mmol mol(Chl)^-1] -> [mol l^-1] for the cytoplasm (derived)
    "E0_QA": -0.140,  # [V] standard electrode potential of the reduction of PS2 plastoquinone A (Lewis2022)
    "E0_PQ": 0.533,  # [V] standard electrode potential of the reduction of free plastoquinone (Lewis2022)
    "E0_PC": 0.350,  # [V] standard electrode potential of the reduction of free plastocyanin (Lewis2022)
    "E0_P700": 0.410,  # [V] standard electrode potential of the reduction of the oxidised PS1 reaction center (Lewis2022)
    "E0_FA": -0.580,  # [V] standard electrode potential of the reduction of PS1 iron-sulfur cluster A (Lewis2022)
    "E0_Fd": -0.410,  # [V] standard electrode potential of the reduction of free ferredoxin (Lewis2022)
    "E0_NADP": -0.113,  # [V] standard electrode potential of the reduction of NADP to NADPH (Falkowski2007)
    "E0_succinate/fumarate": 0.443,  # [V] standard electrode potential of the reduction of fumarate to succinate (Falkowski2007)
    "kH0": 5.000e08,  # [s^-1] rate constant of (unregulated) excitation quenching by heat (Ebenhoh2014)
    "kHst": 1.000e09,  # [s^-1] rate constant of state transition regulated excitation quenching by heat (guess)
    "kF": 6.250e08,  # [s^-1] rate constant of excitation quenching by fluorescence (Ebenhoh2014)
    "k2": 2.500e09,  # [s^-1] rate constant of excitation quenching by photochemistry (Ebenhoh2014,Bernat2009)
    "kPQred": 250.000,  # [mol(Chl) mmol^-1 s^-1] rate constant of PQ reduction via PS2 (Matuszynska2019)
    "kFlvactivation": 0.500,  # [s^-1] rate constant of Flv activation by reduced Fd (manually fitted)
    "kFlvdeactivation": 0.100,  # [s^-1] rate constant of Flv deactivation by oxidised Fd (manually fitted)
    "kCBBdeactivation": 2.500e-03,  # [s^-1] rate constant of CBB deactivation by oxidised Fd (manually fitted)
    "kPCox": 2.500e03,  # [mol(Chl) mmol^-1 s^-1] rate constant of PC oxidation via PS1 (Matuszynska2019)
    "kFdred": 2.500e05,  # [mol(Chl) mmol^-1 s^-1] rate constant of Fd reduction via PS1 (Matuszynska2019)
    "k_ox1": 0.211,  # [mol(Chl)^2 mmol^-2 s^-1] rate constant of oxygen reduction via bd-type (Cyd) terminal oxidases (Ermakova2016)
    "k_NDH": 3.136,  # [mol(Chl) mmol^-1 s^-1] rate constant of PQ reduction by NDH-2 (Cooley2001)
    "k_SDH": 0.430,  # [mol(Chl) mmol^-1 s^-1] rate constant of PQ reduction by SDH (Cooley2001)
    "k_FN_fwd": 3.730,  # [mol(Chl) mmol^-1 s^-1] rate constant of NADP reduction by FNR (Kauny2014)
    "k_FN_rev": 53.364,  # [mol(Chl)^3 mmol^-3 s^-1] rate constant of reverse flux through FNR in darkness (Kauny2014)
    "k_NQ": 100.000,  # [mol(Chl) mmol^-1 s^-1] rate constant of PQ reduction by NDH (Theune2021)
    "kRespiration": 3.506e-06,  # [mol(Chl)^3 mmol^-3 s^-1] rate constant of 3PGA oxidation and fumarate reduction by glycolysis and the TCA cycle (Ermakova2016)
    "kO2out": 4.025e03,  # [s^-1] rate constant of oxygen diffusion out of the cell (Kihara2014)
    "kCCM": 4.025e03,  # [s^-1] rate constant of CO2 diffusion into the cell, assumed to be identical to kO2out (guess)
    "PBS_free": 9.000e-02,  # [unitless] fraction of unbound PBS (Zavrel2023)
    "PBS_PS1": 0.390,  # [unitless] fraction of PBS bound to PSI (Zavrel2023)
    "PBS_PS2": 0.510,  # [unitless] fraction of PBS bound to PSII (Zavrel2023)"
    "Pi_mol": 1.000e-02,  # [mmol mol(Chl)^-1] molar conccentration of phosphate (Matuszynska2019)
    "DeltaG0_ATP": 30.600,  # [kJ mol^-1] energy of ATP formation (Matuszynska2019)
    "HPR": 4.667,  # [unitless] number of protons (14) passing through the ATP synthase per ATP (3) synthesized (Pogoryelov2007)
    "vOxy_max": 16.059,  # [mmol mol(Chl)^-1 s^-1] approximate Rubisco oxygenation rate (Savir2010)
    "KMATP": 24.088,  # [mmol mol(Chl)^-1] order of magnitude of the michaelis constant for ATP consumption in the CBB cycle (Wadano1998,Tsukamoto2013)
    "KMNADPH": 18.066,  # [mmol mol(Chl)^-1] approxiate michaelis constant for NADPH consumption in the CBB cycle (Koksharova1998)
    "KMCO2": 72.264,  # [mmol mol(Chl)^-1] order of magnitude of the michaelis constant for CO2 consumption by cyanobacterial Rubisco (Savir2010)
    "KIO2": 240.880,  # [mmol mol(Chl)^-1] order of magnitude of the michaelis inhibition constant of O2 for CO2 consumption by cyanobacterial Rubisco, assumed equal to KMO2 (Savir2010)
    "KMO2": 240.880,  # [mmol mol(Chl)^-1] order of magnitude of the michaelis constant for O2 consumption by cyanobacterial Rubisco (Savir2010)
    "KICO2": 72.264,  # [mmol l^-1] order of magnitude of the michaelis inhibition constant of CO2 for O2 consumption by cyanobacterial Rubisco, assumed equal to KMCO2 (Savir2010)
    "vCBB_max": 51.301,  # [mmol mol(Chl)^-1 s^-1] approximate maximal rate of the Calvin Benson Bassham cycle (Zavrel2017)
    "kPR": 1.523e-06,  # [mol(Chl)^4 mmol^-4 s^-1] rate constant of (2-phospho)glycolate recycling into 3PGA (Huege2011)
    "cChl": 4.151e-03,  # [mol l^-1] total molar concentration of chlorophyll (derived)
    "CO2ext_pp": 5.000e-02,  # [atm] CO2 partial pressure in 5% CO2 enriched air used for bubbeling (set)
    "S": 35.000,  # [unitless] salinity within a cell (MojicaPrieto2002)
    "k_O2": 12.926,  # [mol(Chl)^2 mmol^-2 s^-1] rate constant of Fd oxidation by Flv 1/3 (Ermakova2016)
    "KHillFdred_CBB": 1.000e-04,  # [mmol^nHillFdred_CBB mol(Chl)^-nHillFdred_CBB] Apparent dissociation constant of reduced ferredoxin for CBB activation (Schuurmans2014)
    "nHillFdred_CBB": 4.000,  # [unitless] Hill coefficient of CBB activation by reduced ferredoxin (guess)
    "fCin": 1032.3825198828176,
    "k_F1": 1.0144691550897802,
    "k_Q": 1789.6877149900638,
    "k_pass": 0.0103757688087645,
    "k_aa": 1.0726342220286855,
    "fluo_influence_ps2": 1.0871604554468057,
    "fluo_influence_ps1": 1.0580747331180056,
    "fluo_influence_pbs": 1.2652931291892002,
    "lcf": 0.4852970468572075,
    "KMPGA": 0.108119035344566,
    "kATPsynth": 10.508685365751944,
    "kATPconsumption": 0.3007216562727887,
    "kNADHconsumption": 10.774127129643528,
    "kUnquench": 0.1084895168082126,
    "KMUnquench": 0.2057753667172297,
    "kQuench": 0.0021298894016293,
    "KHillFdred": 39.138885950985646,
    "nHillFdred": 3.803733201004677,
    "kCBBactivation": 0.0368469911771887,
    "KMFdred": 0.2867372651572032,
    "kOCPactivation": 9.119349718302952e-05,
    "kOCPdeactivation": 0.0013779872861697,
    "OCPmax": 0.2894610142573665,
    "vNQ_max": 47.52413111835536,
    "KMNQ_Qox": 1.225188500783819,
    "KMNQ_Fdred": 1.3324893590389415,
}


###############################################################################
# Data creation
###############################################################################

# def effective_irradiance(I0, depth, absorption_coef, chlorophyll_sample):
#     return (
#         I0
#         * 1
#         / (-absorption_coef * chlorophyll_sample)
#         * np.exp(-depth * absorption_coef * chlorophyll_sample)
#     )


# def get_mean_sample_light(
#     I0: pd.Series,
#     depth: float,
#     absorption_coef: pd.Series,
#     chlorophyll_sample: float,
#     depth0: float = 0,
# ) -> pd.Series:
#     """Calculate the light experienced by an average cell in a sample

#     Args:
#         I0 (Union[pd.Series, np.ArrayLike]): initial light as specific irradiance per wavelength OR a list of such
#         depth (float): depth of the sample in which the irradiance is exponentially attenuated
#         absorption_coef (pd.Series): specific absorption coefficients per wavelength
#         chlorophyll_sample (float): chlorophyll content of the sample
#         depth0 (float, optional): depth of the initial light. Defaults to 0.

#     Returns:
#         pd.Series: adjusted light vector OR a list of such
#     """
#     # Calculate the adjustment for a single irradiace or a list of such
#     if isinstance(I0, pd.Series):
#         I_diff = effective_irradiance(
#             I0, depth, absorption_coef, chlorophyll_sample
#         ) - effective_irradiance(I0, depth0, absorption_coef, chlorophyll_sample)
#         return I_diff / (depth - depth0)

#     elif isinstance(I0, (list, np.ndarray)):
#         if np.all([isinstance(i, pd.Series) for i in I0]):
#             I_diff = [
#                 (
#                     effective_irradiance(
#                         Icurr, depth, absorption_coef, chlorophyll_sample
#                     )
#                     - effective_irradiance(
#                         Icurr, depth0, absorption_coef, chlorophyll_sample
#                     )
#                 )
#                 / (depth - depth0)
#                 for Icurr in I0
#             ]
#             return I_diff

#     raise ValueError("I0 must be a pd.Series or a list of such")


# Light sources
def light_gaussian_led(
    wavelength: float,
    intensity: float = 1.0,
    spread: float = 10,
) -> pd.Series:
    """Get a range of normalised light intensities (400 nm, 700 nm) describing a gaussean LED

    Args
    ----
    wavelength (float): wavelength with maximal intensity
    intensity (float, optional): integrated light intensity. Defaults to 1.
    spread (float, optional): spread (variance) of the distribution. Defaults to 10.

    Returns
    -------
    ArrayLike: an array of wavelength-specific light intensities

    """
    x = np.arange(390, 711)
    light = pd.Series(
        np.exp(-np.power(x - wavelength, 2.0) / (2 * np.power(spread, 2.0))),
        name="gaussian_led",
        index=pd.Index(x, name="wavelength"),
    )
    light = light / simpson(light) * intensity
    return light.loc[400:700]


###############################################################################
# Light functions
###############################################################################


def _get_pigment_association(
    *,
    ps1_ratio: float,
    # data
    ps_comp: pd.DataFrame,
    molar_masses: pd.Series,
    pigment_content: pd.Series,
) -> pd.DataFrame:
    """Create a DataFrame containing the fractions of pigments associated to the main complexes (PS1, PS2. PBS)

    Args:
        ps1_ratio (float): ratio of PS1:PS2
        beta_carotene_method (str): how to calculate the beta carotene energy donation, one of 'original' or 'stoichiometric'
        pigments (pd.DataFrame): DataFrame with the pigment amounts [mg(pigment) mg(chlorophyll a)^-1] in named columns
        verbose (bool): issue a warning if provided and calculated beta carotene amounts don't match

    Returns: pd.DataFrame, with contents like this
    |     |     chla |   beta_carotene |   allophycocyanin |   phycocyanin |
    |:----|---------:|----------------:|------------------:|--------------:|
    | ps1 | 0.215247 |        0.166667 |                 0 |             0 |
    | ps2 | 0.784753 |        0        |                 0 |             0 |
    | pbs | 0        |        0        |                 1 |             1 |

    """
    df = pd.DataFrame(
        np.zeros((3, 4)),
        index=["ps1", "ps2", "pbs"],
        columns=["chla", "beta_carotene", "allophycocyanin", "phycocyanin"],
    )
    df.loc["pbs", ["allophycocyanin", "phycocyanin"]] = 1.0

    # Adapt the ratio tab in _ps_comp
    ps_comp.loc["ratio", "ps1"] = ps1_ratio

    # Chlorophyll a: Fraction of total pool
    chla = ps_comp.loc[["ratio", "n_chla"], :].prod(axis=0)
    chla = chla / chla.sum()

    # Beta carotene: Fraction of total pool wih adjustment factor
    beta_carotene = ps_comp.loc[["ratio", "n_beta_carotene"], :].prod(axis=0)
    beta_carotene = beta_carotene / beta_carotene.sum()

    # Calculate the molar ratio of beta carotene following from the stoichiometry
    bc_stoich = ps_comp.loc[["ratio", "n_beta_carotene"], :].prod(axis=0) * (
        1 / ps_comp.loc["ratio", :].sum()
    )
    bc_stoich = bc_stoich.mul(1 / ps_comp.loc["n_chla", :]).sum()  # type: ignore

    # Compare with the measured value and calculate a proportionality factor
    bc_measured = (
        pigment_content.loc["beta_carotene"]
        * molar_masses["chla"]
        / molar_masses["beta_carotene"]
    )
    bc_fac = np.min(np.array([bc_stoich / bc_measured, 1]))

    # Also take into account that only ps1 is excited by carotenoid absorption

    df.loc[["ps1", "ps2"], "chla"] = chla
    df.loc[["ps1", "ps2"], "beta_carotene"] = beta_carotene * pd.Series(
        {"ps1": bc_fac, "ps2": 0}
    )
    return df


def _absorption_per_complex(
    ps_ratio: float,
    # data
    light_spectrum: pd.Series,
    pigment_content: pd.Series,
    molar_masses: pd.Series,
    abs_coef: pd.DataFrame,
    ps_comp: pd.DataFrame,
) -> tuple[float, float, float]:
    """Compute spectrally integrated light absorption for PSI, PSII, and PBS.

    Weights wavelength-resolved absorption coefficients by pigment content and
    pigment-to-complex association, then integrates over the incident light spectrum.
    Returns absorption values in [mg(Chla) mmol(Chl)^-1 * (absorption unit)] for
    each complex as (ps1, ps2, pbs).
    """
    absorption_per_pigment = (
        (abs_coef * pigment_content)  # nm x pigment
        .mul(light_spectrum, axis=0)  # nm x pigment
        .apply(
            simpson,  # type: ignore
            axis=0,
        )  # pigment
    )
    pigment_association_per_complex = _get_pigment_association(
        ps1_ratio=ps_ratio,
        pigment_content=pigment_content,
        ps_comp=ps_comp,
        molar_masses=molar_masses,
    )  # complex x pigment
    _absorption_per_complex = (
        pigment_association_per_complex.dot(absorption_per_pigment)
        * molar_masses["chla"]
    )  # complex
    return (
        _absorption_per_complex["ps1"],
        _absorption_per_complex["ps2"],
        _absorption_per_complex["pbs"],
    )


###############################################################################
# Rate functions
###############################################################################


def _p_hlumen(
    x: float,
) -> float:
    """Convert lumenal proton concentration [mmol mol(Chl)^-1] to pH using the lumen conversion factor."""
    return -np.log(x * (2.9e-5)) / np.log(10)


def _p_hcytoplasm(
    x: float,
) -> float:
    """Convert cytoplasmic proton concentration [mmol mol(Chl)^-1] to pH using the cytoplasm conversion factor."""
    return -np.log(x * (4.8e-6)) / np.log(10)


def _atp_synthase(
    hi: float,
    ho: float,
    atp: float,
    adp: float,
    delta_g0_atp: float,
    d_g_p_h: float,
    hpr: float,
    pi_mol: float,
    rt: float,
    k_at_psynth: float,
) -> float:
    """Rate of ATP synthesis driven by the transmembrane proton gradient (ΔpH).

    Uses a mass-action kinetic with a pH-dependent equilibrium constant derived
    from the free energy of ATP hydrolysis and the proton motive force across the
    thylakoid membrane. Units: [mmol mol(Chl)^-1 s^-1].
    """
    pHlumen = _p_hlumen(hi)
    p_hcytoplasm = _p_hcytoplasm(ho)
    DG = delta_g0_atp - d_g_p_h * hpr * (p_hcytoplasm - pHlumen)
    Keq = pi_mol * np.exp(-DG / rt)
    return k_at_psynth * (adp - atp / Keq)


def _stoich_hi_at_psynthase(
    hpr: float,
    b_hi: float,
) -> float:
    """Stoichiometric factor for Hi: HPR lumenal protons consumed per ATP synthesized, scaled by lumen buffering capacity."""
    return -hpr / b_hi


def _stoich_ho_at_psynthase(
    hpr: float,
    b_ho: float,
) -> float:
    """Stoichiometric factor for Ho: HPR cytoplasmic protons released per ATP synthesized, scaled by cytoplasm buffering capacity."""
    return hpr / b_ho


def _stoich_ho_v_nad_hconsumption(
    b_ho: float,
) -> float:
    """Stoichiometric factor for Ho: 1 cytoplasmic proton consumed per NADH oxidised, scaled by cytoplasm buffering capacity."""
    return -1 / b_ho


def _atp_consumption(
    atp: float,
    k_at_pconsumption: float,
) -> float:
    """Lumped first-order ATP consumption rate representing cellular maintenance and biosynthesis demand [mmol mol(Chl)^-1 s^-1]."""
    return k_at_pconsumption * atp


def _nadph_consumption(
    nadh: float,
    ho: float,
    k_nad_hconsumption: float,
) -> float:
    """Lumped NADH consumption rate for cytoplasmic biosynthetic reactions, proportional to NADH and cytoplasmic proton concentration [mmol mol(Chl)^-1 s^-1]."""
    return k_nad_hconsumption * nadh * ho


# Add the parameter dG_pH which is used in the calculation of equilibrium constants when protons take part
def _d_g_p_h(
    r: float,
    t: float,
) -> float:
    """Proton free energy coefficient RT·ln(10) [kJ mol^-1], used to convert pH differences to Gibbs free energy contributions."""
    return np.log(10) * r * t


# Add the ratio of photosystems (PS1:PS2) which is used in light handling
def _ps_ratio(
    ps_itot: float,
    psi_itot: float,
) -> float:
    """Molar PSI:PSII ratio corrected for subunit stoichiometry (3 Chl per PSI monomer vs 2 per PSII monomer)."""
    return (ps_itot * 3) / (psi_itot * 2)


# Define the functions used for calculation of pH/molar proton concentrations
def _molar_hlumen(
    h: float,
    cf_lumen: float,
) -> float:
    """Calculate the molar concentration of protons in the thylakoid lumen from the chlorophyll-normalised concentration.

    Uses a conversion factor dependent on the volume of the thylakoid lumen.
    """
    return h * cf_lumen  # (2.9e-5)


def _molar_hcytoplasm(
    h: float,
    cf_cytoplasm: float,
) -> float:
    """
    Calculate the molar concentration of protons in the cytoplasm from the chlorophyll-normalised concentration.

    Uses a conversion factor dependent on the volume of the cytoplasm.
    """
    return h * cf_cytoplasm  # (4.8e-6)


def _p_h(
    h_mol: float,
) -> float:
    """Compute pH from molar proton concentration [mol l^-1]."""
    return -np.log(h_mol) / np.log(10)


def _deltap_h(
    lumen: float,
    cytoplasm: float,
) -> float:
    """Transmembrane ΔpH = pHcytoplasm - pHlumen; positive when lumen is more acidic than cytoplasm."""
    return cytoplasm - lumen


def _ps_normabsorption_tot(
    light_ps1: float,
    light_ps2: float,
    light_ps1_ml: float,
    light_ps2_ml: float,
) -> tuple[float, float]:
    """Sum growth-light and measuring-light absorption for PSI and PSII to give total absorbed excitation per photosystem."""
    return (light_ps1 + light_ps1_ml, light_ps2 + light_ps2_ml)


def _moiety_1(
    part: float,
    total: float,
) -> float:
    """Compute the complementary pool of a conserved moiety: total - part (e.g. PQ_red = PQ_tot - PQ_ox)."""
    return total - part


def _keq_p_qred(
    p_hcytoplasm: float,
    e0_qa: float,
    f: float,
    e0_pq: float,
    d_g_p_h: float,
    rt: float,
) -> float:
    """pH-dependent equilibrium constant for PQ reduction by the PSII primary acceptor QA."""
    DG1 = -e0_qa * f
    DG2 = -2 * e0_pq * f + 2 * d_g_p_h * p_hcytoplasm
    DG = -2 * DG1 + DG2
    return np.exp(-DG / rt)


def _ps2states(
    psi_iq: float,
    pq: float,
    p_qred: float,
    keq_p_qred: float,
    light_ps2: float,
    psi_itot: float,
    k2: float,
    k_f: float,
    k_h0: float,
    k_hst: float,
    k_p_qred: float,
) -> tuple[float, float, float, float]:
    """Analytical quasi-steady-state solution for the four PSII state occupancies.

    Returns (B0, B1, B2, B3) where B0 = dark-relaxed open PSII, B1 = charge-separated
    (closed) PSII with oxidised PQ, B2 = open PSII with reduced PQ, B3 = closed PSII
    with reduced PQ. State-transition quenching shifts PSIIq from the unquenched pool.
    Units: [mmol mol(Chl)^-1].
    """
    # Calculate some rates
    kH = k_h0 + k_hst * (psi_iq / psi_itot)
    k3p = k_p_qred * pq
    k3m = k_p_qred * p_qred / keq_p_qred

    x0 = k2 * k3p
    x1 = k_f * x0
    x2 = kH * x0
    x3 = k_f**2
    x4 = k3p * x3
    x5 = kH**2
    x6 = k3p * x5
    x7 = 2 * k_f * kH
    x8 = k3p * x7
    x9 = light_ps2 * k2
    x10 = k3m * x9
    x11 = k_f * x9
    x12 = kH * x9
    x13 = light_ps2 * k3m
    x14 = k_f * x13
    x15 = kH * x13
    x16 = light_ps2 * k3p
    x17 = k_f * x16
    x18 = kH * x16
    x19 = k2 * k3m
    x20 = k_f * x19
    x21 = kH * x19
    x22 = light_ps2**2 * k2
    x23 = k3m * x3
    x24 = k3m * x5
    x25 = k3m * x7
    x26 = (
        x1
        + x10
        + x11
        + x12
        + x14
        + x15
        + x17
        + x18
        + x2
        + x20
        + x21
        + x22
        + x23
        + x24
        + x25
        + x4
        + x6
        + x8
    ) ** (-1.0)
    return (
        x26
        * (
            psi_itot * x1
            + psi_itot * x2
            + psi_itot * x4
            + psi_itot * x6
            + psi_itot * x8
        ),
        x26 * (psi_itot * x17 + psi_itot * x18),
        x26
        * (
            psi_itot * x11
            + psi_itot * x12
            + psi_itot * x20
            + psi_itot * x21
            + psi_itot * x23
            + psi_itot * x24
            + psi_itot * x25
        ),
        x26 * (psi_itot * x10 + psi_itot * x14 + psi_itot * x15 + psi_itot * x22),
    )


def _psii(
    b1: float,
    k2: float,
) -> float:
    """PSII photochemical rate: electron transfer from water to PQ via charge-separated state B1 [mmol mol(Chl)^-1 s^-1].

    Scaled by 0.5 because k2 is defined per single-electron step (B1→B2) but
    the full PSII reaction extracts 2 electrons per PQ reduction.
    """
    return (
        0.5 * k2 * b1
    )  # k2 is scaled to single electron extraction (B1 -> B2) and has to be scaled by 0.5 for 2-electron PS2 reaction


def _keq_v_sdh(
    p_hcytoplasm: float,
    e0_pq: float,
    f: float,
    e0_succinate_fumarate: float,
    rt: float,
    d_g_p_h: float,
) -> float:
    """pH-dependent equilibrium constant for PQ reduction by succinate dehydrogenase (SDH)."""
    DG1 = -2 * e0_pq * f + 2 * d_g_p_h * p_hcytoplasm
    DG2 = -2 * e0_succinate_fumarate * f + 2 * d_g_p_h * p_hcytoplasm
    DG = DG1 - DG2
    return np.exp(-DG / rt)


def _v_sdh(
    q: float,
    succinate: float,
    fumarate: float,
    pqred: float,
    keq_sdh: float,
    ksdh: float,
) -> float:
    """Electron flow via SDH"""
    return ksdh * (q * succinate - (pqred * fumarate) / (keq_sdh))


def _v_respiration(
    pga: float,
    fumarate: float,
    adp: float,
    nad: float,
    nadp: float,
    k_respiration: float,
    kmpga: float,
) -> float:
    """Approximation of respiration resulting in the consumption of 3PGA with generation of ATP, NADPH, NADH, and succinate (from fumarate)"""
    return k_respiration * fumarate * adp * nad * nadp * pga / (pga + kmpga)


def _vbd(
    o2: float,
    qred: float,
    ho: float,
    kox1: float,
) -> float:  # q, Kbd,   water ignored
    """Electron flow from PQ to O2 via bd-type (Cyd) terminal oxidase, consuming PQH2 and producing water [mmol mol(Chl)^-1 s^-1]."""
    return kox1 * qred * o2 * ho  # - q/Kbd)


def _vb6f(
    pc: float,
    ho: float,
    qred: float,
    kq: float,
) -> float:
    """Electron flow from PQH2 to PC via Cyt b6f complex, pumping protons into the lumen via the Q-cycle [mmol mol(Chl)^-1 s^-1]."""
    return kq * qred * pc * ho


def _vaa(
    pcred: float,
    o2: float,
    ho: float,
    kaa: float,
) -> float:  # pc, hi, Kaa
    """Electron flow from PC to O2 via aa3-type cytochrome c oxidase (COX), consuming O2 and pumping protons [mmol mol(Chl)^-1 s^-1]."""
    return kaa * pcred * o2 * ho


def _keq_fa_fd(
    e0_fa: float,
    f: float,
    e0_fd: float,
    rt: float,
) -> float:
    """Equilibrium constant for electron transfer from PSI iron-sulfur cluster FA to free ferredoxin (Fd)."""
    DG1 = -e0_fa * f
    DG2 = -e0_fd * f
    DG = -DG1 + DG2
    return np.exp(-DG / rt)


def _keq_pcp700(
    e0_pc: float,
    f: float,
    e0_p700: float,
    rt: float,
) -> float:
    """Equilibrium constant for electron transfer from reduced PC to the oxidised PSI reaction centre P700+."""
    DG1 = -e0_pc * f
    DG2 = -e0_p700 * f
    DG = -DG1 + DG2
    return np.exp(-DG / rt)


def _calc_a(
    fd_red: float,
    keq_fa_fd: float,
    k_l1: float,
    k_fdred: float,
) -> float:
    """Intermediate variable a in the analytical QSS solution for PSI states: back-transfer rate from FA to Fd minus light rate."""
    return (k_fdred / keq_fa_fd) * fd_red - k_l1


def _calc_b(
    fd_ox: float,
    k_l1: float,
    k_f1: float,
    k_fdred: float,
) -> float:
    """Intermediate variable b in the analytical QSS solution for PSI states: sum of Fd reduction, fluorescence decay, and light absorption rate constants."""
    return k_fdred * fd_ox + k_f1 + k_l1


def _calc_c(
    pc_ox: float,
    keq_pcp700: float,
    k_l1: float,
    k_p_cox: float,
) -> float:
    """Intermediate variable c in the analytical QSS solution for PSI states: reverse PC-to-P700 transfer rate plus light rate."""
    return (k_p_cox / keq_pcp700) * pc_ox + k_l1


def _calc_d(
    a: float,
    b: float,
    pc_red: float,
    k_f1: float,
    k_p_cox: float,
) -> float:
    """Intermediate variable d in the analytical QSS solution for PSI states: combined PC reduction and fluorescence flux."""
    return k_f1 * a / b + k_p_cox * pc_red


def _calc_f(
    a: float,
    b: float,
    d: float,
) -> float:
    """Intermediate variable f in the analytical QSS solution for PSI states: normalisation factor for PSI state distribution."""
    return (1 + a / b) / d


def _calc_y0(
    b: float,
    c: float,
    f: float,
    ps_itot: float,
    k_l1: float,
    k_f1: float,
) -> float:
    """QSS concentration of open (ground-state) PSI [mmol mol(Chl)^-1]: P700 reduced, FA oxidised, able to absorb light."""
    return (ps_itot * (1 - (k_l1 / b) * (1 - k_f1 * f))) / (1 + c * f)


def _calc_y2(
    y0: float,
    b: float,
    c: float,
    d: float,
    ps_itot: float,
    k_l1: float,
    k_f1: float,
) -> float:
    """QSS concentration of doubly-reduced PSI state Y2 [mmol mol(Chl)^-1]: P700 reduced, FA reduced."""
    return (y0 * c - (k_f1 * k_l1 * ps_itot) / b) / d


def _calc_y1(
    y2: float,
    a: float,
    b: float,
    ps_itot: float,
    k_l1: float,
) -> float:
    """QSS concentration of charge-separated PSI state Y1 [mmol mol(Chl)^-1]: P700 oxidised, FA reduced."""
    return y2 * a / b + (k_l1 * ps_itot) / b


def _ps1_states(
    fd_ox: float,
    fd_red: float,
    pc_ox: float,
    pc_red: float,
    light_ps1: float,
    keq_pcp700: float,
    keq_fa_fd: float,
    ps_itot: float,
    k_f1: float,
    k_p_cox: float,
    k_fdred: float,
) -> tuple[float, float, float]:
    """Analytical quasi-steady-state solution for PSI state occupancies.

    Returns (Y0, Y1, Y2) where Y0 = open PSI (P700 reduced, FA oxidised),
    Y1 = charge-separated PSI (P700 oxidised, FA reduced), and
    Y2 = doubly-reduced PSI (P700 reduced, FA reduced). Units: [mmol mol(Chl)^-1].
    """
    k_l1 = light_ps1

    a = _calc_a(
        fd_red,
        keq_fa_fd,
        k_l1,
        k_fdred,
    )
    b = _calc_b(
        fd_ox,
        k_l1,
        k_f1,
        k_fdred,
    )
    c = _calc_c(
        pc_ox,
        keq_pcp700,
        k_l1,
        k_p_cox,
    )
    d = _calc_d(
        a,
        b,
        pc_red,
        k_f1,
        k_p_cox,
    )
    f = _calc_f(
        a,
        b,
        d,
    )

    y0 = _calc_y0(
        b,
        c,
        f,
        ps_itot,
        k_l1,
        k_f1,
    )
    y2 = _calc_y2(
        y0,
        b,
        c,
        d,
        ps_itot,
        k_l1,
        k_f1,
    )
    y1 = _calc_y1(
        y2,
        a,
        b,
        ps_itot,
        k_l1,
    )
    return y0, y1, y2


def _v_ps1(
    y0: float,
    y1: float,
    light_ps1: float,
    k_f1: float,
) -> float:
    """Net PSI photochemical rate: light absorption by open PSI (Y0) minus back-reaction from charge-separated state (Y1) [mmol mol(Chl)^-1 s^-1]."""
    return light_ps1 * y0 - y1 * k_f1


def _fluorescence_ps2(
    b1: float,
    b3: float,
    b1_tot: float,
    b3_tot: float,
    k_f: float,
    fluo_influence_ps2: float,
) -> float:
    """PSII fluorescence signal: difference between total-light and growth-light-only open PSII states, weighted by fluorescence rate constant kF."""
    return ((b1_tot - b1) * k_f + (b3_tot - b3) * k_f) * fluo_influence_ps2


def _fluorescence_ps1(
    y1: float,
    y1_tot: float,
    k_f1: float,
    fluo_influence_ps1: float,
) -> float:
    """PSI fluorescence signal: difference in charge-separated PSI state Y1 between total-light and growth-light, weighted by k_F1."""
    return (y1_tot - y1) * k_f1 * fluo_influence_ps1


def _fluorescence_tot(
    fps2: float,
    fps1: float,
    fpbs: float,
) -> float:
    """Total chlorophyll fluorescence as the sum of PSII, PSI, and PBS contributions."""
    return fps2 + fps1 + fpbs


def _v_ndh(
    q: float,
    ho: float,
    nadh: float,
    kndh: float,
) -> float:
    """Electron flow via NDH-2"""
    return kndh * q * nadh * ho


def _d_g0_fnr(
    p_hcytoplasm: float,
    e0_fd: float,
    f: float,
    e0_nadp: float,
    d_g_p_h: float,
) -> float:
    """Standard Gibbs free energy [kJ mol^-1] for the FNR reaction: 2 Fd_red + NADP+ → 2 Fd_ox + NADPH + H+."""
    DG1 = -e0_fd * f
    DG2 = -2 * e0_nadp * f + d_g_p_h * p_hcytoplasm
    return -2 * DG1 + DG2


def _d_g_fnr(
    fd_red: float,
    nadp: float,
    fd_ox: float,
    nadph: float,
    d_g0_fnr: float,
    rt: float,
) -> float:
    """Actual Gibbs free energy [kJ mol^-1] of the FNR reaction at current metabolite concentrations."""
    # Caclulate delta G
    s = fd_red**2 * nadp
    P = fd_ox**2 * nadph
    return d_g0_fnr + rt * np.log(P / s)


def _v_fnr(
    fd_red: float,
    nadp: float,
    fd_ox: float,
    nadph: float,
    d_g_fnr: float,
    d_g0_fnr: float,
    k_fn_fwd: float,
    k_fn_rev: float,
    rt: float,
) -> float:
    """Rate of NADPH production by FNR, thermodynamically gated.

    Forward direction (ΔG < 0): Fd_red reduces NADP+ to NADPH.
    Reverse direction (ΔG > 0): NADPH reduces Fd_ox, active in darkness.
    Units: [mmol mol(Chl)^-1 s^-1].
    """
    # Calculate the forward reaction rate
    s = fd_red * nadp
    P = fd_ox * nadph

    # If delta G is positive calculate the reverse rate
    if d_g_fnr > 0:
        Keq = np.exp(-d_g0_fnr / rt)
        return k_fn_rev / Keq * P * (np.exp(-d_g_fnr > 0) - 1)

    return k_fn_fwd * s * (1 - np.exp(d_g_fnr))


def _thylakoid_proton_leakage(
    hi_mol: float,
    ho_mol: float,
    kpass: float,
) -> float:
    """Passive proton flow"""
    return kpass * (hi_mol - ho_mol)


def _v_o2out(
    o2: float,
    o2ext: float,
    k_o2out: float,
) -> float:
    """Exchange of oxygen with the environment.

    Viewed in outward direction, assumes a constant external oxygen concentration
    """
    return k_o2out * (o2 - o2ext)


def _stoich_ho_v_ps2(
    b_ho: float,
) -> float:
    """Stoichiometric factor for Ho: 2 cytoplasmic protons consumed per PSII turnover (water splitting), scaled by cytoplasm buffering capacity."""
    return -2 / b_ho


def _stoich_hi_v_ps2(
    b_hi: float,
) -> float:
    """Stoichiometric factor for Hi: 2 lumenal protons released per PSII turnover (water splitting), scaled by lumen buffering capacity."""
    return 2 / b_hi


def _stoich_ho_v_respiration(
    b_ho: float,
) -> float:
    """Stoichiometric factor for Ho: ~4.9 cytoplasmic protons produced per respiratory turnover, scaled by cytoplasm buffering capacity."""
    return 4.926 / b_ho


def _stoich_ho_vbd(
    b_ho: float,
) -> float:
    """Stoichiometric factor for Ho: 4 cytoplasmic protons consumed per O2 reduced by bd-type oxidase (Cyd), scaled by cytoplasm buffering capacity."""
    return -4 / b_ho


def _stoich_hi_vbd(
    b_hi: float,
) -> float:
    """Stoichiometric factor for Hi: 4 lumenal protons released per O2 reduced by bd-type oxidase (Cyd), scaled by lumen buffering capacity."""
    return 4 / b_hi


def _stoich_hi_vb6f(
    b_hi: float,
) -> float:
    """Stoichiometric factor for Hi: 4 lumenal protons released per Cyt b6f turnover (Q-cycle), scaled by lumen buffering capacity."""
    return 4 / b_hi


def _stoich_ho_vb6f(
    b_ho: float,
) -> float:
    """Stoichiometric factor for Ho: 2 cytoplasmic protons consumed per Cyt b6f turnover (Q-cycle), scaled by cytoplasm buffering capacity."""
    return -2 / b_ho


def _stoich_hi_vaa(
    b_hi: float,
) -> float:
    """Stoichiometric factor for Hi: 1 lumenal proton released per aa3-COX turnover, scaled by lumen buffering capacity."""
    return 1 / b_hi


def _stoich_ho_vaa(
    b_ho: float,
) -> float:
    """Stoichiometric factor for Ho: 5 cytoplasmic protons consumed per aa3-COX turnover, scaled by cytoplasm buffering capacity."""
    return -5 / b_ho


def _stoich_ho_v_fnr(
    b_ho: float,
) -> float:
    """Stoichiometric factor for Ho: 1 cytoplasmic proton consumed per FNR turnover (NADPH production), scaled by cytoplasm buffering capacity."""
    return -1 / b_ho


def _stoich_hi_v_pass(
    b_hi: float,
) -> float:
    """Stoichiometric factor for Hi: 1 lumenal proton lost per passive leakage event, scaled by lumen buffering capacity."""
    return -1 / b_hi


def _stoich_ho_v_pass(
    b_ho: float,
) -> float:
    """Stoichiometric factor for Ho: 1 cytoplasmic proton gained per passive leakage event, scaled by cytoplasm buffering capacity."""
    return 1 / b_ho


def _stoich_ho_v_ndh(
    b_ho: float,
) -> float:
    """Stoichiometric factor for Ho: 1 cytoplasmic proton consumed per NDH-2 turnover (non-electrogenic PQ reduction), scaled by cytoplasm buffering capacity."""
    return -1 / b_ho


def _lumped_photorespiration(
    pg: float,
    atp: float,
    nadph: float,
    nad: float,
    k_pr: float,
) -> float:
    """Calculate the rate of PG recycling"""
    return k_pr * pg * atp * nadph * nad


def _stoich_ho_v_p_rsalv(
    b_ho: float,
) -> float:
    """Stoichiometric factor for Ho: 1 cytoplasmic proton released per photorespiratory salvage turnover, scaled by cytoplasm buffering capacity."""
    return 1 / b_ho


##############################################
# Define functions approximating CBB and oxy
##############################################


#  Calvin-Benson-Bassham cycle (CBB)
def _cbb_cycle(
    f_cbb_energy: float,
    f_cbb_gas: float,
    f_cbb_light: float,
    v_cbb_max: float,
) -> float:
    """Rate of CO2 fixation by the CBB cycle [mmol mol(Chl)^-1 s^-1], limited by energy (ATP/NADPH), gas (CO2/O2), and light-activated enzyme fraction."""
    return v_cbb_max * f_cbb_energy * f_cbb_gas * f_cbb_light


def _rubisco_oxygenation(
    f_oxy_carbon: float,
    f_oxy_energy: float,
    f_oxy_gas: float,
    f_oxy_light: float,
    v_oxy_max: float,
) -> float:
    """Rate of RuBisCO oxygenation producing 2-phosphoglycolate (PG) [mmol mol(Chl)^-1 s^-1], limited by carbon (3PGA), energy, gas (O2/CO2), and light activation."""
    return v_oxy_max * f_oxy_carbon * f_oxy_energy * f_oxy_gas * f_oxy_light


def _cbb_energy_mm(
    atp: float,
    nadph: float,
    kmatp: float,
    kmnadph: float,
) -> float:
    """Michaelis-Menten energy-limitation factor for CBB and Oxy: fraction of maximal rate given current ATP and NADPH concentrations."""
    return atp / (kmatp + atp) * nadph / (kmnadph + nadph)


# Michaelis-Menten with competetive O2 inhibition
def _cbb_gas_mm_o2(
    co2: float,
    o2: float,
    kmco2: float,
    kio2: float,
) -> float:
    """Michaelis-Menten CO2 saturation factor for CBB with competitive O2 inhibition of RuBisCO carboxylation."""
    return 1 * co2 / (co2 + kmco2 * (1 + o2 / kio2))


# Michaelis-Menten with competetive CO2 inhibition
def _oxy_gas_mm_co2(
    o2: float,
    co2: float,
    kmo2: float,
    kico2: float,
) -> float:
    """Michaelis-Menten O2 saturation factor for RuBisCO oxygenation with competitive CO2 inhibition."""
    return 1 * o2 / (o2 + kmo2 * (1 + co2 / kico2))


# Michaelis-Menten
def _oxy_carbon_mm(
    pga: float,
    kmpga: float,
) -> float:
    """Michaelis-Menten carbon-substrate factor for RuBisCO oxygenation: 3PGA saturation representing carbon pool availability."""
    return pga / (kmpga + pga)


def _stoich_ho_v_cbb(
    b_ho: float,
) -> float:
    """Stoichiometric factor for Ho: 5 cytoplasmic protons consumed per CBB cycle turnover, scaled by cytoplasm buffering capacity."""
    return -5 / b_ho


def _stoich_ho_v_oxy(
    b_ho: float,
) -> float:
    """Stoichiometric factor for Ho: 5 cytoplasmic protons consumed per RuBisCO oxygenation turnover, scaled by cytoplasm buffering capacity."""
    return -5 / b_ho


def _cbb_activation(
    cb_ba: float,
    fd_red: float,
    k_cb_bactivation: float,
    km_fdred: float,
) -> float:
    """Rate of CBB enzyme activation by reduced ferredoxin (Fd_red) via the ferredoxin-thioredoxin regulatory pathway [s^-1].

    CBBa approaches Fd_red / (KMFdred + Fd_red) at steady state, linking carbon
    fixation capacity to the redox state of the photosynthetic electron transport chain.
    """
    return k_cb_bactivation * (fd_red / (km_fdred + fd_red) - cb_ba)


def _v_flv_v2(
    fd_red: float,
    o2: float,
    ho: float,
    k_o2: float,
    k_hill_fdred: float,
    n_hill_fdred: float,
) -> float:
    """Electron flow from Fd to O2 via Flv1/3-Heterodimere"""
    return (
        k_o2
        * o2
        * ho
        * 1
        * fd_red**n_hill_fdred
        / (k_hill_fdred + fd_red**n_hill_fdred)
    )


def _co2p_k1(
    t: float,
    s: float,
) -> float:  # [unitless] MojicaPrieto2002
    """First dissociation constant (pK1) of carbonic acid as a function of temperature T [K] and salinity S [unitless] (MojicaPrieto2002)."""
    return (
        -43.6977 - 0.0129037 * s + 1.364e-4 * s**2 + 2885.378 / t + 7.045159 * np.log(t)
    )


def _co2_k_henry(
    t: float,
    s: float,
) -> float:  # [mol l^-1 atm^-1] König2019
    """Henry's law solubility constant for CO2 [mol l^-1 atm^-1] as a function of temperature T [K] and salinity S [unitless] (König2019)."""
    return np.exp(
        -58.0931
        + 90.5069 * 100 / t
        + 22.2940 * np.log(t / 100)
        + s * (0.027766 - 0.025888 * t / 100 + 0.0050578 * (t / 100) ** 2)
    )


def _co2sol(
    t: float,
    s: float,
    co2pp: float,
) -> float:  # [mol l^-1]
    """Dissolved CO2 concentration [mol l^-1] at partial pressure CO2pp [atm], temperature T [K], and salinity S, via Henry's law."""
    return _co2_k_henry(t, s) * co2pp


def _co2aq(
    co2dissolved: float,
    ho: float,
    k: float,
) -> float:  # [mol l^-1]
    """Aqueous CO2 (CO2·H2O) fraction of total dissolved inorganic carbon [mol l^-1], pH-dependent via the first dissociation equilibrium."""
    return co2dissolved * ho / (ho + k)


def _v_ccm_v2(
    co2: float,
    ho: float,
    co2ext_pp: float,
    k_ccm: float,
    f_cin: float,
    t: float,
    s: float,
    c_chl: float,
) -> float:
    """Exchange of CO2 with the environment.

    Viewed in inward direction.
    raises the total cellular CO2 concentration to the maximum allowed by external concentration or solubility
    assumes a constant external CO2 concentration
    """
    # Converssion factor mol l^-1 -> mmol mol(Chl)^-1
    perChl = 1 / c_chl * 1e3

    # Assume the CCM-increased partial pressure within the cell
    co2pp = co2ext_pp * f_cin

    # Calculate the CO2 equilibrium constant
    K1 = 10 ** (-_co2p_k1(t, s)) * perChl  # [mmol mol(Chl)]

    # Calculate the maximally achievable CO2aq concentration
    co2dissolved = _co2sol(
        t,
        s,
        co2pp,
    )
    CO2aq_max = (
        _co2aq(
            co2dissolved,
            ho,
            K1,
        )
        * perChl
    )  # [mmol mol(Chl)]

    return k_ccm * (CO2aq_max - co2)


def _stoich_ho_v_flv_hill(
    b_ho: float,
) -> float:
    """Stoichiometric factor for Ho: 4 cytoplasmic protons consumed per O2 reduced by Flv 1/3, scaled by cytoplasm buffering capacity."""
    return -4 / b_ho


def _stoich_hi_v_nq_mm(
    b_hi: float,
) -> float:
    """Stoichiometric factor for Hi: 1 lumenal proton released per NDH-1 turnover (cyclic electron flow), scaled by lumen buffering capacity."""
    return 1 / b_hi


def _stoich_ho_v_nq_mm(
    b_ho: float,
) -> float:
    """Stoichiometric factor for Ho: 3 cytoplasmic protons consumed per NDH-1 turnover (cyclic electron flow), scaled by cytoplasm buffering capacity."""
    return -3 / b_ho


def _v_nq_mm(
    q_ox: float,
    fd_red: float,
    v_nq_max: float,
    kmnq_qox: float,
    kmnq_fdred: float,
) -> float:
    """Rate of PQ reduction by NDH-1 complex using Fd_red as electron donor; Michaelis-Menten kinetics for both PQ_ox and Fd_red [mmol mol(Chl)^-1 s^-1]."""
    return v_nq_max * q_ox / (kmnq_qox + q_ox) * fd_red / (kmnq_fdred + fd_red)


def _free_pbs(
    pbs_ps1: float,
    pbs_ps2: float,
) -> float:
    """Fraction of free phycobilisomes"""
    return 1 - pbs_ps1 - pbs_ps2


def _ocp_activation(
    ocp: float,
    k_oc_pactivation: float,
    k_oc_pdeactivation: float,
    lcf: float,
    oc_pmax: float,
    # data
    light_spectrum: pd.Series,
    ocp_absorption: pd.Series,
) -> float:
    """Activation of orange carotenoid protein.

    As active, light-involved processwith passive reversal.
    """
    return (
        cast(float, simpson(ocp_absorption.mul(light_spectrum)))
        * lcf
        * k_oc_pactivation
        * (oc_pmax - ocp)
        - k_oc_pdeactivation * ocp
    )


def _fluorescence_pbs_ocp(
    ocp: float,
    pbs_free: float,
    fluo_influence_pbs: float,
    lcf: float,
    ml_absorption_pbs: float,
) -> float:
    """Fluorescence from free PBS, reduced by active OCP quenching; proportional to unquenched fraction (1 - OCP)."""
    return pbs_free * ml_absorption_pbs * fluo_influence_pbs * (1 - ocp) * lcf


def _ps_normabsorption_ocp(
    pbs_ps1: float,
    pbs_ps2: float,
    ocp: float,
    ps_itot: float,
    psi_itot: float,
    lcf: float,
    absorption_ps1: float,
    absorption_ps2: float,
    absorption_pbs: float,
) -> tuple[float, float]:
    """Normalised light absorption per PSI and PSII molecule, including PBS antenna energy transfer attenuated by OCP quenching."""
    light_ps1 = (absorption_ps1 + absorption_pbs * pbs_ps1 * (1 - ocp)) / ps_itot
    light_ps2 = (absorption_ps2 + absorption_pbs * pbs_ps2 * (1 - ocp)) / psi_itot

    return light_ps1 * lcf, light_ps2 * lcf


def _psii_unquench(
    psi_iq: float,
    q_red: float,
    k_unquench: float,
    km_unquench: float,
) -> float:
    """Rate of PSII recovery from the quenched state (PSIIq → PSII), inhibited by reduced PQ via a Hill-type term."""
    nUnquench = 1
    return (
        k_unquench * psi_iq * (1 - q_red**nUnquench / (km_unquench + q_red**nUnquench))
    )


def add_atpase(
    m: Model,
) -> Model:
    """Add the ATP synthase reaction (vATPsynthase) to the model, driven by the transmembrane proton gradient."""
    m.add_reaction(
        "vATPsynthase",
        fn=_atp_synthase,
        args=[
            "Hi",
            "Ho",
            "ATP",
            "ADP",
            "DeltaG0_ATP",
            "dG_pH",
            "HPR",
            "Pi_mol",
            "RT",
            "kATPsynth",
        ],
        stoichiometry={
            "Hi": Derived(fn=_stoich_hi_at_psynthase, args=["HPR", "bHi"]),
            "Ho": Derived(fn=_stoich_ho_at_psynthase, args=["HPR", "bHo"]),
            "ATP": 1,
        },
    )
    return m


def add_consuming_reactions(
    m: Model,
) -> Model:
    """Add lumped ATP and NADH consumption reactions representing cellular maintenance and biosynthesis demand."""
    m.add_reaction(
        "vATPconsumption",
        fn=_atp_consumption,
        args=[
            "ATP",
            "kATPconsumption",
        ],
        stoichiometry={
            "ATP": -1,
        },
    )
    m.add_reaction(
        "vNADHconsumption",
        fn=_nadph_consumption,
        args=[
            "NADH",
            "Ho",
            "kNADHconsumption",
        ],
        stoichiometry={
            "NADH": -1,
            "Ho": Derived(fn=_stoich_ho_v_nad_hconsumption, args=["bHo"]),
        },
    )
    return m


def add_electron_transport_chain(
    m: Model,
    pbs_behaviour: Literal["static", "dynamic"],
) -> Model:
    """Add the complete photosynthetic electron transport chain, proton handling, and associated derived quantities to the model.

    Includes derived variables (pH, equilibrium constants, moieties), QSS surrogates
    for photosystem states, and all electron-transfer reactions from water splitting
    through to Fd reduction, O2 exchange, and CO2 uptake via the CCM.
    """
    m.add_derived(
        "RT",
        fn=fns.mul,
        args=["R", "T"],
    )

    m.add_derived(
        "dG_pH",
        fn=_d_g_p_h,
        args=["R", "T"],
    )

    m.add_derived(
        "ps_ratio",
        fn=_ps_ratio,
        args=["PSItot", "PSIItot"],
    )

    m.add_derived(
        "Hi_mol",
        fn=_molar_hlumen,
        args=["Hi", "cf_lumen"],
    )
    m.add_derived(
        "Ho_mol",
        fn=_molar_hcytoplasm,
        args=["Ho", "cf_cytoplasm"],
    )
    m.add_derived(
        "pHlumen",
        fn=_p_h,
        args=["Hi_mol"],
    )
    m.add_derived(
        "pHcytoplasm",
        fn=_p_h,
        args=["Ho_mol"],
    )
    m.add_derived(
        "dpH",
        fn=_deltap_h,
        args=["pHlumen", "pHcytoplasm"],
    )

    m.add_reaction(
        "OCPactivation",
        fn=_ocp_activation,
        args=[
            "OCP",
            "kOCPactivation",
            "kOCPdeactivation",
            "lcf",
            "OCPmax",
            # data
            "light_spectrum",
            "ocp_absorption",
        ],
        stoichiometry={"OCP": 1},
    )

    ##########################################################################
    # Complex absorption
    ##########################################################################

    m.add_surrogate(
        "absorption_per_complex",
        surrogate=qss.Surrogate(
            model=_absorption_per_complex,
            args=[
                "ps_ratio",
                # data
                "light_spectrum",
                "pigment_content",
                "molar_masses",
                "abs_coef",
                "ps_comp",
            ],
            outputs=[
                "absorption_ps1",
                "absorption_ps2",
                "absorption_pbs",
            ],
        ),
    )
    m.add_surrogate(
        "abs_per_complex_ml",
        surrogate=qss.Surrogate(
            model=_absorption_per_complex,
            args=[
                "ps_ratio",
                # data
                "light_spectrum_measure",
                "pigment_content",
                "molar_masses",
                "abs_coef",
                "ps_comp",
            ],
            outputs=[
                "ml_absorption_ps1",
                "ml_absorption_ps2",
                "ml_absorption_pbs",
            ],
        ),
    )

    ##########################################################################
    # Normal absorption
    ##########################################################################

    m.add_surrogate(
        "ps_normabsorption",
        qss.Surrogate(
            model=_ps_normabsorption_ocp,
            args=[
                "PBS_PS1",
                "PBS_PS2",
                "OCP",
                "PSItot",
                "PSIItot",
                "lcf",
                "absorption_ps1",
                "absorption_ps2",
                "absorption_pbs",
            ],
            outputs=["light_ps1", "light_ps2"],
        ),
    )
    m.add_surrogate(
        "ps_normabsorption_ML",
        qss.Surrogate(
            model=_ps_normabsorption_ocp,
            args=[
                "PBS_PS1",
                "PBS_PS2",
                "OCP",
                "PSItot",
                "PSIItot",
                "lcf",
                "absorption_ps1",
                "absorption_ps2",
                "absorption_pbs",
            ],
            outputs=["light_ps1_ML", "light_ps2_ML"],
        ),
    )
    m.add_surrogate(
        "ps_normabsorption_tot",
        qss.Surrogate(
            model=_ps_normabsorption_tot,
            args=[
                "light_ps1",
                "light_ps2",
                "light_ps1_ML",
                "light_ps2_ML",
            ],
            outputs=["light_ps1_tot", "light_ps2_tot"],
        ),
    )

    ##########################################################################
    # Moieties
    ##########################################################################

    m.add_derived(
        "PQ_red",  # reduced plastoquinone
        fn=_moiety_1,
        args=["PQ_ox", "PQ_tot"],
    )
    m.add_derived(
        "PC_red",  # reduced plastocyanin
        fn=_moiety_1,
        args=["PC_ox", "PC_tot"],
    )
    m.add_derived(
        "Fd_red",  # reduced ferredoxin
        fn=_moiety_1,
        args=["Fd_ox", "Fd_tot"],
    )
    m.add_derived(
        "NADP",
        fn=_moiety_1,
        args=["NADPH", "NADP_tot"],
    )
    m.add_derived(
        "NAD",
        fn=_moiety_1,
        args=["NADH", "NAD_tot"],
    )
    m.add_derived(
        "ADP",
        fn=_moiety_1,
        args=["ATP", "AP_tot"],
    )
    m.add_derived(
        "PSIIq",  # quenched PSII
        fn=_moiety_1,
        args=["PSII", "PSIItot"],
    )
    m.add_derived(
        "Keq_PQred",
        fn=_keq_p_qred,
        args=["pHcytoplasm", "E0_QA", "F", "E0_PQ", "dG_pH", "RT"],
    )

    m.add_derived(
        "Keq_FAFd",
        fn=_keq_fa_fd,
        args=["E0_FA", "F", "E0_Fd", "RT"],
    )

    m.add_derived(
        "Keq_PCP700",
        fn=_keq_pcp700,
        args=["E0_PC", "F", "E0_P700", "RT"],
    )

    if pbs_behaviour == "dynamic":
        m.add_derived(
            "PBS_free",
            fn=_free_pbs,
            args=["PBS_PS1", "PBS_PS2"],
        )

    ##########################################################################
    # Photosystem states
    ##########################################################################

    m.add_surrogate(
        "ps2states",
        qss.Surrogate(
            model=_ps2states,
            args=[
                "PSIIq",
                "PQ_ox",
                "PQ_red",
                "Keq_PQred",
                "light_ps2",
                "PSIItot",
                "k2",
                "kF",
                "kH0",
                "kHst",
                "kPQred",
            ],
            outputs=[
                "B0",
                "B1",
                "B2",
                "B3",
            ],
        ),
    )

    m.add_surrogate(
        "ps2states_tot",
        qss.Surrogate(
            model=_ps2states,
            args=[
                "PSIIq",
                "PQ_ox",
                "PQ_red",
                "Keq_PQred",
                "light_ps2_tot",
                "PSIItot",
                "k2",
                "kF",
                "kH0",
                "kHst",
                "kPQred",
            ],
            outputs=[
                "B0_tot",
                "B1_tot",
                "B2_tot",
                "B3_tot",
            ],
        ),
    )

    m.add_surrogate(
        "ps1states",
        qss.Surrogate(
            model=_ps1_states,
            args=[
                "Fd_ox",
                "Fd_red",
                "PC_ox",
                "PC_red",
                "light_ps1",
                "Keq_PCP700",
                "Keq_FAFd",
                "PSItot",
                "k_F1",
                "kPCox",
                "kFdred",
            ],
            outputs=["Y0", "Y1", "Y2"],
        ),
    )

    m.add_surrogate(
        "ps1states_tot",
        qss.Surrogate(
            model=_ps1_states,
            args=[
                "Fd_ox",
                "Fd_red",
                "PC_ox",
                "PC_red",
                "light_ps1_tot",
                "Keq_PCP700",
                "Keq_FAFd",
                "PSItot",
                "k_F1",
                "kPCox",
                "kFdred",
            ],
            outputs=["Y0_tot", "Y1_tot", "Y2_tot"],
        ),
    )

    m.add_derived(
        "Keq_vSDH",  # equilibrium constant of succinate dehydrogenase
        fn=_keq_v_sdh,
        args=["pHcytoplasm", "E0_PQ", "F", "E0_succinate/fumarate", "RT", "dG_pH"],
    )

    m.add_reaction(
        "vSDH",  # Succinate dehydrogenase
        fn=_v_sdh,
        args=["PQ_ox", "succinate", "fumarate", "PQ_red", "Keq_vSDH", "k_SDH"],
        stoichiometry={
            "PQ_ox": -1,
            "succinate": -1,
            "fumarate": 1,
        },
    )

    m.add_reaction(
        "vRespiration",  # glycolysis and the tricarboxylic acid cycle
        fn=_v_respiration,
        args=["3PGA", "fumarate", "ADP", "NAD", "NADP", "kRespiration", "KMPGA"],
        stoichiometry={
            "3PGA": -1.000,
            "fumarate": -7.402e-02,
            "CO2": 3.000,
            "succinate": 7.402e-02,
            "ATP": 0.567,
            "NADPH": 2.237,
            "NADH": 2.689,
            "Ho": Derived(fn=_stoich_ho_v_respiration, args=["bHo"]),
        },
    )

    m.add_reaction(
        "vbd",  # bd-type terminal oxidase
        fn=_vbd,
        args=["O2", "PQ_red", "Ho", "k_ox1"],
        stoichiometry={
            "PQ_ox": 2,
            "O2": -1,
            "Ho": Derived(fn=_stoich_ho_vbd, args=["bHo"]),
            "Hi": Derived(fn=_stoich_hi_vbd, args=["bHi"]),
        },
    )

    m.add_reaction(
        "vb6f",  # Cytochrome b6f complex
        fn=_vb6f,
        args=["PC_ox", "Ho", "PQ_red", "k_Q"],
        stoichiometry={
            "PQ_ox": 1,
            "Hi": Derived(fn=_stoich_hi_vb6f, args=["bHi"]),
            "PC_ox": -2,
            "Ho": Derived(fn=_stoich_ho_vb6f, args=["bHo"]),
        },
    )

    m.add_reaction(
        "vaa",  # aa3-type terminal oxidase
        fn=_vaa,
        args=["PC_red", "O2", "Ho", "k_aa"],
        stoichiometry={
            "PC_ox": 4,
            "Hi": Derived(fn=_stoich_hi_vaa, args=["bHi"]),
            "O2": -1,
            "Ho": Derived(fn=_stoich_ho_vaa, args=["bHo"]),
        },
    )

    m.add_reaction(
        "vPSIIquench",
        fn=fns.mass_action_2s,
        args=["PSII", "PQ_red", "kQuench"],
        stoichiometry={"PSII": -1},
    )
    m.add_reaction(
        "vPSIIunquench",
        fn=_psii_unquench,
        args=["PSIIq", "PQ_red", "kUnquench", "KMUnquench"],
        stoichiometry={"PSII": 1},
    )

    m.add_reaction(
        "vPS2",  # photosystem ii
        fn=_psii,
        args=["B1", "k2"],
        stoichiometry={
            "PQ_ox": -1,
            "Ho": Derived(fn=_stoich_ho_v_ps2, args=["bHo"]),
            "Hi": Derived(fn=_stoich_hi_v_ps2, args=["bHi"]),
            "O2": 0.5,
        },
    )

    m.add_reaction(
        "vPS1",  # photosystem i
        fn=_v_ps1,
        args=["Y0", "Y1", "light_ps1", "k_F1"],
        stoichiometry={"Fd_ox": -1, "PC_ox": 1},
    )

    m.add_derived(
        "FPS2",
        fn=_fluorescence_ps2,
        args=[
            "B1",
            "B3",
            "B1_tot",
            "B3_tot",
            "kF",
            "fluo_influence_ps2",
        ],
    )

    m.add_derived(
        "FPS1",
        fn=_fluorescence_ps1,
        args=[
            "Y1",
            "Y1_tot",
            "k_F1",
            "fluo_influence_ps1",
        ],
    )

    m.add_derived(
        "FPBS",
        fn=_fluorescence_pbs_ocp,
        args=[
            "OCP",
            "PBS_free",
            "fluo_influence_pbs",
            "lcf",
            "ml_absorption_pbs",
        ],
    )

    m.add_derived(
        "Fluo",
        fn=_fluorescence_tot,
        args=["FPS2", "FPS1", "FPBS"],
    )

    m.add_reaction(
        "vFlv",
        fn=_v_flv_v2,
        args=["Fd_red", "O2", "Ho", "k_O2", "KHillFdred", "nHillFdred"],
        stoichiometry={
            "Fd_ox": 4,
            "O2": -1,
            "Ho": Derived(fn=_stoich_ho_v_flv_hill, args=["bHo"]),
        },
    )

    m.add_reaction(
        "vNDH",
        fn=_v_ndh,
        args=["PQ_ox", "Ho", "NADH", "k_NDH"],
        stoichiometry={
            "PQ_ox": -1,
            "NADH": -1,
            "Ho": Derived(fn=_stoich_ho_v_ndh, args=["bHo"]),
        },
    )

    m.add_derived(
        "dG0_FNR",
        fn=_d_g0_fnr,
        args=["pHcytoplasm", "E0_Fd", "F", "E0_NADP", "dG_pH"],
    )

    m.add_derived(
        "dG_FNR",
        fn=_d_g_fnr,
        args=["Fd_red", "NADP", "Fd_ox", "NADPH", "dG0_FNR", "RT"],
    )

    m.add_reaction(
        "vFNR",
        fn=_v_fnr,
        args=[
            "Fd_red",
            "NADP",
            "Fd_ox",
            "NADPH",
            "dG_FNR",
            "dG0_FNR",
            "k_FN_fwd",
            "k_FN_rev",
            "RT",
        ],
        stoichiometry={
            "Fd_ox": 2,
            "NADPH": 1,
            "Ho": Derived(fn=_stoich_ho_v_fnr, args=["bHo"]),
        },
    )

    m.add_reaction(
        "vPass",  # proton leakage
        fn=_thylakoid_proton_leakage,
        args=["Hi_mol", "Ho_mol", "k_pass"],
        stoichiometry={
            "Hi": Derived(fn=_stoich_hi_v_pass, args=["bHi"]),
            "Ho": Derived(fn=_stoich_ho_v_pass, args=["bHo"]),
        },
    )

    m.add_reaction(
        "vNQ",
        fn=_v_nq_mm,
        args=["PQ_ox", "Fd_red", "vNQ_max", "KMNQ_Qox", "KMNQ_Fdred"],
        stoichiometry={
            "Fd_ox": 2,
            "PQ_ox": -1,
            "Hi": Derived(fn=_stoich_hi_v_nq_mm, args=["bHi"]),
            "Ho": Derived(fn=_stoich_ho_v_nq_mm, args=["bHo"]),
        },
    )

    m.add_reaction(
        "vO2out",  # Oxygen efflux
        fn=_v_o2out,
        args=["O2", "O2ext", "kO2out"],
        stoichiometry={"O2": -1},
    )

    m.add_reaction(
        "vCCM",  # Carbon dioxide concentration by CCM
        fn=_v_ccm_v2,
        args=["CO2", "Ho", "CO2ext_pp", "kCCM", "fCin", "T", "S", "cChl"],
        stoichiometry={"CO2": 1},
    )
    return m


def add_photorespiratory_salvage(
    m: Model,
) -> Model:
    """Adds the lumped reaction recycling (2-phospho)glycolate from photorespiration.

    The standard stoichiometry assumes complete recycling via glycolate dehydrogenase
    and the glycerate pathway (via tartronic semialdehyde):

    2 PG + (NADPH + Ho) + 2 NAD + ATP --> (3PGA + CO2) + NADP + (2 NADH + 2 Ho) + ADP (+ Pi)
    """
    m.add_reaction(
        "vPRsalv",
        fn=_lumped_photorespiration,
        args=["PG", "ATP", "NADPH", "NAD", "kPR"],
        stoichiometry={
            "PG": -2,
            "ATP": -1,
            "NADPH": -1,
            "Ho": Derived(fn=_stoich_ho_v_p_rsalv, args=["bHo"]),
            "NADH": 2,
            "3PGA": 1,
            "CO2": 1,
        },
    )
    return m


def add_cbb_and_oxy(
    m: Model,
) -> Model:
    """Add CBB cycle, RuBisCO oxygenation, and Fd-dependent CBB activation reactions to the model."""
    # Add the factors as algebraic modules
    m.add_derived(
        "f_CBB_energy",
        fn=_cbb_energy_mm,
        args=["ATP", "NADPH", "KMATP", "KMNADPH"],
    )
    m.add_derived(
        "f_CBB_gas",
        fn=_cbb_gas_mm_o2,
        args=["CO2", "O2", "KMCO2", "KIO2"],
    )
    m.add_derived(
        "f_oxy_carbon",
        fn=_oxy_carbon_mm,
        args=["3PGA", "KMPGA"],
    )
    m.add_derived(
        "f_oxy_gas",
        fn=_oxy_gas_mm_co2,
        args=["O2", "CO2", "KMO2", "KICO2"],
    )
    m.add_reaction(
        "vCBBactivation",
        fn=_cbb_activation,
        args=["CBBa", "Fd_red", "kCBBactivation", "KMFdred"],
        stoichiometry={"CBBa": 1},
    )
    m.add_reaction(
        "vCBB",
        fn=_cbb_cycle,
        args=["f_CBB_energy", "f_CBB_gas", "CBBa", "vCBB_max"],
        stoichiometry={
            "CO2": -3,
            "ATP": -8,
            "NADPH": -5,
            "3PGA": 1,
            "Ho": Derived(fn=_stoich_ho_v_cbb, args=["bHo"]),
        },
    )
    m.add_reaction(
        "vOxy",
        fn=_rubisco_oxygenation,
        args=["f_oxy_carbon", "f_CBB_energy", "f_oxy_gas", "CBBa", "vOxy_max"],
        stoichiometry={
            "O2": -3,
            "ATP": -8,
            "NADPH": -5,
            "3PGA": -2,
            "PG": 3,
            "Ho": Derived(fn=_stoich_ho_v_oxy, args=["bHo"]),
        },
    )
    return m


def create_model(
    light_spectrum: pd.Series,  # # [umol(photons) m^-2 s^-1], warm white led @ pfd 100
    light_spectrum_measure: pd.Series,  # [umol(photons) m^-2 s^-1], 625 @ 1
    ocp_absorption: pd.Series,
    abs_coef: pd.DataFrame,
    molar_masses: pd.Series,
    ps_comp: pd.DataFrame,
    pigment_content: pd.Series,
    # variants
    pbs_behaviour: Literal["static", "dynamic"] = "static",
) -> Model:
    """Construct and return the complete Synechocystis photosynthesis ODE model.

    Assembles all sub-models (electron transport chain, ATP synthase, CBB cycle,
    photorespiratory salvage, and lumped consumption reactions) with experimentally
    derived initial conditions and parameters. Light specGtra and pigment data are
    attached as model data rather than parameters to allow dynamic updates.
    """
    m = Model()
    m.add_variables(
        {
            "PSII": 0.415,  # [mmol mol(Chl)^-1] initial concentration of unquenched PSII (guessed from ca. 50% reduced plastoquinone) (guess)
            "O2": 55.402,  # [mmol mol(Chl)^-1] concentration of oxygen in the cell (Kihara2014)
            "PC_ox": 0.157,  # [mmol mol(Chl)^-1] initial concentration of oxidised plastocyanin (aerobic) (Schreiber2017)
            "Fd_ox": 3.237,  # [mmol mol(Chl)^-1] initial concentration of oxidised ferredoxin (aerobic) (Schreiber2017)
            "NADPH": 20.104,  # [mmol mol(Chl)^-1] initial concentration of NADPH (Cooley2001)
            "NADH": 3.574,  # [mmol mol(Chl)^-1] initial concentration of NADH (Tanaka2021)
            "ATP": 172.057,  # [mmol mol(Chl)^-1] initial concentration of ATP (Doello2018)
            "PG": 0.894,  # [mmol mol(Chl)^-1] initial concentration of (2-phospho) glycolate (Huege2011)
            "succinate": 2.000,  # [mmol mol(Chl)^-1] initial concentration of succinate (guess)
            "fumarate": 2.000,  # [mmol mol(Chl)^-1] initial concentration of fumarate (guess)
            "3PGA": 2.000e03,  # [mmol mol(Chl)^-1] initial concentration of 3-phospho glycerate (including all other sugars) (guess)
            "CBBa": 0.000e00,  # [unitless] fraction of Fd-activated, lumped enzymes of the CBB (guess)
            "Hi": 0.217,  # [mmol mol(Chl)^-1] initial concentration of lumenal protons in 10^4 uE cm^-1 s^-1 light (Belkin1987)
            "Ho": 6.932e-03,  # [mmol mol(Chl)^-1] initial concentration of cytoplasmic protons in 10^4 uE cm^-1 s^-1 light (Belkin1987)
            "PQ_ox": 7.202,  # [mmol mol(Chl)^-1] concentration of oxidised, PHOTOACTIVE plastoquinone in 40 umol m^-2 s^-1 irradiation (Khorobrykh2020)
            "CO2": 3.103,  # [mmol mol(Chl)^-1] concentration of CO2 in the cell without activity of the CCM (estimated)
            "OCP": 0.000e00,  # [unitless] initial activity of OCP (guess)
        }
    )
    m.add_parameters(parameters)
    m.add_data("abs_coef", abs_coef)
    m.add_data("light_spectrum", light_spectrum)
    m.add_data("light_spectrum_measure", light_spectrum_measure)
    m.add_data("molar_masses", molar_masses)
    m.add_data("ocp_absorption", ocp_absorption)
    m.add_data("ps_comp", ps_comp)
    m.add_data("pigment_content", pigment_content)

    m = add_electron_transport_chain(m, pbs_behaviour=pbs_behaviour)
    m = add_atpase(m)
    m = add_consuming_reactions(m)
    m = add_cbb_and_oxy(m)
    return add_photorespiratory_salvage(m)
