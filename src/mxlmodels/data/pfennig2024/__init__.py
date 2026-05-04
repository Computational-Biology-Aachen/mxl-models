"""Load functions for model data."""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from mxlmodels.pfennig2024_synechocystis import light_gaussian_led


@dataclass
class Data:
    """Container for model data."""

    light_spectrum: pd.Series
    light_spectrum_measure: pd.Series
    ocp_absorption_per_wavelength: pd.Series
    pigment_abs_coef_per_wavelength: pd.DataFrame
    molar_masses: pd.Series
    ps_comp: pd.DataFrame
    pigment_content: pd.Series


_default = Path(__file__).parent


def load(data_dir: Path = _default) -> Data:
    """Load default Pfennig 2024 data."""
    # relative pigment concentrations in a synechocystis cell (Zavrel2023)
    pigment_content: pd.Series = pd.Series(
        {
            "chla": 1.000,  # [mg(Pigment) mg(Chla)^-1]
            "beta_carotene": 0.176,  # [mg(Pigment) mg(Chla)^-1]
            "allophycocyanin": 1.118,  # [mg(Pigment) mg(Chla)^-1]
            "phycocyanin": 6.765,  # [mg(Pigment) mg(Chla)^-1]
        },
    )

    molar_masses = pd.Series(
        {
            "chla": 893.509,
            "beta_carotene": 536.888,
        }
    )

    ps_comp = pd.DataFrame(
        {
            "ps1": {
                "ratio": 5.0,
                "n_chla": 96.0,
                "beta_carotene_in_membrane": 0.75,
                "n_beta_carotene": 22.0,
            },
            "ps2": {
                "ratio": 1.0,
                "n_chla": 35.0,
                "beta_carotene_in_membrane": 0.75,
                "n_beta_carotene": 11.0,
            },
        }
    )

    # Larger files
    ocp_absorption_per_wavelength: pd.Series = pd.read_csv(
        data_dir / "ocp_absorption.csv", index_col=0
    ).rename(columns={" absorption": "absorption"})["absorption"]

    lights = pd.DataFrame(
        {
            path.stem: pd.read_csv(path, index_col=0).squeeze(axis=1)
            for path in sorted((data_dir / "lights/").glob("*.csv"))
        }
    )

    pigment_abs_coef_per_wavelength = pd.DataFrame(
        {
            pig: pd.read_csv(data_dir / "per_pigment" / f"{pig}.csv", index_col=0)[
                "absorption_coefficient_m2_mgPigment-1"
            ]
            for pig in pigment_content.index
        }
    )

    # Derived
    light_spectrum: pd.Series = lights["warm_white_led"] * 100
    light_spectrum_measure: pd.Series = light_gaussian_led(625, 1)

    return Data(
        light_spectrum=light_spectrum,
        light_spectrum_measure=light_spectrum_measure,
        ocp_absorption_per_wavelength=ocp_absorption_per_wavelength,
        pigment_abs_coef_per_wavelength=pigment_abs_coef_per_wavelength,
        molar_masses=molar_masses,
        ps_comp=ps_comp,
        pigment_content=pigment_content,
    )
