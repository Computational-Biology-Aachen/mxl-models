"""Load functions for model data."""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from mxlmodels.pfennig2024_synechocystis import PIGMENTS, light_gaussian_led


@dataclass
class Data:
    """Container for model data."""

    light_spectrum: pd.Series
    light_spectrum_measure: pd.Series
    ocp_absorption_per_wavelength: pd.Series
    pigment_abs_coef_per_wavelength: pd.DataFrame


_default = Path(__file__).parent


def default(data_dir: Path = _default) -> Data:
    """Load default Pfennig 2024 data."""
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
            for pig in PIGMENTS
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
    )


def lights(data_dir: Path = _default) -> pd.DataFrame:
    """Load the light spectra, one column per light source, scaled by 100."""
    return (
        pd.DataFrame(
            {
                path.stem: pd.read_csv(path, index_col=0).squeeze(axis=1)
                for path in sorted((data_dir / "lights/").glob("*.csv"))
            }
        )
        * 100
    )
