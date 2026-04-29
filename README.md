<p align="center">
    <img src="https://raw.githubusercontent.com/Computational-Biology-Aachen/mxl-models/refs/heads/main/docs/assets/logo.png" width="400px" alt='mxlmodels-logo'>
</p>

# MxlModels

`MxlModels` is a Python package of reference mechanistic models.
It contains the same models as in the [MxlBricks](https://github.com/Computational-Biology-Aachen/mxl-bricks) repo, but written as single, flat files to make inspection easier.

Usually, these here will be created by codegen from [MxlBricks](https://github.com/Computational-Biology-Aachen/mxl-bricks).

## Installation

You can install mxlpy using pip: `pip install mxlmodels`.

Done. Simple as that.

## Models

| Name                 | Description                                                                 |
| -------------------- | --------------------------------------------------------------------------- |
| Yokota 1985          | Photorespiration                                                            |
| Poolman 2000         | CBB cycle, based on Pettersson & Ryde-Pettersson 1988                       |
| Ebenhöh 2011         | PSII & two-state quencher & ATP synthase                                    |
| Ebenhöh 2014         | PETC & state transitions & ATP synthase from Ebenhoeh 2011                  |
| Matuszyńska 2016 NPQ | 2011 + PSII & four-state quencher                                           |
| Matuszyńska 2016 PhD | ?                                                                           |
| Matuszyńska 2019     | Merges PETC (Ebenhöh 2014), NPQ (Matuszynska 2016) and CBB (Poolman 2000)   |
| Saadat 2021          | 2019 + Mehler (Valero ?) & Thioredoxin & extendend PSI states & consumption |
| Ebeling 2026         | unpublishd                                                                  |


## Tool family 🏠

`MxlModels` is part of a larger family of tools that are designed with a similar set of abstractions. Check them out!

- [MxlPy](https://github.com/Computational-Biology-Aachen/MxlPy) is a Python package for mechanistic learning (Mxl)
- [MxlBricks](https://github.com/Computational-Biology-Aachen/mxl-bricks) is a Python package to build mechanistic models composed of pre-defined reactions (bricks)
- [MxlWeb](https://github.com/Computational-Biology-Aachen/mxl-web) brings simulation of mechanistic models to the browser!
- [pysbml](https://github.com/Computational-Biology-Aachen/pysbml) simplifies SBML models for import/export with MxlPy
- [Parameteriser](https://gitlab.com/marvin.vanaalst/parameteriser) looks up kinetic parameters from BRENDA and other databases
