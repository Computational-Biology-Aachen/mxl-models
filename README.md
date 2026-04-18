# MxlModels


MxlModels is a Python package of reference mechanistic models.
It contains the same models as in the [MxlBricks](https://github.com/Computational-Biology-Aachen/mxl-bricks) repo, but written as single, flat files to make inspection easier.

Usually, these here will be created by codegen from [MxlBricks](https://github.com/Computational-Biology-Aachen/mxl-bricks).

## Installation


You can install mxlpy using pip: `pip install mxlmodels`.


If you want access to the sundials solver suite via the [assimulo](https://jmodelica.org/assimulo/) package, we recommend setting up a virtual environment via [pixi](https://pixi.sh/) or [mamba / conda](https://mamba.readthedocs.io/en/latest/) using the [conda-forge](https://conda-forge.org/) channel.

```bash
pixi init
pixi add python assimulo
pixi add --pypi mxlmodels
```


## Development setup

Install pixi [as described in the docs](https://pixi.sh/latest/#installation).

Run

```bash
pixi install
```


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



## Tool family 🏠

`MxlModels` is part of a larger family of tools that are designed with a similar set of abstractions. Check them out!

- [MxlPy](https://github.com/Computational-Biology-Aachen/MxlPy) is a Python package for mechanistic learning (Mxl)
- [MxlBricks](https://github.com/Computational-Biology-Aachen/mxl-bricks) is a Python package to build mechanistic models composed of pre-defined reactions (bricks).
- [MxlWeb](https://github.com/Computational-Biology-Aachen/mxl-web) brings simulation of mechanistic models to the browser!
