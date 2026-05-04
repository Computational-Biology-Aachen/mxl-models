# Light spectra from Fuente2021

Data gathered from the supplementary material to Fuente, D., Lazar, D., Oliver-Villanueva, J. V., & Urchueguía, J. F. (2021). Reconstruction of the absorption spectrum of Synechocystis sp. PCC 6803 optical mutants from the in vivo signature of individual pigments. Photosynthesis Research, 147(1), 75–90. <https://doi.org/10.1007/s11120-020-00799-8>

Extraction software used: <http://www.graphreader.com/>

with options:

```text
Axis Settings (defined by blue rectangle):
    y-high= 100
    y-low= 0
    x-low= 400
    x_high= 700
    Interval: 
        X-axis stepsize
        value= 1
Curve Sampling (using fix-points):
    Machine estimation= Smooth curves
    OR if not leading to curve:
    Point interpolation= Spline
Post-processing:
    None
Data Output:
    Precision, output decimals= 5
```

Afterwards the curves were normalised to a total integral of 1.

## Data

Relative light intensity distributions of different light sources.

| Light                   | File                  |
| ----------------------- | --------------------- |
| Blue-shifted white LED  | cool_white_led.csv    |
| White fluorescent lamp  | fluorescent_lamp.csv  |
| White halogen lamp      | halogen_lamp.csv      |
| White incandescent bulb | incandescent_bulb.csv |
| Solar irradiance        | solar.csv             |
| Red-shifted white LED   | warm_white_led.csv    |
