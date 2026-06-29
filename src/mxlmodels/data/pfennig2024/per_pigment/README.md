# Absorption coefficients from Fuente2021

Data extracted from figure 4 in Fuente, D., Lazar, D., Oliver-Villanueva, J. V., & Urchueguía, J. F. (2021). Reconstruction of the absorption spectrum of Synechocystis sp. PCC 6803 optical mutants from the in vivo signature of individual pigments. Photosynthesis Research, 147(1), 75–90. <https://doi.org/10.1007/s11120-020-00799-8>

## Extraction software

Extraction software used: <http://www.graphreader.com/>

with options:

```text
Axis Settings (defined by blue rectangle):
    y-high= 0.1
    y-low= 0
    x-low= 400
    x_high= 700
    Interval: 
        X-axis stepsize
        value= 1
Curve Sampling (using fix-points):
    Point interpolation= Spline
Post-processing:
    None
Data Output:
    Precision, output decimals= 5
```

## Data

Absorption coefficients [m$^2$ mg(pigment)$^{-1}$] in the wavelengths 400 nm - 700 nm

### Pigments

| Pigment         | Line in Figure    | File                |
| --------------- | ----------------- | ------------------- |
| Chlorophyll a   | solid blue line   | chla.csv            |
| Beta carotenes  | solid green line  | beta_carotene.csv   |
| Allophycocyanin | solid yellow line | allophycocyanin.csv |
| Phycocyanin     | solid red line    | phycocyanin.csv     |
