# Data and code for Abedon et al., 2025

## effort_norm.py

This is the code used to normalize juvenile observations for iNaturalist effort. “physalia_obs.csv” is surveyed for all observations of juvenile _P. physalis_ specimens, resulting in “juvenile_observations.csv”, containing only the juvenile entries. “Effort_norm.py” assigns each juvenile observation a weight using “sand_dollar_all.csv”, “sea_stars_all.csv”, and “hermit_crab_all.csv” as proxies for iNaturalist effort. The result is a weighted juvenile observation dataset that can be used to generate starting points in “sim.py” (see below). 

## Sim.py

This is the main code used to run particle tracking simulations. To run this code, you must use a base, wind, current, and k-value dataset in NetCDF format, a landmask in TIFF format, and a weighted juvenile observation dataset in CSV format. The conda environment used to run this code is found in "sim-env.yml".

### Base Data

The base data should be a zeroed NetCDF file upon which the vector fields are loaded. The code needed to create this file is provided as create_base_data.py in the rep. It is loaded in as a FieldSet with U and V variables and lon, lat, and time dimensions (see setup_fieldset function).

### Wind and Current Data 

Wind and current data should be downloaded from https://data.marine.copernicus.eu/products as described in Abedon et al., 2025. 

### K-values and Land data

Surface $K$ eddy diffusivity values from Groeskamp et al., 2020 for simulating stochasticity can be found in “k_data_surface.nc”. Land data for creating a land mask can be found in “landmask.tif”.
