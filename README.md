# Data and code for Abedon et al., under review

## iNaturalist Data

iNaturalist observations used for the juvenile study (Fig. 2, Abedon et al., under review) are provided in "inat_pphysalis_obs_areaa_all.csv" and include all research-grade _P. physalis_ observations found within Study Area A (Fig. 1, Abedon et al., under review) as exported on 9 September 2024 and amended through October 2024 on 30 April 2026. Juvenile classifications assigned during the study are recorded in the "Classification" column, with the 194 observations classified as juveniles provided separately in "juvenile_observations.csv". iNaturalist observations used for the US East Coast stranding study (Fig. 3 & Fig. 4, Abedon et al., under review) are provided in "inat_pphysalis_obs_usec_2017-2024.csv". The code used to filter Area A observations to the US East Coast study region and time period is provided in "filter_for_eastcoast_stranding_obs.py". Observations are further filtered to  the simulation period (1 Novemeber 2022 - 31 October 2023) for spatiotemporal comparison with simulated strandings in 'sim_obs_spearmen_correlation.py'.

## Juvenile Normalization

The code used to normalize juvenile observations for iNaturalist effort is provided in "effort_norm.py", which assigns each juvenile observation a weight using “sand_dollar_all.csv”, “sea_stars_all.csv”, and “hermit_crab_all.csv” as proxies for iNaturalist effort. The result is a weighted juvenile observation dataset that can be used to generate starting points in “sim.py” (see below). 

## Sim.py

This is the main code used to run particle tracking simulations. To run this code, you must use a base, wind, current, and k-value dataset in NetCDF format, a landmask in TIFF format, and a weighted juvenile observation dataset in CSV format. The conda environment used to run this code is found in "sim-env.yml".

### Base Data

The base data should be a zeroed NetCDF file upon which the vector fields are loaded. The code needed to create this file is provided as create_base_data.py in the rep. It is loaded in as a FieldSet with U and V variables and lon, lat, and time dimensions (see setup_fieldset function).

### Wind and Current Data 

Wind and current data should be downloaded from https://data.marine.copernicus.eu/products as described in Abedon et al., 2025. 

### K-values and Land data

Surface $K$ eddy diffusivity values from Groeskamp et al., 2020 for simulating stochasticity can be found in “k_data_surface.nc”. Land data for creating a land mask can be found in “landmask.tif”.
