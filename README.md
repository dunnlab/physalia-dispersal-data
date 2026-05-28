# Data and code for Abedon et al., under review

## iNaturalist Data

iNaturalist observations used for the juvenile study (Fig. 2, Abedon et al., under review) are provided in "inat_pphysalis_obs_areaa_all.csv" and include all research-grade _P. physalis_ observations found within Study Area A (Fig. 1, Abedon et al., under review) as exported on 9 September 2024 and amended through October 2024 on 30 April 2026. Juvenile classifications assigned during the study are recorded in the "Classification" column, with the 194 observations classified as juveniles provided separately in "juvenile_observations.csv". iNaturalist observations used for the US East Coast stranding study (Fig. 3 & Fig. 4, Abedon et al., under review) are provided in "inat_pphysalis_obs_usec_2017-2024.csv". The code used to filter Area A observations to the US East Coast study region and time period is provided in "filter_for_eastcoast_stranding_obs.py". Observations are further filtered to  the simulation period (1 Novemeber 2022 - 31 October 2023) for spatiotemporal comparison with simulated strandings in "sim_obs_spearmen_correlation.py".

## Juvenile Normalization

The code used to normalize juvenile observations for iNaturalist effort is provided in "effort_norm.py", which assigns each juvenile observation a weight using “sand_dollar_all.csv”, “sea_stars_all.csv”, and “hermit_crab_all.csv” as proxies for iNaturalist effort. The result is a weighted juvenile observation dataset that can be used to generate starting points in “sim.py” (see below). 

## Particle-tracking Model

The primary particle-tracking simulation code is provided in "sim.py". To execute this code, a base grid must first be generated using "create_base_data.py". The CMEMS wind and current datasets used as forcing are available at Zenodo (10.5281/zenodo.20399401), and the landmask is provided as "landmask.tif". Particles are initalized based on an effort-normalized juvenile observation dataset generated using "effort_norm.py". The conda environment required to run this code is found in "sim-env.yml". The simulation is configured for parallel execution on a computing cluster. Each model run outputs particle trajectory data in zarr format and a CSV recording the time and location of strandings events. 

### Base Data

The base data should be a zeroed NetCDF file upon which the vector fields are loaded. The code needed to create this file is provided as "create_base_data.py". It is loaded in as a FieldSet with U and V variables and lon, lat, and time dimensions (see setup_fieldset function).

## Figure Generation & Model Output Analysis

The code used to generate figures in Abedon et al., (under review) is provided. The code for Figure 2 plots the spatial and temporal distribution of juveniles using juvenile_observations.csv". The code for Figure 3 plots the season patterns of strandings on the US East Coast using "inat_pphysalis_obs_usec_2017-2024.csv" and assesses their significance and consistency using Rayleigh tests. The code for Figure 4 compares observed stranding and simulated stranding distributions using model output data and "inat_pphysalis_obs_usec_2017-2024.csv", with the Spearman rank correlation between the two computed in "sim_obs_spearmen_correlation.py". The code for Figure 5 generates static maps of particle trajectories from a single representative model run. The code for Figure 6 plots particle entries into Area C across all 25 model runs, computes Pearson correlations between wind conditions and particle entries using CMEMS wind data, and visualizes the corresponding wind vectors. The mean direction and circular standard deviation of pre-stranding wind vectors are calculated from model output data in "prestranding_winds_analysis.py".

## Supporting Information
