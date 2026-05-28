import numpy as np
import parcels.rng as ParcelsRandom
from datetime import datetime, timedelta
from parcels import FieldSet, ParticleSet, JITParticle, Variable, Field, VectorField
from glob import glob
import os
import logging
import itertools
import rasterio
import csv
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde, wasserstein_distance
import xarray as xr
import math
from scipy.ndimage import distance_transform_edt
import multiprocessing
from multiprocessing import Pool, Manager, cpu_count
from functools import partial
import pandas as pd
from collections import deque

stranded = []


# ---------------------------------------------------------------------------
# Adaptive KDE (Abramson's method, lambda_cap=2)
# ---------------------------------------------------------------------------
class AdaptiveKDE:
    def __init__(self, data, weights=None, alpha=0.5, lambda_cap=2.0):
        self.data = np.asarray(data, dtype=float)
        _, n = self.data.shape

        if weights is None:
            self.weights = np.ones(n) / n
        else:
            w = np.asarray(weights, dtype=float)
            self.weights = w / w.sum()

        self.mu  = self.data.mean(axis=1)
        self.sig = self.data.std(axis=1)
        self.sig[self.sig == 0] = 1.0
        data_std = (self.data - self.mu[:, None]) / self.sig[:, None]

        pilot        = gaussian_kde(data_std, weights=self.weights)
        pilot_density = pilot(data_std)

        log_g    = np.sum(self.weights * np.log(np.maximum(pilot_density, 1e-300)))
        geo_mean = np.exp(log_g)

        raw_lambda    = (pilot_density / geo_mean) ** (-alpha)
        self.lambda_i = np.clip(raw_lambda, 1.0 / lambda_cap, lambda_cap)
        self.base_cov = pilot.covariance

    def resample(self, n_samples):
        d   = self.data.shape[0]
        idx = np.random.choice(len(self.weights), size=n_samples, p=self.weights)
        out_std = np.empty((d, n_samples))
        for k, i in enumerate(idx):
            center     = (self.data[:, i] - self.mu) / self.sig
            cov        = self.lambda_i[i] ** 2 * self.base_cov
            out_std[:, k] = np.random.multivariate_normal(center, cov)
        return out_std * self.sig[:, None] + self.mu[:, None]


# Precompute the KDE and distance map
def precompute_resources(df, land_mask_tif):
    kde_data = np.vstack([
        df['longitude'],
        df['latitude'],
        df['DayOfYear_sin'],
        df['DayOfYear_cos']
    ])
    kde = AdaptiveKDE(kde_data, weights=df['weights'].values, alpha=0.5, lambda_cap=2.0)

    with rasterio.open(land_mask_tif) as src:
        land_mask          = src.read(1)
        distance_from_land = distance_transform_edt(land_mask == 0)
        transform          = src.transform
        width, height      = src.width, src.height

    return kde, distance_from_land, transform, width, height


# Generate starting points from KDE
def create_starting_points(kde, bounds, land_mask_tif, distance_from_land,
                           num_points, min_distance_km=50):
    lon_min, lon_max, lat_min, lat_max = bounds
    points = []

    with rasterio.open(land_mask_tif) as land_mask:
        km_per_pixel        = land_mask.res[0] * 111.0
        min_distance_pixels = int(np.ceil(min_distance_km / km_per_pixel))

        while len(points) < num_points:
            lon, lat, day_sin, day_cos = kde.resample(1).flatten()
            day_of_year = np.arctan2(day_sin, day_cos) * 365 / (2 * np.pi)
            day_of_year = day_of_year if day_of_year > 0 else day_of_year + 365

            if not (lon_min <= lon <= lon_max and lat_min <= lat <= lat_max):
                continue

            row, col = land_mask.index(lon, lat)
            if not (0 <= row < land_mask.height and 0 <= col < land_mask.width):
                continue

            if distance_from_land[row, col] >= min_distance_pixels:
                points.append([lon, lat, day_of_year])

    return points


# Load and preprocess the data
def load_data(file_path):
    df = pd.read_csv(file_path)
    df['DayOfYear'] = pd.to_datetime(df['observed_on']).dt.dayofyear
    df['DayOfYear_sin'] = np.sin(2 * np.pi * df['DayOfYear'] / 365)
    df['DayOfYear_cos'] = np.cos(2 * np.pi * df['DayOfYear'] / 365)
    df['weights'] = df['normalized_weights']
    return df

# Convert meters to latitude/longitude coordinates
def meters_to_coord(lat, u, v):
    lat = np.radians(lat)
    v_lat_change = v / 111111
    u_lon_change = u / (111111 * np.cos(lat))
    return u_lon_change, v_lat_change

# Determine drift angle
def determine_drift_angle(wind_velocity_u, wind_velocity_v):
    wind_velocity = np.sqrt(wind_velocity_u**2 + wind_velocity_v**2)
    drift_angle = 50.8 * math.exp(-0.15 * wind_velocity) - 0.5
    return drift_angle

# Rotate vector by drift angle
def RotateVector(x, y, drift_angle, handedness):
    if handedness == -1:
        drift_angle = -drift_angle

    angle_radians = np.radians(drift_angle)
    cos_angle = np.cos(angle_radians)
    sin_angle = np.sin(angle_radians)
    x_rot = cos_angle * x - sin_angle * y
    y_rot = sin_angle * x + cos_angle * y
    return x_rot, y_rot

def run_simulation(fieldset, offshore_df, trial_number, output_directory,
                   min_longitude=-97.75, max_longitude=-50.25, min_latitude=11.0, max_latitude=51.25):

    global stranded

    stranded_filename = f"stranded_run_{trial_number}.csv"
    output_filename = f"run_{trial_number}_output.zarr"
    os.makedirs(output_directory, exist_ok=True)
    full_output_path = os.path.join(output_directory, output_filename)
    stranded_output_path = os.path.join(output_directory, stranded_filename)

    # Function to write beached particles to a CSV file
    def write_stranded_to_csv(stranded, filename):
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Latitude', 'Longitude', 'time', 'handedness'])
            for item in beached:
                writer.writerow(item)

    pset = pset_setup(offshore_df, fieldset, min_longitude, max_longitude, min_latitude, max_latitude)

    # Delete particles that drift outside the simulation domain
    def handle_out_of_bounds(particle, fieldset, time):
        if particle.lon < particle.min_longitude or particle.lon > particle.max_longitude or particle.lat < particle.min_latitude or particle.lat > particle.max_latitude:
            particle.delete()

    # Flag particles that have stranded
    def land_check(particle, fieldset, time):
        oob_lc = particle.lon < particle.min_longitude or particle.lon > particle.max_longitude or particle.lat < particle.min_latitude or particle.lat > particle.max_latitude
        if oob_lc:
            particle.delete()
        else:
            if fieldset.landmask[time, particle.depth, particle.lat, particle.lon] == 1.0:
                particle.stranded = 1

    # Compute local Smagorinsky diffusivity k from velocity shear and cell area
    def SmagorinskyKernel(particle, fieldset, time):
        oob_smag = (particle.lon < particle.min_longitude or particle.lon > particle.max_longitude or
                    particle.lat < particle.min_latitude or particle.lat > particle.max_latitude)
        if oob_smag:
            particle.delete()
        elif particle.stranded == 0 and time >= particle.start_times:
            dx = 1.0 / 12.0
            dx_m = dx * 111120.0 * math.cos(particle.lat * math.pi / 180.0)  # zonal metres
            dy_m = dx * 111120.0                                             # meridional metres
            u_px, v_px = fieldset.UV_global[time, particle.depth, particle.lat, particle.lon + dx]
            u_mx, v_mx = fieldset.UV_global[time, particle.depth, particle.lat, particle.lon - dx]
            u_py, v_py = fieldset.UV_global[time, particle.depth, particle.lat + dx, particle.lon]
            u_my, v_my = fieldset.UV_global[time, particle.depth, particle.lat - dx, particle.lon]
            dudx = (u_px - u_mx) / (2.0 * dx_m)   # 1/s
            dudy = (u_py - u_my) / (2.0 * dy_m)   # 1/s
            dvdx = (v_px - v_mx) / (2.0 * dx_m)   # 1/s
            dvdy = (v_py - v_my) / (2.0 * dy_m)   # 1/s
            A = fieldset.cell_areas_global[time, particle.depth, particle.lat, particle.lon]  # m²
            particle.k = fieldset.Cs * A * math.sqrt(dudx**2 + 0.5 * (dudy + dvdx)**2 + dvdy**2)  # m²/s

    # Advect particles using ocean current + wind-driven sailing velocity (with handedness-dependent drift angle) + Smagorinsky stochastic noise
    def PhysaliaAdvectionKernel(particle, fieldset, time):
        oob_pa = particle.lon < particle.min_longitude or particle.lon > particle.max_longitude or particle.lat < particle.min_latitude or particle.lat > particle.max_latitude
        if oob_pa:
            particle.delete()
        else:
            if particle.stranded == 0 and time >= particle.start_times:
                wind_coefficient = particle.response_to_wind
                handedness = particle.handedness
                k = particle.k

                (u_current, v_current) = fieldset.UV_global[time, particle.depth, particle.lat, particle.lon]
                (u_wind, v_wind) = fieldset.UV_wind[time, particle.depth, particle.lat, particle.lon]

                wind_velocity = math.sqrt(u_wind**2 + v_wind**2)
                drift_angle = 50.8 * math.exp(-0.15 * wind_velocity) - 0.5

                if handedness == -1:
                    drift_angle = -drift_angle
                angle_radians = drift_angle * math.pi / 180.0
                cos_angle = math.cos(angle_radians)
                sin_angle = math.sin(angle_radians)
                u_wind_rot = cos_angle * u_wind - sin_angle * v_wind
                v_wind_rot = sin_angle * u_wind + cos_angle * v_wind

                u_wind_effective = wind_coefficient * u_wind_rot
                v_wind_effective = wind_coefficient * v_wind_rot

                u_physalia = u_current + u_wind_effective
                v_physalia = v_current + v_wind_effective

                particle.u_wind = u_wind
                particle.v_wind = v_wind
                particle.u_current = u_current
                particle.v_current = v_current

                lat_rad = particle.lat * math.pi / 180.0
                u_physalia_coord = u_physalia / (111111.0 * math.cos(lat_rad))
                v_physalia_coord = v_physalia / 111111.0

                particle.u_movement = u_physalia_coord
                particle.v_movement = v_physalia_coord

                particle_dlon += u_physalia_coord * particle.dt
                particle_dlat += v_physalia_coord * particle.dt

                du = math.sqrt(2.0 * k * math.fabs(particle.dt)) * ParcelsRandom.normalvariate(0, 1)
                dv = math.sqrt(2.0 * k * math.fabs(particle.dt)) * ParcelsRandom.normalvariate(0, 1)

                lat_rad2 = (particle.lat + particle_dlat) * math.pi / 180.0
                particle_dlon += du / (111111.0 * math.cos(lat_rad2))
                particle_dlat += dv / 111111.0

    # Permanently stop the first time-step a stranded particle is detected
    def stranded_check(particle, fieldset, time):
        oob_bc = particle.lon < particle.min_longitude or particle.lon > particle.max_longitude or particle.lat < particle.min_latitude or particle.lat > particle.max_latitude
        if oob_bc:
            particle.delete()
        else:
            if particle.stranded == 1 and particle.stopped == 0:
                particle.stopped = 1

    # Combine all kernels
    combined_kernel = pset.Kernel(handle_out_of_bounds) + pset.Kernel(SmagorinskyKernel) + pset.Kernel(PhysaliaAdvectionKernel) + pset.Kernel(land_check) + pset.Kernel(stranded_check)

    pset.execute(combined_kernel,
                 runtime=timedelta(days=365),
                 dt=timedelta(hours=6),
                 output_file=pset.ParticleFile(name=full_output_path, outputdt=timedelta(hours=24)),
                 verbose_progress=True)

    # Extract stranded particle info from zarr (particle.stopped flips 0→1 at first stranding)
    ds_out = xr.open_zarr(full_output_path)
    stopped = ds_out.stopped.values    # (trajectory, obs)
    lat_out = ds_out.lat.values
    lon_out = ds_out.lon.values
    time_out = ds_out.time.values
    hand_out = ds_out.handedness.values
    stranded_records = []
    for traj in range(stopped.shape[0]):
        idx = np.where(stopped[traj, :] == 1)[0]
        if len(idx) > 0:
            i = idx[0]
            stranded_records.append((lat_out[traj, i], lon_out[traj, i], time_out[traj, i], hand_out[traj, i]))
    ds_out.close()
    with open(stranded_output_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Latitude', 'Longitude', 'time', 'handedness'])
        for item in stranded_records:
            writer.writerow(item)


# Set up the fieldset
def setup_fieldset(land_mask_tif, base_data, current_file, wind_file,
                   time_origin=datetime(2022, 11, 1)):

    with rasterio.open(land_mask_tif) as src:
        landmask_array = src.read(1)
        transform = src.transform

    binary_landmask = np.where(landmask_array == 1, 1, 0).astype(np.float32)
    lon = np.linspace(transform[2], transform[2] + transform[0] * src.width, src.width, endpoint=False)
    lat = np.linspace(transform[5], transform[5] + transform[4] * src.height, src.height, endpoint=False)
    landmask = Field('landmask', binary_landmask, lon=lon, lat=lat, mesh='spherical', transpose=False, interp_method='nearest', time_origin=time_origin)

    # Establish landmask boundaries
    landmask.grid.lon_min = np.min(landmask.lon)
    landmask.grid.lon_max = np.max(landmask.lon)
    landmask.grid.lat_min = np.min(landmask.lat)
    landmask.grid.lat_max = np.max(landmask.lat)

    base_dimensions = {'lon': 'lon', 'lat': 'lat', 'time': 'time'}

    # Create fields directly from NetCDF file
    U = Field.from_netcdf(base_data, variable='U', dimensions=base_dimensions, interp_method='linear', allow_time_extrapolation=True, time_origin=time_origin)
    V = Field.from_netcdf(base_data, variable='V', dimensions=base_dimensions, interp_method='linear', allow_time_extrapolation=True, time_origin=time_origin)

    # Create the fieldset with the base fields
    fieldset = FieldSet(U=U, V=V)

    # GLO12 current
    current_dimensions = {'lon': 'longitude', 'lat': 'latitude', 'time': 'time', 'depth': 'depth'}
    utotal = Field.from_netcdf(current_file, variable='utotal', dimensions=current_dimensions,
                               interp_method='linear', allow_time_extrapolation=True, time_origin=time_origin)
    vtotal = Field.from_netcdf(current_file, variable='vtotal', dimensions=current_dimensions,
                               interp_method='linear', allow_time_extrapolation=True, time_origin=time_origin)

    # Cell areas for Smagorinsky
    lon_arr = utotal.grid.lon
    lat_arr = utotal.grid.lat
    lon_grid, lat_grid = np.meshgrid(lon_arr, lat_arr)
    dlon = np.abs(np.gradient(lon_grid, axis=1))
    dlat = np.abs(np.gradient(lat_grid, axis=0))
    sq_deg_to_sq_m = (1852 * 60) ** 2 * np.cos(np.radians(lat_grid))
    cell_areas_global = Field('cell_areas_global', (dlon * dlat * sq_deg_to_sq_m).astype(np.float32),
                              lon=lon_arr, lat=lat_arr, mesh='spherical', interp_method='nearest',
                              allow_time_extrapolation=True, time_origin=time_origin)

    # CMEMS Wind
    wind_dimensions = {'lon': 'longitude', 'lat': 'latitude', 'time': 'time'}
    u_wind = Field.from_netcdf(wind_file, variable={'u_wnd': 'eastward_wind'}, dimensions=wind_dimensions,
                               interp_method='linear', allow_time_extrapolation=True, time_origin=time_origin)
    v_wind = Field.from_netcdf(wind_file, variable={'v_wnd': 'northward_wind'}, dimensions=wind_dimensions,
                               interp_method='linear', allow_time_extrapolation=True, time_origin=time_origin)

    
    min_longitude = float(max(np.min(utotal.lon), np.min(u_wind.lon)) + 1)
    max_longitude = float(min(np.max(utotal.lon), np.max(u_wind.lon)) - 1)
    min_latitude  = float(max(np.min(utotal.lat), np.min(u_wind.lat)) + 1)
    max_latitude  = float(min(np.max(utotal.lat), np.max(u_wind.lat)) - 1)
    print(f"Longitude bounds: {min_longitude} to {max_longitude}")
    print(f"Latitude bounds: {min_latitude} to {max_latitude}")

    # Register fields in the fieldset
    fieldset.add_field(utotal,          'utotal')
    fieldset.add_field(vtotal,          'vtotal')
    fieldset.add_field(u_wind,          'u_wnd')
    fieldset.add_field(v_wind,          'v_wnd')
    fieldset.add_field(cell_areas_global)
    fieldset.add_field(landmask)

    # --- VectorFields ---
    UV_global = VectorField('UV_global', fieldset.utotal, fieldset.vtotal)
    UV_wind   = VectorField('UV_wind',   fieldset.u_wnd,  fieldset.v_wnd)
    fieldset.add_vector_field(UV_global)
    fieldset.add_vector_field(UV_wind)

    fieldset.add_constant('Cs', 0.1)

    return fieldset, min_latitude, min_longitude, max_latitude, max_longitude

# Set up the particle set
def pset_setup(offshore_df, fieldset, min_longitude=-97.75, max_longitude=-50.25, min_latitude=11.0, max_latitude=51.25):
    class MyParticle(JITParticle):
        stranded = Variable('stranded', initial=0)
        handedness = Variable('handedness', dtype=np.int32, initial=0)
        u_movement = Variable('u_movement', initial=0.0)
        v_movement = Variable('v_movement', initial=0.0)
        u_wind = Variable('u_wind', initial=0.0)
        v_wind = Variable('v_wind', initial=0.0)
        u_current = Variable('u_current', initial=0.0)
        v_current = Variable('v_current', initial=0.0)
        stopped = Variable('stopped', initial=0)
        k = Variable('k', initial=0.0)
        min_longitude = Variable('min_longitude', initial=-99.0)
        max_longitude = Variable('max_longitude', initial=-50)
        min_latitude = Variable('min_latitude', initial=12.0)
        max_latitude = Variable('max_latitude', initial=54.0)
        response_to_wind = Variable('response_to_wind', initial=0.017)
        start_times = Variable('start_times', dtype=np.int32, initial=0)

    latitudes = offshore_df['Latitude'].tolist()
    longitudes = offshore_df['Longitude'].tolist()

    reference_start_time = datetime(2022, 11, 1)
    start_of_2022 = datetime(2022, 1, 1)
    start_of_2023 = datetime(2023, 1, 1)

    start_times = []
    for day_of_year in offshore_df['DayOfYear']:
        if day_of_year > 305:
            day_of_year_date = start_of_2022 + timedelta(days=int(day_of_year) - 1)
        else:
            day_of_year_date = start_of_2023 + timedelta(days=int(day_of_year) - 1)
        relative_seconds = (day_of_year_date - reference_start_time).total_seconds()
        start_times.append(relative_seconds)

    pset = ParticleSet(fieldset=fieldset, pclass=MyParticle, lon=longitudes, lat=latitudes)

    for i, particle in enumerate(pset):
        particle.handedness = np.random.choice([-1, 1])
        particle.start_times = start_times[i]
        particle.response_to_wind = np.random.normal(loc=0.017, scale=0.005)
        particle.min_longitude = min_longitude
        particle.max_longitude = max_longitude
        particle.min_latitude  = min_latitude
        particle.max_latitude  = max_latitude

    return pset

def worker(kde, bounds, land_mask_tif, distance_from_land, num_points, run_number, fieldset,
           output_directory, min_longitude=-97.75, max_longitude=-50.25, min_latitude=11.0, max_latitude=51.25):
    """Generate points and run the simulation for a single worker."""
    np.random.seed(run_number)        # seeds pset_setup numpy calls (handedness, response_to_wind)
    ParcelsRandom.seed(run_number)    # seeds Parcels C-level JIT RNG (normalvariate in kernels)
    # Generate starting points
    points = create_starting_points(
        kde, bounds, land_mask_tif, distance_from_land, num_points)

    # Create a DataFrame for the generated points
    offshore_df = pd.DataFrame(points, columns=['Longitude', 'Latitude', 'DayOfYear'])

    # Run the simulation for these points
    run_simulation(fieldset, offshore_df, run_number, output_directory,
                   min_longitude, max_longitude, min_latitude, max_latitude)

    # Return a status update (optional)
    return f"Run {run_number} complete."



# Main workflow
def parallel_workflow(df, land_mask_tif, base_data, current_file, wind_file,
                      output_directory, num_runs=25, num_points=10000):
    """Execute the workflow in parallel."""
    # Precompute resources
    kde, distance_from_land, transform, width, height = precompute_resources(df, land_mask_tif)

    # Define bounds
    lon_min, lon_max = df['longitude'].min(), df['longitude'].max()
    lat_min, lat_max = df['latitude'].min(), df['latitude'].max()
    bounds = (lon_min, lon_max, lat_min, lat_max)

    # Set up the fieldset once
    fieldset, min_latitude, min_longitude, max_latitude, max_longitude = setup_fieldset(
        land_mask_tif, base_data, current_file, wind_file)

    # Create worker tasks
    worker_partial = partial(
        worker, kde, bounds, land_mask_tif, distance_from_land, num_points,
        fieldset=fieldset, output_directory=output_directory,
        min_longitude=min_longitude, max_longitude=max_longitude,
        min_latitude=min_latitude, max_latitude=max_latitude)

    # Use Pool only when running multiple runs in parallel.
    if num_runs == 1:
        worker_partial(0)
    else:
        num_cores = min(cpu_count(), num_runs)
        with Pool(num_cores) as pool:
            pool.map(worker_partial, range(num_runs))




# Main function
def main():
    # -----------------------------------------------------------------------
    # Update these paths to point to your local copies of the input data
    # -----------------------------------------------------------------------
    obs_file     = 'path/to/weighted_juvenile_obs.csv'      # weighted juvenile observation CSV
    land_mask    = 'path/to/landmask.tif'                   # land mask raster (GeoTIFF)
    base_data    = 'path/to/base_data.nc'                   # base U/V fields (NetCDF)
    current_file = 'path/to/GLO12_currents_cmems_2022-2023.nc'          # 6-hourly merged ocean currents (NetCDF)
    wind_file    = 'path/to/ecmwf_bias_corrected_winds_cmems_2022-2023.nc'             # 6-hourly wind fields (NetCDF)
    output_dir   = 'output/simulation_runs'                 # directory where zarr + stranded CSVs are written
    # -----------------------------------------------------------------------

    # Load and preprocess the data
    df = load_data(obs_file)

    # Run the parallel workflow
    num_runs = 25
    num_points = 10000
    parallel_workflow(df, land_mask, base_data, current_file, wind_file,
                      output_dir, num_runs, num_points)


# Execute the main function if the script is run directly
if __name__ == "__main__":
    main()
