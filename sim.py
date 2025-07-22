import numpy as np
import random
from datetime import datetime, timedelta
from parcels import FieldSet, ParticleSet, ScipyParticle, JITParticle, Variable, Field, VectorField
import os
import rasterio
import csv
from scipy.stats import gaussian_kde
from math import e
from scipy.ndimage import distance_transform_edt
import multiprocessing
from multiprocessing import Pool, Manager, cpu_count
from functools import partial
import pandas as pd

CONFIG = {
    'juvenile_obs' = 'path/to/normalized_juvenile_observations.csv'
    'water_mask' = 'path/to/water_mask.tif'
    'base_data' = 'path/to/base_data.nc'
    'wind_data' = 'path/to/wind_data.nc'
    'current_data' = 'path/to/current_data.nc'
    'k_data' = 'path/to/k_data.nc'
    'land_mask' = 'path/to/land_mask.tif'
    'output_dir' = 'path/to/output_directory'
}

stranded = []

def precompute_resources(df, water_mask_tif):
    """Precompute KDE and distance map."""
    # Generate KDE from juvenile data
    kde_data = np.vstack([
        df['longitude'],
        df['latitude'],
        df['DayOfYear_sin'],
        df['DayOfYear_cos']
    ])
    kde = gaussian_kde(kde_data, weights=df['weights'])
    
    # Precompute distance map
    with rasterio.open(water_mask_tif) as water_mask:
        land_mask = water_mask.read(1)  # 1=land, 0=water
        distance_from_land = distance_transform_edt(land_mask == 0)
        transform = water_mask.transform
        width, height = water_mask.width, water_mask.height

    return kde, distance_from_land, transform, width, height

def push_offshore(lon, lat, water_mask, distance_from_land, min_distance_pixels):
    """Push points offshore if they are too close to land."""
    row, col = water_mask.index(lon, lat)

    # Determine if generated point is within buffer region of land mass
    if not (0 <= row < water_mask.height and 0 <= col < water_mask.width):
        return (False, (lon, lat))

    if distance_from_land[row, col] >= min_distance_pixels:
        return (True, (lon, lat))

    # BFS to find the nearest offshore point
    queue = [(row, col)]
    visited = set(queue)
    while queue:
        r, c = queue.pop(0)
        for dr, dc in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            rr, cc = r + dr, c + dc
            if 0 <= rr < distance_from_land.shape[0] and 0 <= cc < distance_from_land.shape[1]:
                if (rr, cc) not in visited:
                    visited.add((rr, cc))
                    if distance_from_land[rr, cc] >= min_distance_pixels:
                        new_lon, new_lat = water_mask.xy(rr, cc)
                        return (True, (new_lon, new_lat))
                    else:
                        queue.append((rr, cc))

    return (False, (lon, lat))

def create_starting_points(kde, bounds, water_mask_tif, distance_from_land, num_points, min_distance_km=50):
    """Create starting points sampled from KDE and pushed offshore."""
    lon_min, lon_max, lat_min, lat_max = bounds
    points = []

    with rasterio.open(water_mask_tif) as water_mask:
        # Convert buffer distance into pixel representation
        km_per_pixel = water_mask.res[0] * 111.0
        min_distance_pixels = int(np.ceil(min_distance_km / km_per_pixel))
        
        # Generate starting location and time based on sampling of juvenile KDE
        while len(points) < num_points:
            lon, lat, day_sin, day_cos = kde.resample(1).flatten()
            day_of_year = np.arctan2(day_sin, day_cos) * 365 / (2 * np.pi)
            day_of_year = day_of_year if day_of_year > 0 else day_of_year + 365

            # Push generate starting point out of near or onshore location if applicable
            if lon_min <= lon <= lon_max and lat_min <= lat <= lat_max:
                is_valid, (final_lon, final_lat) = push_offshore(
                    lon, lat, water_mask, distance_from_land, min_distance_pixels)
                if is_valid:
                    points.append([final_lon, final_lat, day_of_year])

    return points

def load_data(file_path):
    """Load and preprocess the data"""
    df = pd.read_csv(file_path)
    df['DayOfYear'] = pd.to_datetime(df['observed_on']).dt.dayofyear
    df['DayOfYear_sin'] = np.sin(2 * np.pi * df['DayOfYear'] / 365)
    df['DayOfYear_cos'] = np.cos(2 * np.pi * df['DayOfYear'] / 365)
    df['weights'] = df['normalized_weights']
    return df

def meters_to_coord(lat, u, v):
    """Convert meters to latitude/longitude coordinates"""
    lat = np.radians(lat)
    v_lat_change = v / 111111
    u_lon_change = u / (111111 * np.cos(lat))
    return u_lon_change, v_lat_change

def determine_drift_angle(wind_velocity_u, wind_velocity_v):
    """Determine drift angle"""
    wind_velocity = np.sqrt(wind_velocity_u**2 + wind_velocity_v**2)
    drift_angle = 50.8 * e**(-0.15 * wind_velocity) - 0.5
    return drift_angle

def RotateVector(x, y, drift_angle, handedness):
    """Rotate vector by drift angle"""
    if handedness == -1:
        drift_angle = -drift_angle

    angle_radians = np.radians(drift_angle)
    cos_angle = np.cos(angle_radians)
    sin_angle = np.sin(angle_radians)
    x_rot = cos_angle * x - sin_angle * y
    y_rot = sin_angle * x + cos_angle * y
    return x_rot, y_rot

def run_simulation(fieldset, offshore_df, trial_number):
    
    global stranded
    
    stranded_filename = f"stranded_particles_run_{trial_number}.csv"
    output_filename = f"simulation_run_{trial_number}.zarr"
    output_directory = CONFIG['output_dir']
    full_output_path = os.path.join(output_directory, output_filename)
    stranded_output_path = os.path.join(output_directory, stranded_filename)
    
    # Function to write stranded particles to a CSV file
    def write_stranded_to_csv(stranded, filename):
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Latitude', 'Longitude', 'time', 'handedness'])  # Writing header
            for item in stranded:
                writer.writerow(item)
    
    pset = pset_setup(offshore_df, fieldset)
    
    def handle_out_of_bounds(particle, fieldset, time):
        """Handle out of bound particles"""
        if particle.lon < particle.min_longitude or particle.lon > particle.max_longitude or particle.lat < particle.min_latitude or particle.lat > particle.max_latitude:
            particle.delete()

    def land_check(particle, fieldset, time):
        """Kernel to check if particle is on land"""
        if particle.lon < particle.min_longitude or particle.lon > particle.max_longitude or particle.lat < particle.min_latitude or particle.lat > particle.max_latitude:
            particle.delete()
            return
        if fieldset.landmask[time, particle.depth, particle.lat, particle.lon] == 1.0:
            particle.stranded = 1
    
    def PhysaliaAdvectionKernel(particle, fieldset, time):
        """Physalia advection model"""
        # Ensure particle is within bounds
        if particle.lon < particle.min_longitude or particle.lon > particle.max_longitude or particle.lat < particle.min_latitude or particle.lat > particle.max_latitude:
            particle.delete()
            return
        
        # Advect particle if not stranded and start time has passed
        if particle.stranded == 0 and time >= particle.start_times:
            # Define wind coefficient, handedness and k value
            wind_coefficient = particle.response_to_wind
            handedness = particle.handedness
            k = fieldset.K_WOA[time, particle.depth, particle.lat, particle.lon]
            
            # Find current velocity and wind velocity at particle location
            (u_current, v_current) = fieldset.UV_current[time, particle.depth, particle.lat, particle.lon]
            (u_wind, v_wind) = fieldset.UV_wind[time, particle.depth, particle.lat, particle.lon]

            # Calculate angle of rotation based on wind speed
            drift_angle = determine_drift_angle(u_wind, v_wind)
            
            # Rotate wind vector based on angle of rotation
            u_wind_rot, v_wind_rot = RotateVector(u_wind, v_wind, drift_angle, handedness)
        
            # Calculate effective wind velocity
            u_wind_effective = wind_coefficient * u_wind_rot
            v_wind_effective = wind_coefficient * v_wind_rot
        
            # Calculate physalia velocity by adding current and wind velocities
            u_physalia = u_current + u_wind_effective
            v_physalia = v_current + v_wind_effective
        
            # Store wind and current velocities in particle attributes
            particle.u_wind = u_wind_effective
            particle.v_wind = v_wind_effective
            particle.u_current = u_current
            particle.v_current = v_current
        
            # Convert physalia velocity to coordinate system
            u_physalia_coord, v_physalia_coord  = meters_to_coord(particle.lat, u_physalia, v_physalia)
        
            # Store physalia velocity in particle attributes
            particle.u_movement = u_physalia_coord
            particle.v_movement = v_physalia_coord
        
            # Update particle position based on physalia velocity
            particle.lon += u_physalia_coord * particle.dt
            particle.lat += v_physalia_coord * particle.dt
        
            # Add diffusion to particle position
            du = np.sqrt(2 * k * abs(particle.dt)) * np.random.normal()
            dv = np.sqrt(2 * k * abs(particle.dt)) * np.random.normal()

            # Convert diffusion to coordinate system
            du_deg, dv_deg = meters_to_coord(particle.lat, du, dv)
        
            # Update particle position with diffusion
            particle.lon += du_deg
            particle.lat += dv_deg

            
    def stranded_check(particle, fieldset, time):
        """Kernel to check if particle is stranded"""
        global stranded
        if particle.lon < particle.min_longitude or particle.lon > particle.max_longitude or particle.lat < particle.min_latitude or particle.lat > particle.max_latitude:
            particle.delete()
            return
        if particle.stranded == 1 and particle.stopped == 0:
            stranded.append((particle.lat, particle.lon, time, particle.handedness))
            particle.stopped = 1   

        
    # Combine all kernels
    combined_kernel = pset.Kernel(handle_out_of_bounds) + pset.Kernel(PhysaliaAdvectionKernel) + pset.Kernel(land_check) + pset.Kernel(stranded_check)
    
    # Execute particle set with combined kernel
    pset.execute(combined_kernel,
                 runtime = timedelta(days=365),
                 dt=timedelta(hours=6),
                 output_file=pset.ParticleFile(name=full_output_path, outputdt=timedelta(hours=24)),
                 verbose_progress=False)
    
    # Write stranded particles to CSV
    write_stranded_to_csv(stranded, filename = stranded_output_path)
    
    stranded.clear()

def setup_fieldset(time_origin=datetime(2022, 11, 1)):
    """Setup Parcels field with environmental data"""
    base_data = CONFIG['base_data']
    wind_file = CONFIG['wind_data']
    current_file = CONFIG['current_data']

    # Load land mask raster and convert it to a binary mask
    with rasterio.open(CONFIG['land_mask']) as src:
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
    
    # Define base data variables
    base_filenames = {'U': base_data, 'V': base_data}
    base_data_variables = {
        'U': 'U', 
        'V': 'V'
    }

    # Create fields directly from base NetCDF file
    U = Field.from_netcdf(base_data, variable='U', dimensions=current_dimensions, interp_method='linear', allow_time_extrapolation=True, time_origin=time_origin)
    V = Field.from_netcdf(base_data, variable='V', dimensions=current_dimensions, interp_method='linear', allow_time_extrapolation=True, time_origin=time_origin)

    # Create the fieldset with the newly created fields
    fieldset = FieldSet(U=U, V=V)

    # Define current dimensions and domain
    current_dimensions = {
        'lon': 'lon',
        'lat': 'lat',
        'time': 'time'
    }

    current_domain = {
        'lon_min': -100,
        'lon_max': -50,
        'lat_min': 10,
        'lat_max': 50
    }
    
    # Define wind dimensions
    wind_dimensions = {
        'lon': 'longitude', 
        'lat': 'latitude', 
        'time': 'time'
    }

    # Create wind fields
    eastward_wind = Field.from_netcdf(wind_file, variable='eastward_wind', 
                                      dimensions=wind_dimensions, interp_method='linear', allow_time_extrapolation=True, time_origin=time_origin)
    northward_wind = Field.from_netcdf(wind_file, variable='northward_wind', 
                                       dimensions=wind_dimensions, interp_method='linear', allow_time_extrapolation=True, time_origin=time_origin)

    # Create current fields
    utotal = Field.from_netcdf(current_file, variable='utotal', 
                                        dimensions=current_dimensions, interp_method='linear', allow_time_extrapolation=True, time_origin=time_origin)
    vtotal = Field.from_netcdf(current_file, variable='vtotal', 
                                        dimensions=current_dimensions, interp_method='linear', allow_time_extrapolation=True, time_origin=time_origin)

    # Define K-value dimensions
    k_dimensions = {'lon': 'lon', 'lat': 'lat'}

    # Create field for eddy diffusivity
    k_path = CONFIG['k_data']
    K_WOA = Field.from_netcdf(k_path, variable='K_WOA', dimensions=k_dimensions, interp_method='nearest', allow_time_extrapolation=True, time_origin=time_origin)
    
    # Define simulation boundaries to ensure particles do not go beyond limits of environmental data
    min_longitude, max_longitude = np.min(utotal.lon) + 1, np.max(utotal.lon) - 1
    min_latitude, max_latitude = np.min(utotal.lat) + 1, np.max(utotal.lat) - 1

    # Add environmental fields to fieldset
    fieldset.add_field(eastward_wind, 'eastward_wind')
    fieldset.add_field(northward_wind, 'northward_wind')
    fieldset.add_field(vtotal, 'vtotal')
    fieldset.add_field(utotal, 'utotal')
    fieldset.add_field(landmask)
    fieldset.add_field(K_WOA, 'K_WOA')

    # Define vector fields from U and V components
    UV_current = VectorField('UV_current', fieldset.utotal, fieldset.vtotal)
    UV_wind = VectorField('UV_wind', fieldset.eastward_wind, fieldset.northward_wind)

    # Add the vector fields to the fieldset
    fieldset.add_vector_field(UV_current)
    fieldset.add_vector_field(UV_wind)

    return fieldset, min_latitude, min_longitude, max_latitude, max_longitude

def pset_setup(offshore_df, fieldset):
    """Create particle class and define particle set"""
    class MyParticle(ScipyParticle):
        stranded = Variable('stranded', initial=0)
        handedness = Variable('handedness', dtype=np.int32, initial=0)
        u_movement = Variable('u_movement', initial=0.0)
        v_movement = Variable('v_movement', initial=0.0)
        u_wind = Variable('u_wind', initial=0.0)
        v_wind = Variable('v_wind', initial=0.0)
        u_current = Variable('u_current', initial=0.0)
        v_current = Variable('v_current', initial=0.0)
        stopped = Variable('stopped', initial=0)
        min_longitude = Variable('min_longitude', initial=-98.1875)
        max_longitude = Variable('max_longitude', initial=-51.916656494140625)
        min_latitude = Variable('min_latitude', initial=8.6875)
        max_latitude = Variable('max_latitude', initial=55.1875)
        response_to_wind = Variable('response_to_wind', initial=0.017)
        start_times = Variable('start_times', dtype=np.int32, initial=0)

    # Establish starting locations for particles based on generated list 
    latitudes = offshore_df['Latitude'].tolist()
    longitudes = offshore_df['Longitude'].tolist()

    # Establish starting dates for particles, adjusted to be in datetime
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

    # Create particle set
    pset = ParticleSet(fieldset=fieldset, pclass=MyParticle, lon=longitudes, lat=latitudes)

    # Assign handedness randomly, start times based on generated list, and response to wind
    for i, particle in enumerate(pset):
        particle.handedness = np.random.choice([-1, 1])
        particle.start_times = start_times[i]
        particle.response_to_wind = np.random.normal(loc=0.017, scale=0.005)

    return pset

def worker(kde, bounds, water_mask_tif, distance_from_land, num_points, run_number, fieldset):
    """Generate points and run the simulation for a single worker."""
    # Set random seeds for reproducibility
    random.seed(run_number)
    np.random.seed(run_number)
    
    # Generate starting points
    points = create_starting_points(
        kde, bounds, water_mask_tif, distance_from_land, num_points)

    # Create a DataFrame for the generated points
    offshore_df = pd.DataFrame(points, columns=['Longitude', 'Latitude', 'DayOfYear'])

    # Run the simulation for these points
    run_simulation(fieldset, offshore_df, run_number)

    # Log completed run
    return f"Run {run_number} complete."



# Main workflow
def parallel_workflow(df, water_mask_tif, num_runs=25, num_points=10000):
    """Execute the workflow in parallel."""
    # Precompute resources
    kde, distance_from_land, transform, width, height = precompute_resources(df, water_mask_tif)

    # Define bounds
    lon_min, lon_max = df['longitude'].min(), df['longitude'].max()
    lat_min, lat_max = df['latitude'].min(), df['latitude'].max()
    bounds = (lon_min, lon_max, lat_min, lat_max)

    # Set up the fieldset once
    fieldset, min_latitude, min_longitude, max_latitude, max_longitude = setup_fieldset()

    # Create worker tasks
    worker_partial = partial(
        worker, kde, bounds, water_mask_tif, distance_from_land, num_points, fieldset=fieldset)

    # Execute in parallel
    num_cores = min(cpu_count(), num_runs)
    with Pool(num_cores) as pool:
        pool.map(worker_partial, range(num_runs))

def main():
    """Main function"""
    juvenile_obs = CONFIG['juvenile_obs']
    water_mask_tif = CONFIG['water_mask']

    # Load and preprocess the data
    df = load_data(juvenile_obs)

    # Run the parallel workflow
    num_runs = 25
    num_points = 10000
    parallel_workflow(df, water_mask_tif, num_runs, num_points)


# Execute the main function if the script is run directly
if __name__ == "__main__":
    main()
