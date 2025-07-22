import shapefile
from shapely.geometry import shape
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import numpy as np
import xarray as xr
from datetime import timedelta, datetime
from py_eddy_tracker.dataset.grid import RegularGridDataset

def filter_particles_in_area(zarr_file):
    ds = xr.open_zarr(zarr_file)
    # Filter only include points that enter area of interest (i.e. offshore region of GS)
    '''
    in_area = (ds.lat >= 38.7) & (ds.lon <= -69.7)
    in_area_computed = in_area.any(dim='obs').compute()
    trajectory_ids = ds.trajectory[in_area_computed].values
    filtered_ds = ds.sel(trajectory=trajectory_ids)
    '''
    return ds

# File Paths
ds_particles = filter_particles_in_area("/path/to/simulation_results.zarr")
file_path_adt = "/path/to/adt.nc"
shapefile_path = "/path/to/land.shp"
output_file = "/output/path/anim.mp4"

# Load ADT dataset
ds_adt   = xr.open_dataset(file_path_adt)
time_adt = ds_adt['time'].values 

start_date = np.datetime64("2022-11-01")
end_date   = np.datetime64("2023-10-31")

start_index = np.argmax(time_adt >= start_date)
end_index   = np.argmax(time_adt > end_date) - 1

if end_index == -1:
    end_index = len(time_adt) - 1

# Extract particle information
lat = ds_particles['lat'].values
lon = ds_particles['lon'].values
start_times_raw = ds_particles['start_times'].values
start_times = start_times_raw[:, 0]
start_times_in_days = start_times / (3600 * 24)

base_date = datetime(2022, 11, 1)

# Set up plot
fig, ax = plt.subplots(figsize=(12, 8))

# Load & Plot the Shapefile for Land (Without GeoPandas/Cartopy)
sf = shapefile.Reader(shapefile_path)

land_patches = []
for shape_rec in sf.shapeRecords():
    geom = shape(shape_rec.shape.__geo_interface__)
    
    # If single Polygon
    if geom.geom_type == 'Polygon':
        coords = np.array(geom.exterior.coords)
        patch = Polygon(coords, closed=True)
        land_patches.append(patch)

    # If MultiPolygon (loop over each polygon in the MultiPolygon)
    elif geom.geom_type == 'MultiPolygon':
        for sub_polygon in geom.geoms:
            coords = np.array(sub_polygon.exterior.coords)
            patch = Polygon(coords, closed=True)
            land_patches.append(patch)

# Create a PatchCollection and add to the axis
land_collection = PatchCollection(
    land_patches,
    facecolor='lightgray',  # fill
    edgecolor='black',      # boundaries
    zorder=0
)
ax.add_collection(land_collection)

# 2. Set your bounding box for the region of interest
ax.set_xlim([-81, -72])
ax.set_ylim([28, 36.5])

# 3. Format lat/lon tick labels
def format_lon(x, _):
    if x < 0:
        return f"{abs(x):.0f}°W"
    else:
        return f"{x:.0f}°E"

def format_lat(y, _):
    return f"{y:.0f}°N"

ax.xaxis.set_major_formatter(plt.FuncFormatter(format_lon))
ax.yaxis.set_major_formatter(plt.FuncFormatter(format_lat))

# 4. Initialize scatter plot & placeholders
particle_scatter = ax.scatter([], [], s=10, zorder=5, color='red')
adt_plot = None
eddy_contours = []
colorbar = None

def update(frame_number):
    global adt_plot, eddy_contours, colorbar
    current_date = base_date + timedelta(days=frame_number)
    print(f"Processing frame {frame_number}: {current_date.strftime('%Y-%m-%d')}")

    # --- PLOT PARTICLES ---
    active_particles = np.where(start_times_in_days <= frame_number)[0]
    particle_lons = lon[active_particles, frame_number]
    # Convert from [0, 360] to [-180, 180] for plotting
    particle_lons = np.where(particle_lons > 180, particle_lons - 360, particle_lons)
    particle_lats = lat[active_particles, frame_number]
    particle_scatter.set_offsets(np.c_[particle_lons, particle_lats])

    # --- CLEAR OLD ADT PLOT ---
    if adt_plot:
        for collection in adt_plot.collections:
            collection.remove()
    
    # --- SELECT NEAREST ADT TIME ---
    time_diffs = np.abs(time_adt - np.datetime64(current_date))
    adt_index = np.argmin(time_diffs)
    adt_values = ds_adt['adt'].isel(time=adt_index).values

    # Convert ADT longitudes for display
    adt_lons = ds_adt['longitude'].values
    adt_lons = np.where(adt_lons > 180, adt_lons - 360, adt_lons)

    # --- PLOT ADT CONTOURS ---
    adt_plot = ax.contourf(
        adt_lons,  
        ds_adt['latitude'].values,
        adt_values,
        levels=20,
        cmap="viridis",
        alpha=0.6,
        zorder=1
    )

    # --- COLORBAR (only once) ---
    if colorbar is None:
        cbar = plt.colorbar(adt_plot, ax=ax, orientation='vertical', shrink=0.8, pad=0.02)
        cbar.set_label("ADT (m)", fontsize=14, fontweight='bold')
        colorbar = cbar

    # --- EDDY DETECTION ---
    grid = RegularGridDataset(
        ds_adt,
        "longitude",
        "latitude",
        indexs=dict(time=adt_index),
    )
    grid.bessel_high_filter("adt", 700)
    anticyclones, cyclones = grid.eddy_identification("adt", "ugos", "vgos", current_date, 0.002, shape_error=55)
    combined_eddies = anticyclones.merge(cyclones)

    # Remove old eddy contours
    for contour_list in eddy_contours:
        for contour in contour_list:
            contour.remove()
    eddy_contours.clear()

    # Draw new eddies
    contour_collection = combined_eddies.display(ax=ax, color="k")
    eddy_contours.append(contour_collection)

    # --- UPDATE TITLE ---
    ax.set_title(f"{current_date.strftime('%Y-%m-%d')}", fontsize=16, fontweight='bold')

# How many frames to animate
frames_to_animate = range(start_index, 2)  # e.g., just 2 frames for demo

# Create and save animation
from matplotlib.animation import FuncAnimation
ani = FuncAnimation(fig, update, frames=frames_to_animate, interval=100, repeat=False)
ani.save(output_file, writer="ffmpeg", dpi=300)
plt.close()
