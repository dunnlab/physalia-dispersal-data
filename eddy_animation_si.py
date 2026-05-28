import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.ticker as mticker
import numpy as np
import xarray as xr
from datetime import timedelta, datetime
from py_eddy_tracker.dataset.grid import RegularGridDataset

# ---------------------------------------------------------------------------
# Paths — update before running
# ---------------------------------------------------------------------------
ZARR_FILE   = "/path/to/nowind.zarr"
ADT_FILE    = "/path/to/GLO12_adt_cmems_2022-2023.nc"
OUTPUT_FILE = "/output/eddy_animation.mp4"
# ---------------------------------------------------------------------------

START_DATE = np.datetime64("2022-11-01")
END_DATE   = np.datetime64("2023-10-31")
BASE_DATE  = datetime(2022, 11, 1)

LAT_FILTER = 36          # only animate particles that reach this latitude or higher to focus on GS offshore region
LON_MIN, LON_MAX = 281, 306
LAT_MIN, LAT_MAX = 32, 43


def filter_particles_in_area(zarr_file, lat_threshold=LAT_FILTER):
    ds = xr.open_zarr(zarr_file)
    in_area = (ds.lat >= lat_threshold)
    in_area_computed = in_area.any(dim='obs').compute()
    trajectory_ids = ds.trajectory[in_area_computed].values
    print(f"{len(trajectory_ids)} particles reached lat >= {lat_threshold}")
    return ds.sel(trajectory=trajectory_ids)


def convert_lon_labels(x, pos):
    lon = (x + 180) % 360 - 180
    return f"{abs(int(lon))}°{'W' if lon < 0 else 'E' if lon > 0 else ''}"


def format_lat_labels(y, pos):
    return f"{abs(int(y))}°{'S' if y < 0 else 'N' if y > 0 else ''}"


# --- Load data ---
ds_particles = filter_particles_in_area(ZARR_FILE)

ds_adt   = xr.open_dataset(ADT_FILE)
time_adt = ds_adt['time'].values

start_index = np.argmax(time_adt >= START_DATE)
end_index   = np.argmax(time_adt > END_DATE) - 1
if end_index == -1:
    end_index = len(time_adt) - 1

# Convert ADT longitudes from -180:180 to 0:360
adt_lon_raw     = ds_adt['longitude'].values
adt_lon_360     = np.where(adt_lon_raw < 0, adt_lon_raw + 360, adt_lon_raw)
adt_lon_sort_idx = np.argsort(adt_lon_360)
adt_lon_sorted  = adt_lon_360[adt_lon_sort_idx]

# Consistent colormap scaling across all frames
adt_all  = ds_adt['zos'].isel(time=slice(start_index, end_index + 1)).values
vmin_adt = float(np.nanmin(adt_all))
vmax_adt = float(np.nanmax(adt_all))

# Particle arrays
lat             = ds_particles['lat'].values
lon             = ds_particles['lon'].values
start_times_raw = ds_particles['start_times'].values
start_times     = start_times_raw[:, 0]
start_times_in_days = start_times / (3600 * 24)

# --- Figure setup ---
fig, ax = plt.subplots(figsize=(12, 8))
ax.set_xlim(LON_MIN, LON_MAX)
ax.set_ylim(LAT_MIN, LAT_MAX)
ax.set_xticklabels(ax.get_xticks(), fontweight='bold')
ax.set_yticklabels(ax.get_yticks(), fontweight='bold')
ax.xaxis.set_major_formatter(mticker.FuncFormatter(convert_lon_labels))
ax.yaxis.set_major_formatter(mticker.FuncFormatter(format_lat_labels))

particle_scatter = ax.scatter([], [], s=10, zorder=5, color='red')
adt_plot    = None
eddy_contours = []

sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=vmin_adt, vmax=vmax_adt))
sm.set_array([])
fig.colorbar(sm, ax=ax, label='ADT (m)', pad=0.02)


def update(frame_number):
    global adt_plot, eddy_contours
    print(f"Processing frame {frame_number}")
    current_date = BASE_DATE + timedelta(days=frame_number)

    # Particles
    active_particles = np.where(start_times_in_days <= frame_number)[0]
    particle_lons = [(l + 360) if l < 0 else l for l in lon[active_particles, frame_number]]
    particle_lats = lat[active_particles, frame_number]
    particle_scatter.set_offsets(np.c_[particle_lons, particle_lats])

    # ADT
    if adt_plot is not None:
        adt_plot.remove()
    adt_index  = np.argmin(np.abs(time_adt - np.datetime64(current_date)))
    adt_values = ds_adt['zos'].isel(time=adt_index).values
    adt_plot   = ax.pcolormesh(
        adt_lon_sorted,
        ds_adt['latitude'].values,
        adt_values[:, adt_lon_sort_idx],
        cmap='viridis', alpha=0.6,
        vmin=vmin_adt, vmax=vmax_adt,
        shading='auto', zorder=1,
    )

    # Eddy detection
    grid = RegularGridDataset(ADT_FILE, "longitude", "latitude",
                               indexs=dict(time=adt_index))
    grid.bessel_high_filter("zos", 700)
    grid.add_uv("zos")
    a, c = grid.eddy_identification("zos", "u", "v", current_date, 0.002, shape_error=55)
    combined_eddies = a.merge(c)

    for contour_list in eddy_contours:
        for contour in contour_list:
            contour.remove()
    eddy_contours.clear()
    eddy_contours.append(combined_eddies.display(ax=ax, color="k"))

    ax.set_title(f"{current_date.strftime('%Y-%m-%d')}", fontweight='bold')


frames_to_animate = range(start_index, end_index + 1)
print(f"Animating {len(frames_to_animate)} frames…")

ani = FuncAnimation(fig, update, frames=frames_to_animate, interval=100, repeat=False)
ani.save(OUTPUT_FILE, writer="ffmpeg", dpi=300)
