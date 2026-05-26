import os
import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs

# ---------------------------------------------------------------------------
# Paths — update these before running
# ---------------------------------------------------------------------------
zarr_file  = 'path/to/simulation_run.zarr'   # single-run zarr output from the particle simulation
output_dir = 'path/to/output_figures'        # directory where figures are saved
# ---------------------------------------------------------------------------

SIMULATION_START = pd.Timestamp("2022-11-01")

# Fixed colormap/norm so all four plots and the colorbar share the same scale
# (obs index == days since simulation start, max 365)
CMAP = plt.cm.viridis
NORM = mcolors.Normalize(vmin=0, vmax=365)


# ---------------------------------------------------------------------------
# Data extraction helpers
# ---------------------------------------------------------------------------

def _build_trajectory_df(ds, trajectory_ids, is_beached, trim_at_beach):
    """Extract trajectory rows for the given IDs into a single DataFrame."""
    rows = []
    for tid in trajectory_ids:
        particle = ds.sel(trajectory=tid)
        if trim_at_beach:
            # Keep obs up to and including the first beaching step
            beach_idx = (~is_beached.sel(trajectory=tid)).sum().item()
            particle_slice = particle.isel(obs=slice(0, beach_idx + 1))
        else:
            particle_slice = particle
        trimmed = particle_slice.to_dataframe().reset_index()
        offset_seconds = particle['start_times'].isel(obs=0).item()
        trimmed['start_times'] = SIMULATION_START + pd.to_timedelta(offset_seconds, unit='s')
        rows.append(trimmed)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def filter_beached_in_region(zarr_file, min_lat, max_lat, min_lon=-180, max_lon=180):
    """Return trajectory DataFrame for particles that beached within the lat/lon box."""
    ds = xr.open_zarr(zarr_file, consolidated=False)
    is_beached = ds.beached == 1

    last_valid_idx = (~is_beached).sum(dim='obs') - 1
    last_valid_idx = last_valid_idx.compute()
    last_lon = ds.lon.isel(obs=last_valid_idx).compute()
    last_lat = ds.lat.isel(obs=last_valid_idx).compute()
    actually_beached = is_beached.any(dim='obs').compute()

    mask = (
        actually_beached &
        (last_lat >= min_lat) & (last_lat <= max_lat) &
        (last_lon >= min_lon) & (last_lon <= max_lon)
    ).compute()

    trajectory_ids = ds.trajectory[mask].values
    print(f"  {len(trajectory_ids)} particles matched.")
    return _build_trajectory_df(ds, trajectory_ids, is_beached, trim_at_beach=True)


def filter_nonstranding_entered_box(zarr_file, box_min_lat, box_min_lon):
    """Return full trajectory DataFrame for particles that never beached AND at some
    point entered the region north and east of (box_min_lat, box_min_lon)."""
    ds = xr.open_zarr(zarr_file, consolidated=False)
    is_beached = ds.beached == 1

    never_beached = ~is_beached.any(dim='obs')
    # NaN positions evaluate to False in comparisons, so deleted particles are excluded
    entered_box = ((ds.lat > box_min_lat) & (ds.lon > box_min_lon)).any(dim='obs')

    mask = (never_beached & entered_box).compute()
    trajectory_ids = ds.trajectory[mask].values
    print(f"  {len(trajectory_ids)} particles matched.")
    return _build_trajectory_df(ds, trajectory_ids, is_beached, trim_at_beach=False)


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def plot_trajectories(df, output_file, extent, lat_ticks, lon_ticks, linewidth=0.3, show_endpoints=True):
    """Plot particle trajectories on a Cartopy map and save to output_file."""
    if df.empty:
        print(f"  No data — skipping {output_file}")
        return

    df = df.copy()
    df['obs_numeric'] = pd.to_numeric(df['obs'], errors='coerce')
    df['start_time_days'] = (df['start_times'] - SIMULATION_START).dt.total_seconds() / 86400.0

    fig, ax = plt.subplots(figsize=(12, 8), subplot_kw={'projection': ccrs.PlateCarree()})
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    ax.coastlines()

    gl = ax.gridlines(draw_labels=True)
    gl.top_labels    = False
    gl.right_labels  = False
    gl.xlabel_style  = {'fontsize': 16, 'fontweight': 'bold'}
    gl.ylabel_style  = {'fontsize': 16, 'fontweight': 'bold'}
    gl.xlocator      = plt.FixedLocator(lon_ticks)
    gl.ylocator      = plt.FixedLocator(lat_ticks)

    for _, group in df.groupby('trajectory'):
        times  = group['obs_numeric'].values
        lons   = group['lon'].values
        lats   = group['lat'].values
        colors = CMAP(NORM(times))

        for i in range(len(times) - 1):
            ax.plot(lons[i:i+2], lats[i:i+2],
                    color=colors[i], transform=ccrs.PlateCarree(), linewidth=linewidth)

        if show_endpoints:
            # Start point (circle), colored by launch day
            ax.scatter(lons[0], lats[0],
                       color=CMAP(NORM(group['start_time_days'].values[0])),
                       marker='o', edgecolor='black', s=30,
                       transform=ccrs.PlateCarree(), zorder=5)

            # End point (square), colored by time at last valid position
            if pd.isna(lons[-1]) or pd.isna(lats[-1]):
                vi = (~(pd.isna(lons) | pd.isna(lats))).nonzero()[0][-1]
                end_lon, end_lat, end_time = lons[vi], lats[vi], times[vi]
            else:
                end_lon, end_lat, end_time = lons[-1], lats[-1], times[-1]
            ax.scatter(end_lon, end_lat,
                       color=CMAP(NORM(end_time)),
                       marker='s', edgecolor='black', s=30,
                       transform=ccrs.PlateCarree(), zorder=5)

    plt.tight_layout()
    fig.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {os.path.basename(output_file)}")


def save_colorbar(output_file):
    """Save a standalone horizontal colorbar with month-of-season tick labels."""
    start_date = SIMULATION_START
    end_date   = start_date + pd.Timedelta(days=365)
    dates      = pd.date_range(start_date, end_date, freq='MS')
    positions  = [(d - start_date).days for d in dates]
    labels     = [d.strftime('%b') for d in dates]

    fig_cb, ax_cb = plt.subplots(figsize=(10, 1.8))
    sm = plt.cm.ScalarMappable(cmap=CMAP, norm=NORM)
    sm.set_array([])
    cbar = fig_cb.colorbar(sm, cax=ax_cb, orientation='horizontal')
    cbar.set_label('Time since November 1st, 2022', fontsize=20, fontweight='bold', labelpad=10)
    cbar.set_ticks(positions)
    cbar.set_ticklabels(labels, fontsize=18)
    plt.tight_layout()
    fig_cb.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(fig_cb)
    print(f"  Saved {os.path.basename(output_file)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(output_dir, exist_ok=True)

    lon_ticks = list(np.arange(-95, -50, 5))
    lat_ticks = list(np.arange(20, 55, 5))

    # --- Florida strandings (lat 24.4–30.7, lon -82–-79.5) ---
    print("Filtering Florida strandings...")
    df_fl = filter_beached_in_region(
        zarr_file, min_lat=24.4, max_lat=30.7, min_lon=-82, max_lon=-79.5)
    print("Plotting...")
    plot_trajectories(df_fl,
                      output_file=os.path.join(output_dir, 'trajectories_florida.png'),
                      extent=[-98, -66, 18, 36],
                      lat_ticks=[20, 25, 30],
                      lon_ticks=lon_ticks,
                      linewidth=0.3)

    # --- Georgia to Cape Hatteras strandings (lat 30.7–35.76417, lon <= -75) ---
    print("Filtering Georgia–Cape Hatteras strandings...")
    df_gc = filter_beached_in_region(
        zarr_file, min_lat=30.7, max_lat=35.76417, max_lon=-75)
    print("Plotting...")
    plot_trajectories(df_gc,
                      output_file=os.path.join(output_dir, 'trajectories_ga_to_hatteras.png'),
                      extent=[-92.5, -69, 19, 39],
                      lat_ticks=lat_ticks,
                      lon_ticks=lon_ticks,
                      linewidth=0.3)

    # --- North of Cape Hatteras strandings (lat 35.76417–42) ---
    print("Filtering North-of-Cape-Hatteras strandings...")
    df_noch = filter_beached_in_region(
        zarr_file, min_lat=35.76417, max_lat=42)
    print("Plotting...")
    plot_trajectories(df_noch,
                      output_file=os.path.join(output_dir, 'trajectories_north_hatteras.png'),
                      extent=[-92, -66, 22, 45],
                      lat_ticks=lat_ticks,
                      lon_ticks=lon_ticks,
                      linewidth=0.5)

    # --- Never-stranded particles that entered the box north/east of 23.41412°N, 81.10163°W ---
    print("Filtering never-stranded particles that entered the target box...")
    df_ns = filter_nonstranding_entered_box(
        zarr_file, box_min_lat=23.41412, box_min_lon=-81.10163)
    sampled_ids = np.random.choice(df_ns['trajectory'].unique(), size=500, replace=False)
    df_ns = df_ns[df_ns['trajectory'].isin(sampled_ids)]
    print("Plotting 500 randomly sampled trajectories...")
    plot_trajectories(df_ns,
                      output_file=os.path.join(output_dir, 'trajectories_nonstranding_entered_box.png'),
                      extent=[-98, -52.5, 19, 50],
                      lat_ticks=lat_ticks,
                      lon_ticks=lon_ticks,
                      linewidth=0.3,
                      show_endpoints=False)

    # --- Shared colorbar (same scale for all four plots) ---
    print("Saving colorbar...")
    save_colorbar(os.path.join(output_dir, 'trajectories_colorbar.png'))


if __name__ == '__main__':
    main()
