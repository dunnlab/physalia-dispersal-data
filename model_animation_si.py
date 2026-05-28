#!/usr/bin/env python3
"""
physalia_animation_stranded.py
Animates ONLY particles that eventually strand in target regions,
colored by their eventual stranding region throughout their trajectory.
"""

import os, subprocess, shutil
import numpy as np
import pandas as pd
import xarray as xr
import zarr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import Normalize
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# ============================================================
# PATHS & CONSTANTS
# ============================================================
BASE       = "/directory"
ZARR_PATH  = f"{BASE}/run_(run_number)_output.zarr"
SST_PATH   = f"{BASE}/GLO12_sst_cmems_2022-2023.nc"
WIND_PATH  = f"{BASE}/ecmwf_bias_corrected_winds_cmems_2022-2023.nc"
OUTPUT     = f"{BASE}/model_animation.mp4"

LON0, LON1  =  -99.0,  -55.0
LAT0, LAT1  =   17.0,   48.0
SIM_START   = pd.Timestamp("2022-11-01")
N_DAYS      = 365
SST_VMIN    = 6.0
SST_VMAX    = 30.0
WIND_SKIP   = 8
WIND_CAP    = 15.0
PART_FILTER_LAT = 23.5
PART_FILTER_LON = -82

OCEAN_BG = "#0d1b4a"
LAND_COL = "#e0e0e0"
FIG_BG   = "white"

REGION_COLORS = {
    "Florida":                  "#3260ff",
    "Georgia to Cape Hatteras": "#de2eff",
    "North of Cape Hatteras":   "#ff0d0d",
}

def strand_color(lat):
    if lat < 30.71044:
        return REGION_COLORS["Florida"]
    elif lat < 35.76417:
        return REGION_COLORS["Georgia to Cape Hatteras"]
    else:
        return REGION_COLORS["North of Cape Hatteras"]

# ============================================================
# PARTICLE DATA
# ============================================================
print("Loading zarr …")
store  = zarr.open(ZARR_PATH, mode="r")
lat_z  = store["lat"][:]
lon_z  = store["lon"][:]
st_raw = store["start_times"][:, 0]
start_times_days = st_raw / 86400.0
beached_z = store["beached"][:]

# Initial filter: ever enter target region
valid_pos = ~np.isnan(lat_z)
in_target = valid_pos & (lat_z > PART_FILTER_LAT) & (lon_z > PART_FILTER_LON)
traj_keep = np.any(in_target, axis=1)

lat_all        = lat_z[traj_keep]
lon_all        = lon_z[traj_keep]
beached_all    = beached_z[traj_keep]
start_days_all = start_times_days[traj_keep]

# Find eventual stranding position for each particle
ever_beached    = np.any(beached_all == 1, axis=1)
first_beach_idx = np.where(ever_beached, np.argmax(beached_all == 1, axis=1), 0).astype(int)
beach_lat = lat_all[np.arange(lat_all.shape[0]), first_beach_idx]
beach_lon = lon_all[np.arange(lon_all.shape[0]), first_beach_idx]

# Target stranding: beached in valid regions
excluded = ever_beached & (
    ((beach_lat < 27.3) & (beach_lon > -79.5)) |  # Bahamas
    (beach_lat > 43.0)                            |  # Canada
    (beach_lat < 24.0)                            |  # Cuba
    (beach_lon > -66)                             |  # too far east
    (beach_lon < -81.55459)                          # too far west
)
target_strand = ever_beached & ~excluded

# Target-stranding particles (colored by region)
lat_f        = lat_all[target_strand]
lon_f        = lon_all[target_strand]
beached_f    = beached_all[target_strand]
start_days_f = start_days_all[target_strand]
beach_lat_f  = beach_lat[target_strand]
part_colors  = np.array([strand_color(la) for la in beach_lat_f])

# All other particles that passed the initial filter and never stranded anywhere
other_mask   = ~ever_beached
lat_o        = lat_all[other_mask]
lon_o        = lon_all[other_mask]
beached_o    = beached_all[other_mask]
start_days_o = start_days_all[other_mask]

print(f"  {lat_f.shape[0]} particles stranding in target regions")
print(f"  {lat_o.shape[0]} other particles passing filter")

# ============================================================
# SST
# ============================================================
print("Loading SST …")
ds_sst   = xr.open_dataset(SST_PATH)
sst_surf = ds_sst["thetao"].isel(depth=0).sel(time=slice("2022-11-01", "2023-10-31"))
sst_lat  = sst_surf.latitude.values
sst_lon  = sst_surf.longitude.values
sst_vals = sst_surf.values.astype(np.float32)
N_SST    = sst_vals.shape[0]
print(f"  SST grid: {len(sst_lat)} × {len(sst_lon)}, {N_SST} daily frames")

# ============================================================
# WIND
# ============================================================
print("Loading wind …")
ds_wind     = xr.open_dataset(WIND_PATH)
wind_region = ds_wind.sel(
    time      = slice("2022-11-01", "2023-10-31 23:59"),
    latitude  = slice(LAT0, LAT1),
    longitude = slice(LON0, LON1),
)
wind_sub = wind_region.isel(
    latitude  = slice(None, None, WIND_SKIP),
    longitude = slice(None, None, WIND_SKIP),
)
wind_times_sim = pd.DatetimeIndex(wind_sub.time.values)
wlat = wind_sub.latitude.values
wlon = wind_sub.longitude.values
WLON, WLAT = np.meshgrid(wlon, wlat)

print("  Loading wind arrays into memory …")
u_raw = wind_sub["eastward_wind"].values.astype(np.float32)
v_raw = wind_sub["northward_wind"].values.astype(np.float32)
spd = np.sqrt(u_raw**2 + v_raw**2)
cap = np.minimum(spd, WIND_CAP)
with np.errstate(invalid="ignore"):
    u_disp = np.where(spd > 0.5, u_raw / spd * cap, 0.0).astype(np.float32)
    v_disp = np.where(spd > 0.5, v_raw / spd * cap, 0.0).astype(np.float32)
print(f"  {len(wind_times_sim)} wind frames; quiver grid {WLAT.shape}")

sim_dates    = pd.date_range("2022-11-01", periods=N_DAYS, freq="D")
wind_day_idx = np.array(
    [np.argmin(np.abs(wind_times_sim - t)) for t in sim_dates], dtype=int
)

# ============================================================
# FIGURE
# ============================================================
PROJ = ccrs.PlateCarree()

fig, ax = plt.subplots(figsize=(16, 9), facecolor=FIG_BG,
                       subplot_kw={"projection": PROJ})
fig.subplots_adjust(left=0.06, right=0.97, bottom=0.06, top=0.91)
fig.patch.set_facecolor(FIG_BG)
ax.set_facecolor(OCEAN_BG)
ax.set_extent([LON0, LON1, LAT0, LAT1], crs=PROJ)

# --- SST imshow ---
sst_norm = Normalize(vmin=SST_VMIN, vmax=SST_VMAX)
cmap_sst = plt.cm.RdBu_r.copy()
cmap_sst.set_bad(alpha=0)
temp_plot = ax.imshow(
    np.ma.masked_invalid(sst_vals[0]),
    cmap=cmap_sst, norm=sst_norm,
    extent=[sst_lon[0], sst_lon[-1], sst_lat[0], sst_lat[-1]],
    origin="lower", transform=PROJ,
    interpolation="bilinear",
    zorder=1,
)

# --- Wind quiver ---
wi0  = wind_day_idx[0]
quiv = ax.quiver(
    WLON, WLAT, u_disp[wi0], v_disp[wi0],
    transform=PROJ,
    color="white", alpha=0.28,
    scale=280, scale_units="width",
    width=0.0013, headwidth=3.5, headlength=4.5,
    zorder=2, pivot="mid",
)
ax.quiverkey(quiv, 0.13, 0.035, 10.0, "10 m s⁻¹",
             labelpos="E", color="white", labelcolor="#cccccc",
             fontproperties={"size": 10})

# Green scatter for non-target particles (drawn below region-colored ones)
scat_other  = ax.scatter([], [], s=22, c="#009E73",
                         edgecolors="black", linewidths=0.7,
                         alpha=1.0, zorder=4, transform=PROJ)

# Two scatters: drifting (full size) and stranded (slightly smaller)
# Colors set per-frame from pre-assigned part_colors
scat_drift  = ax.scatter([], [], s=22, c="white",
                         edgecolors="black", linewidths=0.7,
                         alpha=1.0, zorder=5, transform=PROJ)
scat_strand = ax.scatter([], [], s=16, c="white",
                         edgecolors="black", linewidths=0.5,
                         alpha=1.0, zorder=5, transform=PROJ)

# Legend patches
from matplotlib.patches import Patch
legend_handles = [Patch(facecolor=c, edgecolor="black", label=r)
                  for r, c in REGION_COLORS.items()]
legend_handles.append(Patch(facecolor="#009E73", edgecolor="black", label="Did not strand"))
leg = ax.legend(handles=legend_handles, loc="lower right", fontsize=11,
                framealpha=0.85, edgecolor="black", title="Stranding Region")
leg.get_title().set_fontweight("bold")
leg.get_title().set_fontsize(12)

# --- Cartopy land + coastline ---
ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor=LAND_COL, zorder=3)
ax.add_feature(cfeature.COASTLINE.with_scale("50m"), edgecolor="#000000", linewidth=0.8, zorder=4)

# --- Gridlines ---
gl = ax.gridlines(crs=PROJ, draw_labels=True, linewidth=0, color="none",
                  x_inline=False, y_inline=False)
gl.top_labels   = False
gl.right_labels = False
gl.xlabel_style = {"color": "black", "size": 14, "weight": "bold"}
gl.ylabel_style = {"color": "black", "size": 14, "weight": "bold"}
gl.xlocator = mticker.FixedLocator(range(-100, -49, 10))
gl.ylocator = mticker.FixedLocator(range(20, 53, 10))
for sp in ax.spines.values():
    sp.set_edgecolor("black")

# --- Labels ---
date_lbl  = ax.text(0.5, 1.014, "", transform=ax.transAxes,
                    ha="center", va="bottom", fontsize=22,
                    fontweight="bold", color="black")
count_lbl = ax.text(0.015, 0.972, "", transform=ax.transAxes,
                    ha="left", va="top", fontsize=13,
                    fontweight="bold", color="black")

# ============================================================
# RENDER FRAME-BY-FRAME
# ============================================================
FRAME_DIR = f"{BASE}/_frames_stranded"
FPS       = 8
os.makedirs(FRAME_DIR, exist_ok=True)

print(f"Rendering {N_DAYS} frames …")
for d in range(N_DAYS):
    temp_plot.set_data(np.ma.masked_invalid(sst_vals[min(d, N_SST - 1)]))
    quiv.set_UVC(u_disp[wind_day_idx[d]], v_disp[wind_day_idx[d]])

    # Other (green) particles
    started_o = start_days_o <= d
    lat_d_o   = lat_o[:, d]
    lon_d_o   = lon_o[:, d]
    active_o  = started_o & ~np.isnan(lat_d_o)
    other_idx = np.where(active_o)[0]
    scat_other.set_offsets(
        np.column_stack([lon_d_o[other_idx], lat_d_o[other_idx]])
        if len(other_idx) else np.empty((0, 2))
    )

    # Target-stranding particles (region colors)
    started  = start_days_f <= d
    lat_d    = lat_f[:, d]
    lon_d    = lon_f[:, d]
    beached  = started & (beached_f[:, d] == 1)
    drifting = started & ~np.isnan(lat_d) & ~beached

    # Drifting particles
    drift_idx = np.where(drifting)[0]
    scat_drift.set_offsets(
        np.column_stack([lon_d[drift_idx], lat_d[drift_idx]])
        if len(drift_idx) else np.empty((0, 2))
    )
    if len(drift_idx):
        scat_drift.set_facecolor(part_colors[drift_idx])

    # Stranded particles
    strand_idx = np.where(beached)[0]
    scat_strand.set_offsets(
        np.column_stack([lon_d[strand_idx], lat_d[strand_idx]])
        if len(strand_idx) else np.empty((0, 2))
    )
    if len(strand_idx):
        scat_strand.set_facecolor(part_colors[strand_idx])

    date_lbl.set_text(sim_dates[d].strftime("%B %d, %Y"))
    #count_lbl.set_text(f"n = {drifting.sum():,} drifting  |  {beached.sum():,} stranded")

    fig.savefig(f"{FRAME_DIR}/frame_{d:04d}.png", dpi=200)
    print(f"\r  frame {d+1}/{N_DAYS}", end="", flush=True)

print("\nStitching video …")
subprocess.run([
    "ffmpeg", "-y",
    "-framerate", str(FPS),
    "-i", f"{FRAME_DIR}/frame_%04d.png",
    "-vcodec", "h264_videotoolbox",
    "-b:v", "20000k",
    "-pix_fmt", "yuv420p",
    OUTPUT,
], check=True)

shutil.rmtree(FRAME_DIR)
print(f"Saved → {OUTPUT}")
