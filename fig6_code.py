print("********** Script has started **********", flush=True)

import xarray as xr
import pandas as pd
import numpy as np
import os
import re
from glob import glob
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.font_manager import FontProperties
from scipy.stats import pearsonr

# ---------------------------------------------------------------------------
# Paths 
# ---------------------------------------------------------------------------
INPUT_DIR  = "path/to/simulation_runs"
WIND_FILE  = "path/to/ecmwf_bias_corrected_winds_cmems_2022-2023.nc"
OUTPUT_FIG = "/output/fig6.png"
# ---------------------------------------------------------------------------

LAT_THRESHOLD = 39.5
LON_THRESHOLD = -67.7

highlight_bins = [
    ("2022-11-29", "2022-12-13"), ("2022-12-13", "2022-12-27"),
    ("2023-04-04", "2023-04-18"), ("2023-04-18", "2023-05-02"),
    ("2023-05-02", "2023-05-16"), ("2023-05-16", "2023-05-30"),
    ("2023-05-30", "2023-06-13"), ("2023-06-13", "2023-06-27"),
    ("2023-06-27", "2023-07-11"), ("2023-07-11", "2023-07-25"),
    ("2023-07-25", "2023-08-08"), ("2023-08-08", "2023-08-22"),
    ("2023-08-22", "2023-09-05"), ("2023-09-05", "2023-09-19"),
    ("2023-09-19", "2023-10-03"), ("2023-10-03", "2023-10-17"),
    ("2023-10-17", "2023-10-31"),
]


# ---------------------------------------------------------------------------
# Counting logic 
# ---------------------------------------------------------------------------

def get_zarr_files(directory):
    return sorted(glob(os.path.join(directory, "*.zarr")))


def filter_particles_in_area(zarr_file, lat_threshold, lon_threshold):
    ds = xr.open_zarr(zarr_file)

    in_area = (ds.lat >= lat_threshold) & (ds.lon <= lon_threshold)

    stranded_in_area = (
        ds.lat.isnull() & ds.lon.isnull() &
        (ds.lat.shift(obs=1) >= lat_threshold) &
        (ds.lon.shift(obs=1) <= lon_threshold)
    )

    valid_particles = in_area | stranded_in_area
    valid_particles_computed = valid_particles.any(dim='obs').compute()
    trajectory_ids = ds.trajectory[valid_particles_computed].values
    filtered_ds = ds.sel(trajectory=trajectory_ids)

    return filtered_ds.to_dataframe().reset_index()


def count_entries_per_bin(df, lat_threshold, lon_threshold):
    df = df.sort_values(['trajectory', 'obs']).reset_index(drop=True)

    is_inside = (df['lat'] >= lat_threshold) & (df['lon'] <= lon_threshold)
    is_inside_prev = is_inside.groupby(df['trajectory']).shift(1).fillna(False)
    df['entry_event'] = (~is_inside_prev) & is_inside

    df['time'] = pd.to_datetime(df['time'])
    df['time_bin'] = pd.cut(
        df['time'],
        bins=pd.date_range("2022-11-01", "2023-11-01", freq="14D"),
        right=False
    )

    entry_df = df[df['entry_event']]
    return entry_df.groupby('time_bin')['trajectory'].nunique()


def run_counts(input_dir, lat_threshold, lon_threshold):
    zarr_files = get_zarr_files(input_dir)
    all_data = {}
    for zarr_file in zarr_files:
        run_name = os.path.basename(zarr_file).replace(".zarr", "")
        print(f"Processing {zarr_file}...", flush=True)
        df = filter_particles_in_area(zarr_file, lat_threshold, lon_threshold)
        all_data[run_name] = count_entries_per_bin(df, lat_threshold, lon_threshold)
    return pd.DataFrame(all_data).fillna(0)


# ---------------------------------------------------------------------------
# Correlation analysis
# ---------------------------------------------------------------------------

def max_consecutive_southerly(v_series):
    """Longest run of consecutive days with northward (southerly) wind (v > 0)."""
    max_run = current = 0
    for v in v_series:
        if v > 0:
            current += 1
            max_run = max(max_run, current)
        else:
            current = 0
    return max_run


def compute_correlations(v_daily, particle_df):
    """Pearson correlations between wind metrics and mean particle entries per 14-day bin."""
    bins  = pd.date_range("2022-11-01", "2023-11-01", freq="14D")
    times = pd.to_datetime(v_daily["time"].values)
    v_vals = v_daily.values

    mean_counts = particle_df.groupby("Period")["Particle Count"].mean()

    wind_means, consec_days, part_means = [], [], []
    for i in range(len(bins) - 1):
        start, end = bins[i], bins[i + 1]
        mask  = (times >= start) & (times < end)
        v_bin = v_vals[mask]
        key   = pd.Timestamp(start)
        if len(v_bin) == 0 or key not in mean_counts.index:
            continue
        wind_means.append(float(v_bin.mean()))
        consec_days.append(max_consecutive_southerly(v_bin))
        part_means.append(mean_counts[key])

    wind_means  = np.array(wind_means)
    consec_days = np.array(consec_days)
    part_means  = np.array(part_means)
    df          = len(part_means) - 2

    r1, p1 = pearsonr(wind_means,  part_means)
    r2, p2 = pearsonr(consec_days, part_means)

    print("\n--- Wind-Particle Pearson Correlations ---")
    print(f"Mean northward wind vs. entries:        r = {r1:.3f}, df = {df}, p = {p1:.4f}")
    print(f"Consecutive southerly days vs. entries: r = {r2:.3f}, df = {df}, p = {p2:.4f}")
    return r1, p1, r2, p2


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def parse_interval_string(s):
    match = re.match(r"\[(.*), (.*)\)", str(s))
    if match:
        start = pd.to_datetime(match.group(1).strip())
        return pd.Interval(left=start, right=pd.to_datetime(match.group(2).strip()), closed='left')
    return pd.NaT


def prep_particle_df(counts_df):
    counts_df = counts_df.copy()
    counts_df.index = counts_df.index.astype(str)
    counts_df = counts_df.reset_index().rename(columns={"index": "Period"})
    counts_df["Period"] = counts_df["Period"].apply(parse_interval_string)
    counts_df["Period"] = counts_df["Period"].apply(lambda x: x.left if pd.notnull(x) else np.nan)
    counts_df.dropna(subset=["Period"], inplace=True)
    return counts_df.melt(id_vars="Period", var_name="Run", value_name="Particle Count")


def make_plot(particle_df, wind_file, output_fig):
    all_periods = pd.date_range("2022-11-01", "2023-11-01", freq="14D")
    period_lookup = {pd.Timestamp(p): mdates.date2num(p + pd.Timedelta(days=7)) for p in all_periods}
    particle_df["Period_num"] = particle_df["Period"].map(lambda p: period_lookup.get(pd.Timestamp(p), np.nan))
    particle_df.dropna(subset=["Period_num"], inplace=True)

    # Wind data
    ds = xr.open_dataset(wind_file)
    subset = ds.sel(latitude=slice(37, 41), longitude=slice(-74, -69))
    subset = subset.sel(time=slice("2022-11-01", "2023-10-31"))
    daily = subset.resample(time="1D").mean(keep_attrs=True)
    u_daily = daily["eastward_wind"].mean(dim=("latitude", "longitude"))
    v_daily = daily["northward_wind"].mean(dim=("latitude", "longitude"))
    dates_num = mdates.date2num(u_daily["time"].values)
    baseline = np.zeros_like(dates_num)
    max_abs_uv = np.nanmax(np.hypot(u_daily, v_daily))
    ylim = max_abs_uv * 1.15

    compute_correlations(v_daily, particle_df)

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(14, 8), sharex=True, gridspec_kw={'height_ratios': [3, 1]}
    )

    # Top panel
    ax1.scatter(particle_df["Period_num"], particle_df["Particle Count"],
                alpha=0.5, color='#ff0d0d', zorder=4, label="Particles", s=15)

    for dt in all_periods:
        pos = period_lookup[dt]
        group = particle_df[particle_df["Period"] == dt]
        values = group["Particle Count"].values if not group.empty else []
        is_highlight = any(pd.to_datetime(start) <= dt < pd.to_datetime(end) for start, end in highlight_bins)
        box_has_no_face = is_highlight and (len(values) == 0 or np.percentile(values, 75) == 0)
        facecolor = "#ffaaaa" if (is_highlight and not box_has_no_face) else "none"
        line_color = "red" if box_has_no_face else "black"
        ax1.boxplot(
            [values], positions=[pos], widths=10,
            showfliers=False, patch_artist=True,
            boxprops=dict(facecolor=facecolor, color=line_color),
            medianprops=dict(color=line_color),
            whiskerprops=dict(color=line_color),
            capprops=dict(color=line_color),
            zorder=3
        )

    ax1.set_ylabel("Particles Entering Area C", fontsize=20, fontweight='bold')
    ax1.tick_params(axis='y', labelsize=20)
    ax1.tick_params(axis='x', labelsize=0)

    # Bottom panel
    q = ax2.quiver(
        dates_num, baseline, u_daily.values, v_daily.values,
        angles="xy", scale_units="xy", scale=0.9,
        width=0.0025, headlength=0, headwidth=0, headaxislength=0
    )
    ax2.set_ylim(-ylim, ylim)
    ax2.set_yticks([])
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    fig.autofmt_xdate(rotation=0, ha="center")
    ax2.set_xlabel("Month", fontsize=22, fontweight='bold')
    ax2.tick_params(axis='x', labelsize=20)
    ax2.set_xlim([
        mdates.date2num(pd.Timestamp("2022-10-25")),
        mdates.date2num(pd.Timestamp("2023-11-10"))
    ])
    ax2.set_ylabel("Average Wind \n Velocity (m/s)", fontsize=20, fontweight='bold')

    bold_font = FontProperties(weight='bold', size=16)
    ax2.quiverkey(q, X=0.92, Y=0.9, U=10,
                  label="10 m/s", labelpos='E', coordinates='axes', fontproperties=bold_font)

    plt.tight_layout(h_pad=1.5)
    plt.savefig(output_fig, dpi=600, bbox_inches='tight')
    print(f"Figure saved to {output_fig}")
    plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

print("Counting particles...", flush=True)
counts_df = run_counts(INPUT_DIR, LAT_THRESHOLD, LON_THRESHOLD)

print("Preparing data for plotting...", flush=True)
particle_df = prep_particle_df(counts_df)

print("Plotting...", flush=True)
make_plot(particle_df, WIND_FILE, OUTPUT_FIG)

print("\nScript completed!", flush=True)
