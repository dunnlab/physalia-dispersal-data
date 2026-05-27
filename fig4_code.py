import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.patches import Patch
from datetime import datetime

# ---------------------------------------------------------------------------
# Paths — update these before running
# ---------------------------------------------------------------------------
beached_sim_csv  = 'path/to/beached_particles_single_run.csv'  # single-run beaching CSV from simulation
sim_summary_csv  = 'path/to/regional_stranding_summary.csv'    # output of count_strandings_by_region.py
obs_csv          = 'path/to/filtered_stranding_obs.csv'        # pre-filtered iNaturalist stranding CSV
output_dir       = 'path/to/output_figures'
# ---------------------------------------------------------------------------

REGION_COLORS = {
    'Florida':                  '#3260ff',
    'Georgia to Cape Hatteras': '#de2eff',
    'North of Cape Hatteras':   '#ff0d0d',
}
REGIONS      = ['Florida', 'Georgia to Cape Hatteras', 'North of Cape Hatteras']
MONTH_ORDER  = [11, 12, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
MONTH_LABELS = ['Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr',
                'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct']
MONTH_TO_XPOS = {m: i for i, m in enumerate(MONTH_ORDER)}


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def classify_region(lat):
    if lat < 30.71044:
        return 'Florida'
    elif lat < 35.76417:
        return 'Georgia to Cape Hatteras'
    else:
        return 'North of Cape Hatteras'


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2)**2
    return 2 * R * np.arcsin(np.sqrt(a))


def deduplicate_group(group, threshold_km=50):
    """Drop observations within threshold_km of an already-kept point (same-day spatial collapse)."""
    lats = group['Latitude'].values
    lons = group['Longitude'].values
    keep = []
    for i in range(len(lats)):
        too_close = any(haversine_km(lats[i], lons[i], lats[k], lons[k]) < threshold_km for k in keep)
        if not too_close:
            keep.append(i)
    return group.iloc[keep]



def east_coast_bbox(df):
    """Filter to US East Coast bounding box and remove Bahamas."""
    df = df[
        (df['Latitude']  >= 24.7) & (df['Latitude']  <= 42.7) &
        (df['Longitude'] >= -81.5) & (df['Longitude'] <= -69)
    ]
    df = df[~((df['Latitude'] < 27.3) & (df['Longitude'] > -79.5))]
    return df


# ---------------------------------------------------------------------------
# Figure: side-by-side map — simulated vs observed stranding points
# ---------------------------------------------------------------------------

def plot_inat_vs_sim(beached_sim_csv, obs_csv, output_file):
    # --- Simulated beachings (single run, no date filter needed) ---
    sim_df = pd.read_csv(beached_sim_csv)
    sim_df = east_coast_bbox(sim_df)
    sim_df['region'] = sim_df['Latitude'].apply(classify_region)

    # --- Observed strandings: filter to 2022-23 season  ---
    obs_df = pd.read_csv(obs_csv)
    obs_df['observed_on'] = pd.to_datetime(obs_df['observed_on'], errors='coerce')
    obs_df = obs_df[
        (obs_df['observed_on'] >= datetime(2022, 11, 1)) &
        (obs_df['observed_on'] <  datetime(2023, 11, 1))
    ]
    obs_df['region'] = obs_df['Latitude'].apply(classify_region)

    # --- Plot ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 8),
                              subplot_kw={'projection': ccrs.PlateCarree()})
    titles = ['Simulated', 'Observed']
    dfs    = [sim_df, obs_df]

    for i, (df, ax, title) in enumerate(zip(dfs, axes, titles)):
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax.add_feature(cfeature.BORDERS,   linestyle=':', linewidth=0.5)
        ax.set_extent([-82, -65, 24, 45], crs=ccrs.PlateCarree())

        for region, color in REGION_COLORS.items():
            sub = df[df['region'] == region]
            ax.scatter(sub['Longitude'], sub['Latitude'],
                       color=color, s=20, alpha=1,
                       transform=ccrs.PlateCarree(), zorder=5, label=region)

        gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
        gl.top_labels   = False
        gl.right_labels = (i == 1)
        gl.left_labels  = (i == 0)
        gl.xlocator     = mticker.MultipleLocator(5)
        gl.ylocator     = mticker.MultipleLocator(5)
        gl.xlabel_style = {'weight': 'bold', 'size': 14}
        gl.ylabel_style = {'weight': 'bold', 'size': 14}
        ax.set_title(title, fontsize=18, fontweight='bold')

        handles = [Patch(color=c, label=r) for r, c in REGION_COLORS.items()]
        leg = ax.legend(handles=handles, title='Region',
                        loc='lower right', fontsize=12, title_fontsize=14)
        leg.get_title().set_fontweight('bold')

    plt.tight_layout()
    fig.savefig(output_file, dpi=600, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {os.path.basename(output_file)}")


# ---------------------------------------------------------------------------
# Figure C: simulated stranding boxplots (one panel per region)
# ---------------------------------------------------------------------------

def plot_figC_sim_boxplots(sim_summary_csv, output_file):
    sim_df = pd.read_csv(sim_summary_csv)
    sim_df = sim_df[sim_df['Filename'] != 'Average across all simulations'].copy()
    sim_df['month_num'] = pd.to_datetime(sim_df['Month']).dt.month
    sim_df['x_pos']     = sim_df['month_num'].map(MONTH_TO_XPOS)

    fig, axes = plt.subplots(3, 1, figsize=(10, 11), sharex=True)
    plt.subplots_adjust(hspace=0.12)

    for ax, region in zip(axes, REGIONS):
        color = REGION_COLORS[region]
        rdf   = sim_df[sim_df['Region'] == region]

        box_data = [rdf[rdf['x_pos'] == xp]['Beached_Count'].values for xp in range(12)]
        ax.boxplot(
            box_data, positions=range(12), widths=0.55,
            patch_artist=True, showfliers=False,
            medianprops=dict(color='black',  linewidth=2),
            whiskerprops=dict(color='black', linewidth=1.2),
            capprops=dict(color='black',     linewidth=1.2),
            boxprops=dict(facecolor=color, alpha=0.35, edgecolor='black', linewidth=1.2),
        )

        for xp in range(12):
            vals = rdf[rdf['x_pos'] == xp]['Beached_Count'].values
            ax.scatter([xp] * len(vals), vals,
                       color=color, s=18, zorder=4, alpha=0.85, linewidths=0)

        total_per_sim = rdf.groupby('Filename')['Beached_Count'].sum()
        avg_n = total_per_sim.mean()
        sd_n  = total_per_sim.std()
        ax.text(0.985, 0.96, f'Avg N={avg_n:.1f}±{sd_n:.1f}',
                transform=ax.transAxes, ha='right', va='top',
                fontsize=13, fontweight='bold')

        ax.set_ylabel(f'{region}\nSimulated Stranding Events',
                      fontsize=12, fontweight='bold', labelpad=6)
        ax.tick_params(axis='y', labelsize=16)
        ax.set_xlim(-0.7, 11.7)

    axes[-1].set_xticks(range(12))
    axes[-1].set_xticklabels(MONTH_LABELS, fontsize=16)
    axes[-1].set_xlabel('Month', fontsize=18, fontweight='bold')

    fig.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {os.path.basename(output_file)}')


# ---------------------------------------------------------------------------
# Figure D: observed stranding bar charts (one panel per region)
# ---------------------------------------------------------------------------

def plot_figD_obs_bars(obs_csv, output_file):
    obs_df = pd.read_csv(obs_csv)
    obs_df = obs_df.rename(columns={'latitude': 'Latitude', 'longitude': 'Longitude'})
    obs_df['observed_on'] = pd.to_datetime(obs_df['observed_on'], format='mixed')

    # Filter to 2022-23 cohort 
    obs_df = obs_df[
        (obs_df['observed_on'] >= '2022-11-01') &
        (obs_df['observed_on'] <= '2023-10-31')
    ].copy()

    # Uncomment to collapse stranding events within 50 km on the same day
    # obs_df = pd.concat(
    #     [deduplicate_group(g) for _, g in obs_df.groupby(obs_df['observed_on'].dt.date)]
    # ).reset_index(drop=True)
    print(f'  Observations (2022-23 cohort): {len(obs_df)}')

    obs_df['Region']    = obs_df['Latitude'].apply(classify_region)
    obs_df['month_num'] = obs_df['observed_on'].dt.month

    fig, axes = plt.subplots(3, 1, figsize=(9, 11), sharex=True)
    plt.subplots_adjust(hspace=0.12)

    for ax, region in zip(axes, REGIONS):
        color   = REGION_COLORS[region]
        rdf     = obs_df[obs_df['Region'] == region]
        counts  = [len(rdf[rdf['month_num'] == m]) for m in MONTH_ORDER]
        n_total = len(rdf)

        ax.bar(range(12), counts, color=color,
               edgecolor='black', linewidth=0.6, width=0.7)
        ax.text(0.985, 0.96, f'N={n_total}',
                transform=ax.transAxes, ha='right', va='top',
                fontsize=13, fontweight='bold')

        ax.set_ylabel(f'{region}\nObserved Stranding Events',
                      fontsize=12, fontweight='bold', labelpad=6)
        ax.set_xlim(-0.7, 11.7)
        ax.tick_params(axis='y', labelsize=16)

        if region == 'Georgia to Cape Hatteras':
            ax.yaxis.set_major_locator(mticker.MultipleLocator(4))

    axes[-1].set_xticks(range(12))
    axes[-1].set_xticklabels(MONTH_LABELS, fontsize=16)
    axes[-1].set_xlabel('Month', fontsize=18, fontweight='bold')

    fig.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {os.path.basename(output_file)}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(output_dir, exist_ok=True)

    print('=== iNat vs Simulation map ===')
    plot_inat_vs_sim(
        beached_sim_csv, obs_csv,
        output_file=os.path.join(output_dir, 'fig_inat_vs_sim_map.png'))

    print('=== Fig C: simulated boxplots ===')
    plot_figC_sim_boxplots(
        sim_summary_csv,
        output_file=os.path.join(output_dir, 'figC_simulated_boxplots.png'))

    print('=== Fig D: observed bar charts ===')
    plot_figD_obs_bars(
        obs_csv,
        output_file=os.path.join(output_dir, 'figD_observed_bars.png'))


if __name__ == '__main__':
    main()
