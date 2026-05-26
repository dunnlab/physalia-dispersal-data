import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patches as mpatches
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# ---------------------------------------------------------------------------
# Paths — update these before running
# ---------------------------------------------------------------------------
obs_file   = 'path/to/juvenile_observations.csv'  # juvenile iNaturalist observation CSV
output_dir = 'path/to/output_figures'             # directory where figures are saved
# ---------------------------------------------------------------------------

# Cohorts to include in the heatmap grid
COHORTS = ["2019-20", "2020-21", "2021-22", "2022-23", "2023-24"]

# Month display order (Nov–Oct, following the Nov 1 cohort cutoff)
MONTH_ORDER  = [11, 12, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
MONTH_LABELS = ["Nov", "Dec", "Jan", "Feb", "Mar", "Apr",
                "May", "Jun", "Jul", "Aug", "Sep", "Oct"]


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def assign_region(row):
    if row['latitude'] >= 31.0:
        return 'East Coast North of Florida'
    elif row['longitude'] >= -79:
        return 'Caribbean'
    elif 26.0 <= row['latitude'] <= 30.5 and -100 <= row['longitude'] <= -81.6:
        return 'Gulf of Mexico Coast'
    elif row['latitude'] <= 24.0:
        return 'Gulf of Mexico Coast'
    else:
        return 'East Florida Coast & Keys'


REGION_COLORS = {
    'East Florida Coast & Keys':    '#3260ff',
    'Gulf of Mexico Coast':         'mediumpurple',
    'East Coast North of Florida':  'coral',
    'Caribbean':                    'seagreen',
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(output_dir, exist_ok=True)

    # -----------------------------------------------------------------------
    # Load & prepare data
    # -----------------------------------------------------------------------
    df = pd.read_csv(obs_file)
    df['observed_on'] = pd.to_datetime(df['observed_on'], format='mixed')
    df['month'] = df['observed_on'].dt.month
    df['Year']  = df['observed_on'].dt.year

    # Cohort assignment: Nov 1 cutoff
    df['CohortStart'] = df.apply(
        lambda r: r['Year'] if r['month'] >= 11 else r['Year'] - 1, axis=1
    )
    df['CohortLabel'] = df['CohortStart'].apply(lambda y: f"{y}-{str(y + 1)[-2:]}")

    # Region assignment
    df['region'] = df.apply(assign_region, axis=1)

    # -----------------------------------------------------------------------
    # Figure 2: heatmap grid — proportion of annual juvenile obs per month/cohort
    # -----------------------------------------------------------------------
    grid_df = df[df['CohortLabel'].isin(COHORTS)]
    grid = np.zeros((len(COHORTS), len(MONTH_ORDER)))

    for i, cohort in enumerate(COHORTS):
        cohort_df = grid_df[grid_df['CohortLabel'] == cohort]
        total = len(cohort_df)
        if total == 0:
            continue
        for j, month in enumerate(MONTH_ORDER):
            grid[i, j] = len(cohort_df[cohort_df['month'] == month]) / total

    fig1, ax1 = plt.subplots(figsize=(18, 10))
    im = ax1.imshow(grid, aspect='auto', cmap='viridis', vmin=0, vmax=1, origin='upper')
    ax1.set_yticks(range(len(COHORTS)))
    ax1.set_yticklabels(COHORTS, fontsize=24)
    ax1.set_xticks(range(len(MONTH_ORDER)))
    ax1.set_xticklabels(MONTH_LABELS, fontsize=32)
    ax1.set_xlabel('Month', fontsize=34, fontweight='bold')
    ax1.set_ylabel('Year',  fontsize=34, fontweight='bold')
    cb1 = fig1.colorbar(im, ax=ax1, fraction=0.035, pad=0.02)
    cb1.set_label('Proportion of Annual Juvenile Observations', fontsize=24, fontweight='bold')
    cb1.ax.tick_params(labelsize=24)
    plt.tight_layout()
    fig1.savefig(os.path.join(output_dir, 'fig2_juvenile_grid.png'), dpi=600, bbox_inches='tight')
    plt.close(fig1)
    print('Saved fig2_juvenile_grid.png')

    # -----------------------------------------------------------------------
    # Figure 2: map — juvenile observation locations colored by region
    # -----------------------------------------------------------------------
    fig2, ax2 = plt.subplots(figsize=(12, 10),
                              subplot_kw={'projection': ccrs.PlateCarree()})
    ax2.set_extent([-98.5, -62, 17.5, 40], crs=ccrs.PlateCarree())
    ax2.add_feature(cfeature.LAND,      facecolor='white', edgecolor='black', linewidth=0.5)
    ax2.add_feature(cfeature.OCEAN,     facecolor='white')
    ax2.add_feature(cfeature.COASTLINE, edgecolor='black', linewidth=0.8)
    ax2.add_feature(cfeature.BORDERS,   edgecolor='black', linewidth=0.5, linestyle=':')
    ax2.add_feature(cfeature.LAKES,     facecolor='white', edgecolor='black', linewidth=0.4)

    for region, color in REGION_COLORS.items():
        subset = df[df['region'] == region]
        ax2.scatter(subset['longitude'], subset['latitude'],
                    color=color, s=50, alpha=1.0, linewidths=0,
                    transform=ccrs.PlateCarree(), zorder=5)

    legend_handles = [mpatches.Patch(color=c, label=r) for r, c in REGION_COLORS.items()]
    leg2 = ax2.legend(handles=legend_handles, title='Region',
                      title_fontsize=16, fontsize=14,
                      loc='upper right', framealpha=0.9)
    leg2.get_title().set_fontweight('bold')

    gl = ax2.gridlines(draw_labels=True, linewidth=0.5, color='gray',
                       alpha=0.7, linestyle='--', crs=ccrs.PlateCarree())
    gl.xlocator     = plt.FixedLocator(range(-95, -60, 5))
    gl.ylocator     = plt.FixedLocator(range(20, 42, 5))
    gl.top_labels   = False
    gl.right_labels = False
    gl.xlabel_style = {'size': 22, 'weight': 'bold'}
    gl.ylabel_style = {'size': 22, 'weight': 'bold'}

    plt.tight_layout()
    fig2.savefig(os.path.join(output_dir, 'fig2_juvenile_map.png'), dpi=600, bbox_inches='tight')
    plt.close(fig2)
    print('Saved fig2_juvenile_map.png')

    # -----------------------------------------------------------------------
    # Figure 2: stacked bar histogram — juvenile obs per month per region
    # -----------------------------------------------------------------------
    df['month_cat'] = pd.Categorical(df['month'], categories=MONTH_ORDER, ordered=True)
    counts = df.groupby(['month_cat', 'region']).size().unstack(fill_value=0)
    counts = counts.reindex(MONTH_ORDER).reset_index(drop=True)

    fig3, ax3 = plt.subplots(figsize=(12, 8))
    bottom = np.zeros(len(MONTH_ORDER))

    for region, color in REGION_COLORS.items():
        values = counts[region].values if region in counts.columns else np.zeros(len(MONTH_ORDER))
        ax3.bar(range(len(MONTH_ORDER)), values, bottom=bottom,
                color=color, label=region, alpha=0.8)
        bottom += values

    for idx in range(len(MONTH_ORDER)):
        ax3.text(idx, bottom[idx] + 0.5, str(int(bottom[idx])),
                 ha='center', va='bottom', fontsize=18)

    ax3.set_xticks(range(len(MONTH_ORDER)))
    ax3.set_xticklabels(MONTH_LABELS, fontsize=20)
    ax3.set_xlabel('Month',                  fontsize=22, fontweight='bold')
    ax3.set_ylabel('Juvenile Observations',  fontsize=20, fontweight='bold')
    ax3.tick_params(axis='y', labelsize=20)
    ax3.tick_params(axis='x', labelsize=20)
    ax3.set_ylim(0, bottom.max() * 1.2)
    leg3 = ax3.legend(title='Region', fontsize=16, title_fontsize=18)
    leg3.get_title().set_fontweight('bold')

    plt.tight_layout()
    fig3.savefig(os.path.join(output_dir, 'fig3_hist.png'), dpi=600, bbox_inches='tight')
    plt.close(fig3)
    print('Saved fig2_hist.png')


if __name__ == '__main__':
    main()
