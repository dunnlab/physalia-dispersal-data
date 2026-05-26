import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.stats import chi2

# ---------------------------------------------------------------------------
# Paths — update these before running
# ---------------------------------------------------------------------------
obs_file   = 'path/to/inat_physalia_obs_usec_2017-2024.csv'  # pre-filtered iNaturalist observation CSV
output_dir = 'path/to/output_figures'              # directory where figures are saved
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def classify_region(lat):
    if lat < 30.71044:
        return "Florida"
    elif lat < 35.76417:
        return "Georgia to Cape Hatteras"
    else:
        return "North of Cape Hatteras"


def rayleigh_test(angles):
    """Rayleigh test for circular uniformity (Zar 1999 p-value approximation)."""
    n = len(angles)
    C = np.mean(np.cos(angles))
    S = np.mean(np.sin(angles))
    R_bar = np.sqrt(C**2 + S**2)
    Z = n * R_bar**2
    p = np.exp(-Z) * (1 + (2*Z - Z**2) / (4*n) - (24*Z - 132*Z**2 + 76*Z**3 - 9*Z**4) / (288*n**2))
    mean_angle = np.arctan2(S, C) % (2 * np.pi)
    return Z, p, mean_angle, R_bar


def angle_to_date(angle):
    """Convert a circular angle (radians, Nov-1 origin) to a month-day string."""
    return (pd.Timestamp("2000-11-01") + pd.Timedelta(days=int(round(angle / (2 * np.pi) * 365)))).strftime("%b %d")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(output_dir, exist_ok=True)

    # -----------------------------------------------------------------------
    # Load & prepare data
    # -----------------------------------------------------------------------
    df = pd.read_csv(obs_file)
    df = df.rename(columns={"latitude": "Latitude", "longitude": "Longitude"})
    df["observed_on"] = pd.to_datetime(df["observed_on"], format="mixed")
    df["Month"] = df["observed_on"].dt.month
    df["Year"]  = df["observed_on"].dt.year

    # Cohort assignment: Nov 1 cutoff (e.g. Oct 2018 → cohort 2017-18)
    def get_cohort_start(row):
        return row["Year"] if row["Month"] >= 11 else row["Year"] - 1
    df["CohortStart"] = df.apply(get_cohort_start, axis=1)
    df["CohortLabel"]  = df["CohortStart"].apply(lambda y: f"{y}-{str(y + 1)[-2:]}")

    # Day-within-cohort angle for circular statistics
    def day_in_cohort(date, cohort_start):
        return (date - pd.Timestamp(cohort_start, 11, 1)).days
    df["DayInCohort"] = df.apply(lambda r: day_in_cohort(r["observed_on"], r["CohortStart"]), axis=1)
    df["DayAngle"]    = 2 * np.pi * df["DayInCohort"] / 365

    df["Region"] = df["Latitude"].apply(classify_region)

    # -----------------------------------------------------------------------
    # Shared plot settings
    # -----------------------------------------------------------------------
    cohorts      = [f"{y}-{str(y + 1)[-2:]}" for y in range(2017, 2024)]
    month_order  = [11, 12, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    month_labels = ["Nov", "Dec", "Jan", "Feb", "Mar", "Apr", "May",
                    "Jun", "Jul", "Aug", "Sep", "Oct"]
    regions      = ["Florida", "Georgia to Cape Hatteras", "North of Cape Hatteras"]

    # -----------------------------------------------------------------------
    # Rayleigh tests (results printed to stdout) + build heatmap grids
    # -----------------------------------------------------------------------
    grids = []
    for region in regions:
        region_df = df[df["Region"] == region]
        grid = np.zeros((len(cohorts), len(month_order)))

        print(f"\n{'='*60}\n{region}\n{'='*60}")

        # (1) Per-cohort Rayleigh — within-year concentration and peak timing
        r_bars, p_vals, peak_angles = [], [], []
        for i, cohort in enumerate(cohorts):
            cohort_df     = region_df[region_df["CohortLabel"] == cohort]
            cohort_angles = cohort_df["DayAngle"].values
            total = len(cohort_df)
            if total == 0:
                continue
            for j, month in enumerate(month_order):
                grid[i, j] = len(cohort_df[cohort_df["Month"] == month]) / total
            _, p_c, mean_ang, r = rayleigh_test(cohort_angles)
            r_bars.append(r)
            p_vals.append(p_c)
            peak_angles.append(mean_ang)
            print(f"  {cohort}  R̄ = {r:.3f},  p = {p_c:.4f},  peak ≈ {angle_to_date(mean_ang)}")

        fisher_stat = -2 * np.sum(np.log(p_vals))
        fisher_p    = 1 - chi2.cdf(fisher_stat, df=2 * len(p_vals))
        print(f"  Mean R̄ = {np.mean(r_bars):.3f}  (SD = {np.std(r_bars):.3f}),  "
              f"mean p = {np.mean(p_vals):.4f},  Fisher combined p = {fisher_p:.4f}")

        # (2) Rayleigh on annual peak angles — consistency of peak timing across years
        _, p_peaks, mean_peak_angle, R_bar_peaks = rayleigh_test(np.array(peak_angles))
        print(f"  Rayleigh of annual peaks:  R̄ = {R_bar_peaks:.3f},  "
              f"consensus peak ≈ {angle_to_date(mean_peak_angle)},  p = {p_peaks:.4f}")

        # (3) Pooled Rayleigh — overall concentration and peak across all observations
        all_angles = region_df["DayAngle"].values
        _, p_ray, mean_angle_all, R_bar_all = rayleigh_test(all_angles)
        print(f"  Pooled Rayleigh:  R̄ = {R_bar_all:.3f},  "
              f"peak ≈ {angle_to_date(mean_angle_all)},  p = {p_ray:.4f}")

        grids.append(grid)

    # -----------------------------------------------------------------------
    # Figure 3 — heatmap: proportion of annual observations per month per cohort
    # -----------------------------------------------------------------------
    fig3, axes3 = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    for ax, region, grid in zip(axes3, regions, grids):
        ax.imshow(grid, aspect="auto", cmap="viridis", vmin=0, vmax=1, origin="upper")
        ax.set_yticks(range(len(cohorts)))
        ax.set_yticklabels(cohorts, fontsize=18)
        ax.set_title(region, fontsize=26, fontweight="bold")
    axes3[-1].set_xticks(range(len(month_order)))
    axes3[-1].set_xticklabels(month_labels, fontsize=22)
    axes3[-1].set_xlabel("Month", fontsize=30, fontweight="bold")
    fig3.text(-0.02, 0.5, "Year", va="center", ha="center",
              rotation="vertical", fontsize=30, fontweight="bold")
    plt.tight_layout()
    fig3.savefig(os.path.join(output_dir, "fig3_heatmap.png"), dpi=600, bbox_inches="tight")
    plt.close(fig3)
    print("\nSaved fig3_heatmap.png")

    # Standalone colorbar for Figure 3 heatmap
    fig_cb, ax_cb = plt.subplots(figsize=(1.2, 5))
    norm = mpl.colors.Normalize(vmin=0, vmax=1)
    cb   = mpl.colorbar.ColorbarBase(ax_cb, cmap="viridis", norm=norm, orientation="vertical")
    cb.set_label("Proportion of Annual Observations", fontsize=13, fontweight="bold")
    cb.ax.tick_params(labelsize=12)
    fig_cb.tight_layout()
    fig_cb.savefig(os.path.join(output_dir, "fig3_colorbar.png"), dpi=600, bbox_inches="tight")
    plt.close(fig_cb)
    print("Saved fig3_colorbar.png")

    # -----------------------------------------------------------------------
    # Figure 3 - line plot — raw counts per cohort per region
    # -----------------------------------------------------------------------
    region_colors = {
        "Florida":                    "#3260ff",
        "Georgia to Cape Hatteras":   "#de2eff",
        "North of Cape Hatteras":     "#ff0d0d",
    }
    markers = ["o", "s", "^", "D", "v", "P", "X"]

    fig3, axes3 = plt.subplots(3, 1, figsize=(13, 12), sharex=True)
    for ax, region in zip(axes3, regions):
        region_df = df[df["Region"] == region]
        color = region_colors[region]
        for i, cohort in enumerate(cohorts):
            cohort_df = region_df[region_df["CohortLabel"] == cohort]
            counts = [len(cohort_df[cohort_df["Month"] == m]) for m in month_order]
            ax.plot(range(12), counts, linestyle="--", marker=markers[i], color=color,
                    label=cohort, linewidth=1.5, markersize=7)
        ax.set_title(region, fontsize=30, fontweight="bold")
        ax.tick_params(axis="y", labelsize=24)
        if region == "Florida":
            ymax = ax.get_ylim()[1]
            ax.set_yticks(np.arange(0, ymax + 40, 40))

    axes3[-1].set_xticks(range(12))
    axes3[-1].set_xticklabels(month_labels, fontsize=28)
    axes3[-1].set_xlabel("Month", fontsize=30, fontweight="bold")

    handles, labels = axes3[0].get_legend_handles_labels()
    black_handles = [plt.Line2D([0], [0], marker=h.get_marker(), color="black",
                                linestyle="--", markersize=12, linewidth=1.5)
                     for h in handles]
    leg = fig3.legend(black_handles, labels, title="Years", title_fontsize=24,
                      prop={"size": 23}, loc="center right", bbox_to_anchor=(1.22, 0.5))
    leg.get_title().set_fontweight("bold")
    plt.tight_layout()
    fig3.text(-0.005, 0.5, "Number of Stranding Observations", va="center", ha="center",
              rotation="vertical", fontsize=30, fontweight="bold")
    fig3.savefig(os.path.join(output_dir, "fig3_line_plot.png"), dpi=600, bbox_inches="tight")
    plt.close(fig3)
    print("Saved fig3.png")


if __name__ == "__main__":
    main()
