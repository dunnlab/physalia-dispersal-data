import os
import numpy as np
import pandas as pd
from datetime import datetime
from scipy.stats import spearmanr

# ---------------------------------------------------------------------------
# Paths — update before running
# ---------------------------------------------------------------------------
output_dir = "path/to/simulation_runs"  # directory where simulation CSVs and results are saved
inat_file  = "path/to/inat_pphysalis_obs_usec_2017-2024.csv" # pre-filtered iNaturalist observation CSV
# ---------------------------------------------------------------------------

REFERENCE_START = datetime(2022, 11, 1)
LON_MIN, LON_MAX = -81, -66
LAT_MIN, LAT_MAX = 24, 45
BINS = [1, 8, 4]


def filter_within_bounds(df, lon_min, lon_max, lat_min, lat_max):
    return df[
        (df['Longitude'] >= lon_min) & (df['Longitude'] <= lon_max) &
        (df['Latitude']  >= lat_min) & (df['Latitude']  <= lat_max)
    ]


def remove_bahamas(df):
    return df[~((df['Latitude'] < 27.61255) & (df['Longitude'] > -79.7578756))]


def to_density(data, bins):
    """Bin spatiotemporal points and return a normalized flat histogram."""
    hist, _ = np.histogramdd(
        data,
        bins=bins,
        range=[[LON_MIN, LON_MAX], [LAT_MIN, LAT_MAX], [0, 365]],
    )
    total = hist.sum()
    return (hist / total).flatten() if total > 0 else hist.flatten()


def load_inat(inat_file):
    df = pd.read_csv(inat_file, parse_dates=['observed_on'])
    df = df.rename(columns={'latitude': 'Latitude', 'longitude': 'Longitude'})
    df = df[df['observed_on'].dt.year.isin([2022, 2023])]
    df['time'] = (df['observed_on'] - REFERENCE_START).dt.total_seconds() / 86400
    df = df[(df['time'] >= 0) & (df['time'] <= 365)]
    return df[['Longitude', 'Latitude', 'time']].values


def load_sim(run_number, output_dir):
    path = f"{output_dir}/stranded_run_{run_number}.csv"
    df = pd.read_csv(path, parse_dates=['time'])[['Longitude', 'Latitude', 'time']]
    df['time'] = (df['time'] - REFERENCE_START).dt.total_seconds() / 86400
    df = filter_within_bounds(df, LON_MIN, LON_MAX, LAT_MIN, LAT_MAX)
    df = remove_bahamas(df)
    return df[['Longitude', 'Latitude', 'time']].values


def run_all(output_dir, inat_file):
    print("Loading iNat data...")
    data_inat = load_inat(inat_file)
    print(f"  {len(data_inat)} iNat points after filtering")

    hist_inat = to_density(data_inat, BINS)

    results = []
    for run in range(25):
        print(f"Processing run {run}...")
        sim_data = load_sim(run, output_dir)
        if sim_data.size == 0:
            print(f"  No data for run {run}, skipping.")
            results.append({"Run": run, "Spearman Rho": None, "P-Value": None})
            continue

        hist_sim = to_density(sim_data, BINS)
        corr, pvalue = spearmanr(hist_inat, hist_sim)
        print(f"  Rho={corr:.4f}  p={pvalue:.4g}")
        results.append({"Run": run, "Spearman Rho": corr, "P-Value": pvalue})

    df_out = pd.DataFrame(results)

    valid_rhos = df_out["Spearman Rho"].dropna()
    mean_rho = np.tanh(np.mean(np.arctanh(valid_rhos)))
    std_rho  = np.std(valid_rhos)
    print(f"\nMean Rho (Fisher z-transformed): {mean_rho:.4f}")
    print(f"SD Rho (raw):                    {std_rho:.4f}")

    out_csv = os.path.join(output_dir, "spearman_results.csv")
    df_out.to_csv(out_csv, index=False)
    print(f"Results saved to {out_csv}")
    return df_out


if __name__ == "__main__":
    run_all(output_dir, inat_file)
