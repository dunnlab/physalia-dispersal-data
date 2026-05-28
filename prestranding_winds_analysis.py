#!/usr/bin/env python3
"""
stranding_winds_extract_and_average.py
Extracts the 20 wind steps preceding each stranding from simulation zarrs,
then computes regional mean wind statistics for 1, 5, 10, and 20-day windows.
"""

import os
import numpy as np
import pandas as pd
import zarr

# ---------------------------------------------------------------------------
# Paths 
# ---------------------------------------------------------------------------
INPUT_DIR    = "/path/to/simulation_runs"
WINDS_CSV    = "/output/stranding_winds.csv"       # raw extraction output
AVERAGES_CSV = "/output/stranding_wind_averages.csv"  # summary output
# ---------------------------------------------------------------------------

N_PREV  = 20
WINDOWS = [1, 5, 10, 20]
REGIONS = ["Florida", "Georgia to Cape Hatteras", "North of Cape Hatteras"]


def classify_region(lat):
    if lat < 30.71044:
        return "Florida"
    elif lat < 35.76417:
        return "Georgia to Cape Hatteras"
    else:
        return "North of Cape Hatteras"


# ---------------------------------------------------------------------------
# Step 1: Extract winds preceding each stranding from zarr files
# ---------------------------------------------------------------------------

def extract_stranding_winds(input_dir, n_prev=N_PREV):
    all_rows = []

    for run in range(25):
        fname = f"run_{run}_output.zarr"
        fpath = os.path.join(input_dir, fname)

        if not os.path.exists(fpath):
            print(f"  WARNING: {fname} not found, skipping")
            continue

        print(f"Processing run {run} …")
        store        = zarr.open(fpath, mode="r")
        lat          = store["lat"][:]
        lon          = store["lon"][:]
        beached      = store["beached"][:]
        u_wind       = store["u_wind"][:]
        v_wind       = store["v_wind"][:]
        handedness   = store["handedness"][:]
        resp_to_wind = store["response_to_wind"][:]

        n_particles, n_obs = lat.shape

        ever_beached    = np.any(beached == 1, axis=1)
        first_beach_obs = np.where(ever_beached,
                                   np.argmax(beached == 1, axis=1),
                                   -1).astype(int)

        beach_lat = lat[np.arange(n_particles), np.where(ever_beached, first_beach_obs, 0)]
        beach_lon = lon[np.arange(n_particles), np.where(ever_beached, first_beach_obs, 0)]

        excluded = ever_beached & (
            ((beach_lat < 27.3) & (beach_lon > -79.5)) |
            (beach_lat > 43.0)                         |
            (beach_lat < 24.0)                         |
            (beach_lon > -66)                          |
            (beach_lon < -81.55459)
        )

        valid_strand = ever_beached & ~excluded
        print(f"  {valid_strand.sum()} stranded particles after exclusions")

        for i in np.where(valid_strand)[0]:
            t = first_beach_obs[i]

            row = {
                "Run":              run,
                "Particle_ID":      int(i),
                "Strand_Lat":       lat[i, t],
                "Strand_Lon":       lon[i, t],
                "Strand_Obs":       t,
                "Handedness":       handedness[i, 0],
                "Response_to_Wind": resp_to_wind[i, 0],
            }

            for k in range(n_prev):
                src_idx = (t - n_prev) + k
                if src_idx < 0 or src_idx >= n_obs:
                    row[f"u_wind_t-{n_prev - k}"] = np.nan
                    row[f"v_wind_t-{n_prev - k}"] = np.nan
                else:
                    row[f"u_wind_t-{n_prev - k}"] = u_wind[i, src_idx]
                    row[f"v_wind_t-{n_prev - k}"] = v_wind[i, src_idx]

            all_rows.append(row)

        print(f"  Run {run} done")

    return pd.DataFrame(all_rows)


# ---------------------------------------------------------------------------
# Step 2: Compute regional wind averages per window
# ---------------------------------------------------------------------------

def compute_wind_averages(df, windows=WINDOWS, regions=REGIONS):
    df = df.copy()
    df["Region"] = df["Strand_Lat"].apply(classify_region)

    rows = []
    for region in regions:
        sub = df[df["Region"] == region].copy()

        for n in windows:
            u_cols = [f"u_wind_t-{k}" for k in range(1, n + 1)]
            v_cols = [f"v_wind_t-{k}" for k in range(1, n + 1)]

            u_mean = sub[u_cols].mean(axis=1)
            v_mean = sub[v_cols].mean(axis=1)

            has_nan  = sub[u_cols + v_cols].isna().any(axis=1)
            all_zero = (u_mean == 0) & (v_mean == 0)
            valid    = ~has_nan & ~all_zero

            n_valid = int(valid.sum())
            u_valid = u_mean[valid]
            v_valid = v_mean[valid]

            mag      = np.sqrt(u_valid**2 + v_valid**2)
            dirn_met = (270 - np.degrees(np.arctan2(v_valid, u_valid))) % 360

            mean_u = float(u_valid.mean())
            mean_v = float(v_valid.mean())
            sd_u   = float(u_valid.std(ddof=1))
            sd_v   = float(v_valid.std(ddof=1))
            sem_u  = sd_u / np.sqrt(n_valid)
            sem_v  = sd_v / np.sqrt(n_valid)

            mean_mag = float(mag.mean())
            sd_mag   = float(mag.std(ddof=1))
            sem_mag  = sd_mag / np.sqrt(n_valid)

            sin_vals = np.sin(np.radians(dirn_met))
            cos_vals = np.cos(np.radians(dirn_met))
            R        = np.sqrt(sin_vals.mean()**2 + cos_vals.mean()**2)
            mean_dir = float(np.degrees(np.arctan2(sin_vals.mean(), cos_vals.mean())) % 360)
            circ_sd  = float(np.degrees(np.sqrt(-2 * np.log(R))))
            circ_sem = circ_sd / np.sqrt(n_valid)

            rows.append({
                "Region":                region,
                "Window_days":           n,
                "N_particles":           n_valid,
                "Mean_U":                round(mean_u,   4),
                "Mean_V":                round(mean_v,   4),
                "SD_U":                  round(sd_u,     4),
                "SD_V":                  round(sd_v,     4),
                "SEM_U":                 round(sem_u,    4),
                "SEM_V":                 round(sem_v,    4),
                "Mean_Mag":              round(mean_mag, 4),
                "SD_Mag":                round(sd_mag,   4),
                "SEM_Mag":               round(sem_mag,  4),
                "Mean_Dir_met":          round(mean_dir, 2),
                "Mean_Resultant_Length": round(float(R), 4),
                "CircSD_Dir":            round(circ_sd,  2),
                "CircSEM_Dir":           round(circ_sem, 2),
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

print("=== Step 1: Extracting stranding winds ===")
winds_df = extract_stranding_winds(INPUT_DIR)
winds_df.to_csv(WINDS_CSV, index=False)
print(f"Saved raw extraction → {WINDS_CSV}")
print(f"  {len(winds_df)} rows × {len(winds_df.columns)} columns")

print("\n=== Step 2: Computing regional averages ===")
averages_df = compute_wind_averages(winds_df)
pd.set_option("display.width", 120)
print(averages_df.to_string(index=False))
averages_df.to_csv(AVERAGES_CSV, index=False)
print(f"\nSaved averages → {AVERAGES_CSV}")
