import os
import re
import csv
import xarray as xr
import numpy as np
import pandas as pd
from geopy.distance import geodesic

print("SCRIPT STARTED", flush=True)

def extract_run_number(filename):
    """Extract the run number from a filename using regex."""
    match = re.search(r"run_(\d+)", filename)
    return int(match.group(1)) if match else None

def find_tracked_particles(zarr_file, min_longitude, interval_days=15, total_days=365):
    """
    Find particles that never beached and track them at specific time intervals.
    """
    print("Opening Zarr...", flush=True)
    ds = xr.open_zarr(zarr_file, consolidated=False)
    print("Zarr Opened...", flush=True)
    
    # Identify particles that never beached
    never_beached = ~(ds.beached == 1).any(dim="obs")
    
    # Get final position (second-to-last timestep to avoid NaNs)
    final_lon = ds.lon.isel(obs=-2).compute()
    final_lat = ds.lat.isel(obs=-2).compute()

    # Filter particles that end east of the threshold
    valid_particles = never_beached & (final_lon >= min_longitude)
    
    # Select trajectory IDs of non-beached particles
    trajectory_ids = ds.trajectory.where(valid_particles, drop=True).values

    # Define time step indices for tracking (since 1 obs = 1 day, we use direct indexing)
    time_indices = list(range(0, total_days, interval_days))  # Every 15 days

    # Extract particle positions at each time step
    tracked_positions = {
        traj: {
            day: (ds.lon.sel(trajectory=traj).isel(obs=day).compute().item(),
                  ds.lat.sel(trajectory=traj).isel(obs=day).compute().item())
            for day in time_indices if day < len(ds.obs)
        }
        for traj in trajectory_ids
    }

    return tracked_positions, time_indices

def compute_average_distance(wind_zarr, no_wind_zarr, min_longitude, interval_days=15, total_days=365):
    """
    Compute the average distance between matching particles in wind & no-wind simulations
    at multiple time intervals. Returns a dictionary of averages per run.
    """
    wind_positions, time_indices = find_tracked_particles(wind_zarr, min_longitude, interval_days, total_days)
    no_wind_positions, _ = find_tracked_particles(no_wind_zarr, min_longitude, interval_days, total_days)

    avg_distances = {}  # Stores the final average for this run
    per_run_distances = {}  # Fix: Ensure this remains a dictionary

    for day in time_indices:
        distances = []
        common_trajectories = set(wind_positions.keys()).intersection(set(no_wind_positions.keys()))

        for traj in common_trajectories:
            if day in wind_positions[traj] and day in no_wind_positions[traj]:
                wind_coord = wind_positions[traj][day]
                no_wind_coord = no_wind_positions[traj][day]
                dist_km = geodesic(wind_coord[::-1], no_wind_coord[::-1]).km  # Reverse to (lat, lon)
                distances.append(dist_km)

        avg_distance = np.mean(distances) if distances else float('nan')
        avg_distances[day] = avg_distance
        per_run_distances[day] = avg_distance  # Fix: Ensure per_run_distances is a dictionary

    return avg_distances, per_run_distances  # Now correctly returns a dictionary


    return avg_distances, per_run_distances  # Now only returns run-level averages

def process_all_runs(wind_dir, no_wind_dir, min_longitude, output_csv, detailed_output_csv, interval_days=15, total_days=365):
    """
    Process all .zarr runs and compute distance differences at multiple time intervals.
    Outputs both an average distance CSV and a per-run averaged distance CSV.
    """
    print("Listing files...", flush=True)
    wind_files = {extract_run_number(f): f for f in os.listdir(wind_dir) if f.endswith(".zarr")}
    no_wind_files = {extract_run_number(f): f for f in os.listdir(no_wind_dir) if f.endswith(".zarr")}
    print("Files listed...", flush=True)
    common_runs = sorted(set(wind_files.keys()) & set(no_wind_files.keys()))

    overall_distances = {day: [] for day in range(0, total_days, interval_days)}
    per_run_data = []  # Store per-run average distances

    for run in common_runs:
        wind_path = os.path.join(wind_dir, wind_files[run])
        no_wind_path = os.path.join(no_wind_dir, no_wind_files[run])

        print(f"\nProcessing: {wind_files[run]} vs. {no_wind_files[run]}...", flush=True)
        avg_distances, per_run_distances = compute_average_distance(wind_path, no_wind_path, min_longitude, interval_days, total_days)

        for day, distance in avg_distances.items():
            overall_distances[day].append(distance)

        # Store per-run average distances (only one per time step per run)
        for day, avg_distance in per_run_distances.items():
            per_run_data.append([run, day, avg_distance])

    # Compute overall averages for each time interval
    final_distances = {day: np.nanmean(overall_distances[day]) for day in overall_distances}
    print(final_distances, flush=True)

    # Save overall averages to CSV
    df = pd.DataFrame(list(final_distances.items()), columns=["Day", "Average_Distance_km"])
    df.to_csv(output_csv, index=False)
    print(f"\nSaved results to {output_csv}", flush=True)

    # Save per-run average distances to CSV
    df_per_run = pd.DataFrame(per_run_data, columns=["Run", "Day", "Average_Distance_km"])
    df_per_run.to_csv(detailed_output_csv, index=False)
    print(f"\nSaved per-run averages to {detailed_output_csv}", flush=True)

# Example usage
wind_dir = "/gpfs/gibbs/project/dunn/rba27/Buffer_50_Control"
no_wind_dir = "/gpfs/gibbs/project/dunn/rba27/Buffer_50_No_Wind"
output_csv = "control_distances_average.csv"
detailed_output_csv = "control_distances_detailed.csv"

print("Processing all runs...", flush=True)
process_all_runs(wind_dir, no_wind_dir, -81, output_csv, detailed_output_csv)
