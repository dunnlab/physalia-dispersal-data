import pandas as pd
import numpy as np
import os
from scipy.stats import gaussian_kde, wasserstein_distance
import rasterio
from datetime import datetime
from pyemd import emd

def normalize_distribution(data, lon_min, lon_max, lat_min, lat_max, bins):
    """Convert spatiotemporal points into a normalized 4D density grid."""
    hist, edges = np.histogramdd(
        data,
        bins=bins,
        range=[
            [lon_min, lon_max],  # Longitude range
            [lat_min, lat_max],  # Latitude range
            [time_min, time_max] 
        ]
    )
    density = hist / np.sum(hist)
    return density.flatten()  # Flatten into a 1D array

def load_inat_data(inat_file, land_tif):
    """Load iNaturalist data, filter for the year 2023, and determine its bounds."""
    start_date = datetime(2023, 11, 1)
    # Load the data with date parsing
    data = pd.read_csv(inat_file, parse_dates=['observed_on'])
    
    # Filter for records from the year 2023
    data = data[data['observed_on'].dt.year == 2023]
    #data = filter_points_on_land(data, land_tif)

    data['time'] = (data['observed_on'] - reference_start_time).dt.total_seconds()
    data_inat = data[['Longitude', 'Latitude', 'time']].values
    print("iNat time:" data_inat['time'])
    
    # Determine the bounds
    lon_min, lon_max = data_inat[:, 0].min(), data_inat[:, 0].max()
    lat_min, lat_max = data_inat[:, 1].min(), data_inat[:, 1].max()
    lat_min = 30.7
    
    return data_inat, lon_min, lon_max, lat_min, lat_max

def filter_within_bounds(data, lon_min, lon_max, lat_min, lat_max):
    """Filter data within the specified longitude and latitude bounds."""
    return data[
        (data[:, 0] >= lon_min) & (data[:, 0] <= lon_max) & 
        (data[:, 1] >= lat_min) & (data[:, 1] <= lat_max)
    ]

def compute_emd(data_inat, simulated_data, lon_min, lon_max, lat_min, lat_max):
    """Compute the EMD between normalized density grids of two datasets."""
    # Normalize both distributions
    time_min, time_max = data_inat[:, 2].min(), data_inat[:, 2].max()
    
    bins = [50, 50, 20, 20]  # Adjust the number of bins if necessary
    inat_density = normalize_distribution(data_inat, lon_min, lon_max, lat_min, lat_max, time_min, time_max, bins)
    simulated_density = normalize_distribution(simulated_data, lon_min, lon_max, lat_min, lat_max, time_min, time_max, bins)
    print(f"iNat Data Sample:" data_inat[:10])
    print(f"Simulated Data Sample" simulated_data[:10])
    print(f"Total iNat Points:" len(data_inat))
    print(f"Total Simulated Points:" len(simulated_data))

    # Calculate the EMD
    return wasserstein_distance(inat_density, simulated_density)

def process_simulation_result(response, run_number, data_inat, lon_min, lon_max, lat_min, lat_max, output_dir):
    """Process each simulation result and calculate the EMD."""
    file_path = f"{output_dir}/trial_random_response_{response}_run_{run_number}.csv"
    simulated_data = pd.read_csv(file_path)[['Longitude', 'Latitude', 'time']].values
    print("Sim Data:" simulated_data['time'])
    
    # Filter the data within bounds
    filtered_data = filter_within_bounds(simulated_data, lon_min, lon_max, lat_min, lat_max)

    # Calculate the EMD with normalization
    emd = compute_emd(data_inat, filtered_data, lon_min, lon_max, lat_min, lat_max)
    return response, emd



def filter_points_on_land(data, tif_path):
    """Filter rows of the DataFrame that fall on land based on a raster (.tif) mask."""
    land_indices = []
    with rasterio.open(tif_path) as src:
        for i, row in data.iterrows():
            lon, lat = row['Longitude'], row['Latitude']
            try:
                # Sample the raster at the given point
                point_sample = list(src.sample([(lon, lat)]))[0][0]
                if point_sample != src.nodata:  # Check if the point is on land
                    land_indices.append(i)  # Save the index of valid points
            except Exception as e:
                print(f"Skipping point ({lon}, {lat}): {e}")
    
    print(f"Points on land: {len(land_indices)} / {len(data)}")
    return data.loc[land_indices]
 

def calculate_emd_for_all(output_dir, inat_file, land_tif, response_range=(0.01, 0.11)):
    """Calculate EMDs for all simulations and aggregate the results."""
    # Load iNaturalist data
    data_inat, lon_min, lon_max, lat_min, lat_max = load_inat_data(inat_file, land_tif)

    # Prepare response values
    response_values = np.arange(*response_range, 0.02)

    # Collect EMD results
    for run_number in range(5):
        emd_results = []
        for response in response_values:
            response = str(int(response * 1000)).zfill(4)
          # Adjust if you have multiple runs per response
            try:
                result = process_simulation_result(
                    response, run_number, data_inat, lon_min, lon_max, 
                    lat_min, lat_max, output_dir
                )
                emd_results.append(result)
            except Exception as e:
                print(f"Error processing response {response}, run {run_number}: {e}")
                
            sweep_dir = os.path.join(output_dir, "Sweep_CSVs")
            os.makedirs(sweep_dir, exist_ok=True)
            sweep_file = os.path.join(sweep_dir, f"temporal_trial_{run_number}.csv")
            sweep_df = pd.DataFrame(emd_results, columns=["Response", "EMD"])
            sweep_df.to_csv(sweep_file, index=False)

    # Aggregate EMDs by response
    emd_df = pd.DataFrame(emd_results, columns=["Response", "EMD"])
    avg_emd_df = emd_df.groupby("Response").mean().reset_index()

    # Save the averaged EMDs to a CSV file
    output_file = os.path.join(output_dir, "temporal_avg_emd_trial.csv")
    avg_emd_df.to_csv(output_file, index=False)
    print(f"EMD results saved to {output_file}")

    return avg_emd_df

# Main execution
if __name__ == "__main__":
    output_dir = "/gpfs/gibbs/project/dunn/rba27/Parameter_Sweep_Normalized"
    inat_file = "/gpfs/gibbs/project/dunn/rba27/final_inat_ec_obs.csv"
    land_tif = "/gpfs/gibbs/project/dunn/rba27/america_raster2.tif"
    response_range = (0.01, 0.05)  # Adjust as needed

    # Calculate EMDs and print the resulting DataFrame
    avg_emd_df = calculate_emd_for_all(output_dir, inat_file, land_tif, response_range)
    print(avg_emd_df)