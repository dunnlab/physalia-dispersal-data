import pandas as pd
import numpy as np
import os
import rasterio
from datetime import datetime, timedelta
from pyemd import emd
import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature

def visualize_spatial_bins_with_map(data, lon_min, lon_max, lat_min, lat_max, bins):
    """Visualize spatial bins with a map background."""
    # Create a 2D histogram for the bins
    hist, x_edges, y_edges = np.histogram2d(data[:, 0], data[:, 1], bins=bins[:2], 
                                            range=[[lon_min, lon_max], [lat_min, lat_max]])

    # Set up the map
    fig = plt.figure(figsize=(12, 10))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([lon_min, lon_max, lat_min, lat_max])

    # Add features to the map
    ax.add_feature(cfeature.LAND, edgecolor='black', zorder=1)
    ax.add_feature(cfeature.COASTLINE, zorder=2)
    ax.add_feature(cfeature.BORDERS, linestyle=':', zorder=2)
    ax.add_feature(cfeature.OCEAN, zorder=0, alpha=0.3)
    ax.add_feature(cfeature.LAKES, zorder=2, alpha=0.3)
    ax.add_feature(cfeature.RIVERS, zorder=2, alpha=0.3)

    # Plot the bins as a heatmap
    im = ax.imshow(hist.T, origin='lower', aspect='auto', extent=[lon_min, lon_max, lat_min, lat_max], 
                   cmap='viridis', alpha=0.7, zorder=3)

    # Add a colorbar
    cbar = plt.colorbar(im, ax=ax, orientation='vertical', shrink=0.7, pad=0.05)
    cbar.set_label('Bin Count')

    # Add labels
    ax.set_title('Spatial Binning Wind Response = 1.8%')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    
    output_file = "/gpfs/gibbs/project/dunn/rba27/Parameter_Sweep_Normalized/bins_vis_sim.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Figure saved to {output_file}")

def normalize_distribution(data, lon_min, lon_max, lat_min, lat_max, time_min, time_max, bins):
    """
    Convert spatiotemporal points into a normalized 3D density grid.
    """
    hist, edges = np.histogramdd(
        data,
        bins=bins,
        range=[
            [lon_min, lon_max],  # Longitude range
            [lat_min, lat_max],  # Latitude range
            [time_min, time_max],  # Time range
        ]
    )
    print(hist)
    density = hist / np.sum(hist)  # Normalize to sum to 1
    return density.flatten(), hist.shape, edges  # Flatten for EMD and return additional metadata

def filter_within_bounds(data, lon_min, lon_max, lat_min, lat_max):
    """Filter data within the specified longitude and latitude bounds."""
    filtered_data = data[
        (data['Longitude'] >= lon_min) & (data['Longitude'] <= lon_max) & 
        (data['Latitude'] >= lat_min) & (data['Latitude'] <= lat_max)
    ]
    return filtered_data

def compute_cost_matrix(bin_edges):
    """
    Compute the cost matrix based on Euclidean distances between bin centers in 3D space.
    """
    # Compute bin centers for longitude, latitude, and time
    bin_centers = [0.5 * (edges[:-1] + edges[1:]) for edges in bin_edges]
    grid = np.array(np.meshgrid(*bin_centers, indexing="ij"))
    bin_positions = grid.reshape(3, -1).T  # Reshape to list of bin centers

    # Compute cost matrix (pairwise Euclidean distances)
    cost_matrix = np.linalg.norm(bin_positions[:, None, :] - bin_positions[None, :, :], axis=2)
    return cost_matrix


def load_inat_data(inat_file, land_tif):
    """
    Load iNaturalist data, filter for the year 2023, and determine its bounds.
    """
    # Reference start time for simulation
    reference_start_time = datetime(2022, 11, 1)

    # Load the data with date parsing
    data = pd.read_csv(inat_file, parse_dates=['observed_on'])
    print(len(data))
    # Filter for records from the year 2023
    data = data[data['observed_on'].dt.year.isin([2022, 2023])]
    print(len(data))
    # Convert observed times to seconds since the reference start time
    #print("Before conversion:" data['time'][0])
    data['time'] = (data['observed_on'] - reference_start_time).dt.total_seconds()
    data = data[data['time'] >= 0]
    data['time'] = data['time'] / (60 * 60 * 24)
    #print("After conversion:" data['time'][0])
    # Filter points on land (if applicable)
    #data = filter_points_on_land(data, land_tif)
    data = data[data['time'] <= 365]
    # Extract relevant columns
    data_inat = data[['Longitude', 'Latitude', 'time']]

    # Determine bounds for spatial and temporal data
    lon_min, lon_max = data_inat['Longitude'].min(), data_inat['Longitude'].max()
    lat_min, lat_max = data_inat['Latitude'].min(), data_inat['Latitude'].max()
    time_min, time_max = data_inat['time'].min(), data_inat['time'].max()
    lon_max = -66
    lon_min = -81
    lat_max = 40
    lat_min = 24
    print("time_min_inat:", time_min)
    print("time_max_inat:", time_max)
    print("lon_min_inat:", lon_min)
    print("lon_max_inat:", lon_max)
    print("lat_min_inat:", lat_min)
    print("lat_max_inat:", lat_max)
    data_inat = filter_within_bounds(data_inat, lon_min, lon_max, lat_min, lat_max)
    
    data_inat = data_inat[~(
        (data_inat['Latitude'] < 27.61255) & 
        (data_inat['Longitude'] > -79.7578756)
    )]
     
    data_inat = data_inat[['Longitude', 'Latitude', 'time']].values
    print(len(data_inat))
    return data_inat, lon_min, lon_max, lat_min, lat_max, time_min, time_max


def compute_3d_emd(data_inat, simulated_data, bins, lon_min, lon_max, lat_min, lat_max, time_min, time_max):
    """
    Compute 3D Earth Mover's Distance (EMD) for spatiotemporal data.
    """

    print("iNat Data Sample:", data_inat[:10])
    print("Simulated Data Sample", simulated_data[:10])
    print("Total iNat Points:", len(data_inat))
    print("Total Simulated Points:", len(simulated_data))
    # Normalize distributions into histograms
    hist1_flat, _, bin_edges = normalize_distribution(data_inat, lon_min, lon_max, lat_min, lat_max, time_min, time_max, bins)
    hist2_flat, _, _ = normalize_distribution(simulated_data, lon_min, lon_max, lat_min, lat_max, time_min, time_max, bins)
    print("Non-empty bins in iNat:", np.count_nonzero(hist1_flat))
    print("Non-empty bins in Simulated:", np.count_nonzero(hist2_flat))

    # Compute the cost matrix
    cost_matrix = compute_cost_matrix(bin_edges)
    print("Cost matrix shape:", cost_matrix.shape)
    print("Cost matrix sample:", cost_matrix[:5, :5])
    # Compute the EMD using pyemd
    emd_value = emd(hist1_flat, hist2_flat, cost_matrix)
    emd_max = cost_matrix.max()
    emd_normalized = emd_value/emd_max
    return emd_value, emd_normalized


def process_simulation_result(run_number, data_inat, lon_min, lon_max, lat_min, lat_max, time_min, time_max, output_dir):
    """
    Process each simulation result and calculate the 3D EMD.
    """
    simulation_reference_time = datetime(2022, 11, 1)
    
    file_path = f"{output_dir}/beached_buffer_50_run_{run_number}_10000.csv"
    simulated_data = pd.read_csv(file_path)[['Longitude', 'Latitude', 'time']]
    print("Before conversion:", simulated_data['time'][0])
    simulated_data['time'] = simulated_data['time'].apply(lambda t: simulation_reference_time + timedelta(seconds=t))
    print("After conversion:", simulated_data['time'][0])
    common_reference_date = datetime(2022, 11, 1)  # Use the simulation start date or earliest date in both datasets
    simulated_data['time'] = (simulated_data['time'] - common_reference_date).dt.total_seconds()
    simulated_data['time'] = simulated_data['time'] / (60 * 60 * 24)
    print(simulated_data['time'])
    # Print the minimum and maximum time values
    min_time = simulated_data['time'].min()
    max_time = simulated_data['time'].max()
    
    print(f"Minimum time value in simulation data: {min_time}")
    print(f"Maximum time value in simulation data: {max_time}")

    
    # Filter the data within bounds
    filtered_data = filter_within_bounds(simulated_data, lon_min, lon_max, lat_min, lat_max)
    filtered_data = filtered_data[~(
        (filtered_data['Latitude'] < 27.61255) & 
        (filtered_data['Longitude'] > -79.7578756)
    )]
    filtered_data = filtered_data[['Longitude', 'Latitude', 'time']].values
    # If no valid data remains after filtering
    if filtered_data.size == 0:
        print(f"No data for response {response}, run {run_number}.")
        return response, None

    # Compute 3D EMD
    bins = [1, 5, 4]  # Number of bins for longitude, latitude, and time
    '''
    if file_path == f"{output_dir}/temporal_random_response_0018_run_1.csv":
        visualize_spatial_bins_with_map(filtered_data, lon_min, lon_max, lat_min, lat_max, bins)
    '''
    emd_value, emd_normalized = compute_3d_emd(data_inat, filtered_data, bins, lon_min, lon_max, lat_min, lat_max, time_min, time_max)

    return emd_value, emd_normalized


def filter_points_on_land(data, tif_path):
    """
    Filter rows of the DataFrame that fall on land based on a raster (.tif) mask.
    """
    land_indices = []
    with rasterio.open(tif_path) as src:
        for i, row in data.iterrows():
            lon, lat = row['Longitude'], row['Latitude']
            try:
                point_sample = list(src.sample([(lon, lat)]))[0][0]
                if point_sample != src.nodata:  # Check if the point is on land
                    land_indices.append(i)
            except Exception as e:
                print(f"Skipping point ({lon}, {lat}): {e}")

    print(f"Points on land: {len(land_indices)} / {len(data)}")
    return data.loc[land_indices]


def calculate_emd_for_all(output_dir, inat_file, land_tif):
    """
    Calculate EMDs for all simulations and aggregate the results.
    """
    # Load iNaturalist data
    data_inat, lon_min, lon_max, lat_min, lat_max, time_min, time_max = load_inat_data(inat_file, land_tif)
    bins = [1, 5, 4]
    
    # Prepare response values
    #response_values = [round(value, 3) for value in np.arange(0.01, 0.1, 0.002)]
    all_emd_results = []
    # Collect EMD results
    for run_number in range(25):
        print(f"Processing Run: {run_number}")
        # Process each simulation result
        result, normalized_result = process_simulation_result(
            run_number, data_inat, lon_min, lon_max, lat_min, lat_max, time_min, time_max, output_dir
        )
        # Append the result with the run number
        all_emd_results.append({"Run Number": run_number, "EMD": result, "Normalized EMD": normalized_result})

    # Convert results to a DataFrame
    all_emd_df = pd.DataFrame(all_emd_results)

    # Save all results to a single CSV
    output_file = os.path.join(output_dir, "all_emd_results.csv")
    all_emd_df.to_csv(output_file, index=False)
    print(f"All EMD results saved to {output_file}")

    return all_emd_df


# Main execution
if __name__ == "__main__":
    output_dir = "/gpfs/gibbs/project/dunn/rba27/Buffer_50"
    inat_file = "/gpfs/gibbs/project/dunn/rba27/final_inat_ec_obs.csv
    land_tif = "/gpfs/gibbs/project/dunn/rba27/america_raster2.tif"

    # Calculate EMDs and print the resulting DataFrame
    avg_emd_df = calculate_emd_for_all(output_dir, inat_file, land_tif)
    print(avg_emd_df)

