from scipy.stats import gaussian_kde
import numpy as np
import pandas as pd

# Load effort data and construct KDE with DayOfYear as circular data
def load_effort_data_kde(effort_csvs):

    combined_data = []
    
    # Combine all species data into a single array of (lon, lat, DayOfYear_sin, DayOfYear_cos)
    for csv_file in effort_csvs:

        df = pd.read_csv(csv_file)
        
        df['DayOfYear'] = pd.to_datetime(df['observed_on']).dt.dayofyear
        
        # Convert 'DayOfYear' to sine and cosine components to represent circular variable
        df['DayOfYear_sin'] = np.sin(2 * np.pi * df['DayOfYear'] / 365)
        df['DayOfYear_cos'] = np.cos(2 * np.pi * df['DayOfYear'] / 365)
        
        # Combine lon, lat, DayOfYear_sin, and DayOfYear_cos as 4D data points
        combined_data.append(np.vstack([df['longitude'], df['latitude'], df['DayOfYear_sin'], df['DayOfYear_cos']]))
    
    # Combine data from all species into one large array
    combined_data = np.hstack(combined_data)
    
    # Create a 4D KDE based on longitude, latitude, DayOfYear_sin, and DayOfYear_cos
    kde = gaussian_kde(combined_data)
    
    return kde

# Evaluate the effort KDE for a given location and time, using the sine and cosine of DayOfYear
def evaluate_effort_kde(kde, lon, lat, day_of_year):

    # Convert the day_of_year to sine and cosine components for circular KDE
    day_sin = np.sin(2 * np.pi * day_of_year / 365)
    day_cos = np.cos(2 * np.pi * day_of_year / 365)
    
    # Evaluate the KDE at the given lon, lat, day_sin, and day_cos
    effort_density = kde.evaluate([lon, lat, day_sin, day_cos])[0]

    return effort_density

# Assign each observation a weight based on iNat effort in that area
def normalize_juveniles_with_kde(juvenile_df, kde, scale_factor=1):

    normalized_values = []
    
    # Iterate over each juvenile observation
    for _, row in juvenile_df.iterrows():
        lon = row['longitude']
        lat = row['latitude']
        day_of_year = row['DayOfYear']
        
        # Evaluate the KDE at the observation's location and time
        effort_density = evaluate_effort_kde(kde, lon, lat, day_of_year)

        # If the effort density is too small, avoid division by zero
        if effort_density > 0:
            normalized_value = (1 / effort_density) * scale_factor
        else:
            normalized_value = 0

        # Cap normalization weight at 1000 to mitigate outliers
        if normalized_value > 1000:
            normalized_value = 1000

        normalized_values.append(normalized_value)
    
    # Add the normalized weights to the DataFrame
    juvenile_df['normalized_weights'] = normalized_values

    return juvenile_df

# Save the normalized data to CSV
def save_to_csv(normalized_juveniles, output_file):

    # Save the normalized values to a new CSV file
    normalized_juveniles.to_csv(output_file, index=False)

if __name__ == "__main__":

    # Load the classified juvenile data
    file_path = '/path/to/juvenile_observations.csv'
    juvenile_df = pd.read_csv(file_path)
    juvenile_df['observed_on'] = pd.to_datetime(juvenile_df['observed_on'])
    juvenile_df['DayOfYear'] = juvenile_df['observed_on'].dt.dayofyear
    
    # Load and create the effort KDE, treating DayOfYear as circular data
    effort_csvs = [
        '/path/to/hermit_crab_all.csv',
        '/path/to/sand_dollar_all.csv',
        '/path/to/sea_stars_all.csv'
    ]
    effort_kde = load_effort_data_kde(effort_csvs)
    
    # Add normalization weights to the juvenile observations based on effort KDE
    normalized_juveniles = normalize_juveniles_with_kde(juvenile_df, effort_kde)
    
    # Save the normalized data to a new CSV
    save_to_csv(normalized_juveniles, '/path/to/normalized_juvenile_observations.csv')
    
