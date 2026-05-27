import numpy as np
import pandas as pd
import rasterio
from scipy.ndimage import distance_transform_edt

# ---------------------------------------------------------------------------
# Paths — update these before running
# ---------------------------------------------------------------------------
input_csv      = 'path/to/inat_pphysalis_obs_areaa_all.csv'  # iNaturalist export (must have Latitude, Longitude columns)
land_raster    = 'path/to/landmask.tif'              # raster land mask (non-zero = land)
output_csv     = 'path/to/inat_pphysalis_obs_usec_2017-2024.csv'                # filtered output
# ---------------------------------------------------------------------------

df = pd.read_csv(input_csv)

if 'latitude' in df.columns:
    df = df.rename(columns={'latitude': 'Latitude', 'longitude': 'Longitude'})

df['observed_on'] = pd.to_datetime(df['observed_on'], errors='coerce')
df = df[df['observed_on'] >= '2017-11-01'].reset_index(drop=True)
print(f"After date filter:      {len(df)} points")

with rasterio.open(land_raster) as src:
    transform      = src.transform
    land_data      = src.read(1)
    pixel_size_deg = abs(transform.e)
    buffer_deg     = 10 / 111.0

    # Filter 1: keep points on land or within 10 km of land (removes far-offshore)
    dist_to_land = distance_transform_edt(land_data == 0, sampling=[pixel_size_deg, abs(transform.a)])
    filter1      = (land_data != 0) | (dist_to_land <= buffer_deg)

    coords   = list(zip(df['Longitude'], df['Latitude']))
    row_col  = [src.index(x, y) for x, y in coords]
    in_filter1 = np.array([
        filter1[r, c] if 0 <= r < land_data.shape[0] and 0 <= c < land_data.shape[1] else False
        for r, c in row_col
    ])

df = df[in_filter1].reset_index(drop=True)
print(f"After filter 1 (land): {len(df)} points")

with rasterio.open(land_raster) as src:
    transform      = src.transform
    land_data      = src.read(1)
    pixel_size_deg = abs(transform.e)
    buffer_deg     = 10 / 111.0

    # Filter 2: keep points in water or within 10 km of water (removes far-inland)
    dist_to_water = distance_transform_edt(land_data != 0, sampling=[pixel_size_deg, abs(transform.a)])
    filter2       = (land_data == 0) | (dist_to_water <= buffer_deg)

    coords   = list(zip(df['Longitude'], df['Latitude']))
    row_col  = [src.index(x, y) for x, y in coords]
    in_filter2 = np.array([
        filter2[r, c] if 0 <= r < land_data.shape[0] and 0 <= c < land_data.shape[1] else False
        for r, c in row_col
    ])

df = df[in_filter2].reset_index(drop=True)
print(f"After filter 2 (water): {len(df)} points")

# US East Coast bounding box
df = df[
    (df['Latitude']  >= 24.7) & (df['Latitude']  <= 42.7) &
    (df['Longitude'] >= -81.5) & (df['Longitude'] <= -69)
]

# Remove Bahamas
df = df[~((df['Latitude'] < 27.3) & (df['Longitude'] > -79.5))]

print(f"After bbox filter:      {len(df)} points")

df.to_csv(output_csv, index=False)
print(f"Saved to {output_csv}")
