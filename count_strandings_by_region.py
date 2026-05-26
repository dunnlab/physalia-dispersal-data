import os
import pandas as pd
from datetime import datetime, timedelta

def classify_region(lat):
    """Classify region based on latitude."""
    if lat < 30.71044:
        return "Florida"
    elif lat < 35.76417:
        return "Georgia to Cape Hatteras"
    else:
        return "North of Cape Hatteras"

def count_beached_points(directory, output_csv):
    all_data = []
    
    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            file_path = os.path.join(directory, filename)
            try:
                df = pd.read_csv(file_path)

                # Filter for Area B
                df = df[
                    (df["Latitude"] >= 24.7) & (df["Latitude"] <= 42.7) &
                    (df["Longitude"] >= -81.5) & (df["Longitude"] <= -69)
                ]

                # Remove points in the Bahamas
                df = df[
                    ~((df["Latitude"] < 27.3) & (df["Longitude"] > -79.5))
                ]

                # Convert 'time' column from seconds since Nov 1, 2022 to datetime
                if 'time' in df.columns:
                    df['datetime'] = pd.to_datetime(df['time'])

                    # Extract month-year format
                    df['month'] = df['datetime'].dt.strftime('%Y-%m')

                    # Assign regions
                    df["Region"] = df["Latitude"].apply(classify_region)

                    # Count number of beachings per region per month
                    monthly_counts = df.groupby(["Region", "month"]).size().reset_index(name="Beached_Count")

                    # Store data for CSV output
                    for _, row in monthly_counts.iterrows():
                        all_data.append({
                            'Filename': filename,
                            'Region': row['Region'],
                            'Month': row['month'],
                            'Beached_Count': row['Beached_Count']
                        })

            except Exception as e:
                print(f"Error reading {filename}: {e}")

    # Convert to DataFrame
    df_output = pd.DataFrame(all_data)

    # Calculate the average beaching count per region per month across all simulations
    avg_counts = df_output.groupby(["Region", "Month"])["Beached_Count"].mean().reset_index()
    avg_counts["Filename"] = "Average across all simulations"

    # Append the average data to the original DataFrame
    df_output = pd.concat([df_output, avg_counts], ignore_index=True)

    # Save to CSV
    df_output.to_csv(output_csv, index=False)
    print(f"Output saved to {output_csv}")

if __name__ == "__main__":
    directory_path = "path/to/simulation_output"       # directory containing per-run stranding CSVs
    output_csv_path = "path/to/regional_stranding_summary.csv"  # where to write the summary
    count_beached_points(directory_path, output_csv_path)

