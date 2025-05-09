"""
data_utils.py

This module provides utility functions to handle and process traffic data collected
from the San Francisco Municipal Transportation Agency (SFMTA). It supports the
following operations:

1. read_sf_traffic_data: Reading and cleaning raw traffic CSV data files.
2. convert_sf_traffic_csv_format: Converting traffic CSV files from one format to another with specific formatting.
3. extract_sf_traffic_timeslot: Extracting a time-based subset of vehicle data into a new CSV file.
4. read_tnc_stats_data: Reading TNC pickup/dropoff data and filtering it based on a specified time window.
"""


import pandas as pd
from datetime import datetime
import os
from constants.data_constants import SF_TRAFFIC_SFMTA_FOLDER_PATH
import csv
from collections import defaultdict


def read_sf_traffic_data(file_path: str) -> pd.DataFrame:
    """
    Reads and processes raw San Francisco traffic data from a CSV file.

    This function:
    - Parses the data with semicolon delimiter.
    - Converts coordinates and speed to float.
    - Filters out vehicles that never moved (i.e., always speed == 0).
    - Computes relative time from the first timestamp in seconds.

    Parameters:
    ----------
    - file_path : str
        Path to the input CSV file.

    Returns:
    -------
    pandas.DataFrame
        Cleaned DataFrame with the following columns:
        ['timestamp', 'vehicle_id', 'longitude', 'latitude', 'heading', 'speed', 'relative_time']
    """

    df = pd.read_csv(file_path, sep=";")
    df.columns = ["timestamp", "vehicle_id", "longitude", "latitude", "heading", "speed"]
    df['longitude'] = df['longitude'].astype(str).str.replace(',', '.').astype(float)
    df['latitude'] = df['latitude'].astype(str).str.replace(',', '.').astype(float)
    df['speed'] = df['speed'].astype(float)
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Filter out vehicles that have speed == 0 for all their records
    valid_vehicle_ids = df.groupby("vehicle_id")["speed"].max()
    valid_vehicle_ids = valid_vehicle_ids[valid_vehicle_ids > 0].index
    df = df[df["vehicle_id"].isin(valid_vehicle_ids)]

    # Compute relative time
    df['relative_time'] = (df['timestamp'] - df['timestamp'].min()).dt.total_seconds()
    return df


def convert_sf_traffic_csv_format(
        input_csv_path: str,
          output_csv_path: str
          ) -> None:
    """
    Converts a San Francisco traffic CSV from original format to a localized format.

    This function:
    - Renames columns to a standard set.
    - Converts timestamps to ISO 8601 format.
    - Converts decimal separators from '.' to ',' for European locale.
    - Writes the reformatted data to a new file with semicolon delimiter.

    Parameters:
    ----------
    - input_csv_path : str
        Path to the original CSV file using comma as decimal separator.
    - output_csv_path : str
        Path where the reformatted CSV will be saved.

    Returns:
    -------
    None
    """
    # Read original CSV with comma delimiter
    df = pd.read_csv(input_csv_path)

    # Rename columns to match target format
    df.columns = ["timestamp", "vehicle_id", "longitude", "latitude", "heading", "avg_speed"]
    # Convert timestamp to ISO format
    df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.strftime("%Y-%m-%dT%H:%M:%S.000")

    # Convert floats to strings with comma as decimal separator (e.g., -122,401108)
    df["longitude"] = df["longitude"].apply(lambda x: f"{x:.6f}".replace('.', ','))
    df["latitude"] = df["latitude"].apply(lambda x: f"{x:.6f}".replace('.', ','))

    # Write to new file with semicolon delimiter
    df.to_csv(output_csv_path, sep=";", index=False)
    print(f"Formatted CSV saved to {output_csv_path}")


def extract_sf_traffic_timeslot(
        input_csv_path: str, 
        date_str: str, 
        start_time_str: str,
        end_time_str: str, 
        output_csv_folder: str
        ) -> str:
    """
    Extracts a time slot of traffic data for a specific date and time range,
    and saves the filtered data in a structured folder format.

    This function:
    - Filters the dataset to include only records within a given time window on a specific date.
    - Saves the filtered data to a CSV file with semicolon delimiter.
    - Organizes the output into a folder named after the date inside the specified output folder.
    - The file is named using the format: sf_vehicle_{YYMMDD}_{HHHH}.csv.
    - Overwrites the file if it already exists.

    Parameters:
    ----------
    - input_csv_path : str
        Path to the input CSV file with semicolon delimiter.
    - date_str : str
        Date in 'YYYY-MM-DD' format (e.g., '2025-03-25').
    - start_time_str : str
        Start time in 'HH:MM' format (e.g., '08:00').
    - end_time_str : str
        End time in 'HH:MM' format (e.g., '10:00').
    - output_csv_folder : str
        Path to the root output folder where the dated folder and file will be saved.

    Returns:
    -------
    str
        Full path to the saved CSV file containing the filtered data.
    """
    # Read dataset
    df = pd.read_csv(input_csv_path, sep=";")

    # Convert timestamp column to datetime
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Build datetime range from input
    start_dt = datetime.strptime(f"{date_str} {start_time_str}", "%Y-%m-%d %H:%M")
    end_dt = datetime.strptime(f"{date_str} {end_time_str}", "%Y-%m-%d %H:%M")

    # Filter the DataFrame to the time slot
    filtered_df = df[(df["timestamp"] >= start_dt) & (df["timestamp"] < end_dt)]

    # Format date and hour parts for filename
    date_part = datetime.strptime(date_str, "%Y-%m-%d").strftime("%y%m%d")
    start_hour = datetime.strptime(start_time_str, "%H:%M").strftime("%H")
    end_hour = datetime.strptime(end_time_str, "%H:%M").strftime("%H")
    timeslot_part = f"{start_hour}{end_hour}"

    # Create folder path for the specific date
    date_folder_path = os.path.join(output_csv_folder, date_str)
    os.makedirs(date_folder_path, exist_ok=True)  # Will not recreate if already exists

    # Final file path
    filename = f"sf_vehicle_{date_part}_{timeslot_part}.csv"
    output_csv_path = os.path.join(date_folder_path, filename)

    # Save file, overwrite if exists
    filtered_df.to_csv(output_csv_path, sep=";", index=False)
    print(f"✅ Filtered traffic data saved to {output_csv_path}")

    return output_csv_path


def read_tnc_stats_data(
        sf_rides_stats_path: str, 
        start_time_str: str, 
        end_time_str: str
        ) -> dict:
    """
    Reads TNC hourly pickup/dropoff data from a CSV file and filters it based on a specified time window.

    This function:
    - Reads TNC hourly pickup/dropoff data from a CSV file.
    - Filters the data based on the specified time window.
    - Computes total pickups and dropoffs across all zones and selected hours.
    - Returns a nested dictionary with TAZ as keys and hour data as values.

    Parameters:
    ----------
    - sf_rides_stats_path (str): 
        Path to the CSV file with columns: 'taz', 'day_of_week', 'hour', 'pickups', 'dropoffs'.
    - start_time_str (str):
        Start time (HH:MM format).
    - end_time_str (str): 
        End time (HH:MM format). Can wrap around midnight.

    Returns:
    -------
    dict
        Nested dictionary {taz: {hour: {'pickups': x, 'dropoffs': y}}} where `hour` is in standard 0-23 format.
    """
    def parse_hour(time_str):
        # Convert time string (HH:MM) to hour in standard 0-23 format
        return int(datetime.strptime(time_str, "%H:%M").hour)
    
    # Parse start and end times
    start_hour = parse_hour(start_time_str)
    end_hour = parse_hour(end_time_str)
    if start_hour < end_hour:
        selected_hours_std = list(range(start_hour, end_hour))
    else:
        selected_hours_std = list(range(start_hour, 24)) + list(range(0, end_hour))

    # Create a mapping of dataset hours to standard hours
    dataset_hour_map = {h: h % 24 for h in range(3, 27)}
    selected_dataset_hours = {h: std_hour for h, std_hour in dataset_hour_map.items() if std_hour in selected_hours_std}

    # Initialize a dictionary to hold the data
    zone_data = {}

    # Read the CSV file
    with open(sf_rides_stats_path, mode='r') as file:
        reader = csv.DictReader(file, delimiter=',')
        for row in reader:
            # Only process data for Monday (day_of_week == 0)
            # TODO: Extend to other days
            if int(row['day_of_week']) == 0:
                taz = int(row['taz'])
                dataset_hour = int(row['hour'])
                if dataset_hour in selected_dataset_hours:
                    std_hour = selected_dataset_hours[dataset_hour]
                    if taz not in zone_data:
                        zone_data[taz] = {}
                    # Round pickups and dropoffs to the nearest integer
                    row['pickups'] = round(float(row['pickups']))
                    row['dropoffs'] = round(float(row['dropoffs']))
                    zone_data[taz][std_hour] = {
                        'pickups': int(row['pickups']),
                        'dropoffs': int(row['dropoffs'])
                    }
    print(f"✅ TNC stats data read from {sf_rides_stats_path} and filtered to time window {start_time_str} - {end_time_str}")

    # Compute pickups and dropoffs across all zones and selected hours
    total_pickups = sum(hour_data['pickups'] for zone in zone_data.values() for hour_data in zone.values())
    total_dropoffs = sum(hour_data['dropoffs'] for zone in zone_data.values() for hour_data in zone.values())
    print(f"Total pickups:  {total_pickups}")
    print(f"Total dropoffs: {total_dropoffs}")

    return zone_data