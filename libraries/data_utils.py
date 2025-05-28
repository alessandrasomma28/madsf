"""
data_utils.py

This module provides utility functions for downloading, processing, and converting San Francisco data.
It includes utilities for:

1. check_import_traffic: Downloading San Francisco traffic data from the SFMTA API.
2. read_sf_traffic_data: Reading and processing the downloaded traffic data into a structured DataFrame.
3. convert_sf_traffic_csv_format: Converting the traffic data CSV format to a localized format.
4. extract_sf_traffic_timeslot: Extracting a specific time slot of traffic data and saving it in a structured folder format.
5. read_tnc_stats_data: Reading TNC hourly pickup/dropoff data and filtering it based on a specified time window.
"""


from pathlib import Path
import time
from datetime import datetime, timedelta
import os
import csv
import itertools
import sys
import requests
from urllib.parse import quote
import certifi
import threading
import pandas as pd
from constants.data_constants import (SF_TRAFFIC_FOLDER_PATH, SF_TRAFFIC_BASE_URL)


def check_import_traffic(
        start_date_str: str,
        end_date_str: str,
        start_time_str: str,
        end_time_str: str,
        limit: int = 10000000000
    ) -> Path:
    """
    Downloads San Francisco traffic data from the SFMTA API and saves it as a CSV file.
    This function:
    - Constructs a query to fetch vehicle position data within a specified time range.
    - Saves the data to a CSV file in a designated folder, with the format: sf_traffic_data_{YYMMDDHH}_{YYMMDDHH}.csv.

    Parameters:
    ----------
    - start_date_str: str
        Start date in 'YYYY-MM-DD' format (e.g., '2021-03-25').
    - end_date_str: str
        End date in 'YYYY-MM-DD' format (e.g., '2021-03-26').
    - start_time_str: str
        Start time in 'HH:MM' format (e.g., '08:00').
    - end_time_str: str
        End time in 'HH:MM' format (e.g., '10:00').
    - limit: int
        Maximum number of records to fetch.

    Returns:
    -------
    Path
        Full path to the saved CSV file containing the traffic data.

    Raises:
    ------
    - Exception
        If the request to the SFMTA API fails or if the response status code is not 200.
    - ValueError
        If no traffic data is found in the specified interval.
    """
    def _spinner(msg="⬇️  Downloading traffic file from SFMTA..."):
        for char in itertools.cycle("|/-\\"):
            if _spinner_done:
                break
            sys.stdout.write(f"\r{msg} {char}")
            sys.stdout.flush()
            time.sleep(0.1)
    global _spinner_done
    _spinner_done = False

    # Build query and URL
    start_datetime = f"{start_date_str}T{start_time_str}:00"
    end_datetime = f"{end_date_str}T{end_time_str}:00"
    query = f"""
        SELECT
          vehicle_position_date_time,
          vehicle_id,
          loc_x,
          loc_y,
          heading,
          average_speed
        WHERE
          vehicle_position_date_time BETWEEN '{start_datetime}' AND '{end_datetime}'
        LIMIT {limit}
    """
    url = f"{SF_TRAFFIC_BASE_URL}?$query={quote(query)}"

    # Create the folders if they don't exist
    if not os.path.exists(SF_TRAFFIC_FOLDER_PATH):
        os.makedirs(SF_TRAFFIC_FOLDER_PATH)
    sf_traffic_folder = os.path.join(SF_TRAFFIC_FOLDER_PATH, "sfmta_dataset")
    if not os.path.exists(sf_traffic_folder):
        os.makedirs(sf_traffic_folder)

    # Build path and check if the file already exists
    start_date = datetime.strptime(start_date_str, "%Y-%m-%d").strftime("%y%m%d")
    end_date = datetime.strptime(end_date_str, "%Y-%m-%d").strftime("%y%m%d")
    start_hour = datetime.strptime(start_time_str, "%H:%M").strftime("%H")
    end_hour = datetime.strptime(end_time_str, "%H:%M").strftime("%H")
    timeslot = f"{start_date}{start_hour}_{end_date}{end_hour}"
    sf_traffic_file_path = os.path.join(
        sf_traffic_folder,
        f"sf_traffic_data_{timeslot}.csv"
    )
    if os.path.exists(sf_traffic_file_path):
        return sf_traffic_file_path
    
    # Loading spinner
    spinner_thread = threading.Thread(target=_spinner)
    spinner_thread.start()

   # Fetch traffic data
    try:
        response = requests.get(url, verify=certifi.where())
    finally:
        _spinner_done = True
        spinner_thread.join()
        sys.stdout.write("\r✅ Download complete!                                  \n")
    if response.status_code != 200:
        raise Exception(f"Failed to fetch data: {response.status_code} - {response.text}")

    # Check number of rows in the response
    lines = response.text.strip().splitlines()
    num_rows = len(lines) - 1  # Exclude header
    if num_rows == 0:
        raise ValueError("❌ No traffic data found in the specified interval (this is an issue of SFMTA data), please try again with a different time slot\n"
        "You can find a list of days with no traffic detection in the SFMTA data in the 'doc/' directory")

    # Write response to CSV file
    with open(sf_traffic_file_path, "w", encoding="utf-8") as f:
        f.write(response.text)
    print(f"✅ CSV saved to: {sf_traffic_file_path}")

    return Path(sf_traffic_file_path)


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
    file_path : str
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
        start_date_str: str,
        end_date_str: str,
        start_time_str: str,
        end_time_str: str, 
        output_csv_folder: str
    ) -> Path:
    """
    Extracts a time slot of traffic data for a specific date and time range,
    and saves the filtered data in a structured folder format.

    This function:
    - Filters the dataset to include only records within a given time window on a specific date.
    - Saves the filtered data to a CSV file with semicolon delimiter.
    - Organizes the output into a folder named after the date inside the specified output folder.
    - The file is named using the format: sf_vehicle_{YYMMDDYYMMDD}_{HHHH}.csv.
    - Overwrites the file if it already exists.

    Parameters:
    ----------
    - input_csv_path : str
        Path to the input CSV file with semicolon delimiter.
    - start_date_str : str
        Start date in 'YYYY-MM-DD' format (e.g., '2021-03-25').
    - end_date_str : str
        End date in 'YYYY-MM-DD' format (e.g., '2021-03-26').
    - start_time_str : str
        Start time in 'HH:MM' format (e.g., '08:00').
    - end_time_str : str
        End time in 'HH:MM' format (e.g., '10:00').
    - output_csv_folder : str
        Path to the root output folder where the dated folder and file will be saved.

    Returns:
    -------
    Path
        Full path to the saved CSV file containing the filtered data.
    """
    # Read dataset
    df = pd.read_csv(input_csv_path)
    column_renames = {
        "vehicle_position_date_time": "timestamp",
        "vehicle_id": "vehicle_id",
        "loc_x": "longitude",
        "loc_y": "latitude",
        "heading": "heading",
        "average_speed": "speed"
    }
    df.rename(columns=column_renames, inplace=True)

    # Convert timestamp column to datetime
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Build datetime range from input
    start_dt = datetime.strptime(f"{start_date_str} {start_time_str}", "%Y-%m-%d %H:%M")
    end_dt = datetime.strptime(f"{end_date_str} {end_time_str}", "%Y-%m-%d %H:%M")

    # Filter the DataFrame to the time slot
    filtered_df = df[(df["timestamp"] >= start_dt) & (df["timestamp"] < end_dt)]

    # Format date and hour for filename
    start_date = datetime.strptime(start_date_str, "%Y-%m-%d").strftime("%y%m%d")
    end_date = datetime.strptime(end_date_str, "%Y-%m-%d").strftime("%y%m%d")
    start_hour = datetime.strptime(start_time_str, "%H:%M").strftime("%H")
    end_hour = datetime.strptime(end_time_str, "%H:%M").strftime("%H")
    timeslot = f"{start_date}{start_hour}_{end_date}{end_hour}"

    # Create folder path for the specific date
    date_folder_path = os.path.join(output_csv_folder, f"{start_date}-{end_date}")
    os.makedirs(date_folder_path, exist_ok=True)  # Will not recreate if already exists

    # Final file path
    filename = f"sf_vehicle_{timeslot}.csv"
    output_csv_path = os.path.join(date_folder_path, filename)

    # Save file, overwrite if exists
    filtered_df.to_csv(output_csv_path, sep=";", index=False)
    print(f"✅ Filtered traffic data saved to {output_csv_path}")

    return output_csv_path


def read_tnc_stats_data(
        sf_rides_stats_path: str, 
        start_date_str: str,
        end_date_str: str,
        start_time_str: str, 
        end_time_str: str
    ) -> tuple[dict, dict]:
    """
    Reads TNC hourly pickup/dropoff data from a CSV file and filters it based on a specified time window.

    This function:
    - Reads TNC hourly pickup/dropoff data from a CSV file.
    - Filters the data based on the specified time window.
    - Computes total pickups and dropoffs across all zones and selected hours.
    - Returns two nested dictionary with TAZ as keys and hour data as values, one for current hour
      data and one for previous hour data. The latter is useful to compute starting drivers in the simulation.

    Parameters:
    ----------
    - sf_rides_stats_path (str): 
        Path to the CSV file with columns: 'taz', 'day_of_week', 'hour', 'pickups', 'dropoffs'.
    - start_date_str : str
        Start date in 'YYYY-MM-DD' format (e.g., '2021-03-25').
    - end_date_str : str
        End date in 'YYYY-MM-DD' format (e.g., '2021-03-26').
    - start_time_str : str
        Start time in 'HH:MM' format (e.g., '08:00').
    - end_time_str : str
        End time in 'HH:MM' format (e.g., '10:00'). Can wrap around midnight.

    Returns:
    -------
    tuple[dict, dict]
        Nested dictionaries {taz: {hour: {'pickups': x, 'dropoffs': y}}} where `hour` is in standard 0-23 format.
    """
    def parse_hour(time_str):
        # Convert time string (HH:MM) to hour in standard 0-23 format
        return int(datetime.strptime(time_str, "%H:%M").hour)

    start_date = datetime.strptime(start_date_str, "%Y-%m-%d").date()
    end_date = datetime.strptime(end_date_str, "%Y-%m-%d").date()
    num_days = (end_date - start_date).days + 1

    # Parse start and end hours
    start_hour = parse_hour(start_time_str)
    end_hour = parse_hour(end_time_str)

    # Map dataset hours (3–26) to standard 0–23 format
    dataset_hour_map = {h: h % 24 for h in range(3, 27)}

    # Read the CSV file    
    all_rows = []
    with open(sf_rides_stats_path, mode='r') as file:
        reader = csv.DictReader(file, delimiter=',')
        for row in reader:
            row['taz'] = int(row['taz'])
            row['day_of_week'] = int(row['day_of_week'])
            row['hour'] = int(row['hour'])
            row['pickups'] = round(float(row['pickups']))
            row['dropoffs'] = round(float(row['dropoffs']))
            all_rows.append(row)

    # Index by (day_of_week, hour, taz)
    data_by_key = {}
    for row in all_rows:
        key = (row['day_of_week'], row['hour'], row['taz'])
        data_by_key[key] = {'pickups': row['pickups'], 'dropoffs': row['dropoffs']}

    # Initialize dictionaries to hold the output
    zone_data = {}
    zone_data_previous_hour = {}
    # For each simulation day, determine hours to include from that day
    for sim_day_index in range(num_days):
        sim_date = start_date + timedelta(days=sim_day_index)
        sim_day_of_week = sim_date.weekday()
        if sim_day_index == 0:
            selected_std_hours = list(range(start_hour, 24)) if start_hour < 24 else []
        elif sim_day_index == num_days - 1:
            selected_std_hours = list(range(0, end_hour)) if end_hour > 0 else []
        else:
            selected_std_hours = list(range(0, 24))
        selected_dataset_hours = {h: std for h, std in dataset_hour_map.items() if std in selected_std_hours}
        # Filter rows for this day and hour
        for row in all_rows:
            taz = row['taz']
            hour = row['hour']
            day = row['day_of_week']
            if day == sim_day_of_week and hour in selected_dataset_hours:
                std_hour = selected_dataset_hours[hour]
                if taz not in zone_data:
                    zone_data[taz] = {}
                zone_data[taz][std_hour] = {
                    'pickups': row['pickups'],
                    'dropoffs': row['dropoffs']
                }
                # Also get previous hour for this hour (possibly from previous weekday)
                prev_hour = hour - 1
                prev_day = day
                if prev_hour < 3:
                    prev_hour = 26
                    prev_day = (day - 1) % 7
                prev_key = (prev_day, prev_hour, taz)
                if prev_key in data_by_key:
                    if taz not in zone_data_previous_hour:
                        zone_data_previous_hour[taz] = {}
                    zone_data_previous_hour[taz][std_hour] = data_by_key[prev_key]

    print(f"✅ TNC stats data read from {sf_rides_stats_path} and filtered to time window {start_date_str} {start_time_str} - {end_date_str} {end_time_str}")

    # Compute pickups and dropoffs across all zones and selected hours
    total_pickups = sum(hour_data['pickups'] for zone in zone_data.values() for hour_data in zone.values())
    total_dropoffs = sum(hour_data['dropoffs'] for zone in zone_data.values() for hour_data in zone.values())
    print(f"🚕 Total pickups:  {total_pickups}, Total dropoffs: {total_dropoffs}")

    return zone_data, zone_data_previous_hour