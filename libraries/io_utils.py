"""
io_utils.py

This module provides utility functions for validating user input and generating output CSV files.
It includes utilities for:

1. get_valid_date: Validating date input in MM-DD format.
2. get_valid_hour: Validating hour input in HH:MM format.
3. get_valid_int: Validating agents interval input within a specified range.
4. get_valid_scenario: Validating scenario names.
5. get_valid_gui: Validating GUI input (yes/no).
6. generate_output_csv: Generating output CSV files from simulation data, and creating an interactive line plot.
"""


import os
from collections import defaultdict
from datetime import datetime
import xml.etree.ElementTree as ET
import pandas as pd
import plotly.express as px
from constants.sumoenv_constants import SUMO_SCENARIOS_PATH


def get_valid_date(prompt: str) -> str:
    while True:
        date_str = input(prompt).strip()
        try:
            date_obj = datetime.strptime(date_str, "%m-%d")
            if datetime(2021, 1, 1) <= datetime(2021, date_obj.month, date_obj.day) <= datetime(2021, 12, 30):
                return f"2021-{date_str}"
            else:
                print("⚠️  Date must be between 01-01 and 12-30")
        except ValueError:
            print("⚠️  Invalid date format. Use MM-DD")


def get_valid_hour(
        prompt: str,
        start_hour_same_day_check: bool = False
    ) -> str:
    while True:
        hour_str = input(prompt).strip()
        if hour_str.isdigit():
            hour = int(hour_str)
            if start_hour_same_day_check:
                if 0 <= hour <= 22:
                    return f"{hour:02d}:00"
                else:
                    print("⚠️  Since you chose to simulate the same day, start hour must be between 0 and 22")
            else:
                if 0 <= hour <= 23:
                    return f"{hour:02d}:00"
                else:
                    print("⚠️  Hour must be between 0 and 23")
        else:
            print("⚠️  Please enter a valid hour as an integer (e.g., 9 or 17)")


def get_valid_int(
        prompt: str,
        min_val: int,
        max_val: int
    ) -> int:
    while True:
        try:
            val = int(input(prompt))
            if min_val <= val <= max_val:
                return val
            else:
                print(f"⚠️  Value must be between {min_val} and {max_val}")
        except ValueError:
            print("⚠️  Please enter an integer")


def get_valid_scenario(prompt: str) -> str:
    scenarios = ["normal"]
    while True:
        scenario = input(prompt).strip().lower()
        if scenario.isalnum():
            if scenario in scenarios:
                return scenario
            else:
                print(f"⚠️  Invalid scenario name. Available scenarios: {', '.join(scenarios)}")
        else:
            print("⚠️  Scenario name must be alphanumeric")


def get_valid_gui(prompt: str) -> bool:
    while True:
        gui_input = input(prompt).strip().lower()
        if gui_input in ["yes", "no"]:
            return gui_input == "yes"
        else:
            print("⚠️  Please enter 'yes' or 'no'")


def generate_output_csv(
        start_date_str: str,
        end_date_str: str,
        start_time_str: str,
        end_time_str: str,
        scenario: str = "normal",
    ) -> None:
    """
    Processes simulation output files to generate a CSV file containing metrics related to
    passengers, drivers, and traffic vehicles. It also creates an interactive line plot for visualization.
    This function:
    - Loads XML data from the simulation output files: tripinfos.xml, queue.xml, summary.xml and multi_agent_infos.xml.
    - Parses the XML data to extract relevant metrics such as:
        - Passenger requests, departures and arrivals.
        - Durations and route lengths.
        - Driver shift durations, total route lengths, and occupied distances/durations.
        - Average price and surge multiplier.
        - Traffic vehicle durations and route lengths.
        - Average speed and queuing metrics.
    - Stores the extracted metrics in a dictionary, indexed by timestamps.
    - Converts the dictionary into a Pandas DataFrame.
    - Saves the DataFrame to a CSV file.
    - Creates an interactive line plot of the metrics and saves it as an HTML file.

    Parameters:
    ----------
    - start_date_str: str
        Start date in 'YYYY-MM-DD' format (e.g., '2021-03-25').
    - end_date_str : str
        End date in 'YYYY-MM-DD' format (e.g., '2021-03-26').
    - start_time_str : str
        Start time in 'HH:MM' format (e.g., '08:00').
    - end_time_str : str
        End time in 'HH:MM' format (e.g., '10:00').
    - scenario: str
        Scenario name (default is 'normal').

    Returns:
    -------
    None

    Raises:
    ------
    FileNotFoundError: If the specified SUMO scenarios path does not exist or if any of the required XML files are missing.
    """
    print("📊 Generating output CSV and visualization...")
    # Ensure the output directory exists
    if not os.path.exists(SUMO_SCENARIOS_PATH):
        raise FileNotFoundError(f"SUMO scenarios path does not exist: {SUMO_SCENARIOS_PATH}")
    
    # Format date and time parts
    start_date = datetime.strptime(start_date_str, "%Y-%m-%d").strftime("%y%m%d")
    end_date = datetime.strptime(end_date_str, "%Y-%m-%d").strftime("%y%m%d")
    start_hour = datetime.strptime(start_time_str, "%H:%M").strftime("%H")
    end_hour = datetime.strptime(end_time_str, "%H:%M").strftime("%H")

    # Prepare directory and filename
    start_d = start_date.replace("-", "")
    end_d = end_date.replace("-", "")
    start_h = start_hour.replace(":", "")
    end_h = end_hour.replace(":", "")
    timeslot = f"{start_d}{start_h}_{end_d}{end_h}"
    output_path = os.path.join(SUMO_SCENARIOS_PATH, scenario, timeslot)

    # Load XML data
    tripinfos = os.path.join(output_path, "tripinfos.xml")
    if not os.path.exists(tripinfos):
        raise FileNotFoundError(f"File not found: {tripinfos}")
    queue = os.path.join(output_path, "queue.xml")
    if not os.path.exists(queue):
        raise FileNotFoundError(f"File not found: {queue}")
    summary = os.path.join(output_path, "summary.xml")
    if not os.path.exists(summary):
        raise FileNotFoundError(f"File not found: {summary}")
    agents = os.path.join(output_path, "multi_agent_infos.xml")
    if not os.path.exists(agents):
        raise FileNotFoundError(f"File not found: {agents}")

    # Dictionary to hold timestamps and their corresponding metrics
    timestamps = defaultdict(lambda: {
        # Passengers metrics
        "passengers_new": 0,
        "passengers_departures": 0,
        "passengers_arrivals": 0,
        "passengers_unassigned": 0,
        "passengers_assigned": 0,
        "passengers_pickup": 0,
        "passengers_accept": 0,
        "passengers_reject": 0,

        # Drivers metrics
        "drivers_shift_durations": [],
        "drivers_total_lengths": [],
        "drivers_total_durations": [],
        "drivers_occupied_distance": [],
        "drivers_occupied_durations": [],
        "drivers_passengers_served": 0,
        "drivers_idle": 0,
        "drivers_pickup": 0,
        "drivers_busy": 0,
        "drivers_accept": 0,
        "drivers_reject": 0,

        # Rides and providers metrics
        "rides_in_progress": 0,
        "rides_waiting_durations": [],
        "rides_durations": [],
        "rides_lengths": [],
        "expected_rides_durations": 0.0,
        "expected_rides_lengths": 0.0,
        "taxis_dispatched": 0,
        "partial_acceptances": 0,
        "offers_generated": 0,
        "offers_timeout": 0,
        "requests_canceled": 0,
        "requests_not_served": 0,
        "offers_radius": 0.0,
        "offers_price": 0.0,
        "offers_surge_multiplier": 0.0,

        # Traffic metrics
        "traffic_vehicles": 0,
        "traffic_durations": [],
        "traffic_lengths": [],
        "avg_speed": 0,
        "avg_relative_speed": 0,
        "avg_queuing_durations": 0,
        "avg_queuing_length": 0
    })

    # Process tripinfos.xml
    print("Processing tripinfos.xml...")
    tree = ET.parse(tripinfos)
    root = tree.getroot()
    # <personinfo> - rides
    for person in root.findall("personinfo"):
        request = int(float(person.attrib.get("depart")))
        for ride in person.findall("ride"):
            depart = int(float(ride.attrib.get("depart")))
            arrival = int(float(ride.attrib.get("arrival")))
            duration = float(ride.attrib.get("duration"))
            waiting_durations = float(ride.attrib.get("waitingTime"))
            route_length = float(ride.attrib.get("routeLength"))
            for t in range(request, arrival + 1):
                ts = timestamps[t]
                ts["passengers_new"] += 1 if t == request else 0
                ts["passengers_departures"] += 1 if t == depart else 0
                ts["rides_in_progress"] += 1 if depart < t < arrival else 0
                ts["passengers_arrivals"] += 1  if t == arrival else 0
                if t == arrival:
                    ts["rides_waiting_durations"].append(waiting_durations)
                    ts["rides_durations"].append(duration)
                    ts["rides_lengths"].append(route_length)
    # <tripinfo> - traffic and taxis
    for trip in root.findall("tripinfo"):
        if "taxi" in trip.attrib.get("id"):
            depart = int(float(trip.attrib.get("depart")))
            arrival = int(float(trip.attrib.get("arrival")))
            shift_durations = float(trip.attrib.get("duration"))
            route_length = float(trip.attrib.get("routeLength"))
            idle_durations = float(trip.attrib.get("waitingTime"))
            taxi_elem = trip.find("taxi")
            if taxi_elem is not None:
                customers = int(taxi_elem.attrib.get("customers"))
                occupied_distance = float(taxi_elem.attrib.get("occupiedDistance"))
                occupied_durations = float(taxi_elem.attrib.get("occupiedTime"))
            for t in range(depart, arrival + 1):
                ts = timestamps[t]
                ts["drivers_passengers_served"] += customers if t == arrival else 0
                if t == arrival:
                    ts["drivers_shift_durations"].append(shift_durations)
                    ts["drivers_total_lengths"].append(route_length)
                    ts["drivers_total_durations"].append(idle_durations)
                    ts["drivers_occupied_distance"].append(occupied_distance)
                    ts["drivers_occupied_durations"].append(occupied_durations)
        else:
            depart = int(float(trip.attrib.get("depart")))
            arrival = int(float(trip.attrib.get("arrival")))
            duration = float(trip.attrib.get("duration"))
            route_length = float(trip.attrib.get("routeLength"))
            for t in range(depart, arrival + 1):
                ts = timestamps[t]
                ts["traffic_vehicles"] += 1
                if t == arrival:
                    ts["traffic_durations"].append(duration)
                    ts["traffic_lengths"].append(route_length)

    # Process summary.xml
    print("Processing summary.xml...")
    tree = ET.parse(summary)
    root = tree.getroot()
    for step in root.findall("step"):
        timestep = int(float(step.attrib.get("time")))
        ts = timestamps[timestep]
        ts["avg_speed"] = float(step.attrib.get("meanSpeed"))
        ts["avg_relative_speed"] = float(step.attrib.get("meanSpeedRelative"))

    # Process queue.xml
    print("Processing queue.xml...")
    tree = ET.parse(queue)
    root = tree.getroot()
    for data in root.findall("data"):
        timestep = int(float(data.attrib.get("timestep")))
        lanes = data.find("lanes")
        durations, lengths = [], []
        if lanes is not None:
            for lane in lanes.findall("lane"):
                durations.append(float(lane.attrib.get("queueing_time")))
                lengths.append(float(lane.attrib.get("queueing_length")))
        n = len(durations)
        avg_durations = sum(durations) / n if n > 0 else 0.0
        avg_length = sum(lengths) / n if n > 0 else 0.0
        ts = timestamps[timestep]
        ts["avg_queuing_durations"] = avg_durations
        ts["avg_queuing_length"] = avg_length

    # Process multi_agent_infos.xml
    print("Processing multi_agent_infos.xml...")
    tree = ET.parse(agents)
    root = tree.getroot()
    for step in root.findall("step"):
        timestep = int(float(step.attrib.get("timestamp")))
        passengers = step.find("passengers")
        drivers = step.find("drivers")
        offers = step.find("offers")
        rideservices = step.find("rideservices")
        ts = timestamps[timestep]
        if passengers is not None:
            ts["passengers_unassigned"] = int(float(passengers.find("unassigned_requests").text))
            ts["passengers_assigned"] = int(float(passengers.find("assigned_requests").text))
            ts["passengers_pickup"] = int(float(passengers.find("pickup_requests").text))
            ts["passengers_accept"] = int(float(passengers.find("accepted_requests").text))
            ts["passengers_reject"] = int(float(passengers.find("rejected_requests").text))
        if drivers is not None:
            ts["drivers_idle"] = int(float(drivers.find("idle_drivers").text))
            ts["drivers_pickup"] = int(float(drivers.find("pickup_drivers").text))
            ts["drivers_busy"] = int(float(drivers.find("busy_drivers").text))
            ts["drivers_accept"] = int(float(drivers.find("accepted_requests").text))
            ts["drivers_reject"] = int(float(drivers.find("rejected_requests").text))
        if offers is not None:
            ts["expected_rides_durations"] = float(offers.find("avg_expected_time").text)
            ts["expected_rides_lengths"] = float(offers.find("avg_expected_length").text)
            ts["offers_radius"] = float(offers.find("avg_radius").text)
            ts["offers_price"] = float(offers.find("avg_price").text)
            ts["offers_surge_multiplier"] = float(offers.find("avg_surge_multiplier").text)
        if rideservices is not None:
            ts["taxis_dispatched"] = int(float(rideservices.find("dispatched_taxis").text))
            ts["offers_generated"] = int(float(rideservices.find("generated_offers").text))
            ts["offers_timeout"] = int(float(rideservices.find("timeout_offers").text))
            ts["partial_acceptances"] = int(float(rideservices.find("partial_acceptances").text))
            ts["requests_canceled"] = int(float(rideservices.find("requests_canceled").text))
            ts["requests_not_served"] = int(float(rideservices.find("requests_not_served").text))

    # Format results into a DataFrame
    data = []
    for t, stats in sorted(timestamps.items()):
        data.append({
            "timestamp": t,
            "passengers_new": stats["passengers_new"],
            "passengers_departures": stats["passengers_departures"],
            "passengers_arrivals": stats["passengers_arrivals"],
            "passengers_unassigned": stats["passengers_unassigned"],
            "passengers_assigned": stats["passengers_assigned"],
            "passengers_pickup": stats["passengers_pickup"],
            "passengers_accept": stats["passengers_accept"],
            "passengers_reject": stats["passengers_reject"],
            "drivers_shift_duration_avg": sum(stats["drivers_shift_durations"]) / len(stats["drivers_shift_durations"]) if stats["drivers_shift_durations"] else 0,
            "drivers_total_length_avg": sum(stats["drivers_total_lengths"]) / len(stats["drivers_total_lengths"]) if stats["drivers_total_lengths"] else 0,
            "drivers_total_duration_avg": sum(stats["drivers_total_durations"]) / len(stats["drivers_total_durations"]) if stats["drivers_total_durations"] else 0,
            "drivers_occupied_distance_avg": sum(stats["drivers_occupied_distance"]) / len(stats["drivers_occupied_distance"]) if stats["drivers_occupied_distance"] else 0,
            "drivers_occupied_duration_avg": sum(stats["drivers_occupied_durations"]) / len(stats["drivers_occupied_durations"]) if stats["drivers_occupied_durations"] else 0,
            "drivers_passengers_served": stats["drivers_passengers_served"],
            "drivers_idle": stats["drivers_idle"],
            "drivers_pickup": stats["drivers_pickup"],
            "drivers_busy": stats["drivers_busy"],
            "drivers_accept": stats["drivers_accept"],
            "drivers_reject": stats["drivers_reject"],
            "rides_in_progress": stats["rides_in_progress"],
            "rides_waiting_duration_avg": sum(stats["rides_waiting_durations"]) / len(stats["rides_waiting_durations"]) if stats["rides_waiting_durations"] else 0,
            "rides_duration_avg": sum(stats["rides_durations"]) / len(stats["rides_durations"]) if stats["rides_durations"] else 0,
            "rides_length_avg": sum(stats["rides_lengths"]) / len(stats["rides_lengths"]) if stats["rides_lengths"] else 0,
            "rides_duration_expected_avg": stats["expected_rides_durations"],
            "rides_length_expected_avg": stats["expected_rides_lengths"],
            "rides_dispatched": stats["taxis_dispatched"],
            "rides_partial_acceptances": stats["partial_acceptances"],
            "rides_offers_generated": stats["offers_generated"],
            "rides_offers_timeout": stats["offers_timeout"],
            "rides_requests_canceled": stats["requests_canceled"],
            "rides_requests_not_served": stats["requests_not_served"],
            "rides_offers_radius_avg": stats["offers_radius"],
            "rides_offers_price_avg": stats["offers_price"],
            "rides_offers_surge_multiplier_avg": stats["offers_surge_multiplier"],
            "traffic_vehicles": stats["traffic_vehicles"],
            "traffic_duration_avg": sum(stats["traffic_durations"]) / len(stats["traffic_durations"]) if stats["traffic_durations"] else 0,
            "traffic_length_avg": sum(stats["traffic_lengths"]) / len(stats["traffic_lengths"]) if stats["traffic_lengths"] else 0,
            "speed_avg": stats["avg_speed"],
            "speed_relative_avg": stats["avg_relative_speed"],
            "queuing_duration_avg": stats["avg_queuing_durations"],
            "queuing_length_avg": stats["avg_queuing_length"]
        })

    # Create DataFrame and save to CSV
    df = pd.DataFrame(data)
    output_csv_path = os.path.join(output_path, "sf_final_metrics.csv")
    df.to_csv(output_csv_path, index=False)
    print(f"✅ Computed {len(df.columns)-1} metrics, CSV file saved to {output_csv_path}")

    # Create interactive line plot of the metrics
    df.set_index("timestamp", inplace=True)
    title = f"Simulation indicators time series: interactive line plot <br><sup>City: San Francisco, Scenario: {scenario}, From: {start_date_str} {start_time_str} To: {end_date_str} {end_time_str}</sup>"
    output_html_path = os.path.join(output_path, "sf_final_metrics_visualization.html")
    fig = px.line(df, x=df.index, y=df.columns, title=title)
    fig.update_layout(legend_title_text="Simulation metrics")
    fig.write_html(output_html_path)
    print(f"✅ Interactive line plot saved to {output_html_path}")