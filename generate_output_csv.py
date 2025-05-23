"""
generate_output_csv.py

This script processes simulation output files to generate a CSV file containing metrics related to
passengers, drivers, and traffic vehicles. It also creates an interactive line plot for visualization.
The script performs the following steps:
1. Load XML data from the simulation output files: tripinfos.xml, queue.xml, and summary.xml.
2. Parse the XML data to extract relevant metrics such as:
   - Passenger requests, departures, arrivals, and served by drivers.
   - Ride waiting times, durations, and route lengths.
   - Driver shift times, total route lengths, and occupied distances/times.
   - Traffic vehicle durations and route lengths.
   - Average speed and queuing metrics.
3. Store the extracted metrics in a dictionary, indexed by timestamps.
4. Convert the dictionary into a Pandas DataFrame.
5. Save the DataFrame to a CSV file.
6. Create an interactive line plot of the metrics and save it as an HTML file.
"""


from constants.sumoenv_constants import SUMO_SCENARIOS_PATH
import xml.etree.ElementTree as ET
from collections import defaultdict
import pandas as pd
import os
import plotly.express as px

# Set up scenarios, date, timeslot, and output path
scenario = "normal"
date_part = "2021-01-14"
timeslot_part = "09-10"
output_path = os.path.join(SUMO_SCENARIOS_PATH, scenario, date_part, timeslot_part)

# Load XML data
tripinfos = os.path.join(output_path, "tripinfos.xml")
if not os.path.exists(tripinfos):
    print(f"❌ File {tripinfos} does not exist.")
    exit(1)
queue = os.path.join(output_path, "queue.xml")
if not os.path.exists(queue):
    print(f"❌ File {queue} does not exist.")
    exit(1)
summary = os.path.join(output_path, "summary.xml")
if not os.path.exists(summary):
    print(f"❌ File {summary} does not exist.")
    exit(1)

# Dictionary to hold timestamps and their corresponding metrics
timestamps = defaultdict(lambda: {
    "passengers_requests": 0,
    "passengers_departures": 0,
    "passengers_arrivals": 0,
    "passengers_served_by_drivers": 0,
    "ride_waiting_times": [],
    "ride_durations": [],
    "ride_route_lengths": [],
    "num_rides_in_progress": 0,
    "num_drivers_cars": 0,
    "drivers_shift_times": [],
    "drivers_total_route_lengths": [],
    "drivers_total_times": [],
    "drivers_occupied_distance": [],
    "drivers_occupied_time": [],
    "num_traffic_vehicles": 0,
    "traffic_durations": [],
    "traffic_route_lengths": [],
    "avg_speed": 0,
    "avg_relative_speed": 0,
    "avg_queuing_time": 0,
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
        waiting_time = float(ride.attrib.get("waitingTime"))
        route_length = float(ride.attrib.get("routeLength"))
        for t in range(request, arrival + 1):
            ts = timestamps[t]
            ts["passengers_requests"] += 1 if t == request else 0
            ts["passengers_departures"] += 1 if t == depart else 0
            ts["num_rides_in_progress"] += 1 if depart < t < arrival else 0
            ts["passengers_arrivals"] += 1  if t == arrival else 0
            if t == arrival:
                ts["ride_waiting_times"].append(waiting_time)
                ts["ride_durations"].append(duration)
                ts["ride_route_lengths"].append(route_length)
# <tripinfo> - traffic and taxis
for trip in root.findall("tripinfo"):
    if "taxi" in trip.attrib.get("id"):
        depart = int(float(trip.attrib.get("depart")))
        arrival = int(float(trip.attrib.get("arrival")))
        shift_time = float(trip.attrib.get("duration"))
        route_length = float(trip.attrib.get("routeLength"))
        idle_time = float(trip.attrib.get("waitingTime"))
        taxi_elem = trip.find("taxi")
        if taxi_elem is not None:
            customers = int(taxi_elem.attrib.get("customers"))
            occupied_distance = float(taxi_elem.attrib.get("occupiedDistance"))
            occupied_time = float(taxi_elem.attrib.get("occupiedTime"))

        for t in range(depart, arrival + 1):
            ts = timestamps[t]
            ts["num_drivers_cars"] += 1
            ts["passengers_served_by_drivers"] += customers if t == arrival else 0
            if t == arrival:
                ts["drivers_shift_times"].append(shift_time)
                ts["drivers_total_route_lengths"].append(route_length)
                ts["drivers_total_times"].append(idle_time)
                ts["drivers_occupied_distance"].append(occupied_distance)
                ts["drivers_occupied_time"].append(occupied_time)
    else:
        depart = int(float(trip.attrib.get("depart")))
        arrival = int(float(trip.attrib.get("arrival")))
        duration = float(trip.attrib.get("duration"))
        route_length = float(trip.attrib.get("routeLength"))
        for t in range(depart, arrival + 1):
            ts = timestamps[t]
            ts["num_traffic_vehicles"] += 1
            if t == arrival:
                ts["traffic_durations"].append(duration)
                ts["traffic_route_lengths"].append(route_length)

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
    times, lengths = [], []
    if lanes is not None:
        for lane in lanes.findall("lane"):
            times.append(float(lane.attrib.get("queueing_time")))
            lengths.append(float(lane.attrib.get("queueing_length")))
    n = len(times)
    avg_time = sum(times) / n if n > 0 else 0.0
    avg_length = sum(lengths) / n if n > 0 else 0.0
    ts = timestamps[timestep]
    ts["avg_queuing_time"] = avg_time
    ts["avg_queuing_length"] = avg_length

# Format results into a DataFrame
data = []
for t, stats in sorted(timestamps.items()):
    data.append({
        "timestamp": t,
        "passengers_requests": stats["passengers_requests"],
        "passengers_departures": stats["passengers_departures"],
        "passengers_arrivals": stats["passengers_arrivals"],
        "passengers_served_by_drivers": stats["passengers_served_by_drivers"],
        "avg_rides_waiting_time": sum(stats["ride_waiting_times"]) / len(stats["ride_waiting_times"]) if stats["ride_waiting_times"] else 0,
        "avg_rides_duration": sum(stats["ride_durations"]) / len(stats["ride_durations"]) if stats["ride_durations"] else 0,
        "avg_rides_route_length": sum(stats["ride_route_lengths"]) / len(stats["ride_route_lengths"]) if stats["ride_route_lengths"] else 0,
        "num_rides_in_progress": stats["num_rides_in_progress"],
        "num_drivers_cars": stats["num_drivers_cars"],
        "avg_drivers_shift_duration": sum(stats["drivers_shift_times"]) / len(stats["drivers_shift_times"]) if stats["drivers_shift_times"] else 0,
        "avg_drivers_total_route_length": sum(stats["drivers_total_route_lengths"]) / len(stats["drivers_total_route_lengths"]) if stats["drivers_total_route_lengths"] else 0,
        "avg_drivers_total_times": sum(stats["drivers_total_times"]) / len(stats["drivers_total_times"]) if stats["drivers_total_times"] else 0,
        "avg_drivers_occupied_distance": sum(stats["drivers_occupied_distance"]) / len(stats["drivers_occupied_distance"]) if stats["drivers_occupied_distance"] else 0,
        "avg_drivers_occupied_time": sum(stats["drivers_occupied_time"]) / len(stats["drivers_occupied_time"]) if stats["drivers_occupied_time"] else 0,
        "num_traffic_vehicles": stats["num_traffic_vehicles"],
        "avg_traffic_duration": sum(stats["traffic_durations"]) / len(stats["traffic_durations"]) if stats["traffic_durations"] else 0,
        "avg_traffic_route_length": sum(stats["traffic_route_lengths"]) / len(stats["traffic_route_lengths"]) if stats["traffic_route_lengths"] else 0,
        "avg_queuing_time": stats["avg_queuing_time"],
        "avg_queuing_length": stats["avg_queuing_length"],
        "avg_speed": stats["avg_speed"],
        "avg_relative_speed": stats["avg_relative_speed"]
    })

# Create DataFrame and save to CSV
df = pd.DataFrame(data)
output_csv_path = os.path.join(output_path, "final_metrics.csv")
df.to_csv(output_csv_path, index=False)
print(f"✅ CSV file saved in {output_csv_path}")

# Create interactive line plot of the metrics
df.set_index("timestamp", inplace=True)
title = f"Simulation indicators time series: interactive line plot <br><sup>City: San Francisco, Scenario: {scenario}, Date: {date_part}, Timeslot: {timeslot_part}</sup>"
output_html_path = os.path.join(output_path, "metrics_visualization.html")
fig = px.line(df, x=df.index, y=df.columns, title=title)
fig.update_layout(legend_title_text="Simulation metrics")
fig.write_html(output_html_path)
print(f"✅ Interactive line plot saved in {output_html_path}")