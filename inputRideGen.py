# data are indicated for timeslots and taz, meaning that for running an hourly timeslot
# command for generation the taz file required in the parsing function
# python "C:\Program Files (x86)\Eclipse\Sumo\tools\edgesInDistricts.py"
# -n "C:\Users\aless\PycharmProjects\sanfrancisco-ridesharingecosystem\sumoenv\sf.net.xml" -t "C:\Users\aless\PycharmProjects\sanfrancisco-ridesharingecosystem\sumoenv\taz_zones.poly.xml" -o "C:\Users\aless\PycharmProjects\sanfrancisco-ridesharingecosystem\sumoenv\sf.taz.xml"


from libraries.data_utils import read_uber_stats_data
from libraries.sumo_utils import (export_taz_coords, map_coords_to_sumo_edges,
                                  generate_matched_drt_requests, generate_vehicle_start_lanes_from_taz_polygons, get_strongly_connected_edges,
                                  generate_drt_vehicle_instances_from_lanes, get_valid_taxi_edges)
from constants.data_constants import SF_RIDE_STATS_PATH, SF_TAZ_SHAPEFILE_PATH, SF_TAZ_COORDINATES_PATH
from constants.sumoenv_constants import SUMO_TOOLS_LOCAL_PATH, SUMO_NET_PATH
import pandas as pd
import random
import ast

export_taz_coords(SF_TAZ_SHAPEFILE_PATH, SF_TAZ_COORDINATES_PATH)
map_coords_to_sumo_edges(SF_TAZ_COORDINATES_PATH, SUMO_NET_PATH, SF_TAZ_COORDINATES_PATH)

mapping_df = pd.read_csv(SF_TAZ_COORDINATES_PATH, sep=";")
# Extract strongly connected edges
safe_edge_ids = get_strongly_connected_edges(SUMO_NET_PATH)

def filter_polygon_edges(polygon_edge_str):
    """Filter edge list string, keeping only strongly connected edges."""
    edge_list = ast.literal_eval(polygon_edge_str)
    return [e for e in edge_list if e in safe_edge_ids]

def filter_polygon_lanes(polygon_lane_str):
    """Filter lane list string, keeping only lanes whose parent edge is in the strongly connected set."""
    lane_list = ast.literal_eval(polygon_lane_str)
    return [l for l in lane_list if l.split('_')[0] in safe_edge_ids]

# Apply filters
mapping_df['polygon_edge_ids'] = mapping_df['polygon_edge_ids'].apply(filter_polygon_edges)
mapping_df = mapping_df[mapping_df['polygon_edge_ids'].map(lambda x: len(x) > 0)]
mapping_df['polygon_lane_ids'] = mapping_df['polygon_lane_ids'].apply(filter_polygon_lanes)
mapping_df = mapping_df[mapping_df['polygon_lane_ids'].map(lambda x: len(x) > 0)]

# Reconstruct the mapping dictionary
taz_edge_mapping = mapping_df.set_index('TAZ')[['polygon_edge_ids', 'polygon_lane_ids']].dropna().to_dict('index')

start_time = "9:00"
end_time = "10:00"
data=read_uber_stats_data(SF_RIDE_STATS_PATH, start_time, end_time)
# Print the data for debugging
total_pickups = sum(inner[9]['pickups'] for inner in data.values())
total_dropoffs = sum(inner[9]['dropoffs'] for inner in data.values())
print(f"Total pickups:  {total_pickups}")
print(f"Total dropoffs: {total_dropoffs}")

start_lanes = generate_vehicle_start_lanes_from_taz_polygons(
    shapefile_path=SF_TAZ_SHAPEFILE_PATH,
    net_file=SUMO_NET_PATH,
    vehicles_per_taz=2,  # 2 vehicles per TAZ → ~1960 total
    safe_edge_ids=safe_edge_ids
)

# Step 2: Generate trips

number_vehicles_available=1000
generate_drt_vehicle_instances_from_lanes(
    lane_ids=random.sample(start_lanes, number_vehicles_available),
    output_path="fleet_vehicles.rou.xml"
)

# Step 3: Generate matched request file

# Intersect with valid taxi edges
valid_edge_ids = get_valid_taxi_edges(SUMO_NET_PATH, safe_edge_ids)
filtered_valid_edges = [e for e in valid_edge_ids if e in safe_edge_ids]

generate_matched_drt_requests(
    uber_data=data,
    taz_edge_mapping=taz_edge_mapping,
    sim_start_s=0,
    sim_end_s=3600,
    valid_edge_ids=filtered_valid_edges,
    output_path="passenger_requests.rou.xml")