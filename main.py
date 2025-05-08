from libraries.sumo_utils import sf_traffic_map_matching, sf_traffic_od_generation, sf_traffic_routes_generation, \
    export_taz_coords, map_coords_to_sumo_edges, get_strongly_connected_edges, generate_matched_drt_requests, \
    add_sf_traffic_taz_matching, filter_polygon_edges,filter_polygon_lanes, generate_vehicle_start_lanes_from_taz_polygons, \
    generate_drt_vehicle_instances_from_lanes, get_valid_taxi_edges
from constants.data_constants import (SF_TRAFFIC_MAP_MATCHED_FOLDER_PATH, SF_TRAFFIC_VEHICLE_DAILY_FOLDER_PATH,
                                      SF_TRAFFIC_VEHICLE_ONEWEEK_PATH, SF_TAZ_SHAPEFILE_PATH, SF_TAZ_COORDINATES_PATH,
                                      SF_RIDE_STATS_PATH)
from constants.sumoenv_constants import (SUMOENV_PATH, SUMO_NET_PATH, SUMO_BASE_SCENARIO_FOLDER_PATH,
                                         SUMO_CFGTEMPLATE_PATH, SUMO_POLY_PATH)
from libraries.data_utils import convert_sf_traffic_csv_format, extract_sf_traffic_timeslot, read_uber_stats_data
from classes.simulator import Simulator
import pandas as pd
import ast
import random

date = "2021-01-14"
start_time = "9:00"
end_time = "10:00"
radius = 150
sumoSimulator = Simulator(net_file=SUMO_NET_PATH, config_template_path=SUMO_CFGTEMPLATE_PATH, taz_file_path=SUMO_POLY_PATH)

# 1. reading the 10-16 traffic vehicle data set
SF_TRAFFIC_VEHICLE_DAILYHOUR_PATH = extract_sf_traffic_timeslot(input_csv_path=SF_TRAFFIC_VEHICLE_ONEWEEK_PATH,
                                                                date_str=date,
                                                                start_time_str=start_time,end_time_str=end_time,
                                                                output_csv_folder=SF_TRAFFIC_VEHICLE_DAILY_FOLDER_PATH)

# 2. MAP and TAZ matching
SF_TRAFFIC_EDGE_PATH = sf_traffic_map_matching(sf_map_file_path=SUMO_NET_PATH,
                                              sf_real_traffic_data_path=SF_TRAFFIC_VEHICLE_DAILYHOUR_PATH,
                                              date=date, output_folder_path=SF_TRAFFIC_MAP_MATCHED_FOLDER_PATH, radius=150)
add_sf_traffic_taz_matching(SF_TRAFFIC_EDGE_PATH, SF_TAZ_SHAPEFILE_PATH)

# 3. generate ORIGIN-DESTINATION file for each vehicle
SF_TRAFFIC_0D_PATH = sf_traffic_od_generation(sf_real_traffic_edge_path=SF_TRAFFIC_EDGE_PATH,
                         sf_traffic_od_folder_path=SUMO_BASE_SCENARIO_FOLDER_PATH, date=date,
                         start_time=start_time, end_time=end_time)

# 4. generate ROUTES file for traffic
SF_TRAFFIC_ROUTE_PATH = sf_traffic_routes_generation(sf_traffic_od_path=SF_TRAFFIC_0D_PATH,
                                                     sf_traffic_routes_folder_path=SUMO_BASE_SCENARIO_FOLDER_PATH,
                                                     date=date, start_time=start_time, end_time=end_time)

# 5. generate ROUTES file for taxi
# 5.1. MAP and TAZ matching
export_taz_coords(SF_TAZ_SHAPEFILE_PATH, SF_TAZ_COORDINATES_PATH)
map_coords_to_sumo_edges(SF_TAZ_COORDINATES_PATH, SUMO_NET_PATH, SF_TAZ_COORDINATES_PATH)

mapping_df = pd.read_csv(SF_TAZ_COORDINATES_PATH, sep=";")
# Extract strongly connected edges
safe_edge_ids = get_strongly_connected_edges(SUMO_NET_PATH)
# Apply filters
mapping_df['polygon_edge_ids'] = mapping_df['polygon_edge_ids'].apply(
    lambda x: filter_polygon_edges(x, safe_edge_ids)
)
mapping_df = mapping_df[mapping_df['polygon_edge_ids'].map(lambda x: len(x) > 0)]

mapping_df['polygon_lane_ids'] = mapping_df['polygon_lane_ids'].apply(
    lambda x: filter_polygon_lanes(x, safe_edge_ids)
)
mapping_df = mapping_df[mapping_df['polygon_lane_ids'].map(lambda x: len(x) > 0)]
# Reconstruct the mapping dictionary
taz_edge_mapping = mapping_df.set_index('TAZ')[['polygon_edge_ids', 'polygon_lane_ids']].dropna().to_dict('index')

# 5.2. reading uber data
data = read_uber_stats_data(SF_RIDE_STATS_PATH, start_time, end_time)
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

# 5.3. generate taxi trips
number_vehicles_available = 1000
generate_drt_vehicle_instances_from_lanes(
    lane_ids=random.sample(start_lanes, number_vehicles_available),
    output_path="fleet_vehicles.rou.xml"
)

# 5.4. generate matched request file
valid_edge_ids = get_valid_taxi_edges(SUMO_NET_PATH, safe_edge_ids)
filtered_valid_edges = [e for e in valid_edge_ids if e in safe_edge_ids]

generate_matched_drt_requests(
    uber_data=data,
    taz_edge_mapping=taz_edge_mapping,
    sim_start_s=0,
    sim_end_s=3600,
    valid_edge_ids=filtered_valid_edges,
    output_path="passenger_requests.rou.xml")

output_dir_path = sumoSimulator.configure_output_dir(sf_traffic_routes_folder_path=SUMO_BASE_SCENARIO_FOLDER_PATH,
                                                   route_file_path=SF_TRAFFIC_ROUTE_PATH, taxi_route_file_path = "fleet_vehicles.rou.xml",
                                                   passenger_file_path = "passenger_requests.rou.xml", date=date,
                                                   start_time=start_time,end_time=end_time)
sumoSimulator.generate_config()
sumoSimulator.run_simulation(activeGui=True)
