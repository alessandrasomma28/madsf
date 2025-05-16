from libraries.sumo_utils import sf_traffic_map_matching, sf_traffic_od_generation, sf_traffic_routes_generation, \
    export_taz_coords, map_coords_to_sumo_edges, get_strongly_connected_edges, generate_matched_drt_requests, \
    add_sf_traffic_taz_matching, generate_vehicle_start_lanes_from_taz_polygons, generate_drt_vehicle_instances_from_lanes, \
    get_valid_taxi_edges, map_taz_to_edges
from constants.data_constants import (SF_TRAFFIC_MAP_MATCHED_FOLDER_PATH, SF_TRAFFIC_VEHICLE_DAILY_FOLDER_PATH,
                                      SF_TRAFFIC_VEHICLE_ONEWEEK_PATH, SF_TAZ_SHAPEFILE_PATH, SF_TAZ_COORDINATES_PATH,
                                      SF_RIDE_STATS_PATH)
from constants.sumoenv_constants import (SUMO_NET_PATH, SUMO_BASE_SCENARIO_FOLDER_PATH, SUMO_CFGTEMPLATE_PATH, SUMO_POLY_PATH)
from libraries.data_utils import extract_sf_traffic_timeslot, read_tnc_stats_data
from classes.simulator import Simulator
import random


date = "2021-01-14"
start_time = "9:00"
end_time = "10:00"
radius = 150
sumoSimulator = Simulator(
    net_file=SUMO_NET_PATH, 
    config_template_path=SUMO_CFGTEMPLATE_PATH, 
    taz_file_path=SUMO_POLY_PATH
    )

# 0. Get safe edges (i.e., edges that are strongly connected)
safe_edge_ids = get_strongly_connected_edges(sf_map_file_path=SUMO_NET_PATH)

# 1. Read traffic vehicle data set
SF_TRAFFIC_VEHICLE_DAILYHOUR_PATH = extract_sf_traffic_timeslot(
    input_csv_path=SF_TRAFFIC_VEHICLE_ONEWEEK_PATH,
    date_str=date,
    start_time_str=start_time,
    end_time_str=end_time,
    output_csv_folder=SF_TRAFFIC_VEHICLE_DAILY_FOLDER_PATH
    )

# 2. MAP and TAZ matching - traffic
SF_TRAFFIC_EDGE_PATH = sf_traffic_map_matching(
    sf_map_file_path=SUMO_NET_PATH,
    sf_real_traffic_data_path=SF_TRAFFIC_VEHICLE_DAILYHOUR_PATH,
    date_str=date, 
    output_folder_path=SF_TRAFFIC_MAP_MATCHED_FOLDER_PATH,
    radius=radius, 
    safe_edge_ids=safe_edge_ids
    )
add_sf_traffic_taz_matching(
    edge_file_path=SF_TRAFFIC_EDGE_PATH, 
    shapefile_path=SF_TAZ_SHAPEFILE_PATH
    )

# 3. Generate ORIGIN-DESTINATION file for each vehicle
SF_TRAFFIC_0D_PATH = sf_traffic_od_generation(
    sf_real_traffic_edge_path=SF_TRAFFIC_EDGE_PATH,
    sf_traffic_od_folder_path=SUMO_BASE_SCENARIO_FOLDER_PATH,
    date_str=date, 
    start_time_str=start_time, 
    end_time_str=end_time
    )

# 4. Generate ROUTES file for traffic
SF_TRAFFIC_ROUTE_PATH = sf_traffic_routes_generation(
    sf_traffic_od_path=SF_TRAFFIC_0D_PATH,
    sf_traffic_routes_folder_path=SUMO_BASE_SCENARIO_FOLDER_PATH,
    date_str=date, 
    start_time_str=start_time, 
    end_time_str=end_time
    )

# 5. MAP and TAZ matching - TNC
export_taz_coords(
    shapefile_path=SF_TAZ_SHAPEFILE_PATH, 
    output_csv_path=SF_TAZ_COORDINATES_PATH
    )
map_coords_to_sumo_edges(
    taz_csv_path=SF_TAZ_COORDINATES_PATH,
    net_xml_path=SUMO_NET_PATH, 
    output_csv_path=SF_TAZ_COORDINATES_PATH
    )
taz_edge_mapping = map_taz_to_edges(
    taz_csv_path=SF_TAZ_COORDINATES_PATH,
    safe_edge_ids=safe_edge_ids
    )

# 6. Read TNC data
data = read_tnc_stats_data(
    sf_rides_stats_path=SF_RIDE_STATS_PATH,
    start_time_str=start_time, 
    end_time_str=end_time
    )

# 7. Map TAZ polygons to lanes
points_taz = 3   # 3 start lanes per TAZ → ~2800 total
start_lanes = generate_vehicle_start_lanes_from_taz_polygons(
    shapefile_path=SF_TAZ_SHAPEFILE_PATH,
    net_file=SUMO_NET_PATH,
    points_per_taz=points_taz,
    safe_edge_ids=safe_edge_ids
)

# 8. Generate taxi trips
number_vehicles_available = 2000
SF_TNC_FLEET_PATH = generate_drt_vehicle_instances_from_lanes(
    lane_ids=random.sample(start_lanes, number_vehicles_available),
    date_str=date,
    start_time_str=start_time, 
    end_time_str=end_time,
    sf_tnc_fleet_folder_path=SUMO_BASE_SCENARIO_FOLDER_PATH
)

# 9. Get valid edges for taxi routes
valid_edge_ids = get_valid_taxi_edges(
    net_file=SUMO_NET_PATH, 
    safe_edge_ids=safe_edge_ids
    )

# 10. Generate matched DRT requests
SF_TNC_REQUESTS_PATH = generate_matched_drt_requests(
    tnc_data=data,
    taz_edge_mapping=taz_edge_mapping,
    date_str=date,
    start_time_str=start_time, 
    end_time_str=end_time,
    valid_edge_ids=valid_edge_ids,
    sf_requests_folder_path=SUMO_BASE_SCENARIO_FOLDER_PATH
    )

# 11. Configure output directory and run simulation
SF_OUTPUT_DIR_PATH = sumoSimulator.configure_output_dir(
    sf_routes_folder_path=SUMO_BASE_SCENARIO_FOLDER_PATH,
    sf_traffic_route_file_path=SF_TRAFFIC_ROUTE_PATH,
    sf_tnc_fleet_file_path=SF_TNC_FLEET_PATH,
    sf_tnc_requests_file_path=SF_TNC_REQUESTS_PATH,
    date_str=date,
    start_time_str=start_time,
    end_time_str=end_time
    )

# 12. Generate SUMO configuration file
sumoSimulator.generate_config()

# 13. Run simulation
sumoSimulator.run_simulation(activeGui=True)