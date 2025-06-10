import os
from datetime import datetime
from pathlib import Path
from classes.simulator import Simulator
from constants.data_constants import (SF_TRAFFIC_MAP_MATCHED_FOLDER_PATH, SF_RIDE_STATS_PATH, SF_TAZ_SHAPEFILE_PATH,
                                      SF_TRAFFIC_VEHICLE_DAILY_FOLDER_PATH, SF_TAZ_COORDINATES_PATH)
from constants.sumoenv_constants import (SUMO_NET_PATH, SUMO_SCENARIOS_PATH, SUMO_CFGTEMPLATE_PATH, SUMO_POLY_PATH)
from libraries.io_utils import get_valid_date, get_valid_hour, get_valid_scenario, get_valid_str, get_or_prompt, \
    generate_output_csv, load_env, save_to_env, get_valid_mode
from libraries.data_utils import extract_sf_traffic_timeslot, read_tnc_stats_data, check_import_traffic
from libraries.sumo_utils import sf_traffic_map_matching, sf_traffic_od_generation, sf_traffic_routes_generation, \
    export_taz_coords, map_coords_to_sumo_edges, get_strongly_connected_edges, generate_matched_drt_requests, \
    add_sf_traffic_taz_matching, generate_vehicle_start_lanes_from_taz_polygons, generate_drt_vehicle_instances_from_lanes, \
    get_valid_taxi_edges, map_taz_to_edges, compute_requests_vehicles_ratio


# 0. Set initial variables and initialize Simulator class
if not os.path.exists(Path(".env")):
    for var in ["START_DATE", "END_DATE", "START_TIME", "END_TIME", "SCENARIO", "MODE", "ACTIVE_GUI", "VERBOSE"]:
        if var in os.environ:
            del os.environ[var]
print("\n✨ Welcome to the SF Ride-Hailing Digital Mirror Setup! ✨\n")
load_env(override=True)
start_date = get_or_prompt("START_DATE", lambda: get_valid_date("⚙️  Enter simulation start date (MM-DD, between 01-01 and 12-30): "))
start_date_prompt = start_date[5:]
end_date = get_or_prompt("END_DATE", lambda: get_valid_date(f"⚙️  Enter simulation end date (MM-DD, between {start_date_prompt} and 12-30): "))
while datetime.strptime(end_date, "%Y-%m-%d") < datetime.strptime(start_date, "%Y-%m-%d"):
    print("⚠️  End date must be after or equal to start date")
    end_date = get_valid_date(f"⚙️  Enter simulation end date (MM-DD, between {start_date_prompt} and 12-30): ")
    save_to_env("END_DATE", end_date)
if start_date == end_date:
    start_time = get_or_prompt("START_TIME", lambda: get_valid_hour("⚙️  Enter simulation start hour (0-22): ", start_hour_same_day_check=True))
    end_time = get_or_prompt("END_TIME", lambda: get_valid_hour(f"⚙️  Enter simulation hour ({int(start_time[:-3]) + 1}-23): "))
    while int(end_time.split(":")[0]) <= int(start_time.split(":")[0]):
        print("⚠️  End hour must be after start hour")
        end_time = get_valid_hour(f"⚙️  Enter simulation hour ({int(start_time[:-3]) + 1}-23): ")
        save_to_env("END_TIME", end_time)
else:
    start_time = get_or_prompt("START_TIME", lambda: get_valid_hour("⚙️  Enter simulation start hour (0-23): "))
    end_time = get_or_prompt("END_TIME", lambda: get_valid_hour("⚙️  Enter simulation end hour (1-23): ", end_hour_check=True))
scenario = get_or_prompt("SCENARIO", lambda: get_valid_scenario("⚙️  Enter scenario name (normal): "))
mode = get_or_prompt("MODE", lambda: get_valid_mode("⚙️  Enter simulation mode (sumo, multi_agent, social_groups): "))
agents_interval = 60  # Default agents interval
activeGui = get_or_prompt("ACTIVE_GUI", lambda: get_valid_str("⚙️  Do you want to run the simulation with the GUI? (yes/no) ")) == "yes"
verboseMode = get_or_prompt("VERBOSE", lambda: get_valid_str("⚙️  Do you want to run the simulation in verbose mode? (yes/no) ")) == "yes"
SCENARIO_PATH = f"{SUMO_SCENARIOS_PATH}/{scenario}"
os.makedirs(SCENARIO_PATH, exist_ok=True)
radius = 200                        # Radius (meters))for map matching
n_start_lanes = 10                  # Number of possible start lanes for taxis in each TAZ
peak_vehicles = 5700                # Peak number of DRT vehicles in a day
max_vehicles = 45000                # Maximum number of drivers available in one day
# Dispatch algorithm to use (e.g., "traci", "greedy")
if mode in ["multi_agent", "social_groups"]:   
    dispatch_algorithm = "traci"
elif mode == "sumo":
    dispatch_algorithm = "greedy"      
idle_mechanism = "randomCircling"   # Idle mechanism to use (e.g., "randomCircling", "stop")
sumoSimulator = Simulator(
    net_file_path=SUMO_NET_PATH, 
    config_template_path=SUMO_CFGTEMPLATE_PATH, 
    taz_file_path=SUMO_POLY_PATH,
    verbose=verboseMode
    )
print("\n🚀 Computing input for the SF Ride-Hailing Digital Mirror...\n")

# 1. Import traffic data
SF_TRAFFIC_FILE_PATH = check_import_traffic(
    start_date_str=start_date,
    end_date_str=end_date,
    start_time_str=start_time,
    end_time_str=end_time
    )

# 2. Get safe edges (i.e., edges that are strongly connected)
safe_edge_ids = get_strongly_connected_edges(sf_map_file_path=SUMO_NET_PATH)

# 3. Read traffic vehicle data set
SF_TRAFFIC_VEHICLE_DAILYHOUR_PATH = extract_sf_traffic_timeslot(
    input_csv_path=SF_TRAFFIC_FILE_PATH,
    start_date_str=start_date,
    end_date_str=end_date,
    start_time_str=start_time,
    end_time_str=end_time,
    output_csv_folder=SF_TRAFFIC_VEHICLE_DAILY_FOLDER_PATH
    )

# 4. Map and TAZ matching - traffic
start = start_date.replace("-", "")
end = end_date.replace("-", "")
SF_TRAFFIC_EDGE_PATH = sf_traffic_map_matching(
    sf_map_file_path=SUMO_NET_PATH,
    sf_real_traffic_data_path=SF_TRAFFIC_VEHICLE_DAILYHOUR_PATH,
    date_str=f"{start}-{end}",
    output_folder_path=SF_TRAFFIC_MAP_MATCHED_FOLDER_PATH,
    radius=radius, 
    safe_edge_ids=safe_edge_ids
    )
add_sf_traffic_taz_matching(
    edge_file_path=SF_TRAFFIC_EDGE_PATH, 
    shapefile_path=SF_TAZ_SHAPEFILE_PATH
    )

# 5. Generate OD (origin-destination) file for each vehicle
SF_TRAFFIC_0D_PATH = sf_traffic_od_generation(
    sf_real_traffic_edge_path=SF_TRAFFIC_EDGE_PATH,
    sf_traffic_od_folder_path=SCENARIO_PATH,
    start_date_str=start_date, 
    end_date_str=end_date, 
    start_time_str=start_time, 
    end_time_str=end_time
    )

# 6. Generate routes file for traffic
SF_TRAFFIC_ROUTE_PATH = sf_traffic_routes_generation(
    sf_traffic_od_path=SF_TRAFFIC_0D_PATH,
    sf_traffic_routes_folder_path=SCENARIO_PATH,
    start_date_str=start_date, 
    end_date_str=end_date, 
    start_time_str=start_time, 
    end_time_str=end_time
    )

# 7. Map and TAZ matching - TNC
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

# 8. Read TNC data
data, previous_hour_data = read_tnc_stats_data(
    sf_rides_stats_path=SF_RIDE_STATS_PATH,
    start_date_str=start_date,
    end_date_str=end_date,
    start_time_str=start_time, 
    end_time_str=end_time
    )

# 9. Map TAZ polygons to lanes
start_lanes_by_taz = generate_vehicle_start_lanes_from_taz_polygons(
    shapefile_path=SF_TAZ_SHAPEFILE_PATH,
    net_file=SUMO_NET_PATH,
    points_per_taz=n_start_lanes,
    safe_edge_ids=safe_edge_ids
    )

# 10. Compute ratio of TNC requests to traffic vehicles
ratio_requests_vehicles = compute_requests_vehicles_ratio(
    sf_tnc_fleet_folder_path=SF_RIDE_STATS_PATH,
    peak_vehicles=peak_vehicles,
    max_total_drivers=max_vehicles
    )

# 11. Generate taxi trips
SF_TNC_FLEET_PATH = generate_drt_vehicle_instances_from_lanes(
    start_lanes_by_taz=start_lanes_by_taz,
    ratio_requests_vehicles=ratio_requests_vehicles,
    tnc_data=data,
    tnc_previous_hour_data=previous_hour_data,
    start_date_str=start_date,
    end_date_str=end_date,
    start_time_str=start_time, 
    end_time_str=end_time,
    sf_tnc_fleet_folder_path=SCENARIO_PATH,
    idle_mechanism=idle_mechanism
    )

# 12. Get valid edges for taxi routes
valid_edge_ids = get_valid_taxi_edges(
    net_file=SUMO_NET_PATH, 
    safe_edge_ids=safe_edge_ids
    )

# 13. Generate matched DRT requests
SF_TNC_REQUESTS_PATH = generate_matched_drt_requests(
    tnc_data=data,
    taz_edge_mapping=taz_edge_mapping,
    start_date_str=start_date,
    end_date_str=end_date,
    start_time_str=start_time, 
    end_time_str=end_time,
    valid_edge_ids=valid_edge_ids,
    sf_requests_folder_path=SCENARIO_PATH
    )

# 14. Configure output directory and run simulation
SF_OUTPUT_DIR_PATH = sumoSimulator.configure_output_dir(
    sf_routes_folder_path=SCENARIO_PATH,
    sf_traffic_route_file_path=SF_TRAFFIC_ROUTE_PATH,
    sf_tnc_fleet_file_path=SF_TNC_FLEET_PATH,
    sf_tnc_requests_file_path=SF_TNC_REQUESTS_PATH,
    start_date_str=start_date,
    end_date_str=end_date,
    start_time_str=start_time,
    end_time_str=end_time
    )

# 15. Generate SUMO configuration file
sumoSimulator.generate_config(
    dispatch_algorithm=dispatch_algorithm,
    idle_mechanism=idle_mechanism
    )

# 16. Run simulation
sumoSimulator.run_simulation(
    activeGui=activeGui,
    agents_interval=agents_interval,
    dispatch_algorithm=dispatch_algorithm,
    ratio_requests_vehicles=ratio_requests_vehicles,
    mode=mode
    )

# 17. Generate output CSV file
generate_output_csv(
    start_date_str=start_date,
    end_date_str=end_date,
    start_time_str=start_time,
    end_time_str=end_time,
    mode=mode,
    scenario=scenario
    )
