from libraries.sumo_utils import sf_traffic_map_matching, sf_traffic_od_generation, sf_traffic_routes_generation, \
    add_sf_traffic_taz_matching, convert_shapefile_to_sumo_poly_with_polyconvert
from constants.data_constants import (SF_TRAFFIC_MAP_MATCHED_FOLDER_PATH, SF_TRAFFIC_VEHICLE_DAILY_FOLDER_PATH,
                                      SF_TRAFFIC_VEHICLE_ONEWEEK_PATH, SF_TAZ_SHAPEFILE_PATH)
from constants.sumoenv_constants import (SUMOENV_PATH, SUMO_NET_PATH, SUMO_BASE_SCENARIO_FOLDER_PATH,
                                         SUMO_CFGTEMPLATE_PATH, SUMO_POLY_PATH)
from libraries.data_utils import convert_sf_traffic_csv_format, extract_sf_traffic_timeslot
from classes.simulator import Simulator

date = "2021-01-14"
start_time = "9:00"
end_time = "10:00"
radius = 150
sumoSimulator = Simulator(net_file=SUMO_NET_PATH, config_template_path=SUMO_CFGTEMPLATE_PATH, taz_file_path=SUMO_POLY_PATH)

# 1. reading the 10-16 traffic vehicle data set
# TODO: ADJUST CONVERT FUNCTION TO DO LIKE MAP MATCHING: taking a folder path and returning a file path.
#convert_vehicle_csv_format(SF_TRAFFIC_VEHICLE_ONEWEEK_PATH, SF_TRAFFIC_VEHICLE_ONEWEEK_PATH)

SF_TRAFFIC_VEHICLE_DAILYHOUR_PATH = extract_sf_traffic_timeslot(input_csv_path=SF_TRAFFIC_VEHICLE_ONEWEEK_PATH,
                                                                date_str=date,
                                                                start_time_str=start_time,end_time_str=end_time,
                                                                output_csv_folder=SF_TRAFFIC_VEHICLE_DAILY_FOLDER_PATH)

# MAP and TAZ matching
SF_TRAFFIC_EDGE_PATH= sf_traffic_map_matching(sf_map_file_path=SUMO_NET_PATH,
                                              sf_real_traffic_data_path=SF_TRAFFIC_VEHICLE_DAILYHOUR_PATH,
                                              date=date, output_folder_path=SF_TRAFFIC_MAP_MATCHED_FOLDER_PATH, radius=150)
add_sf_traffic_taz_matching(SF_TRAFFIC_EDGE_PATH, SF_TAZ_SHAPEFILE_PATH)

# 2. generate ORIGIN-DESTINATION file for each vehicle
SF_TRAFFIC_0D_PATH= sf_traffic_od_generation(sf_real_traffic_edge_path=SF_TRAFFIC_EDGE_PATH,
                         sf_traffic_od_folder_path=SUMO_BASE_SCENARIO_FOLDER_PATH, date=date,
                         start_time=start_time, end_time=end_time)

# 3. generate ROUTES file
SF_TRAFFIC_ROUTE_PATH = sf_traffic_routes_generation(sf_traffic_od_path=SF_TRAFFIC_0D_PATH,
                                                     sf_traffic_routes_folder_path=SUMO_BASE_SCENARIO_FOLDER_PATH,
                                                     date=date, start_time=start_time, end_time=end_time)



output_dir_path=sumoSimulator.configure_output_dir(sf_traffic_routes_folder_path=SUMO_BASE_SCENARIO_FOLDER_PATH,
                                                   route_file_path=SF_TRAFFIC_ROUTE_PATH, date=date,
                                                   start_time=start_time,end_time=end_time)
sumoSimulator.generate_config()
sumoSimulator.run_simulation(activeGui=True)
