import os.path
from pathlib import Path

##### PROJECT ABSOLUTE PATH #####
path = Path(os.path.abspath(__file__))
projectPath = str(path.parent.parent.absolute())

DATA_FOLDER_PATH = projectPath + "/data"

SF_TRAFFIC_FOLDER_PATH = DATA_FOLDER_PATH +"/sf_traffic"

SF_TRAFFIC_VEHICLE_DAILY_FOLDER_PATH = SF_TRAFFIC_FOLDER_PATH + "/daily_traffic"

SF_TRAFFIC_SFMTA_FOLDER_PATH = SF_TRAFFIC_FOLDER_PATH + "/sfmta_dataset"
SF_TRAFFIC_VEHICLE_ONEWEEK_PATH = SF_TRAFFIC_SFMTA_FOLDER_PATH + "/sf_vehicle_210110_210116.csv"
SF_TRAFFIC_VEHICLE_EXAMPLE_PATH = SF_TRAFFIC_SFMTA_FOLDER_PATH + "/sf_vehicle_010121_0001.csv"

SF_TRAFFIC_MAP_MATCHED_FOLDER_PATH = SF_TRAFFIC_FOLDER_PATH + "/map_matched"

SF_TAZ_FOLDER_PATH = DATA_FOLDER_PATH + "/sf_zones"
SF_TAZ_SHAPEFILE_PATH = SF_TAZ_FOLDER_PATH + "/TAZ981.shp"


SF_RIDE_FOLDER_PATH = DATA_FOLDER_PATH + "/ridehailing_stats"
SF_RIDE_STATS_PATH = SF_RIDE_FOLDER_PATH + "/trip_stats_taz.csv"
SF_TAZ_COORDINATES_PATH = SF_RIDE_FOLDER_PATH + "/taz_coordinates_centroids.csv"



