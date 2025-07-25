import os.path
from pathlib import Path

##### PROJECT ABSOLUTE PATH #####
path = Path(os.path.abspath(__file__))
projectPath = str(path.parent.parent.absolute())

DATA_FOLDER_PATH = projectPath + "/data"
DOC_FOLDER_PATH = projectPath + "/doc"

SF_TRAFFIC_FOLDER_PATH = DATA_FOLDER_PATH + "/sf_traffic"

SF_TRAFFIC_VEHICLE_DAILY_FOLDER_PATH = SF_TRAFFIC_FOLDER_PATH + "/daily_traffic"

SF_TRAFFIC_SFMTA_FOLDER_PATH = SF_TRAFFIC_FOLDER_PATH + "/sfmta_dataset"
SF_TRAFFIC_MAP_MATCHED_FOLDER_PATH = SF_TRAFFIC_FOLDER_PATH + "/map_matched"

SF_TAZ_FOLDER_PATH = DATA_FOLDER_PATH + "/sf_zones"
SF_TAZ_SHAPEFILE_PATH = SF_TAZ_FOLDER_PATH + "/TAZ981.shp"
SF_SFCTA_GEO_PATH = SF_TAZ_FOLDER_PATH + "/sf_sfcta_taz_boundary.geojson"
SF_STANFORD_GEO_PATH = SF_TAZ_FOLDER_PATH + "/sf_stanford_taz_boundary.geojson"
SF_SFCTA_STANFORD_MAPPING_PATH = SF_TAZ_FOLDER_PATH + "/sf_sfcta_stanford_mapping.json"
SF_TAZ_EDGE_MAPPING_PATH = SF_TAZ_FOLDER_PATH + "/sf_taz_to_edges.json"

SF_RIDE_FOLDER_PATH = DATA_FOLDER_PATH + "/ridehailing_stats"
SF_RIDE_STATS_PATH = SF_RIDE_FOLDER_PATH + "/trip_stats_taz.csv"
SF_TAZ_COORDINATES_PATH = SF_RIDE_FOLDER_PATH + "/taz_coordinates_centroids.csv"

SF_TRAFFIC_BASE_URL = "https://data.sfgov.org/resource/x344-v6h6.csv"