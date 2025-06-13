import os.path
from pathlib import Path

##### PROJECT ABSOLUTE PATH #####
path = Path(os.path.abspath(__file__))
projectPath = str(path.parent.parent.absolute())

CONFIG_FOLDER_PATH = projectPath + "/config"

ZIP_ZONES_CONFIG_PATH = CONFIG_FOLDER_PATH + "/zip_zones_config.json"
SCENARIOS_CONFIG_PATH = CONFIG_FOLDER_PATH + "/scenarios_config.json"
PARAMETERS_CONFIG_PATH = CONFIG_FOLDER_PATH + "/parameters_config.json"