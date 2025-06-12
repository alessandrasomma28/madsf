import os.path
from pathlib import Path

##### PROJECT ABSOLUTE PATH #####
path = Path(os.path.abspath(__file__))
projectPath = str(path.parent.parent.absolute())

CONFIG_FOLDER_PATH = projectPath + "/config"

DRIVERS_PERSONALITY = CONFIG_FOLDER_PATH + "/drivers_personality_distribution.json"
DRIVERS_ACCEPTANCE = CONFIG_FOLDER_PATH + "/drivers_acceptance_distribution.json"
PASSENGERS_PERSONALITY = CONFIG_FOLDER_PATH + "/passengers_personality_distribution.json"
PASSENGERS_ACCEPTANCE = CONFIG_FOLDER_PATH + "/passengers_acceptance_distribution.json"
PROVIDERS_CONFIG = CONFIG_FOLDER_PATH + "/providers_config.json"
TIMEOUT_CONFIG = CONFIG_FOLDER_PATH + "/timeout_config.json"