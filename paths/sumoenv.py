import os.path
from pathlib import Path

##### PROJECT ABSOLUTE PATH #####
path = Path(os.path.abspath(__file__))
projectPath = str(path.parent.parent.absolute())

# Path to be changed with the local path containing Eclipse SUMO
SUMO_HOME_PATH = "/opt/homebrew/opt/sumo/share/sumo"
SUMO_TOOLS_LOCAL_PATH = SUMO_HOME_PATH + "/tools"
SUMO_BIN_PATH = SUMO_HOME_PATH + "/bin"

SUMOENV_PATH = projectPath + "/sumoenv"

SUMO_NET_PATH = SUMOENV_PATH + "/sf.net.xml"
SUMO_POLY_PATH = SUMOENV_PATH + "/taz_zones.poly.xml"
SUMO_CFGTEMPLATE_PATH = SUMOENV_PATH + "/sumocfg_template.sumocfg"
SUMO_TAZ_PATH = SUMOENV_PATH + "/sf.taz.xml"
SUMO_SCENARIOS_PATH = SUMOENV_PATH + "/scenarios"