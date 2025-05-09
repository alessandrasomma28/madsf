# SF Digital Mirror

This is a work-in-progress project, feel free to report any inconsistency or bug.

## Description

This repo is composed of 5 folders, a *main.py* and a requirements file.

- `classes/`: contains simulation and multi-agent logic.
- `constants/`: contains paths for better readability.
- `data/`: contains all input data for the simulator. Be sure to have your local copy since git does not allow push of files > 100MB.
- `libraries/`: contains all the utility functions to generate the input for the simulation.
- `sumoenv/`: contains output folder, the SF net files, and the sumocfg file, which is automatically generated in *main.py*.

## How-to-run instructions

1. Install [**SUMO**](https://sumo.dlr.de/docs/Downloads.php) and set [**SUMO_HOME**](https://sumo.dlr.de/docs/Basics/Basic_Computer_Skills.html#sumo_home) environment variable.
2. Open project and create a virtual environment with Python >=3.10.
3. Change paths indicated in `constants/sumoenv_constants.py`.
4. Install requirements from *requirements.txt* file.
5. Run simulation by simply executing *main.py*. If you don't want to use the GUI, simply set `activeGui=False` in *main.py*.
6. Inspect output folder `sumoenv/scenario/normal`.