# SF Digital Mirror

This is a work-in-progress project, feel free to report any inconsistency or bug.

## Why 3 branches?

This repo has 3 branches because they refer to 3 different implementation:

1. `main`. Implementation of standard taxi service with just SUMO logic.
2. `multi-agent`. Implementation of multi-agent based ride-hailing service.
3. `multi-agent-social-groups`. Implementation of multi-agent based ride-hailing service with social groups and consequent handling of accept/reject requests.

So, `multi-agent-social-groups` branch is actually the most recent one, used to complete all the functionalities of the digital mirror. We can consider the other 2 branches `main` and `multi-agent` as completed, and they will serve for comparison purposes. 

## Description

This repo is composed of 6 folders, a *main.py* and a requirements file.

- `classes/`: contains simulation and multi-agent logic.
- `constants/`: contains paths for better readability.
- `data/`: contains all input data for the simulator. **Be sure to have your local copy since git does not allow push of files > 100MB**.
- `libraries/`: contains all the utility functions to generate the input for the simulation.
- `sumoenv/`: contains output folder, the SF net files, and the sumocfg file, which is automatically generated in *main.py*.
- `config/`: contains json files to configurate providers, personalities and acceptances.

## How-to-run instructions

1. Install [**SUMO**](https://sumo.dlr.de/docs/Downloads.php) and set [**SUMO_HOME**](https://sumo.dlr.de/docs/Basics/Basic_Computer_Skills.html#sumo_home) environment variable.
2. Open project and create a virtual environment with Python >=3.10.
3. Change paths indicated in `constants/sumoenv_constants.py`.
4. Install requirements from *requirements.txt* file.
5. Run simulation by simply executing *main.py*. If you don't want to use the GUI, simply set `activeGui=False` in *main.py*.
6. Inspect output folder `sumoenv/scenario/normal`.