# SF Digital Mirror

This is a work-in-progress project, feel free to report any inconsistency or bug.


## Why 3 branches?

This repo has 3 branches because they refer to 3 different implementation:

1. `main`. Implementation of standard taxi service with just *SUMO* logic.
2. `multi-agent`. Implementation of multi-agent based ride-hailing service.
3. `multi-agent-social-groups`. Implementation of multi-agent based ride-hailing service with social groups and consequent handling of accept/reject requests.

So, `multi-agent-social-groups` branch is actually the most recent one, used to complete all the functionalities of the digital mirror. We can consider the other 2 branches `main` and `multi-agent` as completed, and they will serve for comparison purposes. 


## Description

This repo is composed of 7 folders, a *main.py*, a *generate_output_csv.py* and a *requirements.txt* file.

- `classes/`: contains simulation and multi-agent logic.
- `constants/`: contains paths for better readability.
- `data/`: contains all input data for the simulator.
- `libraries/`: contains all the utility functions to generate the input for the simulation.
- `sumoenv/`: contains the output folder, the SF net files, and the `sumocfg` file, which is automatically generated in *main.py*.
- `config/`: contains `.json` files to configurate providers, personalities and acceptances.
- `doc/`: contains additional documentation of the project.


## How-to-run instructions

1. Install [**SUMO**](https://sumo.dlr.de/docs/Downloads.php) and set [**SUMO_HOME**](https://sumo.dlr.de/docs/Basics/Basic_Computer_Skills.html#sumo_home) environment variable. For MacOS users, prefer installation via **Homebrew**. To run the simulation with the GUI, install *SUMO* follow the instructions [here](https://github.com/DLR-TS/homebrew-sumo), then run [*XQuartz*](https://www.xquartz.org/) in background (MacOS users).
2. Open project (`cd/path/to/this/project`), create a virtual environment with Python >=3.10 (`python3.10 -m venv venv`) and activate it (`source venv/bin/activate`).
3. If needed, change paths indicated in `constants/sumoenv_constants.py`.
4. Install requirements from *requirements.txt* file (`pip install -r requirements.txt`).
5. Run simulation by simply executing *main.py*. If you don't want to use the GUI, simply set `activeGui=False` in *main.py*.
6. Follow the instructions printed in the command line to initialize the simulation.
7. Inspect output folder `sumoenv/scenario/{scenario_name}`. The final CSV metrics file will be saved as `sf_final_metrics.csv` and can be inspected with the HTML interactive line plot visualization `sf_final_metrics_visualization.html`.


## Evolvability

The project is designed to be easily extensible and adaptable to different scenarios. The multi-agent model can be easily modified to include new features, without acting directly on the agents. One can simply modify the `.json` files in the `config/` folder to change the parameters of the agents (such as different acceptance probability distribution) or add other components (such as other different ride-hailing providers).


## Classes

- `Simulator`: Creates the *SUMO* configuration and initializes the multi-agent model.
- `Model`: Initializes and computes steps for the agents while running the simulation.
- `Passengers`: Manages unassigned ride requests and interacts with the ride service to accept/reject offers from drivers.
- `Drivers`: Manages idle drivers and interacts with the ride service to accept/reject offers from passengers.
- `RideServices`: Manages ride offers and acceptances between passengers and drivers. When both passenger and driver accept the offer, the ride is dispatched.


## Architecture

All the input files for the simulation are automatically generated in `main.py`. Input files include:
- `sf.net.xml`: *SUMO* network file (**map**) of San Francisco.
- `sf_routes_{date_time}.rou.xml`: *SUMO* routes file for the **traffic** of San Francisco. It contains the trip information for each traffic vehicle.
- `sf_tnc_fleet_{date_time}.rou.xml`: *SUMO* routes file for the **fleet** of TNC drivers of San Francisco. It contains the starting point, the time of departure and the end of the shift for each driver.
- `sf_tnc_passengers_{date_time}.rou.xml`: *SUMO* routes file for the **passengers** ride requests of San Francisco. It contains the time of request, the starting point and the destination for each passenger.

Additionally, `main.py` generates the *SUMO* configuration file `sumo_config.sumocfg`, which contains the simulation parameters and the input files to be used.

The simulation is then run with a real-time interaction between *SUMO* and the multi-agent model. The *SUMO* logic handles the traffic and the routing of all the vehicles, while the multi-agent model handles the ride-hailing service. The two components interact with each other through [*TraCI*](https://sumo.dlr.de/docs/TraCI.html).

![](doc/General_architecture.png)

### Multi-agent model
The multi-agent model is composed of three main components: `Passengers`, `Drivers` and `RideServices`. The logic of the multi-agent model is run every 60 timestamps by default, but it can be changed in the `main.py` file. 

![](doc/Multi-agent_architecture.png)

For each step of the multi-agent logic, *SUMO* logic stops. `Passengers` and `Drivers` are then updated with the current state of the simulation by interacting with *SUMO*. This includes the time of the simulation, the unassigned ride requests, and the driver currently available. Then, for each request, the following actions are performed:
1. `Passengers` ask for a ride to `RideServices`, which forwards the request to `Drivers`.
2. `Drivers` receive the ride request and returns the list of available drivers to `RideServices`.
3. `RideServices` receive the list of available drivers and selects the 8 closest drivers to the request. Then, it generates a ride offer for each driver, containing information about the ride (e.g., time, distance, price, etc.) and sends these 8 offers to `Passengers`.
4. `Passengers` receive the ride offers and select the best one according to the acceptance probability distribution of the social group the passenger belongs to. 
    - If the passenger rejects all the offers, the request is rejected. If the request has exceeded the maximum waiting time, the passenger is removed from the simulation, otherwise the request is kept in the list of unassigned requests. 
    - If the passenger accepts an offer, `Passengers` notify `RideServices`, which forwards the acceptance to `Drivers`.
5. `Drivers` receive the accepted offer and accept/reject it according to the acceptance probability distribution of the social group the driver belongs to. 
    - If the driver rejects the offer, the request is rejected and the driver is marked as unavailable until the next step of the multi-agent logic. The request is kept in the list of unassigned requests.
    - If the driver accepts the offer, `Drivers` notifies `RideServices`, which finally dispatches the ride using the *TraCI* method `dispatchTaxi(request_id, driver_id)`.

When all the requests are processed, the *SUMO* logic is resumed, until the next step of the multi-agent logic.