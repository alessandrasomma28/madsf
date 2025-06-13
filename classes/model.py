"""
model.py

This module defines the Model class, which serves as the core simulation orchestration.
It initializes and computes steps for the agents: Drivers, Passengers, and RideServices.
"""


from pathlib import Path
import time
import json
import os
import sys
sys.path.append(os.path.join(os.environ["SUMO_HOME"], 'tools'))
import traci
from classes.rideservices import RideServices
from classes.passengers import Passengers
from classes.drivers import Drivers
from classes.logger import Logger
from paths.config import PARAMETERS_CONFIG_PATH

class Model:
    sumocfg_path: str
    config_path: str
    end_time: int
    passengers: Passengers
    drivers: Drivers
    rideservices: RideServices
    logger: Logger
    output_dir_path: str
    time: int
    agents_interval: int
    providers: list
    verbose: bool
    ratio_requests_vehicles: float
    mode: str
    scenario: str

    def __init__(
            self,
            sumocfg_path: str,
            end_time: int,
            output_dir_path: str,
            verbose: bool = False,
            ratio_requests_vehicles: float = 1.0,
            mode: str = "social_groups",
            scenario: str = "normal"
        ):
        # Initialize the Model class with the configuration parameters.
        with open(Path(PARAMETERS_CONFIG_PATH), "r") as f:
            self.default_config = json.load(f)
            self.providers = self.default_config["providers"]
            self.drivers_personality_distribution = self.default_config["drivers_personality_distribution"]
            self.drivers_personality_distribution = self.default_config["drivers_personality_distribution"]
            self.drivers_acceptance_distribution = self.default_config["drivers_acceptance_distribution"] 
            self.passengers_personality_distribution = self.default_config["passengers_personality_distribution"]
            self.passengers_acceptance_distribution = self.default_config["passengers_acceptance_distribution"]
            self.drivers_stop_probability = self.default_config["drivers_stop_probability"]
            timeouts = self.default_config["timeouts"]
            self.timeout_p = timeouts["passenger"]
            self.timeout_d = timeouts["driver"]
        self.sumocfg_path = sumocfg_path
        self.end_time = end_time
        self.output_dir_path = output_dir_path
        self.verbose = verbose
        self.ratio_requests_vehicles = ratio_requests_vehicles
        self.logger = Logger(
            self,
            output_dir_path=output_dir_path
            )
        self.passengers = Passengers(
            self,
            logger=self.logger
            )
        self.drivers = Drivers(
            self,
            logger=self.logger
            )
        self.rideservices = RideServices(
            self,
            logger=self.logger
            )
        self.time = 0
        self.agents_interval = 0
        self.mode = mode
        if self.mode == "social_groups":
            self.use_social_groups = True
        else:
            self.use_social_groups = False
        # Check for scenario injection
        full_scenario_config_path = os.path.join(self.output_dir_path, "scenario_parameters_config.json")
        if os.path.exists(full_scenario_config_path):
            with open(Path(full_scenario_config_path), "r") as f:
                self.scenario_config = json.load(f)
        else:
            self.scenario_config = None


    def run(
            self,
            agents_interval: int = 60
        ):
        """
        Runs the simulation with the sumocfg previously generated.

        This function:
        - Performs simulation steps.
        - Handles ride hailing agents every {agents_interval} timestamps.
        - Stops when there are no active persons and vehicles available or when the simulation time exceeds end time plus one hour.

        Parameters
        ----------
        agents_interval: int
            Interval (timestamps) for agents execution.

        Returns
        -------
        None
        """
        self.agents_interval = agents_interval
        self.sumo_time = 0
        self.agents_time = 0
        # If scenario injection is enabled
        if self.scenario_config:
            self.scenario_start = self.scenario_config["trigger_time"]
            self.duration_time = self.scenario_config["duration_time"]
            self.scenario_end = self.scenario_config["trigger_time"] + self.scenario_config["duration_time"]
            while (len(traci.person.getTaxiReservations(3)) > 0 and traci.simulation.getMinExpectedNumber() > 0) or traci.simulation.getTime() < self.end_time + 3600:
                if self.time == self.scenario_start:
                    # Change params
                    self.update_scenario_parameters(self, self.scenario_config)
                    print(f"🚨 Scenario activated! Time: {self.time} seconds")
                if self.time == self.scenario_end:
                    # Restore params
                    self.update_scenario_parameters(self, self.default_config)
                    print(f"🚨 Scenario ended! Time: {self.time} seconds")        
                start_sumo = time.time()
                traci.simulationStep()
                self.time = int(traci.simulation.getTime())
                end_sumo = time.time()
                self.sumo_time += (end_sumo - start_sumo)
                if self.time % agents_interval == 0:
                    start_agents = time.time()
                    self.agents_time 
                    print(f"Simulation time: {self.time} seconds\n")
                    start = time.time()
                    self.passengers.step()
                    end = time.time()
                    print(f"⏱️  Passengers step computed in {round((end - start), 2)} seconds\n")
                    start = time.time()
                    self.drivers.step()
                    end = time.time()
                    print(f"⏱️  Drivers step computed in {round((end - start), 2)} seconds\n")
                    start = time.time()
                    self.rideservices.step()
                    end = time.time()
                    print(f"⏱️  RideServices step computed in {round((end - start), 2)} seconds\n")
                    end_agents = time.time()
                    self.agents_time += (end_agents - start_agents)
            print("✅ Simulation finished!")
            print(f"⏱️  Total SUMO time: {self.sumo_time:.2f} seconds")
            print(f"⏱️  Total agents time: {self.agents_time:.2f} seconds")
            return (self.sumo_time, self.agents_time)
        else:
            # If no scenario injection, run the simulation normally
            while (len(traci.person.getTaxiReservations(3)) > 0 and traci.simulation.getMinExpectedNumber() > 0) or traci.simulation.getTime() < self.end_time + 3600:
                start_sumo = time.time()
                traci.simulationStep()
                self.time = int(traci.simulation.getTime())
                end_sumo = time.time()
                self.sumo_time += (end_sumo - start_sumo)
                if self.time % agents_interval == 0:
                    start_agents = time.time()
                    self.agents_time 
                    print(f"Simulation time: {self.time} seconds\n")
                    start = time.time()
                    self.passengers.step()
                    end = time.time()
                    print(f"⏱️  Passengers step computed in {round((end - start), 2)} seconds\n")
                    start = time.time()
                    self.drivers.step()
                    end = time.time()
                    print(f"⏱️  Drivers step computed in {round((end - start), 2)} seconds\n")
                    start = time.time()
                    self.rideservices.step()
                    end = time.time()
                    print(f"⏱️  RideServices step computed in {round((end - start), 2)} seconds\n")
                    end_agents = time.time()
                    self.agents_time += (end_agents - start_agents)
            print("✅ Simulation finished!")
            print(f"⏱️  Total SUMO time: {self.sumo_time:.2f} seconds")
            print(f"⏱️  Total agents time: {self.agents_time:.2f} seconds")
            return (self.sumo_time, self.agents_time)
    

    def update_scenario_parameters(
            self,
            parameters_config: dict
        ):
        """
        Updates the model parameters based on the provided configuration.

        Parameters
        ----------
        parameters_config: dict
            A dictionary containing the configuration parameters for the model.

        Returns
        -------
        None
        """
        # Update the parameters based on the provided configuration
        for provider in parameters_config["providers"]:
            surge_multiplier = RideServices.get_surge_multiplier(provider)
            parameters_config["providers"][provider]["surge_multiplier"] = surge_multiplier
        self.providers = parameters_config["providers"]
        self.drivers_personality_distribution = parameters_config["drivers_personality_distribution"]
        self.drivers_acceptance_distribution = parameters_config["drivers_acceptance_distribution"]
        self.passengers_personality_distribution = parameters_config["passengers_personality_distribution"]
        self.passengers_acceptance_distribution = parameters_config["passengers_acceptance_distribution"]
        self.drivers_stop_probability = parameters_config["drivers_stop_probability"]
        self.timeout_p = parameters_config["timeouts"]["passenger"]
        self.timeout_d = parameters_config["timeouts"]["driver"]
        # TODO support for flash mob scenario