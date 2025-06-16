"""
model.py

This module defines the Model class, which serves as the core simulation orchestration.
It initializes and computes steps for the agents: Drivers, Passengers, and RideServices.
"""


from pathlib import Path
import time
import json
import numpy as np
import os
import sys
sys.path.append(os.path.join(os.environ["SUMO_HOME"], 'tools'))
import traci
from classes.rideservices import RideServices
from classes.passengers import Passengers
from classes.drivers import Drivers
from classes.logger import Logger
import xml.etree.ElementTree as ET
from paths.data import SF_SFCTA_STANFORD_MAPPING_PATH, SF_TAZ_EDGE_MAPPING_PATH
from paths.config import PARAMETERS_CONFIG_PATH, ZIP_ZONES_CONFIG_PATH
from paths.sumoenv import SUMO_NET_PATH

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

    def __init__(
            self,
            sumocfg_path: str,
            end_time: int,
            output_dir_path: str,
            verbose: bool = False,
            ratio_requests_vehicles: float = 1.0,
            mode: str = "social_groups"
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
            # Load edge speeds from the SUMO network file
            edge_speeds = {}
            tree = ET.parse(SUMO_NET_PATH)
            root = tree.getroot()
            for edge in root.findall('edge'):
                if 'function' in edge.attrib and edge.attrib['function'] == 'internal':
                    continue  # skip internal edges
                for lane in edge.findall('lane'):
                    edge_id = edge.attrib['id']
                    speed = float(lane.attrib['speed'])
                    edge_speeds[edge_id] = speed
            self.original_edge_speeds = edge_speeds
            # Load scenario parameters
            with open(Path(full_scenario_config_path), "r") as f:
                self.scenario_config = json.load(f)
                location = self.scenario_config["location"]
                self.tazs_involved = None
                # Check for location and
                if location == "downtown":
                    with open(Path(SF_TAZ_EDGE_MAPPING_PATH), "r") as f:
                        self.taz_edge_mapping = json.load(f)
                    with open(Path(ZIP_ZONES_CONFIG_PATH), "r") as f:
                        zip_zones = json.load(f)
                        self.tazs_involved = []
                        with open(Path(SF_SFCTA_STANFORD_MAPPING_PATH), "r") as f:
                            sfcta_mapping = json.load(f)
                        for taz in zip_zones[location]:
                            if taz in sfcta_mapping:
                                self.tazs_involved.extend(sfcta_mapping[taz])
                elif location == "midtown":
                    with open(Path(SF_TAZ_EDGE_MAPPING_PATH), "r") as f:
                        self.taz_edge_mapping = json.load(f)
                    with open(Path(ZIP_ZONES_CONFIG_PATH), "r") as f:
                        zip_zones = json.load(f)
                        self.tazs_involved = []
                        self.tazs_involved_less = []
                        with open(Path(SF_SFCTA_STANFORD_MAPPING_PATH), "r") as f:
                            sfcta_mapping = json.load(f)
                        for taz in zip_zones[location]:
                            if taz not in zip_zones["downtown"]:
                                if taz in sfcta_mapping:
                                    self.tazs_involved_less.extend(sfcta_mapping[taz])
                            else:
                                if taz in sfcta_mapping:
                                    self.tazs_involved.extend(sfcta_mapping[taz])
        else:
            self.scenario_config = None
            self.tazs_involved = None
            self.tazs_involved_less = None


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
                    self.update_scenario_parameters(self.scenario_config)
                    print(f"🚨 Scenario {self.scenario_config['name']} activated!\n")
                if self.time == self.scenario_end:
                    # Restore params
                    self.update_scenario_parameters(self.default_config)
                    print(f"🚨 Scenario {self.scenario_config['name']} ended!\n")        
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
        for provider_name, new_params in parameters_config["providers"].items():
            # If provider exists, keep its surge multiplier
            if provider_name in self.providers:
                current_surges = self.providers[provider_name]["surge_multiplier"]
            else:
                current_surges = new_params["surge_multiplier"]
            updated_params = new_params.copy()
            updated_params["surge_multiplier"] = current_surges
            self.providers[provider_name] = updated_params
        self.drivers_personality_distribution = parameters_config["drivers_personality_distribution"]
        self.drivers_acceptance_distribution = parameters_config["drivers_acceptance_distribution"]
        self.passengers_personality_distribution = parameters_config["passengers_personality_distribution"]
        self.passengers_acceptance_distribution = parameters_config["passengers_acceptance_distribution"]
        self.drivers_stop_probability = parameters_config["drivers_stop_probability"]
        self.timeout_p = parameters_config["timeouts"]["passenger"]
        self.timeout_d = parameters_config["timeouts"]["driver"]
        self.slow_down = parameters_config["slow_down_perc"]
        self.slow_mid = parameters_config["slow_mid_perc"]
        if self.slow_down > 0:
            # For each TAZ involved in the scenario, slow down speed of edges
            for taz in self.tazs_involved:
                if str(taz) not in self.taz_edge_mapping:
                    continue
                for edge_id in self.taz_edge_mapping[str(taz)]:
                    speed = max(1, traci.edge.getLastStepMeanSpeed(edge_id) - (traci.edge.getLastStepMeanSpeed(edge_id) * self.slow_down))
                    traci.edge.setMaxSpeed(edge_id, speed)
            for taz in self.tazs_involved_less:
                if str(taz) not in self.taz_edge_mapping:
                    continue
                for edge_id in self.taz_edge_mapping[str(taz)]:
                    speed = max(1, traci.edge.getLastStepMeanSpeed(edge_id) - (traci.edge.getLastStepMeanSpeed(edge_id) * self.slow_mid))
                    traci.edge.setMaxSpeed(edge_id, speed)
        if self.scenario_config["slow_down_perc"] > 0 and self.time == self.scenario_end:
            # For each TAZ involved in the scenario, restore default speed of edges
            for taz in self.tazs_involved:
                if str(taz) not in self.taz_edge_mapping:
                    continue
                for edge_id in self.taz_edge_mapping[str(taz)]:
                    if edge_id in self.original_edge_speeds:
                        traci.edge.setMaxSpeed(edge_id, self.original_edge_speeds[edge_id])
            for taz in self.tazs_involved_less:
                if str(taz) not in self.taz_edge_mapping:
                    continue
                for edge_id in self.taz_edge_mapping[str(taz)]:
                    if edge_id in self.original_edge_speeds:
                        traci.edge.setMaxSpeed(edge_id, self.original_edge_speeds[edge_id])