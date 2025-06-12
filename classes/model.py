"""
model.py

This module defines the Model class, which serves as the core simulation orchestration.
It initializes and computes steps for the agents: Drivers, Passengers, and RideServices.
"""


from pathlib import Path
import time
import json
import traci
from classes.rideservices import RideServices
from classes.passengers import Passengers
from classes.drivers import Drivers
from classes.logger import Logger
from constants.config_constants import (DRIVERS_PERSONALITY, DRIVERS_ACCEPTANCE, PASSENGERS_PERSONALITY,
                                        PASSENGERS_ACCEPTANCE, PROVIDERS_CONFIG, TIMEOUT_CONFIG)

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
    drivers_personality_distribution: list
    drivers_acceptance_distribution: list
    passengers_personality_distribution: list
    passengers_acceptance_distribution: list
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
        with open(Path(DRIVERS_PERSONALITY), "r") as f:
            self.drivers_personality_distribution = json.load(f)
        with open(Path(DRIVERS_ACCEPTANCE), "r") as f:
            self.drivers_acceptance_distribution = json.load(f)
        with open(Path(PASSENGERS_PERSONALITY), "r") as f:
            self.passengers_personality_distribution = json.load(f)
        with open(Path(PASSENGERS_ACCEPTANCE), "r") as f:
            self.passengers_acceptance_distribution = json.load(f)
        with open(Path(PROVIDERS_CONFIG), "r") as f:
            self.providers = json.load(f)
        with open(Path(TIMEOUT_CONFIG), "r") as f:
            timeouts = json.load(f)
            timeout_p = timeouts["passenger"]
            timeout_d = timeouts["driver"]
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
            timeout=timeout_p,
            personality_distribution=self.passengers_personality_distribution,
            acceptance_distribution=self.passengers_acceptance_distribution,
            logger=self.logger
            )
        self.drivers = Drivers(
            self,
            timeout=timeout_d,
            personality_distribution=self.drivers_personality_distribution,
            acceptance_distribution=self.drivers_acceptance_distribution,
            providers=self.providers,
            logger=self.logger
            )
        self.rideservices = RideServices(
            self,
            providers=self.providers,
            logger=self.logger
            )
        self.time = 0
        self.agents_interval = 0
        self.mode = mode
        if self.mode == "social_groups":
            self.use_social_groups = True
        else:
            self.use_social_groups = False


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
        while (len(traci.person.getTaxiReservations(15)) > 0 and traci.simulation.getMinExpectedNumber() > 0) or traci.simulation.getTime() < self.end_time:
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