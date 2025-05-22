"""
model.py

This module defines the Model class, which serves as the core simulation orchestration.
It initializes and computes steps for the agents: Drivers, Passengers, and RideServices.
"""


import traci
from classes.rideservices import RideServices
from classes.passengers import Passengers
from classes.drivers import Drivers
import time
import json
from pathlib import Path
from constants.config_constants import (DRIVERS_PERSONALITY, DRIVERS_ACCEPTANCE, PASSENGERS_PERSONALITY,
                                        PASSENGERS_ACCEPTANCE, PROVIDERS_CONFIG, TIMEOUT_CONFIG)

class Model:
    sumocfg_path: str
    config_path: str
    end_time: int
    passengers: Passengers
    drivers: Drivers
    rideservices: RideServices
    time: int
    drivers_personality_distribution: list
    drivers_acceptance_distribution: list
    passengers_personality_distribution: list
    passengers_acceptance_distribution: list


    def __init__(
            self,
            sumocfg_path: str,
            end_time: int
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
        self.passengers = Passengers(
            self,
            timeout=timeout_p,
            personality_distribution=self.passengers_personality_distribution,
            acceptance_distribution=self.passengers_acceptance_distribution
            )
        self.drivers = Drivers(
            self,
            timeout=timeout_d,
            personality_distribution=self.drivers_personality_distribution,
            acceptance_distribution=self.drivers_acceptance_distribution,
            providers=self.providers
            )
        self.rideservices = RideServices(
            self,
            providers=self.providers
            )
        self.time = 0


    def run(
            self,
            agents_interval: int = 60
        ):
        """
        Runs the simulation with the sumocfg previously generated.

        This function:
        - Perform simulation steps.
        - Handles ride hailing agents every {agents_interval} timestamps.
        - Stops when there are no active persons and vehicles available.

        Parameters:
        ----------
        agents_interval: int
            Interval (timestamps) for agents execution.

        Returns:
        -------
        None
        """
        
        while traci.simulation.getMinExpectedNumber() > 0:
            traci.simulationStep()
            if int(traci.simulation.getTime()) % agents_interval == 0:
                print(f"Simulation time: {traci.simulation.getTime()} seconds\n")
                self.time = traci.simulation.getTime()
                start = time.time()
                self.passengers.step()
                end = time.time()
                print(f"⏱️ Passengers step computed in {round((end - start), 2)} seconds\n")
                start = time.time()
                self.drivers.step()
                end = time.time()
                print(f"⏱️ Drivers step computed in {round((end - start), 2)} seconds\n")
                start = time.time()
                self.rideservices.step()
                end = time.time()
                print(f"⏱️ RideServices step computed in {round((end - start), 2)} seconds\n")