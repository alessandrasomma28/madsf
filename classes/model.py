"""
model.py

This module defines the Model class, which serves as the core simulation orchestration.
It initializes and manages the agents: Driver, Passenger, and RideService.
"""


import traci
from classes.rideservices import RideServices
from classes.passengers import Passengers
from classes.drivers import Drivers
import time

class Model:
    sumocfg_path: str
    end_time: int
    passengers: Passengers
    drivers: Drivers
    rideservices: RideServices
    time: int
    driver_personality_distribution: list
    driver_acceptance_distribution: list
    passenger_personality_distribution: list
    passenger_acceptance_distribution: list


    def __init__(
            self,
            sumocfg_path: str,
            end_time: int
        ):
        self.passenger_personality_distribution = [0.37, 0.45, 0.18]
        self.passenger_acceptance_distribution = {"budget":
                                                  [[-1000,1.5,1], [1.5,1.8,0.9], [1.8,2,0.8], [2,1000,0.7]],
                                                  "normal":
                                                  [[-1000,1.4,1],[1.4,1.6,0.9],[1.6,1.8,0.8],[1.8,2,0.7],[2,2.2,0.6],[2.2,1000,0.5]],
                                                  "greedy":
                                                  [[-1000,1.2,1],[1.2,1.4,0.8],[1.4,1.6,0.6],[1.6,1.8,0.3],[1.8,2,0.2],[2,1000,0.1]]
                                                  }
        self.driver_personality_distribution = [0.21, 0.55, 0.24]
        self.driver_acceptance_distribution = {"budget":
                                               [[-1000,1,0.8], [1,1.2,0.9], [1.2,1.4,0.95], [1.4,1000,1]],
                                               "normal":
                                               [[-1000,1,0.7],[1,1.2,0.8],[1.2,1.4,0.9],[1.4,1.6,0.95],[1.6,1000,1]],
                                               "greedy":
                                               [[-1000,1,0.05],[1,1.2,0.3],[1.2,1.4,0.4],[1.4,1.6,0.5],[1.6,1.8,0.7],[1.8,2,0.8], [2,1000,1]]
                                               }
        self.sumocfg_path = sumocfg_path
        self.end_time = end_time
        self.passengers = Passengers(
            self,
            timeout=900,
            personality_distribution=self.passenger_personality_distribution,
            acceptance_distribution=self.passenger_acceptance_distribution)
        self.drivers = Drivers(
            self,
            timeout=60,
            personality_distribution=self.driver_personality_distribution,
            acceptance_distribution=self.driver_acceptance_distribution)
        self.rideservices = RideServices(self)
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