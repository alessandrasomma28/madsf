"""
model.py

This module defines the Model class, which serves as the core simulation orchestration.
It initializes and manages the agents: Driver, Passenger, and RideService.
"""


import traci
from classes.rideservice import RideService
from classes.passenger import Passenger
from classes.driver import Driver

class Model:
    sumocfg_path: str
    end_time: int
    passenger: Passenger
    driver: Driver
    rideservice: RideService
    time: int


    def __init__(
            self,
            sumocfg_path: str,
            end_time: int
        ):
        self.sumocfg_path = sumocfg_path
        self.end_time = end_time
        self.passenger = Passenger(self, timeout=900)
        self.driver = Driver(self, timeout=60)
        self.rideservice = RideService(self)
        self.time = 0


    def run(self):
        """
        Runs the simulation with the sumocfg previously generated.

        This function:
        - Perform simulation steps handling ride hailing agents.
        - Stops when there are no active persons and vehicles available.

        Returns:
        -------
        None
        """
        
        while traci.simulation.getMinExpectedNumber() > 0:
            traci.simulationStep()
            if int(traci.simulation.getTime()) % 100 == 0:
                print(f"Simulation time: {traci.simulation.getTime()} seconds")
                self.time = traci.simulation.getTime()
                self.passenger.step()
                self.driver.step()
                self.rideservice.step()