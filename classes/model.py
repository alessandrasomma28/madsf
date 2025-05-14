import traci
from classes.rideservice import RideService
from classes.passenger import Passenger
from classes.driver import Driver
from concurrent.futures import ThreadPoolExecutor


class Model:
    def __init__(self, sumocfg_path, end_time):
        self.sumocfg_path = sumocfg_path
        self.driver = Driver(self)
        self.passenger = Passenger(self)
        self.ride_service = RideService(self)
        self.time = 0
        self.end_time = end_time

    def run(self):
        while traci.simulation.getMinExpectedNumber() > 0 and traci.simulation.getTime() < self.end_time+10800:
            traci.simulationStep()
            if traci.simulation.getTime() % 60 == 0:
                print(f"Simulation time: {traci.simulation.getTime()} seconds")
                self.time = traci.simulation.getTime()
                self.driver.step()
                self.passenger.step()
                self.ride_service.step()