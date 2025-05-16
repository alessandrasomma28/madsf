import traci
from classes.rideservice import RideService
from classes.passenger import Passenger
from classes.driver import Driver


class Model:
    def __init__(self, sumocfg_path, end_time):
        self.sumocfg_path = sumocfg_path
        self.driver = Driver(self)
        self.passenger = Passenger(self)
        self.rideservice = RideService(self)
        self.time = 0
        self.end_time = end_time

    def run(self):
        while traci.simulation.getMinExpectedNumber() > 0:
            traci.simulationStep()
            if int(traci.simulation.getTime()) % 10 == 0:
                print(f"Simulation time: {traci.simulation.getTime()} seconds")
                self.time = traci.simulation.getTime()
                self.passenger.step()
                self.driver.step()
                self.rideservice.step()