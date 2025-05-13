import traci
from classes.rideservice import RideService

class Model:
    def __init__(self, sumocfg_path, end_time):
        self.sumocfg_path = sumocfg_path
        self.passengers = {}
        self.drivers = {}
        self.ride_service = RideService(self)
        self.time = 0
        self.end_time = end_time

    def run(self):
        while traci.simulation.getMinExpectedNumber() > 0 and traci.simulation.getTime() < self.end_time+10800:
            if traci.simulation.getTime() % 60 == 0 and traci.simulation.getTime() > 0:
                self.time = traci.simulation.getTime()
                self.ride_service.step(self)

            traci.simulationStep()