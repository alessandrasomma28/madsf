import traci
import random


class Driver:
    def __init__(self, model):
        self.model = model
        self.idle_drivers = set()

    def step(self):
        self.idle_drivers = list(traci.vehicle.getTaxiFleet(0))

        offers = self.model.ride_service.get_offers_for_drivers(self.idle_drivers)
        for driver_id, offer in offers.items():
            # Simulate random acceptance
            if random.random() <= 1.0:
                self.model.ride_service.accept_offer(driver_id, "driver")
