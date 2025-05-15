import traci
import random


class Driver:
    def __init__(self, model):
        self.model = model
        self.idle_drivers = set()
        self.timeout = 120

    def step(self):
        self.idle_drivers = set(traci.vehicle.getTaxiFleet(0))

        for (res_id, driver_id), offer in self.model.rideservice.get_offers_for_drivers(self.idle_drivers).items():
            # Skip if this offer was removed earlier
            if (res_id, driver_id) not in self.model.rideservice.offers and driver_id not in self.idle_drivers:
                continue
            # Simulate random acceptance
            if random.random() <= 1.0:
                self.model.rideservice.accept_offer(res_id, driver_id, "driver")
                self.idle_drivers.discard(driver_id)
                to_remove = [
                    k for k in self.model.rideservice.offers
                    if k[1] == driver_id and k != (res_id, driver_id)
                ]
                for k in to_remove:
                    self.model.rideservice.offers.pop(k, None)