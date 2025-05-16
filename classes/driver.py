import traci
import random
from collections import defaultdict


class Driver:
    def __init__(self, model):
        self.model = model
        self.idle_drivers = set()
        self.timeout = 120

    def step(self):
        self.idle_drivers = set(traci.vehicle.getTaxiFleet(0))

        # Find all pending offers where passenger has already accepted
        pending_offers = {
            (res_id, driver_id): offer
            for (res_id, driver_id), offer in self.model.rideservice.offers.items()
            if (res_id, driver_id) in self.model.rideservice.acceptances
            and "passenger" in self.model.rideservice.acceptances[(res_id, driver_id)][0]
            and driver_id in self.idle_drivers
        }

        print(f"Drivers pending offers: {len(pending_offers)}")

        # Group offers by driver and pick best (lowest distance)
        driver_best_offer = defaultdict(list)
        for (res_id, driver_id), offer in pending_offers.items():
            driver_best_offer[driver_id].append((offer["distance"], res_id))

        accepted = 0
        removed = 0
        print(f"Unique drivers available: {len(driver_best_offer)}")
        for driver_id, offers in driver_best_offer.items():
            best_distance, best_res = min(offers, key=lambda x: x[0])
            self.model.rideservice.accept_offer(best_res, driver_id, "driver")
            accepted+=1
            to_remove = [
                k for k in pending_offers
                if k[1] == driver_id and k[0] != best_res
            ]
            removed+=len(to_remove)
            self.idle_drivers.discard(driver_id)
            for k in to_remove:
                self.model.rideservice.offers.pop(k, None)
        print(f"Drivers accepted {accepted} rides")
        print(f"Removed {removed} duplicated drivers")