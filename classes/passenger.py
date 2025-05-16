import traci
import random


class Passenger:
    def __init__(self, model):
        self.model = model
        self.unassigned_requests = set()
        self.timeout = 900

    def step(self):
        self.unassigned_requests = set(traci.person.getTaxiReservations(3))

        unassigned_requests_ids = {res.id for res in self.unassigned_requests}
        print(f"Passengers pending offers: {len(self.model.rideservice.get_offers_for_passengers(unassigned_requests_ids).items())}")
        # Group offers by reservation
        pending_offers = {}
        for (res_id, driver_id), offer in self.model.rideservice.get_offers_for_passengers(unassigned_requests_ids).items():
            if res_id not in pending_offers:
                pending_offers[res_id] = []
            pending_offers[res_id].append((driver_id, offer))

        accepted = 0
        removed = 0
        used_drivers = set()
        for res_id, driver_offers in pending_offers.items():
            # Sort drivers by distance
            sorted_offers = sorted(driver_offers, key=lambda x: x[1]["distance"])
            best_driver_id = None
            best_offer = None

            # Pick the closest driver not already used
            for driver_id, offer in sorted_offers:
                if driver_id not in used_drivers:
                    best_driver_id = driver_id
                    best_offer = offer
                    break

            # If no available driver, skip
            if best_driver_id is None:
                print(f"No available drivers for reservation {res_id}")
                # Clean up offers for this reservation
                self.unassigned_requests = {
                    r for r in self.unassigned_requests if r.id != res_id
                }
                unassigned_requests_ids.discard(res_id)
                for driver_id, _ in driver_offers:
                    if driver_id != best_driver_id:
                        self.model.rideservice.offers.pop((res_id, driver_id), None)
                        removed += 1
                continue

            # Accept the offer
            self.model.rideservice.accept_offer(res_id, best_driver_id, "passenger")
            used_drivers.add(best_driver_id)
            accepted += 1

            # Clean up offers for this reservation
            self.unassigned_requests = {
                r for r in self.unassigned_requests if r.id != res_id
            }
            unassigned_requests_ids.discard(res_id)
            for driver_id, _ in driver_offers:
                if driver_id != best_driver_id:
                    self.model.rideservice.offers.pop((res_id, driver_id), None)
                    removed += 1
        print(f"Passengers accepted {accepted} rides")
        print(f"Removed {removed} duplicated reservations")
