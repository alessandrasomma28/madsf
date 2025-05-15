import traci
import random


class Passenger:
    def __init__(self, model):
        self.model = model
        self.unassigned_requests = set()
        self.timeout = 120

    def step(self):
        self.unassigned_requests = set(traci.person.getTaxiReservations(3))

        unassigned_requests_ids = {res.id for res in self.unassigned_requests}
        for (res_id, driver_id), offer in self.model.rideservice.get_offers_for_passengers(unassigned_requests_ids).items():
            # Skip if this offer was removed earlier
            if (res_id, driver_id) not in self.model.rideservice.offers and res_id not in self.unassigned_requests:
                continue
            # Simulate random acceptance
            if random.random() <= 1.0:
                self.model.rideservice.accept_offer(res_id, driver_id, "passenger")
                self.unassigned_requests = {
                    r for r in self.unassigned_requests if r.id != res_id
                }
                unassigned_requests_ids.discard(res_id)
                to_remove = [
                    k for k in self.model.rideservice.offers
                    if k[0] == res_id and k != (res_id, driver_id)
                ]
                for k in to_remove:
                    self.model.rideservice.offers.pop(k, None)