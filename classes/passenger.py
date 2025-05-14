import traci
import random


class Passenger:
    def __init__(self, model):
        self.model = model
        self.unassigned_requests = set()

    def step(self):
        self.unassigned_requests = list(traci.person.getTaxiReservations(3))

        unassigned_requests_ids = [res.id for res in self.unassigned_requests]
        offers = self.model.ride_service.get_offers_for_passengers(unassigned_requests_ids)
        for res_id, offer in offers.items():
            # Simulate random acceptance
            if random.random() <= 1.0:
                self.model.ride_service.accept_offer(res_id, "passenger")
