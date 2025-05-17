"""
passenger.py

This module defines the Passenger class, which manages unassigned ride requests and
interacts with the ride service to accept offers from drivers.
It supports the following operations:

1. step: Advances the passenger logic by: (i) updating the set of unassigned requests, (ii) processing unassigned ride requests,
   (iii) evaluating driver offers, (iv) assigning the best offer to each request, and (v) cleaning up redundant offers.
2. get_unassigned_requests: Returns the set of unassigned requests.
3. get_passenger_timeout: Returns the timeout value for the passenger.
"""


from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from classes.model import Model
import traci
from collections import defaultdict


class Passenger:
    def __init__(
            self,
            model: "Model",
            timeout: int
        ):
        self.model = model
        self.unassigned_requests = set()
        self.timeout = timeout


    def step(self) -> None:
        # Get the set of unassigned reservations from TraCI
        self.unassigned_requests = set(traci.person.getTaxiReservations(3))
        # Get ID from each reservation object
        unassigned_requests_ids = {res.id for res in self.unassigned_requests}

        # Group offers by reservation ID
        offers_by_passenger = defaultdict(list)
        for (res_id, driver_id), offer in self.model.rideservice.get_offers_for_passengers(unassigned_requests_ids).items():
            offers_by_passenger[res_id].append((driver_id, offer))

        # Keep track of already assigned drivers
        assigned_drivers = set()
        reservations_to_remove = set()

        # Iterate over grouped offers
        for res_id, offers in offers_by_passenger.items():
            best_offer = None
            best_driver_id = None
            # Sort drivers by closest distance
            for driver_id, offer in sorted(offers, key=lambda x: x[1]["distance"]):
                if driver_id not in assigned_drivers:
                    best_offer = offer
                    best_driver_id = driver_id
                    assigned_drivers.add(driver_id)
                    self.model.rideservice.accept_offer((res_id, driver_id), "passenger")
                    reservations_to_remove.add(res_id)
                    break
                else:
                    continue

            # Remove all other offers for this reservation (except the best one)
            for driver_id, _ in offers:
                if driver_id != best_driver_id:
                    self.model.rideservice.remove_offer((res_id, driver_id))

        self.unassigned_requests = {r for r in self.unassigned_requests if r.id not in reservations_to_remove}


    def get_unassigned_requests(self) -> set:
        """
        Gets the set of unassigned requests.

        Returns:
        -------
        set
            A set containing all the unassigned requests.
        """
        return self.unassigned_requests
    

    def get_passenger_timeout(self) -> int:
        """
        Gets the timeout for passengers.

        Returns:
        -------
        int
            An int containing the max waiting time for passengers.
        """
        return self.timeout
