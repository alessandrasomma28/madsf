"""
passengers.py

This module defines the Passenger class, which manages unassigned ride requests and
interacts with the ride service to accept/reject offers from drivers.
It supports the following operations:

1. step: Advances the passengers logic by:
    (i) updating the set of unassigned requests, 
    (ii) distributing the personalities of new passengers,
    (iii) processing unassigned ride requests,
    (iii) evaluating driver offers,
    (iv) assigning the best offers to each passenger (who either accepts or rejects), and
    (v) cleaning up redundant offers.
2. get_unassigned_requests: Returns the set of unassigned requests.
3. get_passenger_timeout: Returns the timeout value for the passenger.
"""


from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from classes.model import Model
import traci
from collections import defaultdict
import random


class Passengers:
    model: "Model"
    unassigned_requests: set
    timeout: int
    personality_distribution: list
    acceptance_distribution: list
    

    def __init__(
            self,
            model: "Model",
            timeout: int,
            personality_distribution: list,
            acceptance_distribution: list
        ):
        self.model = model
        self.timeout = timeout
        self.personality_distribution = personality_distribution
        self.acceptance_distribution = acceptance_distribution
        self.unassigned_requests = set()
        self.passengers_with_personality = {}  # Maps res_id → "normal", "budget" or "greedy"


    def step(self) -> None:
        # Remove requests if timeout
        now = int(self.model.time)
        for res in traci.person.getTaxiReservations(3):
            if now - int(res.reservationTime) >= self.timeout:
                traci.person.remove(res.persons[0])
        # Get the set of unassigned reservations from TraCI
        self.unassigned_requests = set(traci.person.getTaxiReservations(3))
        # Get ID from each reservation object
        unassigned_requests_ids = {res.id for res in self.unassigned_requests}
        print(f"☝🏻 {len(unassigned_requests_ids)} unassigned requests")
        # Persist personalities
        new_requests = 0
        for res_id in unassigned_requests_ids:
            if res_id not in self.passengers_with_personality:
                new_requests+=1
                probability = random.random()
                for personality, threshold in self.personality_distribution.items():
                    if probability < threshold:
                        self.passengers_with_personality[res_id] = personality
                        break
        print(f"☝🏻 {new_requests} new requests")

        # Group offers by reservation ID
        offers_by_passenger = defaultdict(list)
        for (res_id, driver_id), offer in self.model.rideservices.get_offers_for_passengers(unassigned_requests_ids).items():
            offers_by_passenger[res_id].append((driver_id, offer))

        # Keep track of already assigned drivers
        assigned_drivers = set()
        reservations_to_remove = set()

        # Iterate over grouped offers
        accept = 0
        reject = 0
        removed = 0
        for res_id, offers in offers_by_passenger.items():
            best_driver_id = None
            # Sort by closest drivers (min radius)
            for driver_id, offer in sorted(offers, key=lambda x: x[1]["radius"]):
                if driver_id not in assigned_drivers:
                    # Reject the offer if surge is too low and temporarily remove passenger from available
                    personality = self.passengers_with_personality[res_id]
                    surge = offer["surge"]
                    acceptance = next((perc for low, up, perc in self.acceptance_distribution[personality] if low < surge <= up), None)
                    if random.random() > acceptance:
                        self.model.rideservices.reject_offer((res_id, driver_id))
                        reservations_to_remove.add(res_id)
                        reject+=1
                        continue
                    # Accept the offer
                    best_driver_id = driver_id
                    assigned_drivers.add(driver_id)
                    self.model.rideservices.accept_offer((res_id, driver_id), "passenger")
                    accept+=1
                    reservations_to_remove.add(res_id)
                    break
                else:
                    self.model.rideservices.remove_offer((res_id, driver_id))
                    removed+=1
                    continue

            # Remove all other offers for this reservation (except the best one)
            for driver_id, _ in offers:
                if driver_id != best_driver_id:
                    self.model.rideservices.remove_offer((res_id, driver_id))
                    removed+=1

        print(f"✅ {accept} offers accepted by passengers")
        print(f"📵 {reject} offers rejected by passengers")
        self.unassigned_requests = {r for r in self.unassigned_requests if r.id not in reservations_to_remove}
        print(f"🧹 {removed} duplicated passengers offers removed")


    def get_unassigned_requests(self) -> set:
        """
        Gets the set of unassigned requests.

        Returns:
        -------
        set
            A set containing all the unassigned requests.
        """
        return self.unassigned_requests
    

    def get_passengers_timeout(self) -> int:
        """
        Gets the timeout for passengers.

        Returns:
        -------
        int
            An int containing the max waiting time for passengers.
        """
        return self.timeout
