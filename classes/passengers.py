"""
passengers.py

This module defines the Passenger class, which manages unassigned ride requests and
interacts with the ride service to accept/reject offers from drivers.
It supports the following operations:

1. step: Advances the passengers logic by:
    (i) updating the set of unassigned requests, 
    (ii) distributing the personalities of new passengers,
    (iii) processing unassigned ride requests,
    (iv) evaluating driver offers,
    (v) assigning the best offers to each passenger (who either accepts or rejects), and
    (vi) cleaning up redundant offers.
    (vii) updating the logger with the current state of passengers.
2. get_unassigned_requests: Returns the set of unassigned requests.
3. get_passenger_timeout: Returns the timeout value for the passenger.
4. get_accepted_offers: Returns the number of accepted offers from passengers.
"""


from collections import defaultdict
import random
import traci
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from classes.model import Model
if TYPE_CHECKING:
    from classes.logger import Logger


class Passengers:
    model: "Model"
    unassigned_requests: set
    timeout: int
    personality_distribution: list
    acceptance_distribution: list
    logger: "Logger"

    def __init__(
            self,
            model: "Model",
            timeout: int,
            personality_distribution: list,
            acceptance_distribution: list,
            logger: "Logger"
        ):
        self.model = model
        self.__timeout = timeout
        self.__personality_distribution = personality_distribution
        self.__acceptance_distribution = acceptance_distribution
        self.logger = logger
        self.__unassigned_requests = set()
        self.__passengers_with_personality = {}  # Maps reservation IDs to personalities
        self.__accepted_offers = 0


    def step(self) -> None:
        logged_unassigned = len(set(traci.person.getTaxiReservations(3)))
        logged_assigned = len(set(traci.person.getTaxiReservations(4)))
        logged_pickup = len(set(traci.person.getTaxiReservations(8)))
        # Remove requests if timeout
        now = int(self.model.time)
        for res in traci.person.getTaxiReservations(3):
            if now - int(res.reservationTime) >= self.__timeout:
                traci.person.remove(res.persons[0])
        # Get the set of unassigned reservations from TraCI
        self.__unassigned_requests = set(traci.person.getTaxiReservations(3))
        # Get ID from each reservation object
        unassigned_requests_ids = {res.id for res in self.__unassigned_requests}
        if self.model.verbose:
            print(f"☝🏻 {len(unassigned_requests_ids)} unassigned requests")
        # Assign personalities
        new_requests = 0
        for res_id in unassigned_requests_ids:
            if res_id not in self.__passengers_with_personality:
                new_requests+=1
                probability = random.random()
                for personality, threshold in self.__personality_distribution.items():
                    if probability < threshold:
                        self.__passengers_with_personality[res_id] = personality
                        break
        if self.model.verbose:
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
        self.__accepted_offers = 0
        for res_id, offers in offers_by_passenger.items():
            best_driver_id = None
            # Sort by closest drivers (min radius)
            for driver_id, offer in sorted(offers, key=lambda x: x[1]["radius"]):
                # Reject all offers if surge is too low and temporarily remove passenger from available
                personality = self.__passengers_with_personality[res_id]
                surge = offer["surge"]
                acceptance_ranges = self.__acceptance_distribution[personality]
                acceptance = next((perc for low, up, perc in acceptance_ranges if low < surge <= up), None)
                if random.random() > acceptance:
                    self.model.rideservices.reject_offer((res_id, driver_id))
                    reject+=1
                    break
                else:
                    if driver_id not in assigned_drivers:
                        # Accept the offer
                        best_driver_id = driver_id
                        assigned_drivers.add(driver_id)
                        self.model.rideservices.accept_offer((res_id, driver_id), "passenger")
                        self.__accepted_offers += 1
                        accept+=1
                        reservations_to_remove.add(res_id)
                        break
                    else:
                        if (res_id, driver_id) in self.model.rideservices.get_offers():
                            self.model.rideservices.remove_offer((res_id, driver_id))
                            removed+=1
                        continue
            # Remove all other offers for this reservation
            for driver_id, _ in offers:
                if driver_id != best_driver_id or best_driver_id is None:
                    if (res_id, driver_id) in self.model.rideservices.get_offers():
                        self.model.rideservices.remove_offer((res_id, driver_id))
                        removed+=1

        self.__unassigned_requests = {r for r in self.__unassigned_requests if r.id not in reservations_to_remove}

        if self.model.verbose:
            print(f"✅ {accept} offers accepted by passengers")
            print(f"📵 {reject} offers rejected by passengers")
            print(f"🧹 {removed} duplicated passengers offers removed")

        # Update the logger
        self.logger.update_passengers(
            timestamp = self.model.time,
            unassigned_requests = logged_unassigned,
            assigned_requests = logged_assigned,
            pickup_requests = logged_pickup,
            accepted_requests = accept,
            rejected_requests = reject
        )


    def get_unassigned_requests(self) -> set:
        """
        Gets the set of unassigned requests.

        Returns:
        -------
        set
            A set containing all the unassigned requests.
        """
        return self.__unassigned_requests
    

    def get_passengers_timeout(self) -> int:
        """
        Gets the timeout for passengers.

        Returns:
        -------
        int
            An int containing the max waiting time for passengers.
        """
        return self.__timeout


    def get_accepted_offers(self) -> int:
        """
        Gets the number of accepted offers from passengers.

        Returns:
        -------
        int
            An int containing the total number of accepted offers from passengers.
        """
        return self.__accepted_offers