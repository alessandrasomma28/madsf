"""
passengers.py

This module defines the Passenger class, which manages unassigned ride requests and
interacts with the ride service to accept/reject offers from drivers.
It supports the following operations:

1. step: Advances the passengers logic by:
    (i) Initializing the sets of requests, 
    (ii) Removing timed-out requests,
    (iii) Updating the set of unassigned requests,
    (iv) Assigning personalities to new passengers,
    (v) Processing offers from drivers, either considering social groups or not, and
    (vi) Logging the status of the passengers.
2. get_unassigned_requests: Returns the set of unassigned requests.
3. get_canceled_requests: Returns the set of canceled requests.
"""


import random
from collections import defaultdict
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
        self.__passengers_with_personality = {}     # Maps passengers to personalities
        self.__canceled = set()                     # Set of canceled requests for surge multiplier computation
        self.logger = logger


    def step(self) -> None:
        # --- Initialize step ---
        self.__reservations_status = {s: set(traci.person.getTaxiReservations(s)) for s in [3, 4, 8]}
        self.__logged_unassigned = len(self.__reservations_status[3])
        self.__logged_assigned = len(self.__reservations_status[4])
        self.__logged_pickup = len(self.__reservations_status[8])
        self.__canceled_number = 0
        self.__accept = self.__reject = 0
        # Reset the canceled requests 60 seconds after the surge multiplier computation
        if self.model.time % (300 + self.model.agents_interval) == 0:
            self.__canceled = set()

        # --- Remove timed-out requests ---
        now = int(self.model.time)
        acceptances = self.model.rideservices.get_acceptances()
        for res in self.__reservations_status[3]:
            if now - int(res.reservationTime) >= self.__timeout and all(res.id != key[0] for key in acceptances):
                traci.person.remove(res.persons[0])
                # Add to canceled set for surge multiplier computation
                self.__canceled.add(res)
                # Increment the canceled requests counter for logging
                self.__canceled_number += 1

        # --- Update unassigned requests ---
        self.__unassigned_requests = set(traci.person.getTaxiReservations(3))
        self.__unassigned_request_ids = {res.id for res in self.__unassigned_requests}
        if self.model.verbose:
            print(f"☝🏻 {len(self.__unassigned_request_ids)} unassigned requests")

        # --- Assign personalities ---
        new_requests = 0
        for res_id in self.__unassigned_request_ids:
            if res_id not in self.__passengers_with_personality:
                new_requests += 1
                probability = random.random()
                for personality, threshold in self.__personality_distribution.items():
                    if probability < threshold:
                        self.__passengers_with_personality[res_id] = personality
                        break
        if self.model.verbose:
            print(f"☝🏻 {new_requests} new requests")

        # --- Process offers ---
        # Group offers by passenger
        offers_by_passenger = defaultdict(list)
        all_offer_passengers = self.model.rideservices.get_offers_for_passengers(self.__unassigned_request_ids).items()
        for (res_id, driver_id), offer in all_offer_passengers:
            offers_by_passenger[res_id].append((driver_id, offer))
        # Initialize sets
        assigned_drivers = set()
        reservations_to_remove = set()
        # Process offers for each passenger
        for res_id, offers in offers_by_passenger.items():
            best_driver_id = None
            if self.model.use_social_groups:
                # Initialize sets for tracking accepted and rejected providers
                providers_rejected = set()
                providers_accepted = set()
                # Get personality and acceptance ranges of the passenger
                personality = self.__passengers_with_personality[res_id]
                acceptance_ranges = self.__acceptance_distribution[personality]
                # Sort offers by radius to prioritize closer drivers
                for driver_id, offer in sorted(offers, key=lambda x: x[1]["radius"]):
                    # If all providers have already been rejected, break the loop
                    if providers_rejected.issuperset(self.model.providers):
                        break
                    # If provider has already been rejected, skip to the next offer
                    provider = offer["provider"]
                    if provider in providers_rejected:
                        continue
                    # Compute acceptance probability based on surge and personality
                    surge = offer["surge"]
                    acceptance = next((perc for low, up, perc in acceptance_ranges if low < surge <= up), None)
                    # If provider has already been accepted or acceptance probability is met, accept the offer
                    if provider in providers_accepted or (acceptance is not None and random.random() <= acceptance):
                        if provider not in providers_accepted:
                            providers_accepted.add(provider)
                        # If driver is not assigned accept the offer, else remove the offer and continue
                        if driver_id not in assigned_drivers:
                            assigned_drivers.add(driver_id)
                            self.model.rideservices.accept_offer((res_id, driver_id), "passenger")
                            self.__accept += 1
                            best_driver_id = driver_id
                            break
                        else:
                            self.model.rideservices.remove_offer((res_id, driver_id))
                    else:
                        # If provider is not accepted, reject the offer
                        self.model.rideservices.reject_offer((res_id, driver_id))
                        providers_rejected.add(provider)
                        # Log only the first rejection
                        if len(providers_rejected) == 1:
                            self.__reject += 1
            else:
                # Sort offers by radius to prioritize closer drivers
                for driver_id, _ in sorted(offers, key=lambda x: x[1]["radius"]):
                    # If driver is not assigned accept the offer, else remove the offer and continue
                    if driver_id not in assigned_drivers:
                        assigned_drivers.add(driver_id)
                        self.model.rideservices.accept_offer((res_id, driver_id), "passenger")
                        self.__accept += 1
                        best_driver_id = driver_id
                        break
                    else:
                        self.model.rideservices.remove_offer((res_id, driver_id))
            # If a best driver was found, remove all other offers for this reservation
            for driver_id, _ in offers:
                if driver_id != best_driver_id or best_driver_id is None:
                    self.model.rideservices.remove_offer((res_id, driver_id))
            if best_driver_id:
                reservations_to_remove.add(res_id)
        # Update the internal counter
        self.__unassigned_requests = {r for r in self.__unassigned_requests if r.id not in reservations_to_remove}

        # --- Log status ---
        if self.model.verbose:
            print(f"✅ {self.__accept} offers accepted by passengers")
            if self.model.use_social_groups:
                print(f"📵 {self.__reject} offers rejected by passengers")
                print(f"❌ {self.__canceled_number} requests canceled by passengers")

        self.logger.update_passengers(
            timestamp=self.model.time,
            unassigned_requests=self.__logged_unassigned,
            assigned_requests=self.__logged_assigned,
            pickup_requests=self.__logged_pickup,
            accepted_requests=self.__accept,
            rejected_requests=self.__reject,
            canceled_requests=self.__canceled_number,
        )


    def get_unassigned_requests(self) -> set:
        """
        Gets the set of unassigned requests.
        
        Returns
        -------
        set
            The set of unassigned requests.
        """
        return self.__unassigned_requests

    def get_canceled_requests(self) -> set:
        """
        Gets the set of canceled requests.

        Returns
        -------
        set
            The set of canceled requests.
        """
        return self.__canceled