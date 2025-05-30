"""
drivers.py

This module defines the Drivers class, which manages idle drivers and
interacts with the ride service to accept/reject offers from passengers.
It supports the following operations:

1. step: Advances the drivers logic by:
    (i) updating the set of idle drivers for the providers,
    (ii) distributing the personalities of new idle drivers,
    (iii) processing pending ride offers where the passenger has already accepted,
    (iv) assigning the best offer to each idle driver (who either accepts or rejects), and
    (v) cleaning up redundant offers.
    (vi) updating the logger with the current state of drivers.
2. get_idle_drivers: Returns the set of idle drivers.
3. get_idle_drivers_by_provider: Returns a dictionary containing the sets of available drivers by provider.
4. get_driver_provider: Returns the provider of a specific driver.
5. get_driver_timeout: Returns the timeout value for the driver.
6. get_total_offers: Returns the total number of offers made to drivers.
"""


import random
from collections import defaultdict
import traci
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from classes.model import Model
if TYPE_CHECKING:
    from classes.logger import Logger


class Drivers:
    model: "Model"
    idle_drivers: set
    timeout: int
    personality_distribution: dict
    acceptance_distribution: dict
    providers: dict
    logger: "Logger"

    def __init__(
            self,
            model: "Model",
            timeout: int,
            personality_distribution: dict,
            acceptance_distribution: dict,
            providers: dict,
            logger: "Logger"
        ):
        self.model = model
        self.__timeout = timeout
        self.__personality_distribution = personality_distribution
        self.__acceptance_distribution = acceptance_distribution
        self.__providers = providers
        self.logger = logger
        self.__idle_drivers = set()
        self.__drivers_with_provider = {}  # Maps drivers to providers
        self.__drivers_with_personality = {}  # Maps drivers to personalities


    def step(self) -> None:
        fleet_status = {s: set(traci.vehicle.getTaxiFleet(s)) for s in [0, 1, 2]}
        logged_idle = len(fleet_status[0])
        logged_pickup = len(fleet_status[1])
        logged_busy = len(fleet_status[2])
        # Get the set of idle drivers from TraCI
        self.__idle_drivers = fleet_status[0]
        if self.model.verbose:
            print(f"🚕 {len(self.__idle_drivers)} idle drivers")
        # Assign providers and personalities
        provider_counts = {provider: 0 for provider in self.__providers}
        for driver_id in self.__idle_drivers:
            if driver_id not in self.__drivers_with_provider:
                probability = random.random()
                for provider, config in self.__providers.items():
                    if probability < config["share"]:
                        self.__drivers_with_provider[driver_id] = provider
                        provider_counts[provider] += 1
                        break
            if driver_id not in self.__drivers_with_personality:
                probability = random.random()
                for personality, threshold in self.__personality_distribution.items():
                    if probability < threshold:
                        self.__drivers_with_personality[driver_id] = personality
                        break
        
        # Print number of newly added drivers per provider
        if self.model.verbose:
            for provider, count in provider_counts.items():
                print(f"🚕 {count} drivers assigned to provider '{provider}'")

        # Collect pending offers where passenger has already accepted
        offers_by_driver = defaultdict(list)
        for (res_id, driver_id), offer in self.model.rideservices.get_offers_for_drivers(self.__idle_drivers).items():
            if self.model.rideservices.is_passenger_accepted(res_id, driver_id):
                offers_by_driver[driver_id].append((res_id, offer))

        # Iterate over grouped offers
        accept = 0
        reject = 0
        self.__tot_offers = sum(len(offers) for offers in offers_by_driver.values())
        available_drivers = len(self.__idle_drivers)
        # Dynamic greediness adjustment
        greediness = (self.__tot_offers - available_drivers) / available_drivers if available_drivers > 0 else 0
        greediness = greediness / self.model.ratio_vehicles_requests
        print(f"Total offers: {self.__tot_offers}, Available drivers: {available_drivers}, Greediness: {greediness}")
        for driver_id, offers in offers_by_driver.items():
            # Choose the closest passenger (min radius)
            best_res_id, _ = min(offers, key=lambda x: x[1]["radius"])
            key = (best_res_id, driver_id)
            # Reject the offer if surge is too low and temporarily remove driver from available
            personality = self.__drivers_with_personality[driver_id]
            surge = offers[0][1]["surge"]
            acceptance_ranges = self.__acceptance_distribution[personality]
            acceptance = next((perc for low, up, perc in acceptance_ranges if low < surge <= up), None)
            dynamic_acceptance = max(0, min(1, acceptance - greediness))
            if random.random() > dynamic_acceptance:
                self.model.rideservices.reject_offer(key)
                reject+=1
                self.__idle_drivers.discard(driver_id)
                continue
            # Accept the offer
            self.model.rideservices.accept_offer(key, "driver")
            accept+=1
            self.__idle_drivers.discard(driver_id)

        if self.model.verbose:
            print(f"✅ {accept} offers accepted by drivers")
            print(f"📵 {reject} offers rejected by drivers")

        # Update the logger
        self.logger.update_drivers(
            timestamp = self.model.time,
            idle_drivers = logged_idle,
            pickup_drivers = logged_pickup,
            busy_drivers = logged_busy,
            accepted_requests = accept,
            rejected_requests = reject
        )


    def get_idle_drivers(self) -> set:
        """
        Gets the set of idle drivers.

        Returns:
        -------
        set
            A set containing all the available drivers.
        """
        return self.__idle_drivers
    

    def get_idle_drivers_by_provider(self) -> dict:
        """
        Gets idle drivers according to their provider.

        Returns:
        -------
        dict
            A dict mapping each provider to a set of their idle drivers.
        """
        result = {provider: set() for provider in self.__providers}
        for driver_id in self.__idle_drivers:
            provider = self.__drivers_with_provider.get(driver_id)
            if provider:
                result[provider].add(driver_id)
        return result
    

    def get_driver_provider(
            self,
            driver_id: str
        ) -> str:
        """
        Gets provider of a specific driver.

        Returns:
        -------
        str
            A string containing the provider of the driver.
        """
        return self.__drivers_with_provider.get(driver_id)
    

    def get_drivers_timeout(self) -> int:
        """
        Gets the timeout for drivers.

        Returns:
        -------
        int
            An int containing the max waiting time for drivers.
        """
        return self.__timeout
    

    def get_total_offers(self) -> int:
        """
        Gets the total number of offers made to drivers.

        Returns:
        -------
        int
            An int containing the total number of offers.
        """
        return self.__tot_offers