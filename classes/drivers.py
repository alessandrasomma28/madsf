"""
drivers.py

This module defines the Drivers class, which manages idle drivers and
interacts with the ride service to accept offers from passengers.
It supports the following operations:

1. step: Advances the drivers logic by:
    (i) updating the set of idle drivers for the providers,
    (ii) distributing the personalities of new idle drivers,
    (iii) processing pending ride offers where the passenger has already accepted,
    (iv) assigning the best offer to each idle driver (who either accepts or rejects), and
    (v) cleaning up redundant offers.
2. get_idle_drivers: Returns the set of idle drivers.
3. get_idle_drivers_by_provider: Returns a dictionary containing the sets of available drivers by provider.
4. get_driver_timeout: Returns the timeout value for the driver.
"""


import random
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from classes.model import Model
import traci
from collections import defaultdict


class Drivers:
    model: "Model"
    idle_drivers: set
    timeout: int
    personality_distribution: list
    acceptance_distribution: list
    provider_distribution: list


    def __init__(
            self,
            model: "Model",
            timeout: int,
            personality_distribution: list,
            acceptance_distribution: list,
            provider_distribution: list
        ):
        self.model = model
        self.timeout = timeout
        self.personality_distribution = personality_distribution
        self.acceptance_distribution = acceptance_distribution
        self.provider_distribution = provider_distribution
        self.idle_drivers = set()
        self.drivers_with_provider = {}  # Maps drivers to providers
        self.drivers_with_personality = {}  # Maps drivers to personalities


    def step(self) -> None:
        # Get the set of idle drivers from TraCI
        self.idle_drivers = set(traci.vehicle.getTaxiFleet(0))
        print(f"🚕 {len(self.idle_drivers)} idle drivers")
        # Persist provider assignments and personalities
        new_drivers = 0
        for driver_id in self.idle_drivers:
            if driver_id not in self.drivers_with_provider:
                probability = random.random()
                for provider, threshold in self.provider_distribution.items():
                    if probability < threshold:
                        self.drivers_with_provider[driver_id] = provider
                        break
            if driver_id not in self.drivers_with_personality:
                new_drivers+=1
                probability = random.random()
                for personality, threshold in self.personality_distribution.items():
                    if probability < threshold:
                        self.drivers_with_personality[driver_id] = personality
                        break
        print(f"🚕 {new_drivers} new idle drivers")

        # Collect pending offers where passenger has already accepted
        offers_by_driver = defaultdict(list)
        for (res_id, driver_id), offer in self.model.rideservices.get_offers_for_drivers(self.idle_drivers).items():
            if self.model.rideservices.is_passenger_accepted(res_id, driver_id):
                offers_by_driver[driver_id].append((res_id, offer))

        # Iterate over grouped offers
        accept = 0
        reject = 0
        removed = 0
        for driver_id, offers in offers_by_driver.items():
            # Choose the closest passenger (min radius)
            best_res_id, _ = min(offers, key=lambda x: x[1]["radius"])
            key = (best_res_id, driver_id)
            # Reject the offer if surge is too low and temporarily remove driver from available
            personality = self.drivers_with_personality[driver_id]
            surge = offers[0][1]["surge"]
            acceptance = next((perc for low, up, perc in self.acceptance_distribution[personality] if low < surge <= up), None)
            if random.random() > acceptance:
                self.model.rideservices.reject_offer(key)
                reject+=1
                self.idle_drivers.discard(driver_id)
                # Remove all other offers for the same driver
                for res_id, _ in offers:
                    if res_id != best_res_id:
                        self.model.rideservices.remove_offer((res_id, driver_id))
                        removed+=1
                continue
            # Accept the offer
            self.model.rideservices.accept_offer(key, "driver")
            accept+=1
            self.idle_drivers.discard(driver_id)

            # Remove all other offers for the same driver
            for res_id, _ in offers:
                if res_id != best_res_id:
                    self.model.rideservices.remove_offer((res_id, driver_id))
                    removed+=1

        print(f"✅ {accept} offers accepted by drivers")
        print(f"📵 {reject} offers rejected by drivers")
        print(f"🧹 {removed} duplicated drivers offers removed")


    def get_idle_drivers(self) -> set:
        """
        Gets the set of idle drivers.

        Returns:
        -------
        set
            A set containing all the available drivers.
        """
        return self.idle_drivers
    

    def get_idle_drivers_by_provider(self) -> dict:
        """
        Gets idle drivers according to their provider.

        Returns:
        -------
        dict
            A dict mapping each provider to a set of their idle drivers.
        """
        result = {provider: set() for provider in self.provider_distribution}
        for driver_id in self.idle_drivers:
            provider = self.drivers_with_provider.get(driver_id)
            if provider:
                result[provider].add(driver_id)
        return result
    

    def get_drivers_timeout(self) -> int:
        """
        Gets the timeout for drivers.

        Returns:
        -------
        int
            An int containing the max waiting time for drivers.
        """
        return self.timeout