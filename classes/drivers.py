"""
drivers.py

This module defines the Drivers class, which manages idle drivers and
interacts with the ride service to accept/reject offers from passengers.
It supports the following operations:

1. step: Advances the drivers logic by:
    (i) Initializing the sets of drivers,
    (ii) Updating the idle times of drivers,
    (iii) Assigning providers and personalities to new drivers,
    (iv) Processing offers from the ride service, either considering social groups or not, and
    (v) Logging the status of the drivers.
2. get_idle_drivers: Returns the set of idle drivers.
3. get_idle_drivers_by_provider: Returns a dictionary containing the sets of available drivers by provider.
4. get_driver_provider: Returns the provider of a specific driver.
"""


import random
from collections import defaultdict
import os
import sys
sys.path.append(os.path.join(os.environ["SUMO_HOME"], 'tools'))
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
        self.__drivers_with_provider = {}       # Maps drivers to providers
        self.__drivers_with_personality = {}    # Maps drivers to personalities
        self.__driver_idle_time = {}            # Track how long each driver has been idle
        self.__driver_removal_prob = {}         # Track removal probability for each driver
        self.logger = logger


    def step(self) -> None:
        # --- Initialize step ---
        self.__idle_drivers = set(traci.vehicle.getTaxiFleet(0))
        self.__pickup_drivers = set(traci.vehicle.getTaxiFleet(1))
        self.__busy_drivers = set(traci.vehicle.getTaxiFleet(2))
        self.__logged_idle = len(self.__idle_drivers)
        self.__logged_pickup = len(self.__pickup_drivers)
        self.__logged_busy = len(self.__busy_drivers)
        self.__accept = 0
        self.__reject = 0
        self.__removed = 0
        if self.model.verbose:
            print(f"🚕 {len(self.__idle_drivers)} idle drivers")
        
        # --- Update idle times ---
        for driver_id in self.__idle_drivers:
            self.__driver_idle_time[driver_id] = self.__driver_idle_time.get(driver_id, 0) + self.model.agents_interval

        # --- Assign providers and personalities ---
        provider_counts = {provider: 0 for provider in self.__providers}
        for driver_id in self.__idle_drivers:
            # Assign provider based on the configured market share
            if driver_id not in self.__drivers_with_provider:
                probability = random.random()
                for provider, config in self.__providers.items():
                    if probability < config["share"]:
                        self.__drivers_with_provider[driver_id] = provider
                        provider_counts[provider] += 1
                        break
            # Assign personality based on the configured distribution
            if self.model.use_social_groups and driver_id not in self.__drivers_with_personality:
                probability = random.random()
                for personality, threshold in self.__personality_distribution.items():
                    if probability < threshold:
                        self.__drivers_with_personality[driver_id] = personality
                        break
        if self.model.verbose:
            for provider, count in provider_counts.items():
                print(f"🚕 {count} drivers assigned to provider '{provider}'")
        
        # --- Process offers ---
        # Group offers by driver
        offers_by_driver = defaultdict(list)
        for (res_id, driver_id), offer in self.model.rideservices.get_offers_for_drivers(self.__idle_drivers).items():
            if self.model.rideservices.is_passenger_accepted(res_id, driver_id):
                offers_by_driver[driver_id].append((res_id, offer))
        # Compute demand pressure for dynamic acceptance
        total_requests = len(traci.person.getTaxiReservations(3))
        demand_pressure = (total_requests - len(self.__idle_drivers)) / len(self.__idle_drivers) if self.__idle_drivers else total_requests
        # Process offers for each driver
        for driver_id, offers in offers_by_driver.items():
            # Sort offers by radius and select the min
            best_res_id, offer = min(offers, key=lambda x: x[1]["radius"])
            key = (best_res_id, driver_id)
            if self.model.use_social_groups:
                # Get personality and acceptance ranges of the driver
                personality = self.__drivers_with_personality.get(driver_id)
                surge = offer["surge"]
                acceptance_ranges = self.__acceptance_distribution[personality]
                acceptance = next((perc for low, up, perc in acceptance_ranges if low < surge <= up), 0)
                # Compute dynamic acceptance based on demand pressure
                greediness = demand_pressure / surge
                dynamic_acceptance = max(0, min(1, acceptance - greediness))
                accepted = random.random() <= dynamic_acceptance
            else:
                # Always accept if no social groups
                accepted = True
            # If the offer is accepted notify RideServices and remove the driver from idle drivers
            if accepted:
                self.model.rideservices.accept_offer(key, "driver")
                self.__accept += 1
                self.__idle_drivers.discard(driver_id)
                # Reset the driver's removal probability
                self.__driver_removal_prob[driver_id] = 0.0
            else:
                self.model.rideservices.reject_offer(key)
                self.__reject += 1
                if self.__driver_idle_time.get(driver_id, 0) >= self.__timeout:
                    # If the driver has been inactive for too long, increase removal probability
                    self.__driver_removal_prob[driver_id] = self.__driver_removal_prob.get(driver_id, 0.0) + 0.1
                    if random.random() < self.__driver_removal_prob[driver_id]:
                        traci.vehicle.remove(driver_id)
                        self.__driver_idle_time.pop(driver_id, None)
                        self.__driver_removal_prob.pop(driver_id, None)
                        self.__drivers_with_provider.pop(driver_id, None)
                        self.__drivers_with_personality.pop(driver_id, None)
                        self.__idle_drivers.discard(driver_id)
                        self.__removed += 1
                        continue
                self.__idle_drivers.discard(driver_id)

        # --- Log status ---
        if self.model.verbose:
            print(f"✅ {self.__accept} offers accepted by drivers")
            if self.model.use_social_groups:
                print(f"📵 {self.__reject} offers rejected by drivers")
                print(f"❌ {self.__removed} drivers removed due to inactivity")
        self.logger.update_drivers(
            timestamp=self.model.time,
            idle_drivers=self.__logged_idle,
            pickup_drivers=self.__logged_pickup,
            busy_drivers=self.__logged_busy,
            accepted_requests=self.__accept,
            rejected_requests=self.__reject,
            removed_drivers=self.__removed
        )


    def get_idle_drivers(self) -> set:
        """
        Gets the set of idle drivers.

        Returns
        -------
        set
            The set of driver IDs that are currently idle.
        """
        return self.__idle_drivers

    def get_idle_drivers_by_provider(self) -> dict:
        """
        Gets a dictionary of idle drivers grouped by their providers.

        Returns
        -------
        dict
            A dictionary where keys are provider names and values are sets of driver IDs that are idle under that provider.
        """
        result = {provider: set() for provider in self.__providers}
        for driver_id in self.__idle_drivers:
            provider = self.__drivers_with_provider.get(driver_id)
            if provider:
                result[provider].add(driver_id)
        return result

    def get_driver_provider(self, driver_id: str) -> str | None:
        """
        Gets the provider of a specific driver.

        Parameters
        ----------
        driver_id : str
            The ID of the driver whose provider is to be retrieved.

        Returns
        -------
        str
            The provider of the specified driver, or None if the driver is not found.
        """
        return self.__drivers_with_provider.get(driver_id)