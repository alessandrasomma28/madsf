"""
driver.py

This module defines the Driver class, which manages idle drivers and
interacts with the ride service to accept offers from passengers.
It supports the following operations:

1. step: Advances the passenger logic by: (i) updating the set of idle drivers for both Uber and Lyft,
(ii) processing pending ride offers where the passenger has already accepted,
(iii) assigning the best offer to each idle driver, and (iv) cleaning up redundant offers.
2. get_idle_drivers: Returns the set of idle drivers.
3. get_idle_drivers_by_provider: Returns a dictionary containing the two sets of Uber and Lyft available drivers.
4. get_driver_timeout: Returns the timeout value for the driver.
"""


from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from classes.model import Model
import traci
from collections import defaultdict


class Driver:
    model: "Model"
    idle_drivers: set
    timeout: int
    driver_provider: dict


    def __init__(
            self,
            model: "Model",
            timeout: int
        ):
        self.model = model
        self.idle_drivers = set()
        self.timeout = timeout
        self.driver_provider = {}  # Maps driver_id → "Uber" or "Lyft"


    def step(self) -> None:
        # Get the set of idle drivers from TraCI
        self.idle_drivers = set(traci.vehicle.getTaxiFleet(0))

        # Persist provider assignments
        for driver_id in self.idle_drivers:
            if driver_id not in self.driver_provider:
                self.driver_provider[driver_id] = "Uber" if len(self.driver_provider) % 4 != 3 else "Lyft"

        # Collect pending offers where passenger has already accepted
        offers_by_driver = defaultdict(list)
        for (res_id, driver_id), offer in self.model.rideservice.get_offers_for_drivers(self.idle_drivers).items():
            if self.model.rideservice.is_passenger_accepted(res_id, driver_id):
                offers_by_driver[driver_id].append((res_id, offer))

        # Iterate over grouped offers
        for driver_id, offers in offers_by_driver.items():
            # Choose the closest passenger (min distance) and accept the offer
            best_res_id, _ = min(offers, key=lambda x: x[1]["radius"])
            key = (best_res_id, driver_id)
            self.model.rideservice.accept_offer(key, "driver")
            self.idle_drivers.discard(driver_id)

            # Remove all other offers for the same driver
            for res_id, _ in offers:
                if res_id != best_res_id:
                    self.model.rideservice.remove_offer((res_id, driver_id))


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
            A dict containing the two sets of Uber and Lyft drivers.
        """
        result = {"Uber": set(), "Lyft": set()}
        for driver_id in self.idle_drivers:
            provider = self.driver_provider.get(driver_id)
            if provider:
                result[provider].add(driver_id)
        return result
    

    def get_driver_timeout(self) -> int:
        """
        Gets the timeout for drivers.

        Returns:
        -------
        int
            An int containing the max waiting time for drivers.
        """
        return self.timeout