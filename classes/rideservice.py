"""
rideservice.py

This module defines the RideService class, which manages ride offers and acceptances between passengers and drivers.
It supports the following operations:

1. step: Advances the ride service logic by generating new offers and checking for matches.
2. _generate_offers: Internal method to create offers for unassigned reservations and nearby idle taxis.
3. _check_matches: Internal method to dispatch taxis when both driver and passenger have accepted an offer.
4. accept_offer: Registers an acceptance of an offer by either a driver or passenger.
5. get_offers_for_drivers: Returns offers relevant to the given list of driver IDs.
6. get_offers_for_passengers: Returns offers relevant to the given list of passenger reservation IDs.
7. remove_offer: Removes an offer from the offers dict.
8. is_passenger_accepted: Returns True if the passenger has accepted a specific offer.
9. is_driver_accepted: Returns True if the driver has accepted a specific offer.
10. remove_acceptance: Removes an acceptance from the acceptances dict.
11. compute_offer: Computes travel time, route length, and price of a reservation.
12. compute_surge_multiplier: Computes the price multiplication factor based on the ratio between
    number of pending requests and number of available drivers.
"""


from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from classes.model import Model
from typing import Optional
import traci
import math
import heapq


class RideService:
    model: "Model"
    offers: dict
    acceptances: dict
    uber_base_price: float
    uber_min_price: float
    uber_cost_per_min: float
    uber_cost_per_min: float
    uber_service_fee: float
    uber_surge_multiplier: float
    uber_max_surge: float
    lyft_base_price: float
    lyft_min_price: float
    lyft_cost_per_min: float
    lyft_cost_per_min: float
    lyft_service_fee: float
    lyft_surge_multiplier: float
    lyft_max_surge: float


    def __init__(
            self,
            model: "Model"
        ):
        self.model = model
        self.offers = {}  # key: (res_id, driver_id), value: dict with travel time, route length, price
        self.acceptances = {}  # key: (res_id, driver_id), value: (set of agents, timestamp)

        # Uber fares
        self.uber_base_price = 2.17
        self.uber_min_price = 7.83
        self.uber_cost_per_min = 0.38
        self.uber_cost_per_km = 1.45
        self.uber_service_fee = 5.31
        self.uber_surge_multiplier = 1.0
        self.uber_max_surge = 8.0

        # Lyft fares
        self.lyft_base_price = 2.24
        self.lyft_min_price = 5.00
        self.lyft_cost_per_min = 0.40
        self.lyft_cost_per_km = 1.50
        self.lyft_service_fee = 3.60
        self.lyft_surge_multiplier = 1.0
        self.lyft_max_surge = 5.0


    def step(self):
        self._generate_offers()
        self._check_matches()


    def _generate_offers(self) -> None:
        """
        Generates ride offers by matching unassigned passenger requests with the closest available idle taxis.

        This function:
        - Cleans up partial acceptances that have timed out for either passengers or drivers.
        - Compute the surge multiplier according to unassigned requests and available drivers for both Uber and Lyft.
        - Iterates over all unassigned passenger ride requests.
        - For each request, skips if an offer already exists.
        - Attempts to retrieve the passenger's current position; skips the request if unsuccessful.
        - Calculates the radius from each idle taxi to the passenger's position, skipping taxis
          with unavailable positions and farther than 10km.
        - Selects up to 8 closest taxis.
        - Creates ride offers for each selected taxi, including radius, travel time, route length, price information, and provider.
        - Logs and skips requests if no taxis are available.
        
        Returns:
        -------
        None
        """

        # Load information from driver and passenger agents
        idle_taxis = self.model.driver.get_idle_drivers()
        unassigned = self.model.passenger.get_unassigned_requests()
        timeout_p = self.model.passenger.get_passenger_timeout()
        timeout_d = self.model.driver.get_driver_timeout()
        idle_by_provider = self.model.driver.get_idle_drivers_by_provider()
        uber_drivers = idle_by_provider["Uber"]
        lyft_drivers = idle_by_provider["Lyft"]

        # Compute surge multiplier for both Uber and Lyft
        now = int(self.model.time)
        if (now % 300 == 0):
            self.uber_surge_multiplier = self.compute_surge_multiplier(int(len(unassigned)*0.75), len(uber_drivers), "Uber")
            self.lyft_surge_multiplier = self.compute_surge_multiplier(int(len(unassigned)*0.25), len(lyft_drivers), "Lyft")

        # Clean up partial acceptances if timeout
        expired_keys = [
            key for key, (agents, timestamp) in self.acceptances.items()
            if len(agents) == 1 and (
                ("passenger" in agents and now - timestamp > timeout_p) or
                ("driver" in agents and now - timestamp > timeout_d)
            )
        ]
        for key in expired_keys:
            self.remove_offer(key)
            self.remove_acceptance(key)

        # Iterates over all passenger requests
        existing_res_ids = {r_id for (r_id, _) in self.offers}
        for reservation in unassigned:
            res_id = reservation.id
            if res_id in existing_res_ids:
                print(f"⚠️ Reservation {res_id} already has an offer — skipping")
                continue
            
            # Get passenger position
            try:
                pax_pos = traci.person.getPosition(reservation.persons[0])
            except traci.TraCIException:
                print(f"⚠️ Failed to get position for reservation {res_id}: {reservation}")
                continue

            # Compute radius of each driver from the passenger
            taxis_radius = []
            for taxi_id in idle_taxis:
                try:
                    taxi_pos = traci.vehicle.getPosition(taxi_id)
                    # Euclidean distance
                    radius = math.hypot(taxi_pos[0] - pax_pos[0], taxi_pos[1] - pax_pos[1])
                    if radius <= 10000:  # Only include taxis within 10km
                        taxis_radius.append((radius, taxi_id))
                except traci.TraCIException:
                    print(f"⚠️ Failed to get position for taxi {taxi_id}")
                    continue

            # Get top 8 closest taxis
            closest_taxis = heapq.nsmallest(8, taxis_radius)
            if not closest_taxis:
                print(f"⚠️ No taxis available for reservation {res_id} — skipping")
                continue

            # Create offers
            for radius, taxi_id in closest_taxis:
                offer_key = (res_id, taxi_id)
                from_edge = reservation.fromEdge
                to_edge = reservation.toEdge
                try:
                    if taxi_id in uber_drivers:
                        provider = "Uber"
                        surge_multiplier = self.uber_surge_multiplier
                        travel_time, route_length, price = self.compute_offer(from_edge, to_edge, surge_multiplier, provider)
                    elif taxi_id in lyft_drivers:
                        provider = "Lyft"
                        surge_multiplier = self.lyft_surge_multiplier
                        travel_time, route_length, price = self.compute_offer(from_edge, to_edge, surge_multiplier, provider)
                except traci.TraCIException as e:
                    print(f"⚠️ Failed to compute route for offer {offer_key}: {e}")
                    continue
                self.offers[offer_key] = {
                    "radius": radius,
                    "time": travel_time,
                    "route_length": route_length,
                    "surge": surge_multiplier,
                    "price": price,
                    "provider": provider
                }


    def _check_matches(self) -> None:
        """
        Dispatches taxis for fully accepted matches and cleans up conflicting entries.
        This function:
        - Checks for fully accepted taxi-passenger matches.
        - Dispatches taxis for these matches.
        - Removes all related acceptances to prevent conflicts.

        Returns:
        -------
        None

        Notes:
        A match is considered fully accepted if both the driver and passenger have accepted.
        """
        # Get fully-accepted matches
        matched_keys = [
            (res_id, driver_id)
            for (res_id, driver_id), (agents, _) in self.acceptances.items()
            if "driver" in agents and "passenger" in agents
        ]

        # For each match try to dispatch the taxi
        for res_id, driver_id in matched_keys:
            try:
                traci.vehicle.dispatchTaxi(driver_id, [res_id])
            except traci.TraCIException as e:
                print(f"❌ DispatchTaxi failed: {e} — driver: {driver_id}, res_id: {res_id}")
            except Exception as e:
                print(f"❌ Unknown error during dispatch: {e}")
            finally:
                # Remove all acceptances involving the same reservation or driver
                to_remove = [
                    key for key in self.acceptances
                    if key[0] == res_id or key[1] == driver_id
                ]
                for key in to_remove:
                    self.remove_acceptance(key)


    def accept_offer(
            self,
            key: tuple,
            agent: str
        ) -> None:
        """
        Accepts an offer for a ride by recording the agent's acceptance.
        If both agents associated with the offer have accepted, the offer is removed.

        Parameters:
        ----------
        - key: tuple 
            The unique identifier for the offer.
        - agent: str
            The agent accepting the offer.

        Returns:
        -------
        None
        """
        now = self.model.time
        agents, _ = self.acceptances.setdefault(key, (set(), now))
        agents.add(agent)
        # If there are two agents that means both driver and passenger accepted
        if len(agents) == 2:
            self.remove_offer(key)


    def get_offers_for_drivers(
            self,
            drivers: set
        ) -> dict:
        """
        Gets the dictionary of offers for the specified drivers.

        Parameters:
        ----------
        drivers: set
            The set of driver identifiers to filter offers by.

        Returns:
        -------
        dict
            A dictionary containing offers where the driver identifier (k[1]) is in the provided drivers list.
        """
        return {
            k: v for k, v in self.offers.items()
            if k[1] in drivers
        }


    def get_offers_for_passengers(
            self,
            passengers: set
        ) -> dict:
        """
        Gets the dictionary of offers for the specified passengers.

        Parameters:
        ----------
        passengers: set
            The set of passenger identifiers to filter offers by.

        Returns:
        -------
        dict
            A dictionary containing offers where the passenger identifier (k[0]) is in the provided passengers list.
        """
        return {
            k: v for k, v in self.offers.items()
            if k[0] in passengers
        }


    def remove_offer(
            self,
            key: tuple
        ) -> Optional[dict]:
        """
        Removes a specified offer from offers dictionary.

        Parameters:
        ----------
        key: tuple
            Key (res_id, offer) of the offer to remove

        Returns:
        -------
        None or removed offer value (dict)
        """
        self.offers.pop(key, None)


    def is_passenger_accepted(self,
            res_id: str,
            driver_id: str
        ) -> bool:
        """
        Returns True if the passenger has accepted the offer for (res_id, driver_id).

        Parameters:
        ----------
        - res_id: str
            Unique ID of the passenger reservation
        - driver_id: str
            Unique ID of the driver

        Returns:
        -------
        bool
        """
        key = (res_id, driver_id)
        return (
            key in self.acceptances and
            "passenger" in self.acceptances[key][0]
        )
    
    
    def is_driver_accepted(self,
            res_id: str,
            driver_id: str
        ) -> bool:
        """
        Returns True if the driver has accepted the offer for (res_id, driver_id).

        Parameters:
        ----------
        - res_id: str
            Unique ID of the passenger reservation
        - driver_id: str
            Unique ID of the driver

        Returns:
        -------
        bool
        """
        key = (res_id, driver_id)
        return (
            key in self.acceptances and
            "driver" in self.acceptances[key][1]
        )

    
    def remove_acceptance(
            self,
            key: tuple
        ) -> Optional[tuple]:
        """
        Removes a specified acceptance from acceptance dictionary.

        Parameters:
        ----------
        key: tuple
            Key (res_id, driver_id) of the acceptance to remove

        Returns:
        -------
        None or removed acceptance value (set, int)
        """
        self.acceptances.pop(key, None)


    def compute_offer(
            self,
            from_edge: str,
            to_edge: str,
            surge_multiplier : float,
            provider: str
        ) -> tuple[int, float, float]:
        """
        Computes travel time, distance, and price for a specific offer.

        Parameters:
        ----------
        - from_edge: str
            Departure edge ID of the reservation.
        - to_edge: str
            Arrival edge ID of the reservation.
        - surge_multiplier: float
            Multiplication factor applied to the final price.

        Returns:
        -------
        - travel_time: int
            Estimated travel time in seconds.
        - route_length: float
            Estimated distance in meters.
        - price: float
            Computed price for the ride.
        """
        # Find route and get travel distance (miles) and travel time (seconds)
        route = traci.simulation.findRoute(from_edge, to_edge)
        travel_time = int(route.travelTime)
        route_length = route.length
        # Compute price depending on provider
        if provider == "Uber":
            price = (
                self.uber_base_price + 
                (self.uber_cost_per_min * (travel_time / 60)) +
                (self.uber_cost_per_km * (route_length / 1000))
                ) * surge_multiplier + self.uber_service_fee
            price = max(self.uber_min_price, price)
        elif provider == "Lyft":
            price = (
                self.lyft_base_price + 
                (self.lyft_cost_per_min * (travel_time / 60)) +
                (self.lyft_cost_per_km * (route_length / 1000))
                ) * surge_multiplier + self.lyft_service_fee
            price = max(self.lyft_min_price, price)

        return travel_time, route_length, price
    

    def compute_surge_multiplier(
            self,
            unassigned_requests: set,
            idle_drivers: set,
            provider: str
            ) -> float:
        """
        Computes the surge multiplier based on the ratio of unassigned ride requests
        to available idle drivers. The multiplier ranges from 1.0 (normal conditions)
        to 8.0 for Uber and 5.0 for Lyft (high demand, low supply).

        Parameters:
        ----------
        - unassigned_requests: int
            Number of pending requests.
        - idle_drivers: int
            Number of available drivers.

        Returns
        -------
        surge : float
            Surge price multiplier (clipped between 1.0 and 5.0).
        """
        if provider == "Uber":
            if idle_drivers == 0:
                return self.uber_max_surge  # Max surge if no drivers available
            ratio = unassigned_requests / idle_drivers
            # Linear mapping
            surge = min(self.uber_max_surge, max(1, ratio))
        elif provider == "Lyft":
            if idle_drivers == 0:
                return self.lyft_max_surge  # Max surge if no drivers available
            ratio = unassigned_requests / idle_drivers
            # Linear mapping
            surge = min(self.lyft_max_surge, max(1, ratio))

        return round(surge, 2)