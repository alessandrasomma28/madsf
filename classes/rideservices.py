"""
rideservices.py

This module defines the RideServices class, which manages ride offers and acceptances between passengers and drivers.
It supports the following operations:

1. step: Advances the ride service logic by generating new offers and checking for matches.
2. __generate_offers: Internal method to create offers for unassigned reservations and nearby idle taxis.
3. __check_matches: Internal method to dispatch taxis when both driver and passenger have accepted an offer.
4. __compute_offer: Computes travel time, route length, and price of a reservation.
5. __compute_surge_multiplier: Computes the price multiplication factor based on the ratio between
    number of pending requests and number of available drivers.
6. accept_offer: Registers an acceptance of an offer by either a driver or passenger.
7. reject_offer: Registers a reject of an offer by either a driver or passenger and removes the related offer (and acceptance).
8. get_offers: Returns the dictionary of all offers.
9. get_offers_for_drivers: Returns offers relevant to the given list of driver IDs.
10. get_offers_for_passengers: Returns offers relevant to the given list of passenger reservation IDs.
11. remove_offer: Removes an offer from the offers dict.
12. is_passenger_accepted: Returns True if the passenger has accepted a specific offer.
13. is_driver_accepted: Returns True if the driver has accepted a specific offer.
14. remove_acceptance: Removes an acceptance from the acceptances dict.
"""


import math
import heapq
import traci
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from classes.model import Model
if TYPE_CHECKING:
    from classes.logger import Logger


class RideServices:
    model: "Model"
    providers: dict
    logger: "Logger"

    def __init__(
            self,
            model: "Model",
            providers: dict,
            logger: "Logger"
        ):
        self.model = model
        self.__providers = providers
        self.logger = logger
        self.__expired_offers = []
        self.__expired_acceptances_p = 0
        self.__expired_acceptances_d = 0
        self.__rides_not_served = 0
        self.__generated_offers = 0
        self.__partial_acceptances = 0
        self.__offers = {}  # key: (res_id, driver_id), value: dict with travel time, route length, price
        self.__acceptances = {}  # key: (res_id, driver_id), value: (set of agents, timestamp)


    def step(self):
        self.__generate_offers()
        self.__check_matches()


    def __generate_offers(self) -> None:
        """
        Generates ride offers by matching unassigned passenger requests with the closest available idle taxis.

        This function:
        - Cleans up partial acceptances that have timed out for either passengers or drivers.
        - Compute the surge multiplier according to unassigned requests and available drivers for all providers.
        - Iterates over all unassigned passenger ride requests.
        - For each request, skips if an offer already exists.
        - Attempts to retrieve the passenger's current position; skips the request if unsuccessful.
        - Calculates the radius from each idle taxi to the passenger's position, skipping taxis with unavailable positions or farther than 10km.
        - Selects up to 8 closest taxis.
        - Creates ride offers for each selected taxi, including radius, travel time, route length, price information, and provider.
        - Skips requests if no taxis are available.
        - Updates the logger with average metrics for each provider.
        
        Returns:
        -------
        None
        """

        # Load information from drivers and passengers agents
        idle_taxis = self.model.drivers.get_idle_drivers()
        unassigned = self.model.passengers.get_unassigned_requests()
        timeout_p = self.model.passengers.get_passengers_timeout()
        timeout_d = self.model.drivers.get_drivers_timeout()
        idle_by_provider = self.model.drivers.get_idle_drivers_by_provider()
        self.__expired_acceptances_p = 0
        self.__expired_acceptances_d = 0
        self.__rides_not_served = 0
        self.__generated_offers = 0

        # Compute surge multiplier for all providers
        now = int(self.model.time)
        if (now % 300 == 0):
            # Convert cumulative shares to probabilities
            provider_names = list(self.__providers.keys())
            provider_probs = [self.__providers[provider_names[0]]["share"]] + [
                self.__providers[provider_names[i]]["share"] - self.__providers[provider_names[i-1]]["share"]
                for i in range(1, len(provider_names))
            ]
            # Map provider -> share probability
            shares = dict(zip(provider_names, provider_probs))
            for provider in self.__providers:
                requests_share = int(len(unassigned) * shares[provider])
                idle_count = len(idle_by_provider.get(provider, set()))
                self.__providers[provider]["surge_multiplier"] = self.__compute_surge_multiplier(
                    requests_share, idle_count, provider
                )
        for provider, conf in self.__providers.items():
            print(f"💵 Surge multiplier value for {provider}: {conf['surge_multiplier']}")

        # Clean up partial acceptances if timeout
        to_remove = []
        for key, (agents, timestamp) in self.__acceptances.items():
            if len(agents) == 1:
                if "passenger" in agents and now - timestamp >= timeout_p:
                    self.__expired_acceptances_p += 1
                    to_remove.append(key)
                elif "driver" in agents and now - timestamp >= timeout_d:
                    to_remove.append(key)
                    self.__expired_acceptances_d += 1
        for key in to_remove:
            self.remove_offer(key)
            self.remove_acceptance(key)
        print(f"⌛️ Timeout: {self.__expired_acceptances_p} passengers, {self.__expired_acceptances_d} drivers")

        # Clean up expired offers
        self.__expired_offers = [
            key for key, offer in self.__offers.items()
            if now - offer["timestamp"] >= self.model.agents_interval
        ]
        for key in self.__expired_offers:
            self.remove_offer(key)
        print(f"⌛️ Timeout for {len(self.__expired_offers)} offers")

        # Pre-compute and cache positions of all idle taxis
        taxi_positions = {}
        for taxi_id in idle_taxis:
            try:
                taxi_positions[taxi_id] = traci.vehicle.getPosition(taxi_id)
            except traci.TraCIException:
                print(f"⚠️ Failed to get position for taxi {taxi_id}")

        # Initialize statistics
        expected_time_sum = 0.0
        expected_length_sum = 0.0
        radius_sum = 0.0
        price_sum = 0.0
        surge_multiplier_sum = 0.0

        # Iterates over all passenger requests
        existing_res_ids = {r_id for (r_id, _) in self.__offers}
        for reservation in unassigned:
            res_id = reservation.id
            if res_id in existing_res_ids:
                print(f"⚠️ Reservation {res_id} already has an offer — skipping")
                continue
            person_id = reservation.persons[0]
            # Get passenger position
            try:
                pax_pos = traci.person.getPosition(person_id)
            except traci.TraCIException:
                print(f"⚠️ Failed to get position for reservation {res_id}: {reservation}")
                continue

            # Compute radius of each driver from the passenger
            taxis_radius = []
            for taxi_id, taxi_pos in taxi_positions.items():
                # Bounding box filter (within 10km square)
                if abs(taxi_pos[0] - pax_pos[0]) > 10000 or abs(taxi_pos[1] - pax_pos[1]) > 10000:
                    continue
                radius = math.hypot(taxi_pos[0] - pax_pos[0], taxi_pos[1] - pax_pos[1])
                if radius <= 10000:
                    taxis_radius.append((radius, taxi_id))

            # Get top 8 closest taxis
            closest_taxis = heapq.nsmallest(8, taxis_radius)
            if not closest_taxis:
                self.__rides_not_served += 1
                print(f"⚠️ No taxis available for reservation {res_id} — skipping")
                continue
            # Create offers
            from_edge = reservation.fromEdge
            to_edge = reservation.toEdge
            for radius, taxi_id in closest_taxis:
                offer_key = (res_id, taxi_id)
                provider = self.model.drivers.get_driver_provider(taxi_id)
                surge_multiplier = self.__providers[provider]["surge_multiplier"]
                try:
                    travel_time, route_length, price = self.__compute_offer(from_edge, to_edge, surge_multiplier, provider)
                except traci.TraCIException as e:
                    print(f"⚠️ Failed to compute route for offer {offer_key}: {e}")
                    continue
                self.__offers[offer_key] = {
                    "timestamp": now,
                    "radius": radius,
                    "time": travel_time,
                    "route_length": route_length,
                    "surge": surge_multiplier,
                    "price": price,
                    "provider": provider
                }
                # Update statistics
                expected_time_sum += travel_time
                expected_length_sum += route_length
                radius_sum += radius
                price_sum += price
                surge_multiplier_sum += surge_multiplier
        self.__generated_offers = len(self.__offers)
        print(f"📋 {self.__generated_offers} pending offers")

        # Compute average metrics
        avg_radius = round(radius_sum / len(self.__offers), 2) if len(self.__offers) > 0 else 0.0
        avg_price = round(price_sum / len(self.__offers), 2) if len(self.__offers) > 0 else 0.0
        avg_expected_time = round(expected_time_sum / len(self.__offers), 2) if len(self.__offers) > 0 else 0.0
        avg_expected_length = round(expected_length_sum / len(self.__offers), 2) if len(self.__offers) > 0 else 0.0
        avg_surge_multiplier = round(surge_multiplier_sum / len(self.__offers), 2) if len(self.__offers) > 0 else 0.0
        # Update the logger
        self.logger.update_offer_metrics(
            timestamp=self.model.time,
            avg_expected_time=avg_expected_time,
            avg_expected_length=avg_expected_length,
            avg_radius=avg_radius,
            avg_price=avg_price,
            avg_surge_multiplier=avg_surge_multiplier
        )


    def __check_matches(self) -> None:
        """
        Dispatches taxis for fully accepted matches and cleans up conflicting entries.
        This function:
        - Checks for fully accepted taxi-passenger matches.
        - Dispatches taxis for these matches.
        - Removes all related acceptances to prevent conflicts.
        - Updates the logger with the number of dispatched taxis and other metrics.

        Returns:
        -------
        None

        Notes:
        A match is considered fully accepted if both the driver and passenger have accepted.
        """
        # Get fully-accepted matches
        matched_keys = [
            (res_id, driver_id)
            for (res_id, driver_id), (agents, _) in self.__acceptances.items()
            if "driver" in agents and "passenger" in agents
        ]
        print(f"🚕 Dispatching {len(matched_keys)} taxis")
        # Count partial acceptances for log
        self.__partial_acceptances = sum(
            1 for agents, _ in self.__acceptances.values()
            if len(agents) == 1
        )

        # For each match try to dispatch the taxi
        for res_id, driver_id in matched_keys:
            try:
                traci.vehicle.dispatchTaxi(driver_id, [res_id])
            except traci.TraCIException as e:
                print(f"❌ DispatchTaxi failed: {e} — driver: {driver_id}, res_id: {res_id}")
            except Exception as e:
                print(f"❌ Unknown error during dispatch: {e}")
            finally:
                # Remove all offers and acceptances involving the same reservation or driver
                to_remove = [
                    key for key in self.__acceptances
                    if key[0] == res_id or key[1] == driver_id
                ]
                for key in to_remove:
                    self.remove_offer(key)
                    self.remove_acceptance(key)
        
        # Update the logger
        self.logger.update_rideservices(
            timestamp = self.model.time,
            dispatched_taxis = len(matched_keys),
            generated_offers = self.__generated_offers,
            timeout_offers = len(self.__expired_offers),
            partial_acceptances = self.__partial_acceptances,
            requests_canceled = self.__expired_acceptances_p,
            requests_not_served = self.__rides_not_served
        )


    def __compute_offer(
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
        # Convert units
        travel_minutes = travel_time / 60
        route_km = round(route_length / 1000, 3)
        # Provider-specific pricing table
        config = self.__providers[provider]
        base = config["base_price"]
        cost_per_min = config["cost_per_min"]
        cost_per_km = config["cost_per_km"]
        fee = config["service_fee"]
        min_price = config["min_price"]
        # Compute price
        price = (base + cost_per_min * travel_minutes + cost_per_km * route_km) * surge_multiplier + fee
        price = round(max(min_price, price), 3)

        return travel_time, route_length, price
    

    def __compute_surge_multiplier(
            self,
            unassigned_requests: set,
            idle_drivers: set,
            provider: str
            ) -> float:
        """
        Computes the surge multiplier based on the ratio of unassigned ride requests
        to available idle drivers. The multiplier ranges from 1.0 (normal conditions)
        to max_surge for each provider.

        Parameters:
        ----------
        - unassigned_requests: int
            Number of pending requests.
        - idle_drivers: int
            Number of available drivers.

        Returns
        -------
        surge : float
            Surge price multiplier.
        """
        config = self.__providers[provider]
        if idle_drivers == 0:
            # Max surge if no drivers available
            return config["max_surge"]
        ratio = unassigned_requests / idle_drivers
        surge = min(config["max_surge"], max(1, ratio))
        return round(surge, 2)


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
        agents, _ = self.__acceptances.setdefault(key, (set(), now))
        agents.add(agent)
        # If there are two agents that means both driver and passenger accepted
        if len(agents) == 2:
            self.remove_offer(key)


    def reject_offer(
            self,
            key
        ) -> None:
        """
        Rejects an offer for a ride by removing it from offers.

        Parameters:
        ----------
        key: tuple 
            Key (res_id, driver_id) of the offer to remove.

        Returns:
        -------
        None
        """
        self.remove_offer(key)
        self.remove_acceptance(key)


    def get_offers(
            self
        ) -> dict:
        """
        Gets the dictionary of offers.

        Returns:
        -------
        dict
            A dictionary containing all offers.
        """
        return self.__offers


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
            k: v for k, v in self.__offers.items()
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
            k: v for k, v in self.__offers.items()
            if k[0] in passengers
        }


    def remove_offer(
            self,
            key: tuple
        ) -> None:
        """
        Removes a specified offer from offers dictionary.

        Parameters:
        ----------
        key: tuple
            Key (res_id, offer) of the offer to remove.

        Returns:
        -------
        None
        """
        self.__offers.pop(key, None)


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
            key in self.__acceptances and
            "passenger" in self.__acceptances[key][0]
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
            key in self.__acceptances and
            "driver" in self.__acceptances[key][1]
        )

    
    def remove_acceptance(
            self,
            key: tuple
        ) -> None:
        """
        Removes a specified acceptance from acceptance dictionary.

        Parameters:
        ----------
        key: tuple
            Key (res_id, driver_id) of the acceptance to remove.

        Returns:
        -------
        None
        """
        self.__acceptances.pop(key, None)