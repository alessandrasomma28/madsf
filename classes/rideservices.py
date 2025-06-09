"""
rideservices.py

This module defines the RideServices class, which manages ride offers and acceptances between passengers and drivers.
It supports the following operations:

1. step: Advances the ride services logic by generating new offers and checking for matches.
2. __generate_offers: Internal method to create offers for unassigned reservations and nearby idle taxis.
3. __check_matches: Internal method to dispatch taxis when both driver and passenger have accepted an offer.
4. __compute_offer: Computes travel time, route length, and price of a reservation.
5. __compute_surge_multiplier: Computes the price multiplication factor based on the ratio between
    number of offers for passengers and number of available drivers.
6. accept_offer: Registers an acceptance of an offer by either a driver or passenger.
7. reject_offer: Registers a reject of an offer by either a driver or passenger and removes the related offer (and acceptance).
8. get_offers: Returns the dictionary of all offers.
9. get_offers_for_drivers: Returns offers relevant to the given list of driver IDs.
10. get_offers_for_passengers: Returns offers relevant to the given list of passenger reservation IDs.
11. remove_offer: Removes an offer from the offers dict.
12. is_passenger_accepted: Returns True if the passenger has accepted a specific offer.
13. get_acceptances: Returns the dictionary of all acceptances.
14. remove_acceptance: Removes an acceptance from the acceptances dict.
"""


import math
import heapq
import traci
from collections import defaultdict
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
            logger: "Logger",
        ):
        self.model = model
        self.__providers = providers
        self.logger = logger
        self.__expired_acceptances_p = 0
        self.__expired_acceptances_d = 0
        self.__rides_not_served = 0
        self.__generated_offers = 0
        self.__partial_acceptances = 0
        self.__offers = {}  # key: (res_id, driver_id), value: dict with travel time, route length, price
        self.__acceptances = {}  # key: (res_id, driver_id), value: (set of agents, timestamp)
        self.__miles_radius_max = 16093
        self.__max_offers_per_reservation = 8
        self.__radius_square = self.__miles_radius_max ** 2


    def step(self) -> None:
        self.__generate_offers()
        self.__check_matches()


    def __generate_offers(self) -> None:
        """
        Generates ride offers by matching unassigned passenger requests with the closest available idle taxis.

        This function:
        - Compute the surge multiplier according to unassigned requests and available drivers for all providers.
        - Cleans up partial acceptances that have timed out for either passengers or drivers.
        - Iterates over all unassigned passenger ride requests.
        - Attempts to retrieve the passenger's current position; skips the request if unsuccessful.
        - Calculates the radius from each idle taxi to the passenger's position, skipping taxis with unavailable positions or farther than 10 miles.
        - Selects up to 8 closest taxis.
        - Creates ride offers for each selected taxi, including radius, travel time, route length, price information, and provider.
        - Skips requests if no taxis are available.
        - Updates the logger with average metrics for each provider.
        
        Returns:
        -------
        None
        """

        # Load information from drivers and passengers agents
        now = int(self.model.time)
        idle_taxis = self.model.drivers.get_idle_drivers()
        unassigned = sorted(self.model.passengers.get_unassigned_requests(), key=lambda r: r.reservationTime)
        unassigned_surge = traci.person.getTaxiReservations(3)
        canceled = self.model.passengers.get_canceled_requests()
        idle_by_provider = self.model.drivers.get_idle_drivers_by_provider()

        self.__expired_acceptances_p = 0
        self.__expired_acceptances_d = 0
        self.__rides_not_served = 0
        self.__offers = {}
        self.__generated_offers = 0

        # Compute surge multiplier for all providers
        if now % 300 == 0:
            # Convert cumulative shares to probabilities
            provider_names = list(self.__providers.keys())
            provider_probs = [self.__providers[provider_names[0]]["share"]] + [
                self.__providers[provider_names[i]]["share"] - self.__providers[provider_names[i-1]]["share"]
                for i in range(1, len(provider_names))
            ]
            # Map provider -> share probability
            shares = dict(zip(provider_names, provider_probs))
            for provider in self.__providers:
                requests_share = int((len(unassigned_surge) + len(canceled)) * shares[provider])
                idle_count = len(idle_by_provider.get(provider, set()))
                self.__providers[provider]["surge_multiplier"] = self.__compute_surge_multiplier(
                    requests_share, idle_count, provider
                )
        if self.model.verbose:
            for provider, conf in self.__providers.items():
                print(f"💵 Surge multiplier value for {provider}: {conf['surge_multiplier']}")

        # Clean up partial acceptances if timeout
        to_remove = []
        for key, (agents, timestamp) in self.__acceptances.items():
            if len(agents) == 1:
                if "passenger" in agents and now - timestamp >= self.model.agents_interval:
                    self.__expired_acceptances_p += 1
                    to_remove.append(key)
                elif "driver" in agents and now - timestamp >= self.model.agents_interval:
                    to_remove.append(key)
                    self.__expired_acceptances_d += 1
        for key in to_remove:
            self.remove_acceptance(key)
        if self.model.verbose:
            print(f"⌛️ Timeout: {self.__expired_acceptances_p} passengers, {self.__expired_acceptances_d} drivers")

        # Pre-compute and cache positions of all idle taxis
        taxi_positions = {}
        for taxi_id in idle_taxis:
            try:
                taxi_positions[taxi_id] = traci.vehicle.getPosition(taxi_id)
            except traci.TraCIException:
                print(f"⚠️ Failed to get position for taxi {taxi_id}")

        offer_stats = defaultdict(float)
        tot_res = 0
        # Iterate over all passenger requests
        for reservation in unassigned:
            tot_res += 1
            res_id = reservation.id
            person_id = reservation.persons[0]

            # Get passenger position
            try:
                pax_x, pax_y = traci.person.getPosition(person_id)
            except traci.TraCIException:
                print(f"⚠️ Failed to get position for reservation {res_id}: {reservation}")
                continue

            # Filter nearby taxis using bounding box and Euclidean distance
            taxis_radius = []
            for taxi_id, (tx, ty) in taxi_positions.items():
                dx, dy = tx - pax_x, ty - pax_y
                if abs(dx) > self.__miles_radius_max or abs(dy) > self.__miles_radius_max:
                    continue
                dist_sq = dx * dx + dy * dy
                if dist_sq <= self.__radius_square:
                    taxis_radius.append((dist_sq, taxi_id))
            
            # Get top 8 closest taxis
            closest_taxis = [
                (math.sqrt(dist_sq), taxi_id)
                for dist_sq, taxi_id in heapq.nsmallest(self.__max_offers_per_reservation, taxis_radius)
            ]
            if not closest_taxis:
                self.__rides_not_served += 1
                if self.model.verbose:
                    print(f"⚠️ No taxis available for reservation {res_id} — skipping")
                continue
            
            # Create offers
            from_edge = reservation.fromEdge
            to_edge = reservation.toEdge
            cached_offer_by_provider = {}
            for radius, taxi_id in closest_taxis:
                offer_key = (res_id, taxi_id)
                provider = self.model.drivers.get_driver_provider(taxi_id)
                surge_multiplier = self.__providers[provider]["surge_multiplier"]
                # Check if the offer with the same provider already exists in cache
                if provider in cached_offer_by_provider:
                    travel_time, route_length, price = cached_offer_by_provider[provider]
                else:
                    try:
                        travel_time, route_length, price = self.__compute_offer(from_edge, to_edge, surge_multiplier, provider)
                        cached_offer_by_provider[provider] = (travel_time, route_length, price)
                    except traci.TraCIException as e:
                        print(f"⚠️ Failed to compute route for offer {offer_key}: {e}")
                        continue
                self.__offers[offer_key] = {
                    "timestamp": now,
                    "radius": radius,
                    "time": travel_time * 60,
                    "route_length": route_length * 1000,
                    "surge": surge_multiplier,
                    "price": price,
                    "provider": provider
                }
                # Update statistics
                self.__generated_offers += 1
                offer_stats["radius"] += radius
                offer_stats["price"] += price
                offer_stats["time"] += travel_time
                offer_stats["length"] += route_length
                offer_stats["surge"] += surge_multiplier
                
        if self.model.verbose:
            print(f"📋 {len(self.__offers)} pending offers for {tot_res} reservations")

        # Compute average metrics and update the logger
        self.logger.update_offer_metrics(
            timestamp=self.model.time,
            avg_expected_time=round(offer_stats["time"] / self.__generated_offers, 2) if self.__generated_offers else 0.0,
            avg_expected_length=round(offer_stats["length"] / self.__generated_offers, 2) if self.__generated_offers else 0.0,
            avg_radius=round(offer_stats["radius"] / self.__generated_offers, 2) if self.__generated_offers else 0.0,
            avg_price=round(offer_stats["price"] / self.__generated_offers, 2) if self.__generated_offers else 0.0,
            avg_surge_multiplier=round(offer_stats["surge"] / self.__generated_offers, 2) if self.__generated_offers else 0.0
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
            key for key, (agents, _) in self.__acceptances.items()
            if "driver" in agents and "passenger" in agents
        ]
        if self.model.verbose:
            print(f"🚕 Dispatching {len(matched_keys)} taxis")
        # Count partial acceptances for log
        self.__partial_acceptances = sum(1 for agents, _ in self.__acceptances.values() if len(agents) == 1)

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
                    self.remove_acceptance(key)
        
        # Update the logger
        self.logger.update_rideservices(
            timestamp = self.model.time,
            dispatched_taxis = len(matched_keys),
            generated_offers = self.__generated_offers,
            partial_acceptances = self.__partial_acceptances,
            requests_not_served = self.__rides_not_served + self.__expired_acceptances_p
        )


    def __compute_offer(
            self,
            from_edge: str,
            to_edge: str,
            surge: float,
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
        - surge: float
            Multiplication factor applied to the final price.
        - provider: str
            Provider of the ride service.

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
        travel_min = travel_time / 60
        route_km = round(route_length / 1000, 3)
        # Provider-specific pricing table
        config = self.__providers[provider]
        # Compute price
        price = (
            config["base_price"]
            + config["cost_per_min"] * travel_min
            + config["cost_per_km"] * route_km
        ) * surge + config["service_fee"]
        price = round(max(config["min_price"], price), 3)

        return travel_time, route_length, price


    def __compute_surge_multiplier(
            self,
            pending_requests: int,
            idle_drivers: int,
            provider: str
            ) -> float:
        """
        Computes the surge multiplier based on the ratio of passengers pending requests
        to available idle drivers. The multiplier ranges from 1.0 (normal conditions)
        to max_surge for each provider.

        Parameters:
        ----------
        - pending_requests: int
            Number of passengers pending requests.
        - idle_drivers: int
            Number of available drivers.
        - provider: str
            Provider of the ride service.

        Returns
        -------
        surge: float
            Surge price multiplier.
        """
        config = self.__providers[provider]
        if idle_drivers == 0:
            # Max surge if no drivers available
            return config["max_surge"]
        ratio = pending_requests / idle_drivers
        surge = min(config["max_surge"], max(1, ratio))
        if self.model.verbose:
            print(f"💵 New surge multiplier value for {provider}: {round(surge, 2)} (pending requests: {pending_requests}, available drivers: {idle_drivers})")
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
        return {k: v for k, v in self.__offers.items() if k[1] in drivers}


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
        return {k: v for k, v in self.__offers.items() if k[0] in passengers}


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
        return "passenger" in self.__acceptances.get(key, ({}, 0))[0]
    

    def get_acceptances(
            self
        ) -> dict:
        """
        Gets the dictionary of acceptances.

        Returns:
        -------
        dict
            A dictionary containing all acceptances.
        """
        return self.__acceptances

    
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