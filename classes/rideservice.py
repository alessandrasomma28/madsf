from mesa import Agent

class RideSharingService(Agent):
    """
    RideSharingService acts as a centralized dispatcher for ride-matching.

    Attributes:
        pending_requests (list): Queue of unmatched PassengerAgents.
        available_drivers (list): List of available DriverAgents.
    """
    def __init__(self, unique_id, model):
        """
        Initialize a RideSharingService.

        Args:
            unique_id (int): Unique identifier for the service agent.
            model (Model): The simulation model instance.
        """
        super().__init__(unique_id, model)
        self.pending_requests = []
        self.available_drivers = []

    def register_request(self, passenger):
        """
        Add a passenger to the queue of ride requests.

        Args:
            passenger (PassengerAgent): Passenger needing a ride.
        """
        self.pending_requests.append(passenger)

    def register_driver(self, driver):
        """
        Add a driver to the list of available drivers.

        Args:
            driver (DriverAgent): Driver available for matching.
        """
        self.available_drivers.append(driver)

    def step(self):
        """
        Called at each simulation step. Matches available drivers to waiting passengers
        and calculates fare including surge pricing.
        """
        matches = self._match_passengers_to_drivers()
        for passenger, driver in matches:
            passenger.match_with_driver(driver.unique_id)
            driver.available = False
            driver.current_passenger = passenger.unique_id
            fare = self._calculate_fare(passenger, driver)
            driver.income += fare

    def _match_passengers_to_drivers(self): # Minimize distance
        """
        Match passengers to drivers in FIFO order.

        Returns:
            list of tuples: Each tuple is (PassengerAgent, DriverAgent)
        """
        matches = []
        while self.pending_requests and self.available_drivers:
            passenger = self.pending_requests.pop(0)
            driver = self.available_drivers.pop(0)
            matches.append((passenger, driver))
        return matches

    def _calculate_fare(self, passenger, driver):
        """
        Compute fare based on base fare, distance, and surge multiplier.

        Args:
            passenger (PassengerAgent): The matched passenger.
            driver (DriverAgent): The matched driver.

        Returns:
            float: Calculated fare.
        """
        base_fare = 3.0
        surge = self._compute_surge(passenger.origin_taz)
        distance = 5  # Placeholder for route distance
        fare = base_fare + (distance * 1.5 * surge)
        return fare

    def _compute_surge(self, taz):
        """
        Compute a surge multiplier for the given TAZ.

        Args:
            taz (int): Traffic Analysis Zone.

        Returns:
            float: Surge multiplier.
        """
        demand = sum(p.origin_taz == taz for p in self.model.passengers)
        supply = sum(d.current_taz == taz for d in self.model.drivers if d.available)
        if supply == 0:
            return 2.0
        ratio = demand / supply
        return min(1.0 + 0.5 * ratio, 3.0)

    def complete_ride(self, passenger):
        # TODO
        return

    def assign_ride(self, driver):
        # Assign the first pending request
        if self.pending_requests:
            return self.pending_requests.pop(0)
        return None