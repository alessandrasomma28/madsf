from mesa import Agent

class Driver(Agent):
    """
    Driver represents a ride-hailing driver.

    Attributes:
        current_taz (int): TAZ where the driver currently is.
        (income (float): Total income earned by the driver.)
        available (bool): Whether the driver is available to take a ride.
        (status (str): Idle, Pickup, Moving, OnRoad.)
        current_passenger (int): ID of the current passenger if matched.
    """

    def __init__(self, unique_id, model, current_taz, vehicle_id):
        super().__init__(unique_id, model)
        self.current_taz = current_taz
        self.income = 0.0
        self.available = True
        self.current_passenger = None
        self.vehicle_id = vehicle_id

    def step(self):
        if self.available:
            self.current_passenger = self.model.rideservice.assign_ride(self)
            if self.current_passenger:
                self.available = False
                self.set_route_to(self.current_passenger.origin_edge)

        if self.current_passenger:
            if self.reached(self.current_passenger.origin_edge):
                self.set_route_to(self.current_passenger.destination_edge)

            if self.reached(self.current_passenger.destination_edge):
                self.available = True
                self.model.rideservice.complete_ride(self.current_passenger)
                self.current_passenger = None

    def set_route_to(self, destination_edge):
        # Integrate with SUMO via traci
        return

    def reached(self, target_edge):
        # Check if the SUMO vehicle reached the edge
        return False