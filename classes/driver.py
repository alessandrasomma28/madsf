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

    def __init__(self, vid, model, start_lane, depart_time):
        super().__init__(vid, model)
        self.start_lane = start_lane
        self.depart_time = depart_time
        self.status = "idle"
        self.current_passenger = None
        self.income = 0.0

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