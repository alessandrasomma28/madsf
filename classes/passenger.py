from mesa import Agent
import random

class Passenger(Agent):
    """
    Passenger represents a passenger requesting a ride.

    Attributes:
        origin_taz (int): TAZ from where the passenger starts.
        destination_taz (int): TAZ where the passenger wants to go.
        request_time (int): Simulation time when request was made.
        wait_time (int): Total time waited until matched.
        matched (bool): Whether the passenger has been matched to a driver.
        driver_id (int): ID of the matched driver.
    """

    def __init__(self, pid, model, request_time, origin, destination):
        super().__init__(pid, model)
        self.request_time = request_time
        self.origin = origin
        self.destination = destination
        self.wait_time = self.model.time - self.request_time
        self.status = "waiting"
        self.assigned_driver = None


    def step(self, model):
        if self.status == "waiting":
            self.wait_time = model.time - self.request_time
            if self.wait_time > 900:
                self.status = "timed_out"
                print(f"[!] Passenger {self.id} timed out after {self.wait_time}s")


    def match_with_driver(self, driver_id):
        """
        Mark the passenger as matched with a driver.

        Args:
            driver_id (int): ID of the matched driver.
        """
        self.driver_id = driver_id
        self.matched = True