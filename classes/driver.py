from mesa import Agent

class Driver(Agent):
    """
    Driver represents a ride-hailing driver.

    Attributes:
        current_taz (int): TAZ where the driver currently is.
        available (bool): Whether the driver is available to take a ride.
        current_passenger (int): ID of the current passenger if matched.
    """