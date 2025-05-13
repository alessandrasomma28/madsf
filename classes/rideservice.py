import traci
import math

class RideService:
    """
    This class manages the ride service operations, including dispatching taxis to unassigned reservations.
    It uses the SUMO TraCI API to interact with the simulation environment.
    """
    def __init__(self, model):
        self.model = model

    def step(self, model):
        self._dispatch_taxis()

    def _dispatch_taxis(self):
        """
        Dispatch taxis to unassigned reservations based on proximity.
        This function checks for idle taxis and unassigned reservations, and assigns the closest taxi to each reservation.
        """
        # Get the list of idle taxis and unassigned reservations
        idle_taxis = list(traci.vehicle.getTaxiFleet(0))
        unassigned_reservations = traci.person.getTaxiReservations(3)
        if not idle_taxis or not unassigned_reservations:
            return

        assignments = []  # (taxiID, reservationID)

        # Iterate over unassigned reservations and find the closest idle taxi to each reservation
        for reservation in unassigned_reservations:
            pax_id = reservation.persons[0]
            try:
                pax_pos = traci.person.getPosition(pax_id)
            except traci.TraCIException as e:
                print(f"⚠️ Could not get position for {pax_id}: {e}")
                continue
            
            best_taxi = None
            best_distance = float("inf")
            for taxi_id in idle_taxis:
                taxi_pos = traci.vehicle.getPosition(taxi_id)
                # Calculate the distance between the taxi and the passenger with Euclidean distance
                dist = math.hypot(taxi_pos[0] - pax_pos[0], taxi_pos[1] - pax_pos[1])
                # Check if this taxi is closer than the best found so far
                if dist < best_distance:
                    best_distance = dist
                    best_taxi = taxi_id

            if best_taxi:
                # Assign the best taxi to the reservation
                assignments.append((best_taxi, reservation.id))
                # Remove assigned taxi from the list of idle taxis
                idle_taxis.remove(best_taxi)

        # Perform the dispatches
        for taxi_id, res_id in assignments:
            try:
                traci.vehicle.dispatchTaxi(taxi_id, [res_id])
            except traci.TraCIException as e:
                print(f"⚠️ Dispatch failed ({taxi_id} → {res_id}): {e}")