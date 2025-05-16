import traci
import math
import heapq
from collections import defaultdict


class RideService:
    def __init__(self, model):
        self.model = model
        self.offers = {}  # key: (res_id, driver_id), value: dict with time, distance, price
        self.acceptances = {}  # key: (res_id, driver_id), value: (set of agents, timestamp)


    def step(self):
        print(f"🧍 {len(set(traci.person.getTaxiReservations(3)))} unassigned passengers")
        print(f"🧍 {len(set(traci.person.getTaxiReservations(4)))} assigned passengers")
        print(f"🧍 {len(set(traci.person.getTaxiReservations(8)))} picked-up passengers")
        print(f"🚕 {len(set(traci.vehicle.getTaxiFleet(0)))} idle taxis")
        print(f"🚕 {len(set(traci.vehicle.getTaxiFleet(1)))} pick-up taxis")
        print(f"🚕 {len(set(traci.vehicle.getTaxiFleet(2)))} occupied taxis")
        self._generate_offers()
        print(f"Generated {len(self.offers)} offers")
        self._check_matches()
        print(f"🧍 {len(set(traci.person.getTaxiReservations(3)))} unassigned passengers")
        print(f"🧍 {len(set(traci.person.getTaxiReservations(4)))} assigned passengers")
        print(f"🧍 {len(set(traci.person.getTaxiReservations(8)))} picked-up passengers")
        print(f"🚕 {len(set(traci.vehicle.getTaxiFleet(0)))} idle taxis")
        print(f"🚕 {len(set(traci.vehicle.getTaxiFleet(1)))} pick-up taxis")
        print(f"🚕 {len(set(traci.vehicle.getTaxiFleet(2)))} occupied taxis")

    def _generate_offers(self):
        idle_taxis = self.model.driver.idle_drivers
        unassigned = self.model.passenger.unassigned_requests
        now = self.model.time
        timeout_p = self.model.passenger.timeout
        timeout_d = self.model.driver.timeout

        # Clean up partial acceptances
        for key, (agents, timestamp) in list(self.acceptances.items()):
            age = now - timestamp
            if len(agents) == 1:
                if "passenger" in agents and age > timeout_p:
                    self.acceptances.pop(key, None)
                    self.offers.pop(key, None)
                elif "driver" in agents and age > timeout_d:
                    self.acceptances.pop(key, None)
                    self.offers.pop(key, None)

        for reservation in unassigned:
            res_id = reservation.id
            # Skip reservation if it's already being considered
            if any(res_id == r_id for (r_id, _) in self.offers):
                print(f"⚠️ Reservation {res_id} already has an offer — skipping")
                continue
            try:
                pax_pos = traci.person.getPosition(reservation.persons[0])
            except traci.TraCIException:
                print(f"⚠️ Failed to get position for reservation {res_id}: {reservation}")
                continue

            taxis_with_dist = []
            for taxi_id in idle_taxis:
                try:
                    taxi_pos = traci.vehicle.getPosition(taxi_id)
                    dist = math.hypot(taxi_pos[0] - pax_pos[0], taxi_pos[1] - pax_pos[1])
                    taxis_with_dist.append((dist, taxi_id))
                except traci.TraCIException:
                    print(f"⚠️ Failed to get position for taxi {taxi_id}: {reservation}")
                    continue

            # Get top 8 closest taxis using heapq
            closest_taxis = [(dist, taxi_id) for (dist, taxi_id) in taxis_with_dist if dist <= 10000]
            closest_taxis = heapq.nsmallest(8, taxis_with_dist)

            # Create offers
            if closest_taxis:
                for dist, taxi_id in closest_taxis:
                    offer_key = (res_id, taxi_id)
                    self.offers[offer_key] = {
                        "time": 300,
                        "distance": dist,
                        "price": 10
                    }
            else:
                print(f"⚠️ No taxis available for reservation {res_id} — skipping")
                continue


    def _check_matches(self):
        reservations = traci.person.getTaxiReservations(3)
        valid_res_ids = {r.id for r in reservations}
        matched = [
            key for key, (agents, _) in self.acceptances.items()
            if "driver" in agents and "passenger" in agents
        ]
        print(f"✅ Accepted {len(matched)} trips (fully matched)")

        for (res_id, driver_id), (agents, _) in list(self.acceptances.items()):
            if "driver" in agents and "passenger" in agents:
                if res_id not in valid_res_ids:
                    print(f"⚠️ Reservation {res_id} no longer valid — cleaning up")
                    self.acceptances.pop((res_id, driver_id), None)
                    self.offers.pop((res_id, driver_id), None)
                    continue
                try:
                    if not driver_id in traci.vehicle.getTaxiFleet(0):
                        print(f"⚠️ Driver {driver_id} no longer exists — skipping dispatch")
                        self.acceptances.pop((res_id, driver_id), None)
                        self.offers.pop((res_id, driver_id), None)
                        continue
                    if res_id not in valid_res_ids:
                        print(f"⚠️ Reservation {res_id} vanished since last check — skipping dispatch")
                        self.acceptances.pop((res_id, driver_id), None)
                        self.offers.pop((res_id, driver_id), None)
                        continue

                    traci.vehicle.dispatchTaxi(driver_id, [res_id])

                except traci.TraCIException as e:
                    print(f"❌ DispatchTaxi failed: {e} — driver: {driver_id}, res_id: {res_id}")
                except Exception as e:
                    print(f"❌ Unknown error during dispatch: {e}")
                finally:
                    # Remove other complete and partial matches for same passenger or driver
                    for k in list(self.acceptances):
                        if (k[0] == res_id or k[1] == driver_id):
                            self.acceptances.pop(k, None)
                            self.offers.pop(k, None)


    def get_offers_for_drivers(self, drivers):
        return {
            k: v for k, v in self.offers.items()
            if k[1] in drivers
        }

    def get_offers_for_passengers(self, passengers):
        return {
            k: v for k, v in self.offers.items()
            if k[0] in passengers
        }
    
    def accept_offer(self, res_id, driver_id, agent):
        now = self.model.time
        key = (res_id, driver_id)
        agents, _ = self.acceptances.setdefault(key, (set(), now))
        agents.add(agent)
        if len(agents) == 2:
            self.offers.pop(key, None)