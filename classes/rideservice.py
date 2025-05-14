import traci
import math
from collections import defaultdict


class RideService:
    def __init__(self, model):
        self.model = model
        self.offers = {}  # key: (res_id, driver_id), value: dict with time, distance, price
        self.acceptances = defaultdict(set)  # "driver"/"passenger" keys


    def step(self):
        self._generate_offers()
        self._check_matches()

    def _generate_offers(self):
        idle_taxis = self.model.driver.idle_drivers
        unassigned = self.model.passenger.unassigned_requests

        for reservation in unassigned:
            res_id = reservation.id
            try:
                pax_pos = traci.person.getPosition(reservation.persons[0])
            except traci.TraCIException:
                print(f"⚠️ Failed to get position for reservation {res_id}: {reservation}")
                continue

            taxis_sorted = []
            for taxi_id in idle_taxis:
                if any(res_id == r_id for (r_id, _) in self.offers):  
                    print(f"⚠️ Reservation {res_id} already has an offer — skipping")
                    continue  # already being considered
                try:
                    taxi_pos = traci.vehicle.getPosition(taxi_id)
                    dist = math.hypot(taxi_pos[0] - pax_pos[0], taxi_pos[1] - pax_pos[1])
                    taxis_sorted.append((dist, taxi_id))
                except traci.TraCIException:
                    print(f"⚠️ Failed to get position for taxi {taxi_id}: {reservation}")
                    continue

            taxis_sorted.sort()
            for _, taxi_id in taxis_sorted[:8]:
                offer_key = (res_id, taxi_id)
                self.offers[offer_key] = {
                    "time": 300,  # dummy
                    "distance": dist,
                    "price": 10 # dummy
                }


    def _check_matches(self):
        reservations = traci.person.getTaxiReservations(3)
        valid_res_ids = {r.id for r in reservations}

        for (res_id, driver_id), agents in list(self.acceptances.items()):
            if "driver" in agents and "passenger" in agents:
                if res_id not in valid_res_ids:
                    print(f"⚠️ Reservation {res_id} no longer valid — cleaning up")
                    self.acceptances.pop((res_id, driver_id), None)
                    continue

                try:
                    if not driver_id in traci.vehicle.getTaxiFleet(0):
                        print(f"⚠️ Driver {driver_id} no longer exists — skipping dispatch")
                        continue
                    if res_id not in valid_res_ids:
                        print(f"⚠️ Reservation {res_id} vanished since last check — skipping dispatch")
                        continue

                    print(f"🚕 Dispatching driver {driver_id} to reservation {res_id}")
                    traci.vehicle.dispatchTaxi(driver_id, [res_id])

                except traci.TraCIException as e:
                    print(f"❌ DispatchTaxi failed: {e} — driver: {driver_id}, res_id: {res_id}")
                except Exception as e:
                    print(f"❌ Unknown error during dispatch: {e}")
                finally:
                    self.acceptances.pop((res_id, driver_id), None)


    def get_offers_for_drivers(self, drivers):
        return {
            k[1]: v for k, v in self.offers.items()
            if k[1] in drivers
        }

    def get_offers_for_passengers(self, passengers):
        return {
            k[0]: v for k, v in self.offers.items()
            if k[0] in passengers
        }
    
    def accept_offer(self, agent_id, agent):
        for (res_id, driver_id) in list(self.offers.keys()):
            if (agent == "driver" and driver_id == agent_id) or (agent == "passenger" and res_id == agent_id):
                self.acceptances[(res_id, driver_id)].add(agent)

                if agent == "driver" and driver_id in self.model.driver.idle_drivers:
                    self.model.driver.idle_drivers.remove(driver_id)
                elif agent == "passenger":
                    self.model.passenger.unassigned_requests = [
                        r for r in self.model.passenger.unassigned_requests if r.id != res_id
                    ]

                # Remove other offers for the same agent
                to_remove = [
                    key for key in self.offers
                    if ((agent == "driver" and key[1] == agent_id) or
                        (agent == "passenger" and key[0] == agent_id)) and
                        key != (res_id, driver_id)
                ]
                for key in to_remove:
                    self.offers.pop(key, None)
                return