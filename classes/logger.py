"""
logger.py

This class defines a Logger that writes multi-agent simulation data to an XML file at each timestamp.
"""


from pathlib import Path
import os
import xml.etree.ElementTree as ET
import xml.dom.minidom as minidom
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from classes.model import Model


class Logger:
    model: "Model"
    output_path: str

    def __init__(
            self,
            model: "Model",
            output_dir_path: str
        ):
        self.model = model
        self.output_path = Path(os.path.join(output_dir_path, "multi_agent_infos.xml"))
        if os.path.exists(self.output_path):
            os.remove(self.output_path)
        self.root = ET.Element("MultiAgentLog")
        self.tree = ET.ElementTree(self.root)
        self.__write()


    def update_passengers(
            self,
            timestamp: int,
            unassigned_requests: int,
            assigned_requests: int,
            accepted_requests: int,
            rejected_requests: int,
            canceled_requests: int
            ) -> None:
        entry = ET.SubElement(self.root, "step", timestamp=str(timestamp))
        passengers_el = ET.SubElement(entry, "passengers")
        ET.SubElement(passengers_el, "unassigned_requests").text = str(unassigned_requests)
        ET.SubElement(passengers_el, "assigned_requests").text = str(assigned_requests)
        ET.SubElement(passengers_el, "accepted_requests").text = str(accepted_requests)
        ET.SubElement(passengers_el, "rejected_requests").text = str(rejected_requests)
        ET.SubElement(passengers_el, "canceled_requests").text = str(canceled_requests)
        self.__write()


    def update_drivers(
            self,
            timestamp: int,
            idle_drivers: int,
            pickup_drivers: int,
            busy_drivers: int,
            accepted_requests: int,
            rejected_requests: int,
            removed_drivers: int
            ) -> None:
        entry = ET.SubElement(self.root, "step", timestamp=str(timestamp))
        drivers_el = ET.SubElement(entry, "drivers")
        ET.SubElement(drivers_el, "idle_drivers").text = str(idle_drivers)
        ET.SubElement(drivers_el, "pickup_drivers").text = str(pickup_drivers)
        ET.SubElement(drivers_el, "busy_drivers").text = str(busy_drivers)
        ET.SubElement(drivers_el, "accepted_requests").text = str(accepted_requests)
        ET.SubElement(drivers_el, "rejected_requests").text = str(rejected_requests)
        ET.SubElement(drivers_el, "removed_drivers").text = str(removed_drivers)
        self.__write()


    def update_rideservices(
            self,
            timestamp: int,
            dispatched_taxis: int,
            generated_offers: int,
            partial_acceptances: int,
            requests_not_served: int
            ) -> None:
        entry = ET.SubElement(self.root, "step", timestamp=str(timestamp))
        rideservices_el = ET.SubElement(entry, "rideservices")
        ET.SubElement(rideservices_el, "dispatched_taxis").text = str(dispatched_taxis)
        ET.SubElement(rideservices_el, "generated_offers").text = str(generated_offers)
        ET.SubElement(rideservices_el, "partial_acceptances").text = str(partial_acceptances)
        ET.SubElement(rideservices_el, "requests_not_served").text = str(requests_not_served)
        self.__write()


    def update_offer_metrics(
            self,
            timestamp: int,
            avg_expected_time: float,
            avg_expected_length: float,
            avg_radius: float,
            avg_price: float,
            avg_surge_multiplier: float,
            offers_by_provider: dict,
            surge_by_provider: dict
        ) -> None:
        entry = ET.SubElement(self.root, "step", timestamp=str(timestamp))
        offers_el = ET.SubElement(entry, "offers")
        ET.SubElement(offers_el, "avg_expected_time").text = str(avg_expected_time)
        ET.SubElement(offers_el, "avg_expected_length").text = str(avg_expected_length)
        ET.SubElement(offers_el, "avg_radius").text = str(avg_radius)
        ET.SubElement(offers_el, "avg_price").text = str(avg_price)
        ET.SubElement(offers_el, "avg_surge_multiplier").text = str(avg_surge_multiplier)
        if offers_by_provider:
            providers_el = ET.SubElement(offers_el, "offers_by_provider")
            for provider, count in offers_by_provider.items():
                ET.SubElement(providers_el, "provider", name=provider, count=str(count))
        if surge_by_provider:
            surge_el = ET.SubElement(offers_el, "surge_by_provider")
            for provider, avg_surge in surge_by_provider.items():
                ET.SubElement(surge_el, "provider", name=provider, avg_surge=str(avg_surge))
        self.__write()


    def __write(self):
        # Writes XML to file
        rough_string = ET.tostring(self.root, encoding="utf-8")
        reparsed = minidom.parseString(rough_string)
        pretty_xml = reparsed.toprettyxml(indent="  ")
        with open(self.output_path, "w", encoding="utf-8") as f:
            f.write(pretty_xml)