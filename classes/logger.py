"""
logger.py

This class defines a Logger that writes multi-agent simulation data to an XML file at each timestamp.
"""


import xml.etree.ElementTree as ET
import xml.dom.minidom as minidom
from pathlib import Path
import os
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
        self._write()


    def update_passengers(
            self,
            timestamp: int,
            unassigned_requests: int,
            assigned_requests: int,
            pickup_requests: int,
            accepted_requests: int,
            rejected_requests: int
            ) -> None:
        entry = ET.SubElement(self.root, "step", timestamp=str(timestamp))
        passengers_el = ET.SubElement(entry, "passengers")
        ET.SubElement(passengers_el, "unassigned_requests").text = str(unassigned_requests)
        ET.SubElement(passengers_el, "assigned_requests").text = str(assigned_requests)
        ET.SubElement(passengers_el, "pickup_requests").text = str(pickup_requests)
        ET.SubElement(passengers_el, "accepted_requests").text = str(accepted_requests)
        ET.SubElement(passengers_el, "rejected_requests").text = str(rejected_requests)
        self._write()


    def update_drivers(
            self,
            timestamp: int,
            idle_drivers: int,
            pickup_drivers: int,
            busy_drivers: int,
            accepted_requests: int,
            rejected_requests: int
            ) -> None:
        entry = ET.SubElement(self.root, "step", timestamp=str(timestamp))
        drivers_el = ET.SubElement(entry, "drivers")
        ET.SubElement(drivers_el, "idle_drivers").text = str(idle_drivers)
        ET.SubElement(drivers_el, "pickup_drivers").text = str(pickup_drivers)
        ET.SubElement(drivers_el, "busy_drivers").text = str(busy_drivers)
        ET.SubElement(drivers_el, "accepted_requests").text = str(accepted_requests)
        ET.SubElement(drivers_el, "rejected_requests").text = str(rejected_requests)
        self._write()


    def update_rideservices(
            self,
            timestamp: int,
            dispatched_taxis: int,
            timeout_offers: int,
            requests_canceled: int,
            requests_not_served: int
            ) -> None:
        entry = ET.SubElement(self.root, "step", timestamp=str(timestamp))
        rideservices_el = ET.SubElement(entry, "rideservices")
        ET.SubElement(rideservices_el, "dispatched_taxis").text = str(dispatched_taxis)
        ET.SubElement(rideservices_el, "timeout_offers").text = str(timeout_offers)
        ET.SubElement(rideservices_el, "requests_canceled").text = str(requests_canceled)
        ET.SubElement(rideservices_el, "requests_not_served").text = str(requests_not_served)
        self._write()


    def _write(self):
        """Write XML tree to file."""
        rough_string = ET.tostring(self.root, encoding="utf-8")
        reparsed = minidom.parseString(rough_string)
        pretty_xml = reparsed.toprettyxml(indent="  ")
        with open(self.output_path, "w", encoding="utf-8") as f:
            f.write(pretty_xml)