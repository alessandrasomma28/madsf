import os
from pathlib import Path
from datetime import datetime
import traci
import sumolib
from libraries.sumo_utils import convert_shapefile_to_sumo_poly_with_polyconvert
from constants.sumoenv_constants import SUMOENV_PATH
from constants.data_constants import SF_TAZ_SHAPEFILE_PATH
from typing import Optional


class Simulator:
    net_file_path: Path
    route_file_path: Path
    output_dir_path: Path
    end_time: int
    sumocfg_file_path: Path
    taz_file_path: Path


    def __init__(
            self,
            net_file: str, 
            config_template_path: str,
            taz_file_path: Optional[str]
            ):
        self.net_file_path = Path(net_file).resolve()
        self.config_template_path = Path(config_template_path).resolve()
        if taz_file_path is None:
            self.taz_file_path = convert_shapefile_to_sumo_poly_with_polyconvert(
                net_file=net_file,
                shapefile_path=SF_TAZ_SHAPEFILE_PATH,
                output_dir=SUMOENV_PATH
            )
        else:
            self.taz_file_path = Path(taz_file_path).resolve()
        self.sumocfg_file_path = None
        self.route_file_path = None
        self.output_dir_path = None
        self.end_time = None


    def configure_output_dir(
            self, 
            sf_routes_folder_path: str, 
            sf_traffic_route_file_path: str, 
            sf_tnc_fleet_file_path: str, 
            sf_tnc_requests_file_path: str, 
            date_str: str, 
            start_time_str: str, 
            end_time_str: str
            ):
        """
        Builds output dir path and route file path based on date and time slot.

        This function:
        - Creates a folder structure based on the date and time slot.
        - Sets the output directory path, route file paths, date part, and timeslot part.
        - Sets the route file paths for traffic, taxi, and passenger data.
        - Sets the simulation end time in seconds.

        Parameters:
        -----------
        - sf_routes_folder_path: str 
            Base scenario root folder.
        - sf_traffic_route_file_path: str
            Path to the route file.
        - sf_tnc_fleet_file_path: str
            Path to the taxi route file.
        - sf_tnc_requests_file_path: str
            Path to the passenger requests file.
        - date_str: str
            Date in 'YYYY-MM-DD' format.
        - start_time_str: str
            Start time in 'HH:MM' format.
        - end_time_str: str
            End time in 'HH:MM' format.

        Returns:
        --------
        str 
            Path to the generated directory
        """
        date_part = datetime.strptime(date_str, "%Y-%m-%d").strftime("%y%m%d")
        start_hour = datetime.strptime(start_time_str, "%H:%M").strftime("%H")
        end_hour = datetime.strptime(end_time_str, "%H:%M").strftime("%H")
        timeslot_folder = f"{start_hour}-{end_hour}"
        timeslot_part = f"{start_hour}{end_hour}"

        # Only calculate the path — don't create it
        full_folder_path = Path(sf_routes_folder_path, date_str, timeslot_folder).absolute()

        # Set internal state
        self.output_dir_path = full_folder_path
        self.route_file_path = [
            Path(sf_traffic_route_file_path).absolute(),
            Path(sf_tnc_fleet_file_path).absolute(),
            Path(sf_tnc_requests_file_path).absolute()
        ]
        self.date_part = date_part
        self.timeslot_part = timeslot_part

        # Calculate end time in seconds
        sim_start = datetime.strptime(start_time_str, "%H:%M")
        sim_end = datetime.strptime(end_time_str, "%H:%M")
        self.end_time = int((sim_end - sim_start).total_seconds())

        return self.output_dir_path


    def generate_config(self):
        """
        Generates a SUMO configuration (.sumocfg) file from a template, using dynamic values
        for net file, route file, additional files, output directory, and simulation end time.

        The generated config file is saved in the `output_dir_path` with the name `sumo_config.sumocfg`.

        Template placeholders expected in the config template:
            - {net_file}
            - {route_file}
            - {tazpoly_file}
            - {output_dir}
            - {end_time}

        Returns:
        -------
        None

        Raises:
        -------
        ValueError: If output_dir_path is not set before calling this method.
        """
        if self.output_dir_path is None:
            raise ValueError("output_dir_path must be configured before generating the config file.")
        if not self.route_file_path:
            raise ValueError("route_file_path must be set before generating the config file.")
        if not self.end_time:
            raise ValueError("end_time must be set before generating the config file.")


        # Read the template
        with open(self.config_template_path, 'r') as f:
            template = f.read()
        if isinstance(self.route_file_path, list):
            route_file_str = ", ".join([p.as_posix() for p in self.route_file_path])
        else:
            route_file_str = self.route_file_path.as_posix()
        # Format it with current instance values
        config_content = template.format(
            net_file=self.net_file_path.as_posix(),
            route_file=route_file_str,
            tazpoly_file=self.taz_file_path.as_posix(),
            output_dir=self.output_dir_path.as_posix(),
            end_time=self.end_time
        )

        # Define the config file path inside output dir
        config_filename = "sumo_config.sumocfg"
        generated_config_path = self.output_dir_path / config_filename
        self.sumocfg_file_path = generated_config_path

        # Write it
        with open(generated_config_path, 'w') as f:
            f.write(config_content)

        print(f"Generated SUMO config at: {generated_config_path}")


    def run_simulation(
            self,
            activeGui: bool = False
            ):
        """
        Runs the SUMO simulation using the generated configuration file.

        Parameters:
        -----------
        - activeGui (bool): 
            If True, runs with SUMO-GUI. If False, runs headless.

        Raises:
        -------
        - FileNotFoundError: If the SUMO config file is not generated.
        - ImportError: If traci or libsumo is not installed or SUMO_HOME is not set.
        """
        if not self.sumocfg_file_path or not self.sumocfg_file_path.exists():
            raise FileNotFoundError("SUMO configuration file not found. Please generate it first.")

        try:
            import traci
            from sumolib import checkBinary
        except ImportError:
            raise ImportError("SUMO Python tools not found. Install with: pip install eclipse-sumo")

        # Select the right binary
        sumo_binary = checkBinary("sumo-gui" if activeGui else "sumo")

        # Start simulation with traci
        traci.start([sumo_binary, "-c", str(self.sumocfg_file_path)])

        print("Simulation started...")

        try:
            while traci.simulation.getMinExpectedNumber() > 0 and traci.simulation.getTime() < self.end_time+10800:
                if traci.simulation.getTime() % 1000 == 0:
                    print("Simulation time:", traci.simulation.getTime())
                    print("Expected number of vehicles:", traci.simulation.getMinExpectedNumber())
                    print("Pending vehicles:", len(traci.vehicle.getIDList()))
                    print("Pending persons:", len(traci.person.getIDList()))
                traci.simulationStep()

            print("Simulation finished.")
        finally:
            traci.close()

