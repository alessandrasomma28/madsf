"""
sumo_utils.py

This module provides tools to perform map-matching and route processing on San Francisco
real traffic data using a SUMO network. It includes utilities for:

1. get_strongly_connected_edges: Identifying strongly connected edges in a SUMO network.
2. get_nearest_edge: Finding the nearest edge in the SUMO network to a given geographic location.
3. sf_traffic_map_matching: Map-matching raw GPS traffic data to network edges.
4. sf_traffic_od_generation: Generating Origin-Destination (OD) files from matched data.
5. add_sf_traffic_taz_matching: Adding TAZ information to OD trips based on GPS coordinates and TAZ polygons.
6. sf_traffic_routes_generation: Creating SUMO-compatible route files (in XML) from OD data.
7. convert_shapefile_to_sumo_poly_with_polyconvert: Converting a shapefile to a SUMO polygon file.
8. export_taz_coords: Extracting polygon boundary and centroid coordinates from a TAZ shapefile.
9. map_coords_to_sumo_edges: Mapping TAZ polygon and centroid coordinates to nearest SUMO edges and lanes.
10. map_taz_to_edges: Mapping TAZ IDs to SUMO edges.
11. generate_vehicle_start_lanes_from_taz_polygons: Mapping points inside each TAZ polygon to the nearest lane.
12. compute_requests_vehicles_ratio: Computing the ratio of TNC requests to TNC vehicles.
13. generate_drt_vehicle_instances_from_lanes: Generating a DRT fleet file with <vType> and <vehicle> entries.
14. get_valid_taxi_edges: Getting valid edges for taxi routes.
15. generate_matched_drt_requests: Generating matched DRT requests based on TNC data and TAZ mapping.
16. filter_polygon_edges: Filter edge list string, keeping only strongly connected edges
17. filter_polygon_lanes: Filter lane list string, keeping only those lanes whose parent edge is in the strongly connected set.
"""


from pathlib import Path
from typing import Optional
from collections import defaultdict
import random
from itertools import cycle
from datetime import datetime, time, timedelta
import os
import ast
import subprocess
import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET
import geopandas as gpd
from shapely.geometry import Point, MultiPolygon, Polygon
from scipy.spatial import cKDTree
from sumolib import net
import traci
from constants.sumoenv_constants import SUMO_BIN_PATH
from libraries.data_utils import read_sf_traffic_data


def get_strongly_connected_edges(sf_map_file_path: str) -> set:
    """
    Identifies strongly connected edges in a SUMO network.

    This function:
    - Reads a SUMO network XML file.
    - Constructs a directed graph of edges.
    - Identifies strongly connected edges using depth-first search (DFS).
    - Returns a set of edge IDs that are strongly connected.

    Parameters:
    ----------
    sf_map_file_path : str
        Path to the SUMO network XML file.

    Returns:
    -------
    set
        A set of edge IDs (str) strongly connected.
    
    Notes:
    Strongly connected edges are those that can be reached from each other in both directions.
    """
    # Load the SUMO network
    net_data = net.readNet(sf_map_file_path)
    edges = net_data.getEdges()
    # Adjacency dictionaries for forward and reverse graph traversal
    forward_graph = {}
    reverse_graph = {}
    # Empty adjacency lists for each edge
    for edge in edges:
        eid = edge.getID()
        forward_graph[eid] = set()
        reverse_graph[eid] = set()
    for edge in edges:
        eid = edge.getID()
        for lane in edge.getLanes():
            for conn in lane.getOutgoing():
                # Get the destination edge of this connection
                to_obj = conn.getTo()
                to_edge = to_obj.getEdge() if hasattr(to_obj, "getEdge") else to_obj
                to_eid = to_edge.getID()
                # Add forward and reverse connections
                forward_graph[eid].add(to_eid)
                reverse_graph[to_eid].add(eid)

    # Depth-first search to find all reachable nodes from a start node
    def dfs(graph, start):
        stack = [start]
        visited = set()
        while stack:
            node = stack.pop()
            if node not in visited:
                visited.add(node)
                stack.extend(graph.get(node, []))
        return visited

    # Compute reachable nodes in forward and reverse directions
    all_nodes = list(forward_graph.keys())
    forward_reachable = dfs(forward_graph, all_nodes[0])
    reverse_reachable = dfs(reverse_graph, all_nodes[0])

    # Strongly connected edges are those reachable in both directions
    strongly_connected = forward_reachable & reverse_reachable
    print(f"✅ Strongly connected edges count: {len(strongly_connected)} over {len(all_nodes)} total edges")
    
    return strongly_connected


def get_nearest_edge(
        network: net, 
        lon: float, 
        lat: float, 
        radius: float, 
        safe_edge_ids: set = None
    ) -> Optional[str]:
    """
    Finds the nearest edge in the SUMO network to a given geographic location.

    This function:
    - Converts geographic coordinates (longitude, latitude) to network XY coordinates.
    - Searches for neighboring edges within a specified radius.
    - Returns the ID of the nearest edge if found, otherwise None.

    Parameters:
    ----------
    - network : sumolib.net.Net
        The loaded SUMO network.
    - lon : float
        Longitude of the point.
    - lat : float
        Latitude of the point.
    - radius : float
        Search radius for neighboring edges (in SUMO units, typically meters).
    - safe_edge_ids : set or None, optional
        Set of edge IDs to consider as valid (e.g., strongly connected edges).
        If None, all edges are considered.

    Returns:
    -------
    str or None
        The ID of the nearest edge if found, otherwise None.
    """
    # Convert geographic coordinates (lon, lat) to network XY coordinates
    x, y = network.convertLonLat2XY(lon, lat)
    # Get neighboring edges within the specified radius
    edges = network.getNeighboringEdges(x, y, radius)
    # If any edges are found, sort them by distance and return the closest edge's ID
    if edges:
        # Filter only strongly connected edges if a filter is provided
        if safe_edge_ids is not None:
            edges = [(edge, dist) for edge, dist in edges if edge.getID() in safe_edge_ids]
        if edges:
            distancesAndEdges = sorted([(dist, edge) for edge, dist in edges], key=lambda x: x[0])
            return distancesAndEdges[0][1].getID()
    return None


def sf_traffic_map_matching(
        sf_map_file_path: str, 
        sf_real_traffic_data_path: str, 
        date_str: str,
        output_folder_path: str, 
        radius: float, 
        safe_edge_ids: set = None
    ) -> Path:
    """
    Performs map matching by assigning each GPS point to its nearest road network edge.

    This function:
    - Reads raw GPS traffic data.
    - Matches each point to the closest edge in the SUMO network.
    - Saves the resulting file into a subfolder with a custom filename.
    - Creates output directories if they do not exist.
    - Overwrites the output CSV if it already exists.

    Parameters:
    ----------
    - sf_map_file_path: str
        Path to the SUMO network XML file.
    - sf_real_traffic_data_path : str
        Path to the raw traffic CSV file (preprocessed with read_sf_traffic_data).
    - date_str: str
        The date string (format: 'YYYY-MM-DD') used to name the subfolder.
    - output_folder_path: str
        The base folder where the output subfolder will be created.
    - radius: float
        Radius (in meters) to search for neighboring edges.
    - safe_edge_ids: set or None, optional
        Set of edge IDs to consider as valid (e.g., strongly connected edges).
        If None, all edges are considered.

    Returns:
    -------
    Path
        Path to the saved output CSV file with edge IDs.
    """
    # Load the SUMO network
    network = net.readNet(sf_map_file_path)

    # Read the traffic data
    df = read_sf_traffic_data(sf_real_traffic_data_path)

    # Map match each GPS point to the nearest edge
    df['edge_id'] = df.apply(
        lambda row: get_nearest_edge(network, row['longitude'], row['latitude'], radius=radius, safe_edge_ids=safe_edge_ids),
        axis=1
    )

    # Build output paths using pathlib
    output_base = Path(output_folder_path)
    output_base.mkdir(parents=True, exist_ok=True)

    date_subfolder = output_base / date_str
    date_subfolder.mkdir(parents=True, exist_ok=True)

    input_filename = Path(sf_real_traffic_data_path).stem
    output_csv_path = date_subfolder / f"{input_filename}_edges.csv"

    # Save the updated DataFrame (overwrite if exists)
    df.to_csv(output_csv_path, index=False, sep=";")
    print(f"✅ Mapped traffic coordinates to SUMO edges. Saved to: {output_csv_path}")

    return output_csv_path


def sf_traffic_od_generation(
        sf_real_traffic_edge_path: str, 
        sf_traffic_od_folder_path: str,
        start_date_str: str,
        end_date_str: str,
        start_time_str: str, 
        end_time_str: str
    ) -> Path:
    """
    Generates an Origin-Destination (OD) file from map-matched traffic data.

    This function:
    - Reads the map-matched traffic data with edge IDs.
    - Groups the data by vehicle ID.
    - For each vehicle, extracts the first and last edge IDs (origin and destination),
      and captures the corresponding timestamps.
    - Saves the resulting OD data in a structured folder format (sf_traffic_od_{YYMMDD}_{HHHH}.csv).

    Parameters:
    ----------
    - sf_real_traffic_edge_path : str
        Path to the input CSV containing traffic data with edge IDs.
    - sf_traffic_od_folder_path : str
        Path to the output folder where the OD file will be saved.
    - start_date_str : str
        Start date in 'YYYY-MM-DD' format (e.g., '2021-03-25').
    - end_date_str : str
        Start date in 'YYYY-MM-DD' format (e.g., '2021-03-26').
    - start_time_str : str
        Start time in 'HH:MM' format (e.g., '08:00').
    - end_time_str : str
        End time in 'HH:MM' format (e.g., '10:00').

    Returns:
    -------
    Path
        Full path to the saved OD CSV file.
    """
    # Read the CSV file
    df = pd.read_csv(sf_real_traffic_edge_path, sep=";")

    # Ensure the timestamp column is in datetime format
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Sort the data by vehicle_id and relative_time (ensuring chronological order)
    df = df.sort_values(['vehicle_id', 'relative_time'])

    # Build a list for the OD data
    od_data = []
    for vehicle_id, group in df.groupby('vehicle_id'):
        group = group.sort_values('relative_time')
        origin_row = group.iloc[0]
        destination_row = group.iloc[-1]
        od_data.append({
            'vehicle_id': vehicle_id,
            'origin_edge_id': origin_row['edge_id'],
            'origin_taz_id': origin_row['assigned_taz'],
            'destination_edge_id': destination_row['edge_id'],
            'destination_taz_id': destination_row['assigned_taz'],
            'origin_starting_time': origin_row['timestamp'],
            'destination_ending_time': destination_row['timestamp']
        })

    od_df = pd.DataFrame(od_data)

    # Format date and time parts
    start_date = datetime.strptime(start_date_str, "%Y-%m-%d").strftime("%y%m%d")
    end_date = datetime.strptime(end_date_str, "%Y-%m-%d").strftime("%y%m%d")
    start_hour = datetime.strptime(start_time_str, "%H:%M").strftime("%H")
    end_hour = datetime.strptime(end_time_str, "%H:%M").strftime("%H")

    # Prepare directory and filename
    start_d = start_date.replace("-", "")
    end_d = end_date.replace("-", "")
    start_h = start_hour.replace(":", "")
    end_h = end_hour.replace(":", "")
    timeslot = f"{start_d}{start_h}_{end_d}{end_h}"
    date_folder = os.path.join(sf_traffic_od_folder_path, timeslot)
    os.makedirs(date_folder, exist_ok=True)
    filename = f"sf_traffic_od_{timeslot}.csv"
    full_path = os.path.join(date_folder, filename)

    # Save to file (overwrite if exists)
    od_df.to_csv(full_path, sep=";", index=False)
    print(f"✅ Origin-Destination traffic file saved to {full_path}")

    return full_path


def add_sf_traffic_taz_matching(
        edge_file_path: str, 
        shapefile_path: str, 
        lat_col: str = "latitude", 
        lon_col: str = "longitude",
        zone_column: str = "assigned_taz",
        zone_id_field: str = "TAZ"
    ) -> None:
    """
    Adds TAZ info to OD trips based on origin GPS coords and TAZ polygons.

    This function:
    - Reads a CSV file with latitude and longitude columns.
    - Loads a shapefile containing TAZ polygons.
    - Assigns TAZ IDs to the CSV based on the polygons.
    - If a point is not contained in any polygon, assigns the nearest TAZ centroid.
    - Saves the updated CSV with a new TAZ column.

    Parameters:
    ----------
    - edge_file_path: str
        CSV with columns latitude and longitude.
    - shapefile_path: str
        Path to .shp file.
    - lat_col: str
        Name of latitude column.
    - lon_col: str
        Name of longitude column.
    - zone_column: str
        New column to assign TAZ ID.
    - zone_id_field: str
        Field name in shapefile for TAZ ID.

    Returns:
    -------
    None

    Raises:
    ------
    - ValueError: If lat_col or lon_col are missing in edge_file_path.
    - ValueError: If zone_id_field is not found in shapefile_path.
    """
    df = pd.read_csv(edge_file_path, sep=';')
    if lat_col not in df or lon_col not in df:
        raise ValueError(f"Missing '{lat_col}' or '{lon_col}' in the edge file.")

    # Convert to GeoDataFrame
    geometry = [Point(lon, lat) for lon, lat in zip(df[lon_col], df[lat_col])]
    od_gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")

    zones_gdf = gpd.read_file(shapefile_path).to_crs("EPSG:4326")

    if zone_id_field not in zones_gdf.columns:
        raise ValueError(f"Zone ID field '{zone_id_field}' not found in the shapefile.")

    # Assigning zones by containment
    od_gdf[zone_column] = None
    for idx, row in od_gdf.iterrows():
        match = zones_gdf[zones_gdf.contains(row.geometry)]
        if not match.empty:
            od_gdf.at[idx, zone_column] = match.iloc[0][zone_id_field]

    # TAZ unmatched are filled with the nearest TAZ centroid
    missing = od_gdf[zone_column].isna()
    if missing.any():
        centroids = np.array([[geom.centroid.x, geom.centroid.y] for geom in zones_gdf.geometry])
        coords = np.array([[geom.x, geom.y] for geom in od_gdf.loc[missing, 'geometry']])
        tree = cKDTree(centroids)
        _, nearest_idxs = tree.query(coords)
        nearest_zone_ids = zones_gdf.iloc[nearest_idxs][zone_id_field].values
        od_gdf.loc[missing, zone_column] = nearest_zone_ids

    od_gdf.drop(columns="geometry").to_csv(edge_file_path, sep=';', index=False)
    print(f"✅ TAZ assignment complete. Saved to: {edge_file_path}")


def sf_traffic_routes_generation(
        sf_traffic_od_path: str, 
        sf_traffic_routes_folder_path: str,
        start_date_str: str, 
        end_date_str: str,
        start_time_str: str, 
        end_time_str: str
    ) -> Path:
    """
    Generates a SUMO-compatible route file (XML) from an OD file.

    This function:
    - Reads the OD CSV file.
    - Creates a SUMO route XML structure.
    - Saves the route file in a structured folder format in
      {sf_traffic_routes_folder_path}/{date}/{HH}-{HH}/sf_routes_{YYMMDD}_{HHHH}.rou.xml

    Each trip contains:
    - Vehicle ID
    - Departure time (seconds from simulation start)
    - Origin edge
    - Destination edge
    - Expected arrival time (seconds from simulation start)
    
    Parameters:
    ----------
    - sf_traffic_od_path: str
        Path to the OD CSV file.
    - sf_traffic_routes_folder_path: str
        Root folder path where the generated route file will be saved.
    - start_date_str: str
        Start date in 'YYYY-MM-DD' format (e.g., '2021-03-25').
    - end_date_str: str
        Start date in 'YYYY-MM-DD' format (e.g., '2021-03-26').
    - start_time_str: str
        Start time in 'HH:MM' format (e.g., '08:00').
    - end_time_str: str
        End time in 'HH:MM' format (e.g., '10:00').

    Returns:
    -------
    Path
        Full path to the saved XML file.
    """
    # Load OD data
    df = pd.read_csv(sf_traffic_od_path, sep=";")
    df['origin_starting_time'] = pd.to_datetime(df['origin_starting_time'])
    df['destination_ending_time'] = pd.to_datetime(df['destination_ending_time'])
    df.sort_values('origin_starting_time', inplace=True)

    # Determine simulation start time (reference zero)
    sim_start = df['origin_starting_time'].min().replace(microsecond=0)

    # Create XML structure
    root = ET.Element("routes")

    for _, row in df.iterrows():
        if pd.notna(row['origin_edge_id']) and pd.notna(row['destination_edge_id']):
            depart = int((row['origin_starting_time'] - sim_start).total_seconds())
            arrival = int((row['destination_ending_time'] - sim_start).total_seconds())

            ET.SubElement(root, "trip", {
                "id": str(row['vehicle_id']),
                "depart": str(depart),
                "from": row['origin_edge_id'],
                "to": row['destination_edge_id'],
                "arrival": str(arrival)
            })
        else:
            if pd.isna(row['origin_edge_id']):
                print(f"Skipping vehicle {row['vehicle_id']} due to missing origin edge ID(s).")
            else:
                print(f"Skipping vehicle {row['vehicle_id']} due to missing destination edge ID(s).")

    # Format parts for folder structure and filename
    start_date = datetime.strptime(start_date_str, "%Y-%m-%d").strftime("%y%m%d")
    end_date = datetime.strptime(end_date_str, "%Y-%m-%d").strftime("%y%m%d")
    start_hour = datetime.strptime(start_time_str, "%H:%M").strftime("%H")
    end_hour = datetime.strptime(end_time_str, "%H:%M").strftime("%H")

    # Create full folder path: root/timeslot/
    start_d = start_date.replace("-", "")
    end_d = end_date.replace("-", "")
    start_h = start_hour.replace(":", "")
    end_h = end_hour.replace(":", "")
    timeslot = f"{start_d}{start_h}_{end_d}{end_h}"
    full_folder_path = os.path.join(sf_traffic_routes_folder_path, timeslot)
    os.makedirs(full_folder_path, exist_ok=True)

    # Filename and full path
    filename = f"sf_routes_{timeslot}.rou.xml"
    full_path = os.path.join(full_folder_path, filename)

    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    with open(full_path, "wb") as f:
        tree.write(f, encoding="UTF-8", xml_declaration=True)

    return full_path


def convert_shapefile_to_sumo_poly_with_polyconvert(
        net_file: str, 
        shapefile_path: str, 
        output_dir: str,
        type_field: str = "zone", 
        id_field: str = "TAZ",
        output_filename: str = "taz_zones.poly.xml"
    ) -> Path:
    """
    This function uses the polyconvert tool to convert a shapefile into a SUMO polygon file.

    Parameters:
    ----------
    - net_file: str
        Path to the SUMO network file (.net.xml).
    - shapefile_path: str
        Path to the .shp file (will strip .shp automatically).
    - output_dir: str
        Output folder where the .poly.xml will be saved.
    - type_field: str
        Attribute column to use as the polygon type.
    - id_field: str
        Attribute column to use as polygon ID.
    - output_filename: str
        Name of the resulting .poly.xml file.

    Returns:
    -------
    Path
        Full path to the generated .poly.xml file.

    Raises:
    ------
    RuntimeError: If polyconvert fails.
    """

    polyconvert_bin = os.path.join(SUMO_BIN_PATH, "polyconvert.exe")
    net_file = os.path.abspath(net_file)
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / output_filename

    # Remove .shp extension to get prefix path
    shapefile_prefix = os.path.splitext(os.path.abspath(shapefile_path))[0]

    command = [
        polyconvert_bin,
        "--net-file", net_file,
        "--shapefile-prefixes", shapefile_prefix,
        "--shapefile.id-column", id_field,
        "--shapefile.type-columns", type_field,
        "--color", "255,255,255",
        "--shapefile.fill", "false",
        "--output-file", str(output_path)
    ]

    try:
        subprocess.run(command, check=True)
        print(f"Poly file generated at: {output_path}")
    except subprocess.CalledProcessError as e:
        print("Polyconvert failed!")
        print(f"Command: {' '.join(command)}")
        print(f"Return code: {e.returncode}")
        raise RuntimeError(f"There was a problem with polyconvert.")

    return output_path


def export_taz_coords(
        shapefile_path: str, 
        output_csv_path: str
    ) -> None:
    """
    Extracts polygon boundary and centroid coordinates from a TAZ shapefile.

    This function:
    - Reads a TAZ shapefile.
    - Extracts the polygon coordinates and centroid coordinates.
    - Saves the data to a CSV file with columns: TAZ, polygon_coords, centroid_coords.

    Parameters:
    ----------
    - shapefile_path: str
        Full path to the input TAZ shapefile (.shp).
    - output_csv_path: str
        Path where the output CSV will be saved. If the file exists, it will be overwritten.

    Returns:
    -------
    None

    Notes:
    - Assumes the shapefile has a column named 'TAZ' for unique zone IDs.
    - If a geometry is empty or invalid, centroid_coords will be None.
    """
    # Load the shapefile
    gdf = gpd.read_file(shapefile_path)
    # Ensure coordinates are in WGS84 (lat/lon)
    gdf = gdf.to_crs(epsg=4326)

    # --- Coordinate Extraction ---
    def extract_coords(geometry):
        if geometry is None or geometry.is_empty:
            return None
        if geometry.geom_type == 'Polygon':
            return list(geometry.exterior.coords)
        elif geometry.geom_type == 'MultiPolygon':
            return [list(poly.exterior.coords) for poly in geometry.geoms]
        return None

    def extract_centroid(geometry):
        if geometry and not geometry.is_empty:
            centroid = geometry.centroid
            return (centroid.x, centroid.y)
        return None

    # Extract data
    gdf['polygon_coords'] = gdf['geometry'].apply(extract_coords)
    gdf['centroid_coords'] = gdf['geometry'].apply(extract_centroid)

    # Ensure TAZ is sorted from 1 to 981
    gdf = gdf.sort_values(by='TAZ')  # Adjust if your ID column is named differently

    # Write to CSV, overwrite if exists
    gdf[['TAZ', 'polygon_coords', 'centroid_coords']].to_csv(output_csv_path,  sep=';', index=False)
    print(f"✅ Exported TAZ data to: {output_csv_path}")


def map_coords_to_sumo_edges(
        taz_csv_path: str, 
        net_xml_path: str, 
        output_csv_path: str
    ) -> None:
    """
    Maps each coordinate from TAZ polygon and centroid to nearest SUMO edge and lane.

    This function:
    - Reads a CSV file with TAZ polygon and centroid coordinates.
    - For each polygon, finds all edges and lanes that touch the polygon.
    - For each centroid, finds the nearest edge and lane.
    - Saves the results in a new CSV file with additional columns:
        - 'polygon_edge_ids': List of unique edge IDs touched by polygon coords.
        - 'polygon_lane_ids': List of unique lane IDs touched by polygon coords.
        - 'centroid_edge_id': Edge ID mapped to the centroid.
        - 'centroid_lane_id': Lane ID mapped to the centroid.

    Parameters:
    ----------
    - taz_csv_path: str
        Path to the TAZ CSV file with 'polygon_coords' and 'centroid_coords'.
    - net_xml_path: str
        Path to the SUMO network.xml.
    - output_csv_path: str
        Path to save the output CSV with mappings.

    Returns:
    -------
    None
    """
    df = pd.read_csv(taz_csv_path, sep=';')
    network = net.readNet(net_xml_path)

    poly_edge_ids_list = []
    poly_lane_ids_list = []
    centroid_edge_id_list = []
    centroid_lane_id_list = []

    for idx, row in df.iterrows():
        try:
            # Extract polygon and centroid coordinates
            polygon_coords = ast.literal_eval(row['polygon_coords'])
            centroid_coord = ast.literal_eval(row['centroid_coords'])

            # If MultiPolygon-style, flatten
            if isinstance(polygon_coords[0], list):
                polygon_coords = [pt for sublist in polygon_coords for pt in sublist]

            # For polygon: get all edge and lane IDs
            poly_edges = set()
            poly_lanes = set()

            for lon, lat in polygon_coords:
                edge_id = get_nearest_edge(network, lon, lat, radius=50)
                if edge_id:
                    poly_edges.add(edge_id)
                    lanes = network.getEdge(edge_id).getLanes()
                    for lane in lanes:
                        poly_lanes.add(lane.getID())

            # For centroid
            lon_c, lat_c = centroid_coord
            centroid_edge_id = get_nearest_edge(network, lon_c, lat_c, radius=50)
            if centroid_edge_id:
                centroid_lanes = network.getEdge(centroid_edge_id).getLanes()
                centroid_lane_id = centroid_lanes[0].getID() if centroid_lanes else None
            else:
                centroid_lane_id = None

        except Exception as e:
            poly_edges, poly_lanes = set(), set()
            centroid_edge_id, centroid_lane_id = None, None

        poly_edge_ids_list.append(list(poly_edges))
        poly_lane_ids_list.append(list(poly_lanes))
        centroid_edge_id_list.append(centroid_edge_id)
        centroid_lane_id_list.append(centroid_lane_id)

    # Add columns to DataFrame
    df['polygon_edge_ids'] = poly_edge_ids_list
    df['polygon_lane_ids'] = poly_lane_ids_list
    df['centroid_edge_id'] = centroid_edge_id_list
    df['centroid_lane_id'] = centroid_lane_id_list

    df.to_csv(output_csv_path, sep=';', index=False)
    print(f"✅ Mapped polygon + centroid to SUMO edges. Saved to: {output_csv_path}")


def map_taz_to_edges(
        taz_csv_path: str, 
        safe_edge_ids: set = None
    ) -> dict:
    """
    Maps TAZ polygons to their corresponding edges and lanes.

    This function:
    - Reads a CSV file with TAZ polygon and centroid coordinates.
    - For each TAZ, finds all edges and lanes that touch the polygon.
    - For each centroid, finds the nearest edge and lane.
    - Returns a dictionary mapping TAZ IDs to their corresponding edge and lane IDs.
    - Filters out entries with empty edge and lane lists.
    - If safe_edge_ids is provided, only includes edges in that set.

    Parameters:
    ----------
    - taz_csv_path: str
        Path to the TAZ CSV file with 'polygon_edge_ids' and 'centroid_edge_id'.
    - safe_edge_ids: set or None, optional
        Set of edge IDs to consider as valid (e.g., strongly connected edges).
        If None, all edges are considered.

    Returns:
    -------
    dict
        Dictionary mapping TAZ IDs to their corresponding edge and lane IDs.
    """
    df = pd.read_csv(taz_csv_path, sep=';')
    taz_edge_mapping = {}
    for _, row in df.iterrows():
        taz_id = row['TAZ']
        polygon_edge_ids = ast.literal_eval(row['polygon_edge_ids'])
        polygon_lane_ids = ast.literal_eval(row['polygon_lane_ids'])
        centroid_edge_id = row['centroid_edge_id']
        centroid_lane_id = row['centroid_lane_id']
        # Filter only strongly connected edges
        if safe_edge_ids is not None:
            polygon_edge_ids = [e for e in polygon_edge_ids if e in safe_edge_ids]
            polygon_lane_ids = [l for l in polygon_lane_ids if l.split('_')[0] in safe_edge_ids]

        taz_edge_mapping[taz_id] = {
            'polygon_edge_ids': polygon_edge_ids,
            'polygon_lane_ids': polygon_lane_ids,
            'centroid_edge_id': centroid_edge_id,
            'centroid_lane_id': centroid_lane_id
        }
        # Keep only entries with non-empty edge and lane lists
        taz_edge_mapping = {
            k: v for k, v in taz_edge_mapping.items()
            if len(v['polygon_edge_ids']) > 0 and len(v['polygon_lane_ids']) > 0
        }

    return taz_edge_mapping


def generate_vehicle_start_lanes_from_taz_polygons(
        shapefile_path: str,
        net_file: str,
        points_per_taz: int = 5,
        safe_edge_ids: set = None
    ) -> list:
    """
    Samples points inside each TAZ polygon and maps them to the nearest lane using SUMO net.

    This function:
    - Reads a TAZ shapefile and converts it to WGS84 coordinates.
    - For each TAZ polygon, samples random points.
    - For each point, finds the nearest edge and lane in the SUMO network.
    - Returns a list of lane IDs where vehicles should be placed for each TAZ.
    - If safe_edge_ids is provided, only considers edges in that set.

    Parameters:
    ----------
    - shapefile_path: str
        Path to TAZ polygon shapefile.
    - net_file: str
        Path to SUMO net.xml.
    - points_per_taz: int
        Number of vehicle start points per TAZ.
    - safe_edge_ids: set or None, optional
        Set of edge IDs to consider as valid (e.g., strongly connected edges).
        If None, all edges are considered.

    Returns:
    -------
    dict
        Dictionary mapping TAZ IDs to lists of lane IDs where vehicles should be placed.
    """

    gdf = gpd.read_file(shapefile_path)
    # Ensure coordinates are in WGS84 (lat/lon)
    gdf = gdf.to_crs(epsg=4326)
    network = net.readNet(net_file)

    print("🚕 Sampling vehicle start points...")

    lanes_by_taz = defaultdict(list)
    for _, row in gdf.iterrows():
        geom = row['geometry']
        taz_id = str(row["TAZ"])
        if isinstance(geom, MultiPolygon):
            geom = max(geom.geoms, key=lambda a: a.area)
        if not isinstance(geom, Polygon):
            continue

        minx, miny, maxx, maxy = geom.bounds
        points = []
        attempts = 0
        while len(points) < points_per_taz and attempts < 100:
            x = random.uniform(minx, maxx)
            y = random.uniform(miny, maxy)
            pt = Point(x, y)
            if geom.contains(pt):
                points.append(pt)
            attempts += 1

        for pt in points:
            lon, lat = pt.x, pt.y
            edge_id = get_nearest_edge(network, lon, lat, radius=100, safe_edge_ids=safe_edge_ids)
            if edge_id:
                lanes = network.getEdge(edge_id).getLanes()
                if lanes:
                    lanes_by_taz[taz_id].append(lanes[0].getID())

    total_lanes = sum(len(l) for l in lanes_by_taz.values())
    print(f"✅ Found {total_lanes} start lanes across TAZs.")

    return dict(lanes_by_taz)


def compute_requests_vehicles_ratio(
        sf_tnc_fleet_folder_path: str,
        peak_vehicles: int,
        max_total_drivers: int
    ) -> float:
    """
    Computes the ratio of requests to vehicles for each hour of the day.
    This function:
    - Reads a CSV file containing hourly pickup requests.
    - Groups the data by hour and computes the average pickups across all TAZs and days.
    - Computes the final requests to drivers ratio for each hour.

    Parameters:
    ----------
    - sf_tnc_fleet_folder_path: str
        Path to the CSV file containing hourly pickup requests.
    - peak_vehicles: int
        Number of vehicles available during peak hours (e.g., 7 pm).
    - max_total_drivers: int
        Maximum number of drivers available across all hours for one day.

    Returns:
    -------
    float
        The requests to drivers ratio.
    """
    df = pd.read_csv(sf_tnc_fleet_folder_path)
    df["hour"] = df["hour"] % 24
    # Group by hour and compute the average pickups across all TAZs and days
    sum_pickups_per_hour_day = df.groupby(["day_of_week", "hour"])["pickups"].sum().reset_index()
    avg_pickups_per_hour = sum_pickups_per_hour_day.groupby("hour")["pickups"].mean().reset_index()
    # Estimate driver needs assuming peak_vehicles at peak hour
    max_pickups = avg_pickups_per_hour["pickups"].max()
    avg_pickups_per_hour["estimated_vehicles"] = (avg_pickups_per_hour["pickups"] / max_pickups) * peak_vehicles
    # Normalize the number of vehicles per hour to sum to max_total_drivers
    total_raw_drivers = avg_pickups_per_hour["estimated_vehicles"].sum()
    scale_factor = max_total_drivers / total_raw_drivers
    avg_pickups_per_hour["normalized_vehicles"] = avg_pickups_per_hour["estimated_vehicles"] * scale_factor
    # Compute hourly requests-to-drivers ratio
    avg_pickups_per_hour["requests_to_drivers_ratio"] = (
        avg_pickups_per_hour["pickups"] / avg_pickups_per_hour["normalized_vehicles"]
    )
    
    return avg_pickups_per_hour["requests_to_drivers_ratio"].mean()


def generate_drt_vehicle_instances_from_lanes(
        start_lanes_by_taz: dict,
        ratio_vehicles_requests: float,
        tnc_data: dict,
        tnc_previous_hour_data: dict,
        start_date_str: str,
        end_date_str: str,
        start_time_str: str,
        end_time_str: str,
        sf_tnc_fleet_folder_path: str,
        idle_mechanism: str
    ) -> Path:
    """
    Generates a DRT fleet file with <vType> and <vehicle> entries, and dummy routes.

    This function:
    - Reads hourly requests data from a dictionary.
    - Calculates the total number of vehicles needed based on the ratio of requests to drivers.
    - Allocates vehicles per TAZ and hour based on the requests.
    - Generates 50% of drivers from the previous hour at simulation start.
    - Creates a SUMO .rou.xml file with <vehicle> entries for each allocated vehicle.

    Parameters:
    ----------
    - start_lanes_by_taz: dict
        Dictionary mapping TAZ IDs to lists of start lane IDs where vehicles should be placed.
    - ratio_vehicles_requests: float
        Ratio of requests to drivers, used to determine the number of vehicles.
    - tnc_data: dict
        Dictionary containing hourly requests data, where keys are hours (0-23) and values are the number of requests for that hour.
    - tnc_previous_hour_data: dict
        Dictionary containing hourly requests data for the previous hour, used to calculate the number of starting vehicles needed.
    - start_date_str: str
        Start date in 'YYYY-MM-DD' format (e.g., '2021-03-25').
    - end_date_str: str
        End date in 'YYYY-MM-DD' format (e.g., '2021-03-26').
    - start_time_str: str
        Start time in 'HH:MM' format (e.g., '08:00').
    - end_time_str: str
        End time in 'HH:MM' format (e.g., '10:00').
    - sf_tnc_fleet_folder_path: str
        Path to save the resulting SUMO .rou.xml file with <vehicle> entries.
    - idle_mechanism: str
        Idling mechanism for taxis ("stop" or "randomCircling").

    Returns:
    -------
    Path
        Full path to the saved XML file.

    Raises:
    ------
    ValueError: If idle_mechanism is not "stop" or "randomCircling".
    """
    # Compute simulation start time
    sim_start = datetime.strptime(f"{start_date_str} {start_time_str}", "%Y-%m-%d %H:%M")

    # Flatten pickups across all TAZs and hours
    total_requests = 0
    requests_per_taz_hour = {}
    requests_per_hour = {}
    requests_per_taz = {}
    # Calculate total requests and requests per TAZ and hour
    for taz, taz_data in tnc_data.items():
        requests_per_taz[taz] = 0
        requests_per_taz_hour[taz] = {}
        for hour, metrics in taz_data.items():
            pickups = metrics.get('pickups', 0)
            requests_per_taz_hour[taz][hour] = pickups
            requests_per_hour[hour] = requests_per_hour.get(hour, 0) + pickups
            requests_per_taz[taz] += pickups
            total_requests += pickups

    # Calculate total number of vehicles needed based on the ratio
    total_vehicles = round(total_requests / ratio_vehicles_requests)
    # Determine how many vehicles to assign per hour and per TAZ
    raw_allocations = defaultdict(dict)
    fractional_parts = []
    # int truncation will leave leftover vehicles to be distributed later
    for taz, hour_data in requests_per_taz_hour.items():
        for hour, pickups in hour_data.items():
            proportion = pickups / total_requests if total_requests > 0 else 0
            exact = proportion * total_vehicles
            base = int(exact)
            frac = exact - base
            raw_allocations[taz][hour] = base
            fractional_parts.append((frac, taz, hour))
    # Distribute leftover vehicles to largest fractional parts
    vehicles_per_taz_hour = defaultdict(dict)
    total_assigned = sum(base for taz_hours in raw_allocations.values() for base in taz_hours.values())
    remaining = total_vehicles - total_assigned
    # Sort by descending fractional parts
    fractional_parts.sort(reverse=True)
    # Assign the remaining vehicles to the TAZ-hour pairs with the largest fractional parts
    for i in range(remaining):
        _, taz, hour = fractional_parts[i]
        raw_allocations[taz][hour] += 1
    # Now we have a complete allocation of vehicles per TAZ and hour
    for taz, hours in raw_allocations.items():
        for hour, count in hours.items():
            vehicles_per_taz_hour[taz][hour] = count
    
    eco = int(total_vehicles // 3.5)
    diesel = int(total_vehicles // 3.5)
    gas = int(total_vehicles // 3.5)
    zero = total_vehicles - eco - diesel - gas
    vehicle_types = [
        {
            "id": "electric_eco",
            "count": eco,
            "color": "0,1,0",
            "emissionClass": "Energy",
            "vClass": "taxi"
        },
        {
            "id": "diesel_normal",
            "count": diesel,
            "color": "1,0,0",
            "emissionClass": "HBEFA3/PC_D_EU4",
            "vClass": "taxi"
        },
        {
            "id": "gas_modern",
            "count": gas,
            "color": "0,0,1",
            "emissionClass": "HBEFA3/PC_G_EU6",
            "vClass": "taxi"
        },
        {
            "id": "zero_emis",
            "count": zero,
            "color": "0.5,0.5,0",
            "emissionClass": "Zero",
            "vClass": "taxi"
        }
    ]
    root = ET.Element("routes")
    vehicle_counter = 0
    # Define <vType> entries
    for vt in vehicle_types:
        vtype = ET.SubElement(root, "vType", {
            "id": vt["id"], 
            "vClass": vt["vClass"],
            "color": vt["color"], 
            "emissionClass": vt["emissionClass"]
        })
        ET.SubElement(vtype, "param", key="has.taxi.device", value="true")
        ET.SubElement(vtype, "param", key="device.taxi.pickUpDuration", value="30")
        ET.SubElement(vtype, "param", key="device.taxi.dropOffDuration", value="30")
    
    # Generate vehicles per TAZ and hour based on request proportions
    types_cycle = cycle(vehicle_types)
    vehicle_counter = 0
    for taz, hour_data in vehicles_per_taz_hour.items():
        lanes_for_taz = start_lanes_by_taz.get(str(taz), [])
        if not lanes_for_taz:
            print(f"No start lanes found for TAZ {taz}")
            continue
        lane_index = 0
        total_lanes = len(lanes_for_taz)
        for hour, num_vehicles in hour_data.items():
            for _ in range(num_vehicles):
                lane_id = lanes_for_taz[lane_index % total_lanes]
                lane_index += 1
                vt = next(types_cycle)
                edge_id = lane_id.split("_")[0]
                depart_date = sim_start.date()
                if hour < sim_start.hour:
                    depart_date += timedelta(days=1)
                depart_datetime = datetime.combine(depart_date, time(hour=hour))
                base_offset = int((depart_datetime - sim_start).total_seconds())
                depart_seconds = base_offset + random.randint(0, 3599)
                if idle_mechanism == "randomCircling":
                    vehicle = ET.SubElement(root, "vehicle", {
                        "id": f"taxi_{vehicle_counter}",
                        "depart": f"{depart_seconds:.2f}",
                        "type": vt["id"]
                    })
                    # ➔ Dummy initial route = just the edge where the vehicle is starting
                    ET.SubElement(vehicle, "route", {"edges": edge_id})
                    ET.SubElement(vehicle, "param", key="device.taxi.end", value=str(generate_work_duration()))
                elif idle_mechanism == "stop":
                    trip = ET.SubElement(root, "trip", {
                        "id": f"taxi_{vehicle_counter}",
                        "depart": f"{depart_seconds:.2f}",
                        "type": vt["id"],
                        "personCapacity": "4"
                    })
                    # Lane where the vehicle is starting
                    ET.SubElement(trip, "stop", {"lane": lane_id, "triggered": "person"})
                    ET.SubElement(trip, "param", key="device.taxi.end", value=str(generate_work_duration()))
                else:
                    raise ValueError("Invalid idle mechanism, please choose 'stop' or 'randomCircling'")
                vehicle_counter += 1

    # Add 50% of previous hour's vehicles at time 0
    prev_total_pickups = sum(metrics.get('pickups', 0) for taz_data in tnc_previous_hour_data.values() for metrics in taz_data.values())
    total_prev_vehicles = int((prev_total_pickups / ratio_vehicles_requests) * 0.5)
    # int truncation will leave leftover vehicles to be distributed later
    raw_prev_allocations = []
    for taz, hour_data in tnc_previous_hour_data.items():
        for hour, metrics in hour_data.items():
            pickups = metrics.get('pickups', 0)
            if pickups == 0:
                continue
            exact = pickups / prev_total_pickups * total_prev_vehicles
            base = int(exact)
            frac = exact - base
            raw_prev_allocations.append((frac, taz, hour, base))
    # Distribute leftover vehicles to largest fractional parts
    allocated_prev = sum(base for (_, _, _, base) in raw_prev_allocations)
    leftover = total_prev_vehicles - allocated_prev
    raw_prev_allocations.sort(reverse=True)
    for i in range(leftover):
        frac, taz, hour, base = raw_prev_allocations[i]
        raw_prev_allocations[i] = (frac, taz, hour, base + 1)
    # Generate starting vehicles at time 0
    for (_, taz, hour, count) in raw_prev_allocations:
        lanes_for_taz = start_lanes_by_taz.get(str(taz), [])
        if not lanes_for_taz:
            continue
        lane_index = 0
        total_lanes = len(lanes_for_taz)
        for _ in range(count):
            lane_id = lanes_for_taz[lane_index % total_lanes]
            lane_index += 1
            vt = next(types_cycle)
            edge_id = lane_id.split("_")[0]
            if idle_mechanism == "randomCircling":
                vehicle = ET.SubElement(root, "vehicle", {
                    "id": f"taxi_{vehicle_counter}",
                    "depart": "0.00",
                    "type": vt["id"]
                })
                ET.SubElement(vehicle, "route", {"edges": edge_id})
                ET.SubElement(vehicle, "param", key="device.taxi.end", value=str(generate_work_duration()))
            elif idle_mechanism == "stop":
                trip = ET.SubElement(root, "trip", {
                    "id": f"taxi_{vehicle_counter}",
                    "depart": "0.00",
                    "type": vt["id"],
                    "personCapacity": "4"
                })
                ET.SubElement(trip, "stop", {"lane": lane_id, "triggered": "person"})
                ET.SubElement(trip, "param", key="device.taxi.end", value=str(generate_work_duration(starting = True)))
            else:
                raise ValueError("Invalid idle mechanism, please choose 'stop' or 'randomCircling'")
            vehicle_counter += 1

    start_date = datetime.strptime(start_date_str, "%Y-%m-%d").strftime("%y%m%d")
    end_date = datetime.strptime(end_date_str, "%Y-%m-%d").strftime("%y%m%d")
    start_hour = datetime.strptime(start_time_str, "%H:%M").strftime("%H")
    end_hour = datetime.strptime(end_time_str, "%H:%M").strftime("%H")

    # Create full folder path: root/timeslot/
    start_d = start_date.replace("-", "")
    end_d = end_date.replace("-", "")
    start_h = start_hour.replace(":", "")
    end_h = end_hour.replace(":", "")
    timeslot = f"{start_d}{start_h}_{end_d}{end_h}"
    full_folder_path = os.path.join(sf_tnc_fleet_folder_path, timeslot)
    os.makedirs(full_folder_path, exist_ok=True)

    # Filename and full path
    filename = f"sf_tnc_fleet_{timeslot}.rou.xml"
    full_path = os.path.join(full_folder_path, filename)

    # Write XML
    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    tree.write(full_path, encoding="utf-8", xml_declaration=True)

    print(f"✅ DRT vehicle fleet written to: {full_path} ({vehicle_counter} vehicles)")

    return full_path


def get_valid_taxi_edges(
        net_file: str,
        safe_edge_ids: set = None
    ) -> set:
    """
    Extracts usable edge IDs where taxis can drive. Uses sumolib (static network loading).

    This function:
    - Reads the SUMO network file.
    - Iterates through edges and checks if they are valid for taxi routing.
    - Filters out edges that are internal, dead-end, or not strongly connected.
    - Returns a set of valid edge IDs.

    Parameters:
    ----------
    - net_file: str
        Path to the SUMO .net.xml file.
    - safe_edge_ids: set or None, optional
        Set of edge IDs to consider as valid (e.g., strongly connected edges).
        If None, all edges are considered.

    Returns:
    -------
    set
        Set of valid edge IDs.
    """
    # Load the SUMO network
    sfnet = net.readNet(net_file)
    valid_edges = set()

    for edge in sfnet.getEdges():
        if edge.isSpecial():
            continue  # Skip internal/junction edges
        if not edge.getOutgoing():
            continue  # Skip dead-end edges
        if safe_edge_ids is not None and edge.getID() not in safe_edge_ids:
            continue  # Skip not strongly connnected edges

        # Check if any lane in this edge allows taxi or passenger
        for lane in edge.getLanes():
            allowed_classes = getattr(lane, '_allowed', [])
            if "passenger" in allowed_classes or "taxi" in allowed_classes:
                valid_edges.add(edge.getID())
                break  # No need to check other lanes once we find one good

    print(f"✅ Found {len(valid_edges)} valid taxi edges.")

    return valid_edges


def generate_matched_drt_requests(
        tnc_data: dict,
        taz_edge_mapping: dict,
        start_date_str: str,
        end_date_str: str,
        start_time_str: str,
        end_time_str: str,
        sf_requests_folder_path: str,
        valid_edge_ids: set
    ) -> Path:
    """
    Generates DRT requests for SUMO from TNC data and TAZ-edge mapping,
    ensuring all persons depart from and arrive at valid, reachable edges.

    This function:
    - Reads a nested dictionary of TNC pickups and dropoffs per TAZ and hour.
    - For each TAZ, randomly selects an edge from the mapping.
    - For each pickup, finds a valid dropoff edge in a different TAZ.
    - Creates a SUMO route XML structure with <person> entries.
    - Each person is assigned a departure time and a ride from pickup to dropoff.
    - The output is saved to the specified path.

    Parameters:
    ----------
    - tnc_data: dict
        Nested dictionary of Uber pickups and dropoffs per TAZ and hour.
    - taz_edge_mapping: dict
        Mapping {taz_id: {'centroid_edge_id': edge_id}}.
    - start_date_str: str
        Start date in 'YYYY-MM-DD' format (e.g., '2021-03-25').
    - end_date_str: str
        End date in 'YYYY-MM-DD' format (e.g., '2021-03-26').
    - start_time_str: str
        Start time in 'HH:MM' format (e.g., '08:00').
    - end_time_str: str
        End time in 'HH:MM' format (e.g., '10:00').
    - sf_requests_folder_path: str
        Path to save the resulting SUMO .rou.xml file with <person> requests.
    - valid_edge_ids: set
        Set of SUMO edge IDs validated for taxi routing (connected, non-junction, drivable).

    Returns:
    -------
    Path
        Full path to the saved XML file.
    """
    person_elements = []
    person_id = 0

    # Compute simulation start time
    sim_start = datetime.strptime(f"{start_date_str} {start_time_str}", "%Y-%m-%d %H:%M")

    # Build pickup list
    pickups_by_hour = []
    for taz, hour_data in tnc_data.items():
        if taz not in taz_edge_mapping:
            print(f"Pickup warning: TAZ {taz} not found in edge mapping.")
            continue
        edge = random.choice(taz_edge_mapping[taz]['polygon_edge_ids'])
        if edge not in valid_edge_ids or edge.startswith(":"):
            print(f"Pickup warning: Edge {edge} not valid for TAZ {taz}.")
            continue
        for hour, stats in hour_data.items():
            # Roll over to next day if hour is earlier than sim start hour
            pickup_date = sim_start.date()
            if hour < sim_start.hour:
                pickup_date += timedelta(days=1)
            pickup_datetime = datetime.combine(pickup_date, time(hour=hour))
            base_offset = int((pickup_datetime - sim_start).total_seconds())
            for _ in range(stats['pickups']):
                # Interpret 'hour' as actual hour-of-day
                depart_time = base_offset + random.randint(0, 3599)
                pickups_by_hour.append({
                    "taz": taz,
                    "edge": edge,
                    "depart_time": depart_time
                })

    # Build dropoff list
    dropoffs_by_taz = defaultdict(list)
    for taz, hour_data in tnc_data.items():
        if taz not in taz_edge_mapping:
            print(f"Dropoff warning: TAZ {taz} not found in edge mapping.")
            continue
        edge = random.choice(taz_edge_mapping[taz]['polygon_edge_ids'])
        if edge not in valid_edge_ids or edge.startswith(":"):
            print(f"Dropoff warning: Edge {edge} not valid for TAZ {taz}.")
            continue
        for hour, stats in hour_data.items():
            dropoffs_by_taz[taz].extend([edge] * stats['dropoffs'])

    dropoff_pool = [(taz, edge) for taz, edges in dropoffs_by_taz.items() for edge in edges]
    random.shuffle(dropoff_pool)
    dropoff_iter = iter(dropoff_pool)

    # Generate requests
    for pickup in pickups_by_hour:
        from_taz = pickup['taz']
        from_edge = pickup['edge']
        depart_time = pickup['depart_time']
        found = False
        while True:
            try:
                to_taz, to_edge = next(dropoff_iter)
                if to_taz != from_taz:
                    found = True
                    break
            except StopIteration:
                break
        if not found:
            continue
        person = ET.Element("person", {
            "id": f"p_{person_id}",
            "depart": str(depart_time) + ".00",
        })
        ET.SubElement(person, "ride", {
            "from": from_edge,
            "to": to_edge,
            "lines": "taxi"
        })
        person_elements.append((depart_time, person))
        person_id += 1
    person_elements.sort(key=lambda x: x[0])
    root = ET.Element("routes")
    for _, person in person_elements:
        root.append(person)

    start_date = datetime.strptime(start_date_str, "%Y-%m-%d").strftime("%y%m%d")
    end_date = datetime.strptime(end_date_str, "%Y-%m-%d").strftime("%y%m%d")
    start_hour = datetime.strptime(start_time_str, "%H:%M").strftime("%H")
    end_hour = datetime.strptime(end_time_str, "%H:%M").strftime("%H")

    # Create full folder path: root/timeslot/
    start_d = start_date.replace("-", "")
    end_d = end_date.replace("-", "")
    start_h = start_hour.replace(":", "")
    end_h = end_hour.replace(":", "")
    timeslot = f"{start_d}{start_h}_{end_d}{end_h}"

    full_folder_path = os.path.join(sf_requests_folder_path, timeslot)
    os.makedirs(full_folder_path, exist_ok=True)

    # Filename and full path
    filename = f"sf_passenger_requests_{timeslot}.rou.xml"
    full_path = os.path.join(full_folder_path, filename)

    # Write XML
    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    tree.write(full_path, encoding="utf-8", xml_declaration=True)
    print(f"✅ DRT passenger requests written to: {full_path} | Total requests generated: {person_id}")

    return full_path


def filter_polygon_edges(
        polygon_edge_str: str,
        safe_edge_ids: set
        ) -> list:
    """Filters edge list string, keeping only strongly connected edges."""
    edge_list = ast.literal_eval(polygon_edge_str)
    return [e for e in edge_list if e in safe_edge_ids]


def filter_polygon_lanes(
        polygon_lane_str: str,
        safe_edge_ids: set
        ) -> list:
    """Filters lane list string, keeping only lanes whose parent edge is in the strongly connected set."""
    lane_list = ast.literal_eval(polygon_lane_str)
    return [l for l in lane_list if l.split('_')[0] in safe_edge_ids]


def generate_work_duration(starting: bool = False) -> int:
    """
    Generates a taxi work duration (in hours) based on the following distribution:
    - 51% work between 30 minutes and 2 hours
    - 30% work between 2 and 5 hours
    - 12% work between 5 and 7 hours
    - 7% work between 7 and 8 hours

    Parameters:
    ----------
    - starting: bool
        If True, the duration is generated for the vehicles at the simulation start.

    Returns:
    ------ 
    int
        Duration in seconds.
    """
    r = random.random()
    if starting:
        if r < 0.58:
            return round(random.uniform(1800, 3600))
        elif r < 0.88:
            return round(random.uniform(3601, 14400))
        else:
            return round(random.uniform(14401, 21600))
    else:
        if r < 0.51:
            return round(random.uniform(1800, 7200))
        elif r < 0.81:
            return round(random.uniform(7201, 18000))
        elif r < 0.93:
            return round(random.uniform(18001, 25200))
        else:
            return round(random.uniform(25201, 28800))