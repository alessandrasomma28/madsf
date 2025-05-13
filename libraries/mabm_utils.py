import xml.etree.ElementTree as ET

def parse_taxis_and_passengers(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    taxis = []
    passengers = []

    for trip in root.findall('trip'):
        taxi_id = trip.get('id')
        lane = trip.find('stop').get('lane')
        taxis.append((taxi_id, lane))

    for person in root.findall('person'):
        person_id = person.get('id')
        ride = person.find('ride')
        from_edge = ride.get('from')
        to_edge = ride.get('to')
        passengers.append((person_id, from_edge, to_edge))

    return taxis, passengers