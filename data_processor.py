import re
import math
import json
from smartdata import DB_Record, Unit
from sniffer_cleaner import remove_content_after_pattern
import pyproj

wgs84_to_ecef = pyproj.Transformer.from_crs(
    {"proj":'latlong', "ellps":'WGS84', "datum":'WGS84'},
    {"proj":'geocent', "ellps":'WGS84', "datum":'WGS84'},
)

ecef_to_wgs84 = pyproj.Transformer.from_crs(
    {"proj":'geocent', "ellps":'WGS84', "datum":'WGS84'},
    {"proj":'latlong', "ellps":'WGS84', "datum":'WGS84'},
)

data_model = {
    "IMU_ACC_x": 0,
    "IMU_ACC_Y": 1,
    "IMU_ACC_Z": 2,
    "SPEED_X": 3,
    "SPEED_Y": 4,
    "SPEED_Z": 5,
    "YAW_RATE": 36,
    "PITCH_RATE": 37,
    "ROLL_RATE": 38,
    "DRAG": 6,
    "GPS_X": 7,
    "GPS_Y": 8,
    "GPS_Z": 9,
    "GEAR": 10,
    "ENGINE_RPM": 11,
    "YAW": 13,
    "PITCH": 14,
    "ROLL": 15,
    #"DYNAMICS_STATE": 16,
    #"DESTINATION": 41,
    #"CAMERA_IMAGE": 17,
    #"LIDAR_CLOUD_POINT": 18,
    #"RADAR_CLOUD_POINT": 19,
    #"LIST_OF_OBJECTS_CAM": 20,
    #"LIST_OF_OBJECTS_LIDAR": 21,
    #"LIST_OF_OBJECTS_RADAR": 22,
    #"LIST_OF_OBJECTS_MAP": 24,
    #"ETSI_CAM": 26,
    #"LIST_OF_OBJECTS_FUSER": 23,
    #"ETSI_DENM": 27,
    #"ETSI_CPM": 28,
    #"LIST_OF_STATE_PATH": 29,
    #"LIST_OF_STATE_PLAN": 30,
    #"CONTROLLER": 39,
    "STEER": 31,
    "THROTTLE": 32,
    "BRAKE": 33,
    "REVERSE": 34,
    "HAND_BRAKE": 35,
    "FUEL": 42,
    "CO2_EMISSION": 43,
    "SUSPENSION_PREDICTOR": 47,
    "SUSPENSION_PREDICTOR_P": 48,
    "VISUAL_VIBRATION_DETECTION": 52,
    "MASS": 53,
    "ODOMETER": 54,
}

def parse_controle_txt(content):
    pattern = re.compile(
        r"seq (\d+)-(\d+):\s*mass = (\d+)\s*friction = ([\d.]+)",
        re.MULTILINE
    )
    result = {}
    for match in pattern.finditer(content):
        start, end = int(match.group(1)), int(match.group(2))
        mass = int(match.group(3))
        friction = float(match.group(4))
        result[(start, end)] = {"mass": mass, "friction": friction}
    return result

def get_entry_for_index(parsed_dict, index):
    for (start, end), entry in parsed_dict.items():
        if start <= index <= end:
            return entry
    return None

def convert_to_valid_json(text):
    # Strip the outer brackets and any whitespace
    text = text.strip()[1:-1]
    
    # Split by comma to get individual key-value pairs
    pairs = text.split(',')
    
    # Process each pair to format it as "key":"value"
    json_pairs = []
    for pair in pairs:
        if '=' in pair:
            key, value = pair.split('=', 1)
            # Add quotes around the key and the value
            json_pairs.append(f'"{key.strip()}":"{value.strip()}"')

    # Join the pairs with commas and wrap in curly braces to form a JSON object
    return f"{{{','.join(json_pairs)}}}"

def get_entries_from_dirty_sniffer_log(file_path, save_path):
    ENTRY_REGEX = re.compile(
        r"\(u=\{[^}]+\}=>\d+,d=(?P<d>\d+),t=(?P<t>\d+),sig=(?P<sig>\d+)\)={(?P<v>[^}]+)}",
        re.MULTILINE
    )
    
    clean_sniffer = remove_content_after_pattern(file_path, save_path)
    start_string = "Log Start:\n"
    log_start = clean_sniffer.find(start_string)
    if log_start != -1:
        clean_sniffer = clean_sniffer[log_start + len(start_string):]

    entries = {}
    for match in ENTRY_REGEX.finditer(clean_sniffer):
        val_str = match.group("v").strip()
        val_str = val_str.replace('\n', '').replace('\t', '')
        
        # Check if it's a list
        if val_str.startswith('['):
            val_str = val_str[1:-1].strip() # remove brackets
            
            # Split into individual objects
            items_str = val_str.split('},')
            
            list_v = []
            for item in items_str:
                if not item: continue
                item = item.replace('{','').replace('}','').strip()
                
                # Use the robust converter
                json_obj_str = convert_to_valid_json(f"{{{item}}}")
                try:
                    list_v.append(json.loads(json_obj_str))
                except json.JSONDecodeError as e:
                    print(f"Error decoding item: {json_obj_str} -> {e}")

            value = list_v
        else:
            try:
                value = float(val_str)
            except ValueError:
                value = val_str

        u_match = re.search(r'u={([^}]+)}', match.group(0))
        unit_val = u_match.group(1) if u_match else ''
        
        # This part of the regex is tricky, so we get the raw 'u' value
        u_int_match = re.search(r"=>(\d+),d=", match.group(0))
        u = int(u_int_match.group(1)) if u_int_match else 0
        
        entry = DB_Record(
            unit=Unit(u), 
            dev=int(match.group("d")), 
            t=int(match.group("t")), 
            signature=int(match.group("sig")), 
            value=value
        )
        
        if entry.signature not in entries:
            entries[entry.signature] = {}
        if entry.dev not in entries[entry.signature]:
            entries[entry.signature][entry.dev] = []
        entries[entry.signature][entry.dev].append(entry)
        
    return entries

def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371e3  # metres
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = math.sin(delta_phi / 2) * math.sin(delta_phi / 2) + \
        math.cos(phi1) * math.cos(phi2) * \
        math.sin(delta_lambda / 2) * math.sin(delta_lambda / 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return R * c

def find_map_feature(lat, lon, map_features):
    # This function uses a simplified approach. For more accuracy, especially with rotated features,
    # you might need a more sophisticated geometric library.
    for feature_type, features in map_features.items():
        for feature in features:
            if feature_type == 'pothole':
                feature_lat = feature.get('latitude')
                feature_lon = feature.get('longitude')
                if feature_lat and feature_lon:
                    distance = haversine_distance(lat, lon, feature_lat, feature_lon)
                    radius = feature.get('r', 5) 
                    if distance <= radius:
                        return f"pothole_{feature.get('class', 'unknown')}"

            elif feature_type == 'ramp':
                start = feature.get('start')
                end = feature.get('end')
                if start and end:
                    s_lat, s_lon = start.get('latitude', 0), start.get('longitude', 0)
                    e_lat, e_lon = end.get('latitude', 0), end.get('longitude', 0)
                    
                    in_lon = (s_lon <= lon <= e_lon) or (e_lon <= lon <= s_lon)
                    in_lat = (s_lat <= lat <= e_lat) or (e_lat <= lat <= s_lat)

                    if in_lon and in_lat:
                        return f"ramp_{feature.get('class', 'unknown')}"
                        
            elif feature_type in ['speedbump', 'elevated_crosswalk', 'cut']:
                feature_lat = feature.get('latitude')
                feature_lon = feature.get('longitude')
                if feature_lat and feature_lon:
                    length = feature.get('l', 2.0) / 2.0 / 111139 
                    width = feature.get('r', 2.0) / 2.0 / 111139
                    
                    in_lon = (feature_lon - width <= lon <= feature_lon + width)
                    in_lat = (feature_lat - length <= lat <= feature_lat + length)

                    if in_lon and in_lat:
                        return f"{feature_type}_{feature.get('class', 'unknown')}"
    return None

def parse_egos_ECEF_position(egos, is_time_step_fix, ts_step, signature):
    """
    Parses DB_Record of mv and update the position in ECEF coordinates.
    """
    t0 = egos[0].t
    for index, mv in enumerate(egos):
        ego_data = None
        if isinstance(mv.value, list) and mv.value:
            # It's a list, take the first element which should be a dict
            ego_data = mv.value[0]
        elif isinstance(mv.value, dict):
            # It's a dictionary
            ego_data = mv.value

        if ego_data and isinstance(ego_data, dict):
            lon = float(ego_data.get('lon', 0)) / 100000000.
            lat = float(ego_data.get('lat', 0)) / 100000000.
            alt = float(ego_data.get('alt', 0)) / 1000.
            x, y, z = wgs84_to_ecef.transform(lat, lon, alt)
            mv.x = x
            mv.y = y
            mv.z = z
        else:
            # If there's no data or it's not a dict, set coordinates to None
            mv.x = None
            mv.y = None
            mv.z = None

        if (is_time_step_fix):
            mv.t = t0 + index * ts_step
        mv.signature = signature
    return egos 