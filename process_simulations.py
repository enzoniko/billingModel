import os
import json
import math
import pandas as pd
from pyproj import Proj, transform
import numpy as np

# Data model mapping sensor names to their device IDs
DATA_MODEL = {
    "IMU_ACC_x": 0, "IMU_ACC_Y": 1, "IMU_ACC_Z": 2,
    "SPEED_X": 3, "SPEED_Y": 4, "SPEED_Z": 5,
    "DRAG": 6, "GPS_X": 7, "GPS_Y": 8, "GPS_Z": 9,
    "GEAR": 10, "ENGINE_RPM": 11, "YAW": 13, "PITCH": 14, "ROLL": 15, "STEER": 31,
    "THROTTLE": 32, "BRAKE": 33, "REVERSE": 34, "HAND_BRAKE": 35,
    "YAW_RATE": 36, "PITCH_RATE": 37, "ROLL_RATE": 38, "FUEL": 42, "CO2_EMISSION": 43, "ODOMETER": 54,
}

# Define WGS84 and ECEF projections for coordinate conversion
wgs84 = Proj(proj='latlong', ellps='WGS84', datum='WGS84')
ecef = Proj(proj='geocent', ellps='WGS84', datum='WGS84')

def get_ecef_from_wgs84(lat, lon, alt):
    """
    Converts WGS84 coordinates to ECEF, applying specific truncation.
    """
    lon = float(int(float(lon) * 100000000.) / 100000000.)
    lat = float(int(float(lat) * 100000000.) / 100000000.)
    alt = float(int(float(alt) * 1000.) / 1000.)
    return transform(wgs84, ecef, lon, lat, alt, radians=False)

def get_dist_sq(p1, p2):
    """Calculates the squared Euclidean distance between two 3D points."""
    return (p1[0] - p2[0])**2 + (p1[1] - p2[1])**2 + (p1[2] - p2[2])**2

def get_point_to_line_segment_dist_sq(p, a, b):
    """
    Calculates the squared distance from a point p to a line segment ab.
    Returns the squared distance and the closest point on the segment.
    """
    l2 = get_dist_sq(a, b)
    if l2 == 0.0:
        return get_dist_sq(p, a), a
    
    p_vec = np.array(p)
    a_vec = np.array(a)
    b_vec = np.array(b)
    
    t = max(0, min(1, np.dot(p_vec - a_vec, b_vec - a_vec) / l2))
    projection = a_vec + t * (b_vec - a_vec)
    
    return get_dist_sq(p, projection.tolist()), projection.tolist()


def get_map_context(vehicle_pos, map_features_ecef):
    """
    Determines the map feature context for a given vehicle position.
    """
    # Check ramps first
    for feature in map_features_ecef.get("ramp", []):
        dist_sq, _ = get_point_to_line_segment_dist_sq(vehicle_pos, feature['start_ecef'], feature['end_ecef'])
        if dist_sq <= (feature['w'] / 2)**2:
            return feature['id']
            
    # Check other features with a radius
    for feature_type in ["pothole", "speedbump", "elevated_crosswalk", "cut"]:
        for feature in map_features_ecef.get(feature_type, []):
            dist_sq = get_dist_sq(vehicle_pos, feature['center_ecef'])
            if dist_sq <= feature['r']**2:
                return feature['id']
                
    return "road"


def process_simulation_folder(folder_path, output_dir):
    """
    Processes a single simulation folder, generates, and saves a CSV file.
    """
    sim_id = os.path.basename(folder_path).replace("simulations", "")
    print(f"--- Processing simulation {sim_id} ---")
    
    context_path = os.path.join(folder_path, "context.json")
    data_path = os.path.join(folder_path, "data")

    if not os.path.exists(context_path) or not os.path.exists(data_path):
        print(f"Skipping {folder_path}: missing context.json or data directory.")
        return

    # 1. Load and parse context.json
    with open(context_path, 'r') as f:
        context_data = json.load(f)

    vehicle_mass = context_data["vehicles"][0]["body"]["mass"]
    map_friction = context_data["environment"]["simulatorSettings"]["map_friction"]
    map_features = context_data["environment"]["location"]["map_features"]
    
    # 2. Pre-process map features to get ECEF coordinates and unique IDs
    map_features_ecef = {}
    for feature_type, features in map_features.items():
        map_features_ecef[feature_type] = []
        for feature in features:
            unique_id = f"{feature_type}_{feature.get('class', '')}"
            processed_feature = {"type": feature_type}
            
            if 'start' in feature and 'end' in feature: # Ramp-like features
                processed_feature['start_ecef'] = get_ecef_from_wgs84(
                    feature['start']['latitude'], feature['start']['longitude'], feature['start']['altitude']
                )
                processed_feature['end_ecef'] = get_ecef_from_wgs84(
                    feature['end']['latitude'], feature['end']['longitude'], feature['end']['altitude']
                )
                processed_feature['w'] = feature.get('w', 0)
                unique_id += f"_{feature.get('ry', 0)}_{feature.get('rz', 0)}"

            else: # Point-based features
                processed_feature['center_ecef'] = get_ecef_from_wgs84(
                    feature['latitude'], feature['longitude'], feature['altitude']
                )
                processed_feature['r'] = feature.get('r', feature.get('l', 0)) # use r or l
                unique_id += f"_{feature.get('r', feature.get('l', 0))}"
            
            processed_feature['id'] = unique_id
            map_features_ecef[feature_type].append(processed_feature)

    # 3. Load sensor data
    dev_to_name = {v: k for k, v in DATA_MODEL.items()}
    sensor_data = {}

    # Find the expected length from a reliable sensor file first
    expected_length = 0
    for filename in os.listdir(data_path):
        if filename.endswith(".json"):
            with open(os.path.join(data_path, filename), 'r') as f:
                 readings = json.load(f)
                 if readings.get('series'):
                     expected_length = len(readings['series'])
                     break
    
    if expected_length == 0:
        print(f"Skipping {folder_path}: No data series found in any file.")
        return

    for filename in os.listdir(data_path):
        if filename.endswith(".json"):
            with open(os.path.join(data_path, filename), 'r') as f:
                try:
                    readings = json.load(f)
                    if not readings.get('series'):
                        continue
                    
                    dev_id = readings['series'][0]['dev']
                    if dev_id in dev_to_name:
                        sensor_name = dev_to_name[dev_id]
                        
                        values = [r['value'] for r in readings['series']]
                        
                        # Pad or truncate to expected length
                        if len(values) > expected_length:
                            values = values[:expected_length]
                        elif len(values) < expected_length:
                            values.extend([np.nan] * (expected_length - len(values)))

                        sensor_data[sensor_name] = values
                        
                        # Special case for GPS: extract x, y, z
                        if dev_id == 7: # GPS_X
                            sensor_data['x'] = [r['x'] for r in readings['series']]
                            sensor_data['y'] = [r['y'] for r in readings['series']]
                            sensor_data['z'] = [r['z'] for r in readings['series']]
                            
                            if len(sensor_data['x']) > expected_length:
                                sensor_data['x'] = sensor_data['x'][:expected_length]
                                sensor_data['y'] = sensor_data['y'][:expected_length]
                                sensor_data['z'] = sensor_data['z'][:expected_length]
                            elif len(sensor_data['x']) < expected_length:
                                pad_count = expected_length - len(sensor_data['x'])
                                sensor_data['x'].extend([np.nan] * pad_count)
                                sensor_data['y'].extend([np.nan] * pad_count)
                                sensor_data['z'].extend([np.nan] * pad_count)


                except (json.JSONDecodeError, KeyError) as e:
                    print(f"Warning: Could not process {filename}. Error: {e}")

    if not sensor_data:
        print(f"Skipping {folder_path}: No valid sensor data found.")
        return
        
    df = pd.DataFrame(sensor_data)

    # 4. Data Transformation
    # Fill NA values that may result from sensor series having different lengths
    df = df.fillna(method='ffill').fillna(method='bfill')

    # Calculate speed
    df['SPEED'] = np.sqrt(df['SPEED_X']**2 + df['SPEED_Y']**2 + df['SPEED_Z']**2)
    df = df.drop(columns=['SPEED_X', 'SPEED_Y', 'SPEED_Z'])
    
    # Get map context
    # Ensure x,y,z are not NaN before applying context function
    df = df.dropna(subset=['x', 'y', 'z'])
    if df.empty:
        print(f"Skipping {folder_path}: No valid GPS data to determine context.")
        return
        
    df['context'] = df.apply(lambda row: get_map_context([row['x'], row['y'], row['z']], map_features_ecef), axis=1)

    # 5. Save to CSV
    output_filename = f"simulation_{sim_id}_mass_{int(vehicle_mass)}_friction_{map_friction}.csv"
    output_path = os.path.join(output_dir, output_filename)
    df.to_csv(output_path)
    print(f"Successfully created {output_path}")


def main():
    """
    Main function to find and process all simulation folders.
    """
    base_dir = os.getcwd()
    output_dir = os.path.join(base_dir, "processed_data")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    all_dirs = [d for d in os.listdir(base_dir) if os.path.isdir(d) and d.startswith("simulations")]
    
    for folder_name in sorted(all_dirs):
        folder_path = os.path.join(base_dir, folder_name)
        try:
            process_simulation_folder(folder_path, output_dir)
        except Exception as e:
            print(f"FATAL: An unexpected error occurred processing {folder_name}: {e}")

if __name__ == "__main__":
    main() 