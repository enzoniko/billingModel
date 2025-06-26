import os
import json
import numpy as np
from pyproj import Proj, transform

# --- Copied from process_simulations.py for self-containment ---
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

def calculate_distance(p1, p2):
    """Calculates the Euclidean distance between two 3D points."""
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2 + (p1[2] - p2[2])**2)
# --- End of Copied Section ---


def main():
    """
    Main diagnostic function to test context assignment by converting
    the vehicle's own WGS84 path to ECEF.
    """
    SIMULATION_FOLDER = "simulations100"
    TARGET_FEATURE_TYPE = "pothole"
    TARGET_FEATURE_INDEX = 0
    
    # DEV IDs for GNSS data
    # From context.json: longitude is dev 7, latitude is dev 8, altitude is dev 9
    LON_DEV_ID = 7
    LAT_DEV_ID = 8
    ALT_DEV_ID = 9

    print(f"--- Running WGS84-based Diagnostic for: {SIMULATION_FOLDER} ---")

    # --- 1. Load Context and Target Feature ---
    context_path = os.path.join(SIMULATION_FOLDER, "context.json")
    if not os.path.exists(context_path):
        print(f"ERROR: Cannot find '{context_path}'")
        return

    with open(context_path, 'r') as f:
        context_data = json.load(f)

    try:
        target_feature_wgs84 = context_data["environment"]["location"]["map_features"][TARGET_FEATURE_TYPE][TARGET_FEATURE_INDEX]
        print(f"Target Feature Loaded: A '{TARGET_FEATURE_TYPE}' at lat/lon ({target_feature_wgs84['latitude']}, {target_feature_wgs84['longitude']})")
    except (KeyError, IndexError):
        print(f"ERROR: Could not find target feature in context.json")
        return
        
    feature_ecef = get_ecef_from_wgs84(
        target_feature_wgs84['latitude'],
        target_feature_wgs84['longitude'],
        target_feature_wgs84['altitude']
    )
    print(f"Target Feature ECEF Coords: {feature_ecef}")

    # --- 2. Load Vehicle Path Data (Lat, Lon, Alt) ---
    data_path = os.path.join(SIMULATION_FOLDER, "data")
    sensor_files = {
        LAT_DEV_ID: None,
        LON_DEV_ID: None,
        ALT_DEV_ID: None
    }

    for filename in os.listdir(data_path):
        with open(os.path.join(data_path, filename), 'r') as f:
            readings = json.load(f)
            if readings.get('series'):
                dev_id = readings['series'][0]['dev']
                if dev_id in sensor_files:
                    sensor_files[dev_id] = [r['value'] for r in readings['series']]
    
    lats = sensor_files[LAT_DEV_ID]
    lons = sensor_files[LON_DEV_ID]
    alts = sensor_files[ALT_DEV_ID]

    if not all((lats, lons, alts)):
        print("ERROR: Could not load latitude, longitude, or altitude data for the vehicle.")
        return
    
    print(f"Loaded {len(lats)} points from vehicle's WGS84 path.")

    # --- 3. Process Path and Find Minimum Distance ---
    min_distance = float('inf')
    
    for i in range(len(lats)):
        try:
            vehicle_pos_wgs = (lats[i], lons[i], alts[i])
            vehicle_pos_ecef = get_ecef_from_wgs84(*vehicle_pos_wgs)
            
            distance = calculate_distance(vehicle_pos_ecef, feature_ecef)
            if distance < min_distance:
                min_distance = distance
        except Exception as e:
            print(f"Warning: Could not process point {i}. Data: {vehicle_pos_wgs}. Error: {e}")
            continue

    # --- 4. Print Summary ---
    print("\n--- Diagnostic Summary ---")
    print(f"The minimum distance found between the vehicle and the target '{TARGET_FEATURE_TYPE}' was:")
    print(f"==> {min_distance:.2f} meters")
    print("--------------------------")
    
    feature_radius = target_feature_wgs84.get('r', 0)
    if min_distance <= feature_radius:
        print("\nConclusion: SUCCESS! The vehicle's path passed within the feature's radius.")
        print("The coordinate mismatch is confirmed as the root cause. The proposed fix is correct.")
    else:
        print("\nConclusion: The vehicle path still does not intersect the feature.")
        print("The coordinate mismatch may not be the only issue. The feature's dimensions or location could also be a factor.")


if __name__ == "__main__":
    main() 