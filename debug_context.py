import os
import json
import numpy as np
from process_simulations import get_ecef_from_wgs84

def calculate_distance(p1, p2):
    """Calculates the Euclidean distance between two 3D points."""
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2 + (p1[2] - p2[2])**2)

def main():
    """
    Main diagnostic function to analyze the distance between a vehicle's path
    and a specific map feature.
    """
    SIMULATION_FOLDER = "simulations100"
    TARGET_FEATURE_TYPE = "pothole" # You can change this to "ramp", etc.
    TARGET_FEATURE_INDEX = 0      # 0 for the first feature of that type

    print(f"--- Running Diagnostic for: {SIMULATION_FOLDER} ---")

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
        print(f"ERROR: Could not find feature of type '{TARGET_FEATURE_TYPE}' at index {TARGET_FEATURE_INDEX} in context.json")
        return
        
    # Convert feature's location to ECEF
    feature_ecef = get_ecef_from_wgs84(
        target_feature_wgs84['latitude'],
        target_feature_wgs84['longitude'],
        target_feature_wgs84['altitude']
    )
    print(f"Target Feature ECEF Coords: {feature_ecef}")

    # --- 2. Load Vehicle Path Data ---
    data_path = os.path.join(SIMULATION_FOLDER, "data")
    gps_file_found = False
    vehicle_path_ecef = []

    for filename in os.listdir(data_path):
        if filename.endswith(".json"):
            with open(os.path.join(data_path, filename), 'r') as f:
                readings = json.load(f)
                if readings.get('series') and readings['series'][0]['dev'] == 7: # dev 7 is GPS_X
                    vehicle_path_ecef = [[r['x'], r['y'], r['z']] for r in readings['series']]
                    gps_file_found = True
                    print(f"Loaded {len(vehicle_path_ecef)} points from vehicle's GPS path.")
                    # Print the first coordinate pair for comparison
                    if vehicle_path_ecef:
                        print(f"Vehicle's First ECEF Coords: {vehicle_path_ecef[0]}")
                    break
    
    if not gps_file_found:
        print("ERROR: Could not find the GPS data file (dev id 7).")
        return

    # --- 3. Calculate Distances and Find Minimum ---
    min_distance = float('inf')
    
    for i, vehicle_pos in enumerate(vehicle_path_ecef):
        distance = calculate_distance(vehicle_pos, feature_ecef)
        if distance < min_distance:
            min_distance = distance

    # --- 4. Print Summary ---
    print("\n--- Diagnostic Summary ---")
    print(f"The minimum distance found between the vehicle and the target '{TARGET_FEATURE_TYPE}' was:")
    print(f"==> {min_distance:.2f} meters")
    print("--------------------------")
    
    if min_distance > 50: # Arbitrary large number
        print("\nConclusion: The minimum distance is very large.")
        print("This strongly suggests a COORDINATE SYSTEM MISMATCH.")
        print("The vehicle's (x, y, z) path is likely not in the same ECEF coordinate system as the map features.")
    else:
        print("\nConclusion: The vehicle passed close to the feature.")
        print("If context is still 'road', the issue is likely with the feature's dimensions (e.g., radius 'r' or width 'w' is too small).")


if __name__ == "__main__":
    main() 