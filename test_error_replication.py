import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pyproj import Proj, transform

# --- Setup for Coordinate Transformations ---
wgs84 = Proj(proj='latlong', ellps='WGS84', datum='WGS84')
ecef = Proj(proj='geocent', ellps='WGS84', datum='WGS84')

def get_ecef_from_faulty_transform(lat_deg, lon_deg, alt):
    """
    REPLICATES THE ERROR: Treats degree input as if it were radians.
    """
    # The colleague's code was expecting radians but received degrees.
    # So we call transform with radians=True, forcing pyproj to interpret
    # the degree values as if they were radians.
    return transform(wgs84, ecef, lon_deg, lat_deg, alt, radians=True)


def main():
    """
    Main function to visualize if replicating the suspected error
    causes the vehicle path and map features to align.
    """
    SIMULATION_FOLDER = "simulations100"
    OUTPUT_FILENAME = "error_replication_visualization.png"

    print(f"--- Creating Error Replication Visualization for: {SIMULATION_FOLDER} ---")

    # --- 1. Load Context and Map Features ---
    context_path = os.path.join(SIMULATION_FOLDER, "context.json")
    with open(context_path, 'r') as f:
        context_data = json.load(f)
    map_features = context_data["environment"]["location"]["map_features"]

    # --- 2. Load Vehicle Path (ECEF x, y) ---
    data_path = os.path.join(SIMULATION_FOLDER, "data")
    vehicle_path_x = []
    vehicle_path_y = []

    for filename in os.listdir(data_path):
        if filename.endswith(".json"):
            with open(os.path.join(data_path, filename), 'r') as f:
                readings = json.load(f)
                if readings.get('series') and readings['series'][0]['dev'] == 7: # dev 7 is GPS_X
                    vehicle_path_x = [r['x'] for r in readings['series']]
                    vehicle_path_y = [r['y'] for r in readings['series']]
                    print(f"Loaded {len(vehicle_path_x)} points from vehicle's GPS path.")
                    break
    
    if not vehicle_path_x:
        print("ERROR: Could not find the GPS data file.")
        return

    # --- 3. Setup Plot ---
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.set_title(f"Error Replication for {SIMULATION_FOLDER}")
    ax.set_xlabel("ECEF X Coordinate")
    ax.set_ylabel("ECEF Y Coordinate")
    ax.set_aspect('equal', adjustable='box')
    
    # --- 4. Plot Vehicle Path ---
    ax.plot(vehicle_path_x, vehicle_path_y, label="Vehicle Path", color="blue", zorder=2)
    ax.scatter(vehicle_path_x[0], vehicle_path_y[0], c='green', s=100, label='Start', zorder=5)
    ax.scatter(vehicle_path_x[-1], vehicle_path_y[-1], c='red', s=100, label='End', zorder=5)

    # --- 5. Process and Plot Map Features using the FAULTY function ---
    feature_handles = []
    for feature_type, features in map_features.items():
        if not features: continue
        color = 'gray'
        if 'pothole' in feature_type: color = 'red'
        if 'ramp' in feature_type: color = 'purple'
        if 'speedbump' in feature_type: color = 'orange'
        feature_handles.append(patches.Patch(color=color, label=f'{feature_type.capitalize()}s'))
        
        for feature in features:
            if 'start' in feature: # Ramp-like
                start_ecef = get_ecef_from_faulty_transform(feature['start']['latitude'], feature['start']['longitude'], feature['start']['altitude'])
                end_ecef = get_ecef_from_faulty_transform(feature['end']['latitude'], feature['end']['longitude'], feature['end']['altitude'])
                ax.plot([start_ecef[0], end_ecef[0]], [start_ecef[1], end_ecef[1]], color=color, linewidth=feature.get('w', 1) * 0.5, zorder=1, alpha=0.7)
            else: # Point-like
                center_ecef = get_ecef_from_faulty_transform(feature['latitude'], feature['longitude'], feature['altitude'])
                radius = feature.get('r', feature.get('l', 1))
                circle = patches.Circle((center_ecef[0], center_ecef[1]), radius=radius, color=color, zorder=1, alpha=0.7)
                ax.add_patch(circle)

    ax.legend(handles=feature_handles + ax.get_legend_handles_labels()[0])
    ax.grid(True)
    
    # --- 6. Save Plot ---
    plt.savefig(OUTPUT_FILENAME)
    print(f"\nSuccessfully created visualization: '{OUTPUT_FILENAME}'")

if __name__ == "__main__":
    main() 