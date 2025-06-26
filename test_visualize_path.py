import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from process_simulations import get_ecef_from_wgs84

def main():
    """
    Main function to visualize the vehicle path and map features in a top-down
    2D ECEF plot.
    """
    SIMULATION_FOLDER = "simulations100"
    OUTPUT_FILENAME = "path_visualization.png"

    print(f"--- Creating Visualization for: {SIMULATION_FOLDER} ---")

    # --- 1. Load Context and Map Features ---
    context_path = os.path.join(SIMULATION_FOLDER, "context.json")
    if not os.path.exists(context_path):
        print(f"ERROR: Cannot find '{context_path}'")
        return

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
        print("ERROR: Could not find the GPS data file (dev id 7).")
        return

    # --- 3. Setup Plot ---
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.set_title(f"Vehicle Path vs. Map Features for {SIMULATION_FOLDER} (ECEF Top-Down)")
    ax.set_xlabel("ECEF X Coordinate")
    ax.set_ylabel("ECEF Y Coordinate")
    ax.set_aspect('equal', adjustable='box')
    
    # --- 4. Plot Vehicle Path ---
    ax.plot(vehicle_path_x, vehicle_path_y, label="Vehicle Path", color="blue", zorder=2)
    ax.scatter(vehicle_path_x[0], vehicle_path_y[0], c='green', s=100, label='Start', zorder=5)
    ax.scatter(vehicle_path_x[-1], vehicle_path_y[-1], c='red', s=100, label='End', zorder=5)

    # --- 5. Process and Plot Map Features ---
    feature_handles = []
    
    for feature_type, features in map_features.items():
        if not features: continue
        
        # Color and label for the legend (one per feature type)
        color = 'gray'
        if 'pothole' in feature_type: color = 'red'
        if 'ramp' in feature_type: color = 'purple'
        if 'speedbump' in feature_type: color = 'orange'
        
        feature_handles.append(patches.Patch(color=color, label=f'{feature_type.capitalize()}s'))
        
        for feature in features:
            if 'start' in feature: # Ramp-like
                start_ecef = get_ecef_from_wgs84(feature['start']['latitude'], feature['start']['longitude'], feature['start']['altitude'])
                end_ecef = get_ecef_from_wgs84(feature['end']['latitude'], feature['end']['longitude'], feature['end']['altitude'])
                ax.plot([start_ecef[0], end_ecef[0]], [end_ecef[1], end_ecef[1]], color=color, linewidth=feature.get('w', 1) * 0.5, zorder=1, alpha=0.7)
            else: # Point-like
                center_ecef = get_ecef_from_wgs84(feature['latitude'], feature['longitude'], feature['altitude'])
                radius = feature.get('r', feature.get('l', 1))
                circle = patches.Circle((center_ecef[0], center_ecef[1]), radius=radius, color=color, zorder=1, alpha=0.7)
                ax.add_patch(circle)

    ax.legend(handles=feature_handles + ax.get_legend_handles_labels()[0])
    ax.grid(True)
    
    # --- 6. Save and Show Plot ---
    plt.savefig(OUTPUT_FILENAME)
    print(f"\nSuccessfully created visualization: '{OUTPUT_FILENAME}'")
    # plt.show() # Commented out for automated environments


if __name__ == "__main__":
    main() 