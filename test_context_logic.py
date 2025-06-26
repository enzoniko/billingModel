import numpy as np
from process_simulations import get_ecef_from_wgs84, get_map_context

def pre_process_mock_features(mock_features):
    """
    Helper function to convert mock WGS84 features to ECEF,
    mimicking the logic in the main script.
    """
    map_features_ecef = {}
    for feature_type, features in mock_features.items():
        map_features_ecef[feature_type] = []
        for feature in features:
            unique_id = f"{feature_type}_{feature.get('class', '')}"
            processed_feature = {"type": feature_type}
            
            if 'start' in feature and 'end' in feature: # Ramp-like
                processed_feature['start_ecef'] = get_ecef_from_wgs84(
                    feature['start']['latitude'], feature['start']['longitude'], feature['start']['altitude']
                )
                processed_feature['end_ecef'] = get_ecef_from_wgs84(
                    feature['end']['latitude'], feature['end']['longitude'], feature['end']['altitude']
                )
                processed_feature['w'] = feature.get('w', 0)
                unique_id += f"_{feature.get('ry', 0)}_{feature.get('rz', 0)}"

            else: # Point-based
                processed_feature['center_ecef'] = get_ecef_from_wgs84(
                    feature['latitude'], feature['longitude'], feature['altitude']
                )
                processed_feature['r'] = feature.get('r', 0)
                unique_id += f"_{feature.get('r', 0)}"

            processed_feature['id'] = unique_id
            map_features_ecef[feature_type].append(processed_feature)
    return map_features_ecef

def run_test(test_name, result, expected):
    """Helper to print test results in a standard format."""
    if result == expected:
        print(f"[PASS] {test_name}")
    else:
        print(f"[FAIL] {test_name}")
        print(f"  - Expected: {expected}")
        print(f"  - Got:      {result}")
        return 1
    return 0

def main():
    print("--- Running Context Logic Validation Tests ---")
    
    # 1. Define Mock Map Features in WGS84
    # A pothole centered at (lat=10, lon=10, alt=0) with a 5-meter radius
    # A ramp starting at (lat=20, lon=20, alt=10) and ending at (lat=20, lon=21, alt=10) with a 4-meter width
    mock_map_features_wgs84 = {
        "pothole": [{
            "latitude": 10.0,
            "longitude": 10.0,
            "altitude": 0,
            "r": 5, # 5-meter radius
            "class": "test_hole"
        }],
        "ramp": [{
            "start": {"latitude": 20.0, "longitude": 20.0, "altitude": 10},
            "end":   {"latitude": 20.0, "longitude": 20.0001, "altitude": 10}, # ~11 meters away in longitude
            "w": 4, # 4-meter width
            "class": "test_ramp",
            "ry": 0,
            "rz": 0
        }]
    }

    # 2. Pre-process features to get ECEF coordinates
    mock_features_ecef = pre_process_mock_features(mock_map_features_wgs84)
    pothole_id = mock_features_ecef["pothole"][0]['id']
    ramp_id = mock_features_ecef["ramp"][0]['id']

    # Get ECEF coordinates for our test points
    pothole_center_ecef = mock_features_ecef["pothole"][0]['center_ecef']
    ramp_start_ecef = mock_features_ecef["ramp"][0]['start_ecef']
    ramp_end_ecef = mock_features_ecef["ramp"][0]['end_ecef']
    
    # Calculate a point in the middle of the ramp
    ramp_mid_ecef = (np.array(ramp_start_ecef) + np.array(ramp_end_ecef)) / 2

    fail_count = 0

    # 3. Define Test Cases
    print("\n--- Testing Pothole Detection ---")
    # Test 1: Vehicle exactly at the center of the pothole
    result = get_map_context(pothole_center_ecef, mock_features_ecef)
    fail_count += run_test("Center of Pothole", result, pothole_id)
    
    print("\n--- Testing Ramp Detection ---")
    print("Testing ramp start...")
    # Test 3: Vehicle exactly at the start of the ramp
    result = get_map_context(ramp_start_ecef, mock_features_ecef)
    fail_count += run_test("Start of Ramp", result, ramp_id)
    
    print("Testing ramp middle...")
    # Test 4: Vehicle in the middle of the ramp
    result = get_map_context(ramp_mid_ecef.tolist(), mock_features_ecef)
    fail_count += run_test("Middle of Ramp", result, ramp_id)

    print("Testing off ramp...")
    # Test 5: A point slightly to the side of the ramp's center-line, but still on it (width is 4m, so 1m off is on)
    # To do this, we need a vector perpendicular to the ramp's direction. This is tricky in 3D.
    # A simpler way is to test a point that is clearly off the ramp line.
    # Let's create a point 10 meters "above" the ramp start in the Z-axis
    off_ramp_point = [ramp_start_ecef[0], ramp_start_ecef[1], ramp_start_ecef[2] + 10]
    result = get_map_context(off_ramp_point, mock_features_ecef)
    fail_count += run_test("Off the side of Ramp", result, "road")


    # 4. Final Summary
    print("\n--- Test Summary ---")
    if fail_count == 0:
        print("All tests passed successfully!")
    else:
        print(f"{fail_count} test(s) failed.")
    print("--------------------")


if __name__ == "__main__":
    main() 