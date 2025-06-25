# -*- coding: utf-8 -*-

import json
import re
import base64
import time
import sys
import struct
import pandas as pd
import pyproj
from data_processor import (
    parse_controle_txt,
    get_entry_for_index,
    get_entries_from_dirty_sniffer_log,
    find_map_feature,
    ecef_to_wgs84,
    data_model,
    parse_egos_ECEF_position
)

''' Define constants '''
LOG_PATH = "./log.txt"
#clean log
with open(LOG_PATH, "w") as log_file:
    log_file.write("")
DEBUG = True

def debug(*args):
    if DEBUG:
        print(*args)
    
    output = ' '.join(map(str, args))
    output = output + "\n"
    with open(LOG_PATH, "a") as log_file:
        log_file.write(output)

def process_log_and_generate_csv(
    sniffer_log_path,
    controle_file_path,
    map_features_path,
    output_csv_path,
    run_index,
    is_time_step_fix=True,
    ts_step=100000
):
    """
    Process a single log file and generate a CSV with mass, friction, and context columns.
    Only includes sensors that are defined in the current data_model.
    
    Args:
        sniffer_log_path: Path to the sniffer log file
        controle_file_path: Path to controle.txt file
        map_features_path: Path to map_features.json file
        output_csv_path: Path for the output CSV file
        run_index: Index of the current run
        is_time_step_fix: Whether to use fixed time steps
        ts_step: Time step value
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        with open(controle_file_path, 'r') as f:
            controle_content = f.read()
        controle_data = parse_controle_txt(controle_content)
    except FileNotFoundError:
        debug(f"Error: controle.txt not found at {controle_file_path}")
        return False

    try:
        with open(map_features_path, 'r') as f:
            map_features = json.load(f)
    except FileNotFoundError:
        debug(f"Error: map_features.json not found at {map_features_path}")
        return False

    # Get run parameters (mass and friction)
    run_params = get_entry_for_index(controle_data, run_index)
    if not run_params:
        debug(f"No control parameters found for run {run_index}")
        return False

    entries_by_sig = get_entries_from_dirty_sniffer_log(sniffer_log_path, 'data_small/clean_sniffer.log')
    
    if not entries_by_sig:
        debug(f"No entries found in log file {sniffer_log_path}")
        return False

    csv_data = []
    
    # Create device ID to sensor name mapping only for active sensors
    dev_to_name = {v: k for k, v in data_model.items()}
    active_sensor_names = set(data_model.keys())
    active_device_ids = set(data_model.values())
    
    debug(f"Active sensors: {sorted(active_sensor_names)}")
    debug(f"Active device IDs: {sorted(active_device_ids)}")
    
    for signature, sig_entries in entries_by_sig.items():
        debug(f"Processing signature {signature} for run {run_index}")

        if 16 not in sig_entries:
            debug(f"No ego motion vectors (dev=16) found for signature {signature}. Skipping.")
            continue
        
        all_records = []
        for records in sig_entries.values():
            all_records.extend(records)
        all_records.sort(key=lambda r: r.t)

        egos = [r for r in all_records if r.dev == 16]
        parse_egos_ECEF_position(egos, is_time_step_fix, ts_step, signature)

        # Initialize current state only for active sensors
        current_state = {}
        for sensor_name in active_sensor_names:
            current_state[sensor_name] = None

        for record in all_records:
            # Only process records for active device IDs
            if record.dev in active_device_ids and record.dev in dev_to_name:
                dev_name = dev_to_name[record.dev]
                if isinstance(record.value, list) and not record.value:
                    current_state[dev_name] = None
                else:
                    current_state[dev_name] = record.value
            elif record.dev not in [16] and record.dev not in active_device_ids:
                # Log discarded devices (except dev=16 which is special)
                debug(f"Discarding device {record.dev} (not in active data_model)")

            if record.dev == 16:
                row = {
                    'timestamp': record.t,
                    'run_index': run_index,
                    'signature': signature,
                    'mass': run_params['mass'],
                    'friction': run_params['friction']
                }
                
                # Only add active sensor data to the row
                for sensor_name in active_sensor_names:
                    row[sensor_name] = current_state[sensor_name]

                if record.x and record.y and record.z:
                    lat, lon, alt = ecef_to_wgs84.transform(record.x, record.y, record.z)
                    row['lat'] = lat
                    row['lon'] = lon
                    row['alt'] = alt
                    
                    # Find map feature context
                    map_feature = find_map_feature(lat, lon, map_features)
                    row['context'] = map_feature if map_feature else 'road'
                else:
                    row['lat'] = None
                    row['lon'] = None
                    row['alt'] = None
                    row['context'] = 'road'  # Default when no GPS coordinates
                
                vx = current_state.get("SPEED_X", 0) or 0
                vy = current_state.get("SPEED_Y", 0) or 0
                vz = current_state.get("SPEED_Z", 0) or 0
                row['fused_speed'] = (vx**2 + vy**2 + vz**2)**0.5
                
                csv_data.append(row)

        debug(f"Finished processing signature {signature} for run {run_index}")

    if csv_data:
        df = pd.DataFrame(csv_data)
        
        # Define column order - only include active sensors
        core_cols = ['timestamp', 'run_index', 'signature', 'mass', 'friction']
        sensor_cols = sorted(active_sensor_names)  # Only active sensors
        location_cols = ['lat', 'lon', 'alt', 'context', 'fused_speed']
        
        # Ensure all columns exist (but only for active sensors)
        all_cols = core_cols + sensor_cols + location_cols
        for col in all_cols:
            if col not in df.columns:
                df[col] = None
        
        # Order columns and save
        df = df[all_cols]
        df.to_csv(output_csv_path, index=False)
        debug(f"Successfully generated {output_csv_path} with {len(df)} rows and {len(df.columns)} columns")
        debug(f"Included sensor columns: {sensor_cols}")
        return True
    else:
        debug(f"No data collected for run {run_index}")
        return False

def process_multiple_logs(
    max_logs=None,
    start_run=1,
    end_run=220,
    missing_runs=None,
    output_dir="output_csvs",
    controle_file_path='controle.txt',
    map_features_path='map_features.json'
):
    """
    Process multiple log files and generate individual CSV files for each.
    Only includes sensors that are defined in the current data_model.
    
    Args:
        max_logs: Maximum number of logs to process (None for all available)
        start_run: Starting run number (default: 1)
        end_run: Ending run number (default: 220)
        missing_runs: List of run numbers to skip (default: [54, 55])
        output_dir: Directory to save CSV files (default: "output_csvs")
        controle_file_path: Path to controle.txt file
        map_features_path: Path to map_features.json file
    
    Returns:
        tuple: (success_count, total_attempts, output_files)
    """
    import os
    
    if missing_runs is None:
        missing_runs = [54, 55]
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Log which sensors are active
    active_sensors = sorted(data_model.keys())
    debug(f"Processing with active sensors: {active_sensors}")
    debug(f"Total active sensors: {len(active_sensors)}")
    
    success_count = 0
    total_attempts = 0
    output_files = []
    processed_runs = []
    
    for i in range(start_run, end_run + 1):
        if max_logs and success_count >= max_logs:
            debug(f"Reached maximum number of logs to process: {max_logs}")
            break
            
        if i in missing_runs:
            continue
        
        log_file_path = f'data_small/sniffer_run_{i}.log'
        output_csv_path = f'{output_dir}/sniffer_run_{i}.csv'
        total_attempts += 1
        
        # Check if the log file exists before processing
        if not os.path.exists(log_file_path):
            debug(f"Log file not found: {log_file_path}, skipping.")
            continue

        debug(f"Processing run {i}...")
        try:
            success = process_log_and_generate_csv(
                sniffer_log_path=log_file_path,
                controle_file_path=controle_file_path,
                map_features_path=map_features_path,
                output_csv_path=output_csv_path,
                run_index=i
            )
            if success:
                success_count += 1
                processed_runs.append(i)
                output_files.append(output_csv_path)
            else:
                debug(f"Failed to process run {i}")
        except Exception as e:
            debug(f"Error processing run {i}: {e}")
            continue

    debug(f"\nProcessing complete:")
    debug(f"  - Successfully processed: {success_count} files")
    debug(f"  - Total attempts: {total_attempts}")
    debug(f"  - Processed runs: {processed_runs}")
    debug(f"  - Output files: {len(output_files)}")
    debug(f"  - Active sensors used: {len(active_sensors)}")
    
    print(f"Successfully processed {success_count} log files")
    print(f"Generated {len(output_files)} CSV files in '{output_dir}' directory")
    print(f"Using {len(active_sensors)} active sensors from data_model")
    if processed_runs:
        print(f"Processed runs: {processed_runs[:10]}{'...' if len(processed_runs) > 10 else ''}")
    
    return success_count, total_attempts, output_files

# Legacy function for backward compatibility (but not used anymore)
def reset_collected_data():
    """Legacy function for backward compatibility - no longer needed"""
    pass

if __name__ == "__main__":
    # Process all available log files, creating individual CSV files
    process_multiple_logs()

