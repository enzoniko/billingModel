# IoT Log Sender Tests

This directory contains comprehensive tests for the `iot_log_sender.py` module.

## Overview

The tests validate the functionality of processing IoT sensor log files and generating **individual CSV files** for each processed log file. The module has been adapted to process a configurable number of log files and create separate CSV files with enhanced data including mass, friction, and context information.

## Key Features Tested

### 1. Single Log File Processing
- **Test**: `test_process_single_log_file`
- **Purpose**: Validates that exactly one log file can be processed
- **Validation**: Checks CSV structure, data types, content integrity, and new columns

### 2. Multiple Log Files with Limits
- **Test**: `test_process_multiple_logs_with_limit`
- **Purpose**: Tests processing multiple files with a maximum limit
- **Validation**: Ensures the limit is respected and individual CSV outputs are correct

### 3. Enhanced Data Validation
- **Test**: Various validation checks in `_validate_csv_structure`
- **Purpose**: Validates mass, friction, and context columns
- **Validation**: Ensures data integrity and correct map feature detection

### 4. Error Handling
- **Test**: `test_missing_files_handling`
- **Purpose**: Tests graceful handling of missing files
- **Validation**: Ensures no crashes when files don't exist

## Modified `iot_log_sender.py` Features

### New Function: `process_log_and_generate_csv()`
```python
process_log_and_generate_csv(
    sniffer_log_path,       # Path to sniffer log file
    controle_file_path,     # Path to controle.txt
    map_features_path,      # Path to map_features.json  
    output_csv_path,        # Output CSV file path
    run_index,              # Run index number
    is_time_step_fix=True,  # Fixed time steps
    ts_step=100000          # Time step value
)
```

### Enhanced Function: `process_multiple_logs()`
```python
process_multiple_logs(
    max_logs=None,                    # Limit number of logs to process
    start_run=1,                      # Starting run number
    end_run=220,                      # Ending run number
    missing_runs=None,                # List of runs to skip
    output_dir="output_csvs",         # Output directory for CSV files
    controle_file_path='controle.txt', # Path to control file
    map_features_path='map_features.json' # Path to map features
)
```

### Key Changes from Previous Version:
- **Individual CSV Files**: Creates one CSV per log file instead of one combined file
- **Mass Column**: Extracted from `controle.txt` based on run number (8300, 10900, or 13500)
- **Friction Column**: Extracted from `controle.txt` based on run number (0.5, 0.75, or 1.0)
- **Context Column**: Identifies map features ("pothole_large", "ramp_asc", "speedbump_raised", etc.) or defaults to "road"
- **Enhanced Location Data**: Improved GPS coordinate processing and map feature detection

## Running the Tests

### Run All Tests
```bash
python -m pytest tests/test_iot_log_sender.py -v
```

### Run Only Integration Tests
```bash
python -m pytest tests/test_iot_log_sender.py -m integration -v
```

### Run Only Unit Tests
```bash
python -m pytest tests/test_iot_log_sender.py -m unit -v
```

### Run Single Log Test (Standalone)
```bash
python tests/run_single_log_test.py
```

## Test Output Example

When running the single log test, you should see output similar to:
```
=== Single Log File Processing Test ===
1. Testing single log file processing...
   - Processed 1 files successfully
   - Attempted 1 files
   - Generated 1 CSV files
2. Validating CSV output...
   - CSV file: sniffer_run_1.csv
   - CSV contains 127 rows and 56 columns
   - Mass values: [8300]
   - Friction values: [1.0]
   - Context values: ['road']
3. Sample data:
          timestamp  run_index  mass  friction context       lat       lon    alt
0  1750721688185866          1  8300       1.0    road  0.000026  0.000055  0.394
✅ TEST PASSED: Single log file processed successfully and CSV is well-formed!
```

## CSV Output Structure

Each generated CSV includes:
- **Core fields**: `timestamp`, `run_index`, `signature`, `mass`, `friction`
- **Sensor data**: All fields from the `data_model` mapping (IMU, GPS, speed, etc.)
- **Location fields**: `lat`, `lon`, `alt`, `context`, `fused_speed`

### New Enhanced Columns:
- **`mass`**: Vehicle mass from controle.txt (8300, 10900, or 13500 kg)
- **`friction`**: Road friction coefficient from controle.txt (0.5, 0.75, or 1.0)
- **`context`**: Current road context based on GPS coordinates:
  - `"road"` - Default when no map feature detected
  - `"pothole_small"`, `"pothole_medium"`, `"pothole_large"` - Pothole features
  - `"ramp_asc"`, `"ramp_desc"` - Ramp features  
  - `"speedbump_raised"`, `"speedbump_sunken"` - Speed bump features
  - `"elevated_crosswalk"` - Crosswalk features
  - `"cut"` - Road cut features

## Output Files

The script now generates:
```
output_csvs/
├── sniffer_run_1.csv     # Data from log file 1
├── sniffer_run_2.csv     # Data from log file 2
├── sniffer_run_3.csv     # Data from log file 3
└── ...                   # One CSV per processed log file
```

Instead of a single combined `data.csv` file.

## Test Categories

- **Unit Tests** (`@pytest.mark.unit`): Fast tests that don't require external files
- **Integration Tests** (`@pytest.mark.integration`): Tests that require actual log files and dependencies

## Dependencies

The tests require:
- `pytest`
- `pandas`
- `tempfile` (for temporary test outputs)
- All dependencies from the main `iot_log_sender.py` module
- Access to `controle.txt` and `map_features.json` files

## File Structure

```
tests/
├── conftest.py                    # Pytest configuration and fixtures
├── test_iot_log_sender.py         # Main test suite
├── run_single_log_test.py         # Standalone single log test
└── README.md                      # This documentation
```

## Data Sources

The enhanced CSV files pull data from:
- **Log files**: `data_small/sniffer_run_*.log` - Sensor and vehicle telemetry
- **Control file**: `controle.txt` - Mass and friction parameters per run
- **Map features**: `map_features.json` - Road features for context detection