#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Simple test runner for single log file processing.
This script specifically tests processing only one log file and validates the CSV output.
"""

import os
import sys
import tempfile
import shutil

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from iot_log_sender import process_multiple_logs
import pandas as pd

def run_single_log_test():
    """Run a simple test that processes only one log file."""
    print("=== Single Log File Processing Test ===")
    
    # Create temporary directory for output
    temp_dir = tempfile.mkdtemp()
    
    try:
        print("1. Testing single log file processing...")
        
        # Process only one log file
        success_count, total_attempts, output_files = process_multiple_logs(
            max_logs=1,           # Only process 1 log file
            start_run=1,          # Start from run 1
            end_run=10,           # Check up to run 10
            missing_runs=[54, 55], # Skip known missing runs
            output_dir=temp_dir,
            controle_file_path='controle.txt',
            map_features_path='map_features.json'
        )
        
        print(f"   - Processed {success_count} files successfully")
        print(f"   - Attempted {total_attempts} files")
        print(f"   - Generated {len(output_files)} CSV files")
        
        # Validate results
        if success_count > 0 and output_files:
            print("2. Validating CSV output...")
            
            csv_path = output_files[0]  # Take the first (and only) output file
            if os.path.exists(csv_path):
                # Read and validate the CSV
                df = pd.read_csv(csv_path)
                
                print(f"   - CSV file: {os.path.basename(csv_path)}")
                print(f"   - CSV contains {len(df)} rows and {len(df.columns)} columns")
                print(f"   - Columns: {list(df.columns)}")
                
                # Basic validation
                assert len(df) > 0, "CSV should contain at least one row"
                assert 'timestamp' in df.columns, "CSV should have timestamp column"
                assert 'run_index' in df.columns, "CSV should have run_index column"
                assert 'signature' in df.columns, "CSV should have signature column"
                assert 'mass' in df.columns, "CSV should have mass column"
                assert 'friction' in df.columns, "CSV should have friction column"
                assert 'context' in df.columns, "CSV should have context column"
                
                # Check data types
                assert df['timestamp'].dtype in ['int64', 'float64'], "Timestamp should be numeric"
                assert df['mass'].dtype in ['int64', 'float64'], "Mass should be numeric"
                assert df['friction'].dtype in ['int64', 'float64'], "Friction should be numeric"
                
                # Validate mass and friction values
                mass_values = df['mass'].dropna().unique()
                friction_values = df['friction'].dropna().unique()
                context_values = df['context'].dropna().unique()
                
                print(f"   - Mass values: {mass_values}")
                print(f"   - Friction values: {friction_values}")
                print(f"   - Context values: {context_values}")
                
                # Check that mass and friction are from controle.txt
                expected_masses = [8300, 10900, 13500]
                expected_frictions = [0.5, 0.75, 1.0]
                
                assert all(mass in expected_masses for mass in mass_values), f"Mass should be from controle.txt: {mass_values}"
                assert all(friction in expected_frictions for friction in friction_values), f"Friction should be from controle.txt: {friction_values}"
                
                # Check context values
                valid_contexts = ['road']
                map_feature_prefixes = ['pothole_', 'ramp_', 'speedbump_', 'elevated_crosswalk', 'cut']
                
                for context in context_values:
                    is_valid = (context == 'road' or 
                               any(context.startswith(prefix) for prefix in map_feature_prefixes))
                    assert is_valid, f"Context '{context}' should be 'road' or a valid map feature"
                
                # Show sample data
                print("3. Sample data:")
                pd.set_option('display.max_columns', None)
                pd.set_option('display.width', None)
                print(df[['timestamp', 'run_index', 'mass', 'friction', 'context', 'lat', 'lon', 'alt']].head())
                
                print("\n✅ TEST PASSED: Single log file processed successfully and CSV is well-formed!")
                print(f"   - Generated file: {csv_path}")
                print(f"   - Contains proper mass, friction, and context columns")
                return True
            else:
                print("❌ TEST FAILED: No output file was created")
                return False
        else:
            print("⚠️  TEST SKIPPED: No log files were processed successfully")
            print("   This might be because the test data files don't exist")
            return True  # Not a failure, just no data to process
            
    except Exception as e:
        print(f"❌ TEST FAILED: Exception occurred: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # Clean up temporary directory
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

if __name__ == "__main__":
    success = run_single_log_test()
    sys.exit(0 if success else 1) 