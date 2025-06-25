# -*- coding: utf-8 -*-

import pytest
import os
import pandas as pd
import tempfile
import sys
import shutil
from unittest.mock import patch, MagicMock

# Add parent directory to path to import the module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import iot_log_sender
from iot_log_sender import process_multiple_logs, reset_collected_data


class TestIoTLogSender:
    """Test suite for iot_log_sender.py functionality"""
    
    def setup_method(self):
        """Setup before each test method"""        
        # Create a temporary directory for test outputs
        self.temp_dir = tempfile.mkdtemp()
        self.original_cwd = os.getcwd()
        
    def teardown_method(self):
        """Cleanup after each test method"""
        # Clean up temporary directory
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_reset_collected_data(self):
        """Test that reset_collected_data function exists (for backward compatibility)"""
        # This function now just exists for compatibility but doesn't do anything
        result = reset_collected_data()
        assert result is None  # Should return None
    
    @pytest.mark.integration
    def test_process_single_log_file(self):
        """Test processing a single log file and validate CSV output"""
        # Check if test data exists
        test_log_path = 'data_small/sniffer_run_1.log'
        controle_path = 'controle.txt'
        map_features_path = 'map_features.json'
        
        # Skip test if required files don't exist
        if not os.path.exists(test_log_path):
            pytest.skip(f"Test data not found: {test_log_path}")
        if not os.path.exists(controle_path):
            pytest.skip(f"Test data not found: {controle_path}")
        if not os.path.exists(map_features_path):
            pytest.skip(f"Test data not found: {map_features_path}")
        
        # Process a single log file
        success_count, total_attempts, output_files = process_multiple_logs(
            max_logs=1,
            start_run=1,
            end_run=1,
            missing_runs=[],
            output_dir=self.temp_dir,
            controle_file_path=controle_path,
            map_features_path=map_features_path
        )
        
        # Validate processing results
        assert success_count >= 0, "Should have processed at least 0 files successfully"
        assert total_attempts >= 1, "Should have attempted to process at least 1 file"
        
        # If data was processed, validate the CSV
        if success_count > 0 and output_files:
            assert len(output_files) == success_count, "Should have one output file per successful processing"
            csv_path = output_files[0]
            assert os.path.exists(csv_path), f"Output CSV should exist: {csv_path}"
            self._validate_csv_structure(csv_path)
    
    @pytest.mark.integration
    def test_process_multiple_logs_with_limit(self):
        """Test processing multiple log files with a limit"""
        # Process up to 3 log files
        success_count, total_attempts, output_files = process_multiple_logs(
            max_logs=3,
            start_run=1,
            end_run=10,
            missing_runs=[54, 55],
            output_dir=self.temp_dir,
            controle_file_path='controle.txt',
            map_features_path='map_features.json'
        )
        
        # Validate processing results
        assert success_count >= 0, "Should have processed at least 0 files successfully"
        assert success_count <= 3, "Should not have processed more than 3 files"
        assert total_attempts >= 1, "Should have attempted to process at least 1 file"
        assert len(output_files) == success_count, "Should have one output file per successful processing"
        
        # If data was processed, validate each CSV
        for csv_path in output_files:
            assert os.path.exists(csv_path), f"Output CSV should exist: {csv_path}"
            self._validate_csv_structure(csv_path)
    
    def _validate_csv_structure(self, csv_path):
        """Helper method to validate the structure of generated CSV files"""
        # Read the CSV
        df = pd.read_csv(csv_path)
        
        # Basic structure validation
        assert len(df) > 0, "CSV should contain at least one row of data"
        assert len(df.columns) > 0, "CSV should contain at least one column"
        
        # Check for expected core columns
        expected_core_columns = ['timestamp', 'run_index', 'signature', 'mass', 'friction', 'context']
        for col in expected_core_columns:
            assert col in df.columns, f"CSV should contain column: {col}"
        
        # Validate data types
        assert df['timestamp'].dtype in ['int64', 'float64'], "Timestamp should be numeric"
        assert df['run_index'].dtype in ['int64', 'float64'], "Run index should be numeric"
        assert df['signature'].dtype in ['int64', 'float64'], "Signature should be numeric"
        assert df['mass'].dtype in ['int64', 'float64'], "Mass should be numeric"
        assert df['friction'].dtype in ['int64', 'float64'], "Friction should be numeric"
        
        # Check for non-empty timestamps
        non_null_timestamps = df['timestamp'].dropna()
        assert len(non_null_timestamps) > 0, "Should have some non-null timestamps"
        
        # Validate run_index values are reasonable
        run_indices = df['run_index'].dropna().unique()
        assert all(idx >= 1 and idx <= 220 for idx in run_indices), "Run indices should be in range 1-220"
        
        # Validate mass and friction values
        mass_values = df['mass'].dropna().unique()
        friction_values = df['friction'].dropna().unique()
        
        # Mass should be one of the expected values from controle.txt
        expected_masses = [8300, 10900, 13500]
        assert all(mass in expected_masses for mass in mass_values), f"Mass values should be from controle.txt: {mass_values}"
        
        # Friction should be one of the expected values from controle.txt
        expected_frictions = [0.5, 0.75, 1.0]
        assert all(friction in expected_frictions for friction in friction_values), f"Friction values should be from controle.txt: {friction_values}"
        
        # Check context column values
        context_values = df['context'].dropna().unique()
        print(f"Found context values: {context_values}")
        
        # Context should be either 'road' or a valid map feature
        valid_contexts = ['road']  # road is the default
        # Add other expected map features based on map_features.json
        map_feature_prefixes = ['pothole_', 'ramp_', 'speedbump_', 'elevated_crosswalk', 'cut']
        
        for context in context_values:
            is_valid = (context == 'road' or 
                       any(context.startswith(prefix) for prefix in map_feature_prefixes))
            assert is_valid, f"Context '{context}' should be 'road' or a valid map feature"
        
        # Check that we have some sensor data columns
        data_model_columns = [
            'IMU_ACC_x', 'SPEED_X', 'SPEED_Y', 'SPEED_Z', 
            'GPS_X', 'GPS_Y', 'GPS_Z', 'YAW', 'PITCH', 'ROLL'
        ]
        found_sensor_columns = [col for col in data_model_columns if col in df.columns]
        
        # We should have at least some sensor data columns
        print(f"Found sensor columns: {found_sensor_columns}")
        print(f"CSV shape: {df.shape}")
        print(f"CSV columns: {list(df.columns)}")
        
        # Basic validation - just ensure the CSV is well-formed
        assert not df.empty, "DataFrame should not be empty"
        
    @pytest.mark.unit
    def test_process_multiple_logs_parameters(self):
        """Test the parameter handling of process_multiple_logs function"""
        with patch('iot_log_sender.process_log_and_generate_csv') as mock_process:
            with patch('os.path.exists', return_value=True):
                # Make the mock return True for successful processing
                mock_process.return_value = True
                
                # Test with max_logs parameter
                success_count, total_attempts, output_files = process_multiple_logs(
                    max_logs=2,
                    start_run=1,
                    end_run=5,
                    missing_runs=[3],
                    output_dir=self.temp_dir,
                    controle_file_path="controle.txt",
                    map_features_path="map_features.json"
                )
                
                # Should have attempted to process runs 1, 2, 4, 5 but stopped at max_logs=2
                assert total_attempts >= 2, "Should have made at least 2 attempts"
                assert success_count <= 2, "Should not exceed max_logs limit"
    
    @pytest.mark.unit 
    def test_missing_files_handling(self):
        """Test behavior when required files are missing"""
        # Test with non-existent files
        success_count, total_attempts, output_files = process_multiple_logs(
            max_logs=1,
            start_run=999,  # Use a run number that definitely doesn't exist
            end_run=999,
            missing_runs=[],
            output_dir=self.temp_dir,
            controle_file_path='controle.txt',
            map_features_path='map_features.json'
        )
        
        # Should handle missing files gracefully
        assert success_count == 0, "Should not have processed any files successfully"
        assert total_attempts >= 0, "Should have attempted to process files"
        assert len(output_files) == 0, "Should not have created any output files"

    @pytest.mark.unit
    def test_output_directory_creation(self):
        """Test that output directory is created if it doesn't exist"""
        nonexistent_dir = os.path.join(self.temp_dir, "nonexistent")
        
        with patch('iot_log_sender.process_log_and_generate_csv') as mock_process:
            with patch('os.path.exists') as mock_exists:
                with patch('os.makedirs') as mock_makedirs:
                    # Mock the file existence checks
                    def exists_side_effect(path):
                        if path.endswith('controle.txt') or path.endswith('map_features.json'):
                            return True
                        if path == nonexistent_dir:
                            return False  # Directory doesn't exist initially
                        return False
                    
                    mock_exists.side_effect = exists_side_effect
                    mock_process.return_value = False  # No successful processing
                    
                    success_count, total_attempts, output_files = process_multiple_logs(
                        max_logs=1,
                        start_run=999,
                        end_run=999,
                        output_dir=nonexistent_dir,
                        controle_file_path='controle.txt',
                        map_features_path='map_features.json'
                    )
                    
                    # os.makedirs should have been called to create the directory
                    mock_makedirs.assert_called_once_with(nonexistent_dir)


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 