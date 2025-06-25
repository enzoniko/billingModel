# -*- coding: utf-8 -*-

import pytest
import sys
import os

# Add the parent directory to the Python path so we can import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

@pytest.fixture(scope="session")
def test_data_dir():
    """Fixture that provides the path to test data directory"""
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

@pytest.fixture(scope="session") 
def project_root():
    """Fixture that provides the path to project root directory"""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__))) 