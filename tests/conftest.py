"""Pytest configuration file."""

import os
import sys
import pytest

# Add the project root directory to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)


@pytest.fixture(autouse=True)
def setup_test_env():
    """Setup test environment."""
    # Add any environment setup needed for tests
    pass
