import pytest
import tempfile
import shutil
import numpy as np
from pathlib import Path


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    shutil.rmtree(temp_path)


@pytest.fixture
def sample_input_data():
    """Create sample input data for CNN testing."""
    return np.random.rand(10, 28, 28, 1).astype(np.float32)


@pytest.fixture
def sample_output_data():
    """Create sample output labels for CNN testing."""
    return np.eye(10)[np.random.randint(0, 10, 10)]


@pytest.fixture
def mock_config():
    """Provide a mock configuration dictionary."""
    return {
        "learning_rate": 0.01,
        "batch_size": 32,
        "epochs": 10,
        "activation": "relu",
    }


@pytest.fixture
def sample_conv_layer():
    """Create a sample convolutional layer configuration."""
    return {
        "num_filters": 32,
        "filter_size": 3,
        "stride": 1,
        "padding": 1,
    }


@pytest.fixture
def sample_pool_layer():
    """Create a sample pooling layer configuration."""
    return {
        "pool_size": 2,
        "stride": 2,
        "mode": "max",
    }


@pytest.fixture(autouse=True)
def reset_random_seed():
    """Reset random seed before each test for reproducibility."""
    np.random.seed(42)


@pytest.fixture
def capture_stdout(monkeypatch):
    """Capture stdout for testing print statements."""
    import io
    import sys
    
    captured_output = io.StringIO()
    monkeypatch.setattr(sys, 'stdout', captured_output)
    
    yield captured_output
    
    captured_output.close()


@pytest.fixture
def mock_file_operations(mocker):
    """Mock file I/O operations."""
    mock_open = mocker.mock_open()
    mocker.patch('builtins.open', mock_open)
    return mock_open


@pytest.fixture(scope="session")
def test_data_dir():
    """Path to test data directory."""
    return Path(__file__).parent / "test_data"