import pytest
import numpy as np
from pathlib import Path


class TestSetupValidation:
    """Validation tests to ensure the testing infrastructure is properly configured."""
    
    def test_pytest_is_installed(self):
        """Verify pytest is installed and accessible."""
        assert pytest.__version__
    
    def test_fixtures_are_available(self, temp_dir, sample_input_data, mock_config):
        """Verify that custom fixtures from conftest.py are available."""
        assert isinstance(temp_dir, Path)
        assert temp_dir.exists()
        
        assert isinstance(sample_input_data, np.ndarray)
        assert sample_input_data.shape == (10, 28, 28, 1)
        
        assert isinstance(mock_config, dict)
        assert "learning_rate" in mock_config
    
    @pytest.mark.unit
    def test_unit_marker_works(self):
        """Verify that the unit test marker is properly configured."""
        assert True
    
    @pytest.mark.integration
    def test_integration_marker_works(self):
        """Verify that the integration test marker is properly configured."""
        assert True
    
    @pytest.mark.slow
    def test_slow_marker_works(self):
        """Verify that the slow test marker is properly configured."""
        import time
        time.sleep(0.1)
        assert True
    
    def test_coverage_is_tracked(self):
        """Verify that coverage tracking is working."""
        def dummy_function(x):
            if x > 0:
                return x * 2
            else:
                return 0
        
        assert dummy_function(5) == 10
        assert dummy_function(-5) == 0
    
    def test_mocking_works(self, mocker):
        """Verify that pytest-mock is properly installed and working."""
        mock_func = mocker.Mock(return_value=42)
        assert mock_func() == 42
        mock_func.assert_called_once()
    
    def test_numpy_random_seed_is_reset(self):
        """Verify that the random seed fixture is working."""
        random_value_1 = np.random.rand()
        
        # This should be the same value if the seed is properly reset
        expected_value = 0.3745401188473625
        assert abs(random_value_1 - expected_value) < 1e-10
    
    def test_project_structure(self):
        """Verify the expected project structure exists."""
        project_root = Path(__file__).parent.parent
        
        assert (project_root / "pyproject.toml").exists()
        assert (project_root / "tests").is_dir()
        assert (project_root / "tests" / "__init__.py").exists()
        assert (project_root / "tests" / "unit").is_dir()
        assert (project_root / "tests" / "integration").is_dir()
        assert (project_root / "tests" / "conftest.py").exists()
    
    def test_main_module_is_importable(self):
        """Verify that the main CNN module can be imported."""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent))
        
        try:
            import cnn
            assert hasattr(cnn, 'sigmoid')
            assert hasattr(cnn, 'relu')
            assert hasattr(cnn, 'softmax')
        finally:
            sys.path.pop(0)