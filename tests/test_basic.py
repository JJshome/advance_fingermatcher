"""
Basic tests for advance_fingermatcher package
"""

import pytest
import sys
import subprocess
import importlib


def test_package_import():
    """Test that the main package can be imported."""
    import advance_fingermatcher
    assert hasattr(advance_fingermatcher, '__version__')
    assert advance_fingermatcher.__version__ == "1.0.1"


def test_package_info():
    """Test package metadata."""
    import advance_fingermatcher
    info = advance_fingermatcher.get_package_info()
    assert info['name'] == 'advance_fingermatcher'
    assert info['version'] == '1.0.1'
    assert 'author' in info
    assert 'url' in info


def test_dependency_check():
    """Test dependency checking functionality."""
    import advance_fingermatcher
    deps = advance_fingermatcher.check_dependencies()
    
    assert 'missing_core' in deps
    assert 'missing_optional' in deps
    assert 'all_core_available' in deps
    assert isinstance(deps['missing_core'], list)
    assert isinstance(deps['missing_optional'], list)


def test_cli_help():
    """Test CLI help command."""
    result = subprocess.run(
        [sys.executable, '-m', 'advance_fingermatcher.cli', '--help'], 
        capture_output=True, text=True
    )
    assert result.returncode == 0
    assert 'Advanced Fingerprint Matcher CLI' in result.stdout


def test_cli_demo():
    """Test CLI demo command."""
    result = subprocess.run(
        [sys.executable, '-m', 'advance_fingermatcher.cli', 'demo'], 
        capture_output=True, text=True
    )
    assert result.returncode == 0
    assert 'Advanced Fingerprint Matcher Demo' in result.stdout


def test_cli_version():
    """Test CLI version command."""
    result = subprocess.run(
        [sys.executable, '-m', 'advance_fingermatcher.cli', 'version'], 
        capture_output=True, text=True
    )
    assert result.returncode == 0
    assert '1.0.1' in result.stdout


def test_fingermatcher_command():
    """Test fingermatcher console script."""
    result = subprocess.run(
        ['fingermatcher', '--help'], 
        capture_output=True, text=True
    )
    assert result.returncode == 0
    assert 'Advanced Fingerprint Matcher CLI' in result.stdout


def test_fingermatcher_demo_command():
    """Test fingermatcher demo console script."""
    result = subprocess.run(
        ['fingermatcher', 'demo'], 
        capture_output=True, text=True
    )
    assert result.returncode == 0
    assert 'Advanced Fingerprint Matcher Demo' in result.stdout


def test_safe_imports():
    """Test that imports handle missing dependencies gracefully."""
    import advance_fingermatcher
    
    # These should not raise exceptions even if dependencies are missing
    try:
        advance_fingermatcher.print_system_info()
    except Exception as e:
        pytest.fail(f"print_system_info() raised {e}")


def test_logging_setup():
    """Test that logging is properly configured."""
    import logging
    import advance_fingermatcher
    
    logger = logging.getLogger('advance_fingermatcher')
    assert logger.level == logging.INFO
    assert len(logger.handlers) > 0


if __name__ == '__main__':
    # Run tests directly
    pytest.main([__file__, '-v'])
