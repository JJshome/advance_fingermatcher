"""
Utility modules for fingerprint matching system.

This package contains:
- BatchProcessor: Batch processing of multiple fingerprints
- Visualizer: Visualization tools for results
- Logger: Logging utilities
"""

try:
    from .batch_processor import BatchProcessor
    from .visualizer import Visualizer
    from .logger import get_logger
except ImportError:
    # Handle import errors gracefully
    import logging
    logger = logging.getLogger(__name__)
    logger.warning("Some utility modules could not be imported")
    
    class BatchProcessor:
        def __init__(self):
            pass
    
    class Visualizer:
        def __init__(self):
            pass
    
    def get_logger(name):
        return logging.getLogger(name)

__all__ = ["BatchProcessor", "Visualizer", "get_logger"]
