"""
API modules for fingerprint matching system.

This package contains:
- FastAPI server implementation
- REST API endpoints
- Request/response models
"""

try:
    from .server import create_app
except ImportError:
    # Handle import errors gracefully
    import logging
    logger = logging.getLogger(__name__)
    logger.warning("API modules could not be imported - FastAPI may not be installed")
    
    def create_app():
        raise ImportError("FastAPI not available")

__all__ = ["create_app"]
