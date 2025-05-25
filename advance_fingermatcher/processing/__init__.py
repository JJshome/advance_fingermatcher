"""
Image processing modules for fingerprint enhancement and preprocessing.

This package contains:
- ImagePreprocessor: Basic image preprocessing operations
- ImageEnhancer: Advanced ridge enhancement techniques
"""

try:
    from .preprocessor import ImagePreprocessor
    from .enhancer import ImageEnhancer
except ImportError:
    # Handle import errors gracefully
    import logging
    logger = logging.getLogger(__name__)
    logger.warning("Some processing modules could not be imported")
    
    class ImagePreprocessor:
        def __init__(self):
            pass
    
    class ImageEnhancer:
        def __init__(self):
            pass

__all__ = ["ImagePreprocessor", "ImageEnhancer"]
