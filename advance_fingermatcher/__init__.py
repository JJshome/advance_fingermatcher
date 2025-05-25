"""
Advance Fingerprint Matcher

Advanced High-Performance Fingerprint Matching System using Deep Learning and Computer Vision.

Authors: JJshome
License: MIT
Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "JJshome"
__email__ = "contact@jjshome.com"
__license__ = "MIT"

from .core.matcher import FingerprintMatcher
from .core.feature_extractor import FeatureExtractor
from .core.minutiae_detector import MinutiaeDetector
from .processing.preprocessor import ImagePreprocessor
from .processing.enhancer import ImageEnhancer
from .utils.batch_processor import BatchProcessor
from .utils.visualizer import Visualizer
from .api.server import create_app

__all__ = [
    "FingerprintMatcher",
    "FeatureExtractor",
    "MinutiaeDetector",
    "ImagePreprocessor",
    "ImageEnhancer",
    "BatchProcessor",
    "Visualizer",
    "create_app",
]

# Package metadata
__title__ = "advance-fingermatcher"
__description__ = "Advanced High-Performance Fingerprint Matching System"
__url__ = "https://github.com/JJshome/advance_fingermatcher"
__version_info__ = tuple(int(i) for i in __version__.split("."))

# Configuration
DEFAULT_CONFIG = {
    "model_path": "models/",
    "temp_dir": "temp/",
    "log_level": "INFO",
    "gpu_enabled": True,
    "batch_size": 32,
    "match_threshold": 0.85,
    "quality_threshold": 0.7,
}
