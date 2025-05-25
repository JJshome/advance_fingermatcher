"""
Advanced Fingerprint Matcher - Main Package
==========================================

This package provides state-of-the-art fingerprint matching capabilities including:
- Deep learning-based feature extraction
- Graph neural network matching
- Ultra-fast 1:N search with learned indexing
- Advanced quality assessment
- Multi-modal fusion techniques
"""

__version__ = "1.0.1"
__author__ = "JJshome"
__email__ = "advance@fingermatcher.com"

# Safe imports with error handling
try:
    from .advanced_matcher import (
        AdvancedFingerprintMatcher,
        MatchingResult,
        SearchResultAdvanced,
        create_advanced_matcher
    )
    _HAS_ADVANCED_MATCHER = True
except ImportError:
    _HAS_ADVANCED_MATCHER = False

try:
    from .algorithms.enhanced_bozorth3 import (
        EnhancedBozorth3Matcher,
        EnhancedMinutia,
        MinutiaPair
    )
    _HAS_BOZORTH3 = True
except ImportError:
    _HAS_BOZORTH3 = False

# Core exports - only include what's actually available
__all__ = ['__version__', '__author__', '__email__']

if _HAS_ADVANCED_MATCHER:
    __all__.extend([
        'AdvancedFingerprintMatcher',
        'MatchingResult', 
        'SearchResultAdvanced',
        'create_advanced_matcher'
    ])

if _HAS_BOZORTH3:
    __all__.extend([
        'EnhancedBozorth3Matcher',
        'EnhancedMinutia',
        'MinutiaPair'
    ])

# Package metadata
PACKAGE_INFO = {
    'name': 'advance_fingermatcher',
    'version': __version__,
    'description': 'Advanced fingerprint matching with deep learning and graph neural networks',
    'author': __author__,
    'email': __email__,
    'url': 'https://github.com/JJshome/advance_fingermatcher',
    'license': 'MIT',
    'keywords': [
        'fingerprint', 'biometrics', 'matching', 'deep-learning', 
        'graph-neural-networks', 'computer-vision', 'pattern-recognition'
    ]
}


def get_version():
    """Get package version"""
    return __version__


def get_package_info():
    """Get complete package information"""
    return PACKAGE_INFO.copy()


def check_dependencies():
    """Check if all required dependencies are installed"""
    import importlib
    
    # Core dependencies that should be available
    core_packages = [
        ('numpy', 'numpy'),
        ('click', 'click'),
        ('Pillow', 'PIL')
    ]
    
    # Optional advanced dependencies  
    optional_packages = [
        ('torch', 'torch'),
        ('opencv-python', 'cv2'),
        ('scikit-image', 'skimage'),
        ('scipy', 'scipy'),
        ('scikit-learn', 'sklearn'),
        ('faiss-cpu', 'faiss'),
        ('matplotlib', 'matplotlib')
    ]
    
    missing_core = []
    missing_optional = []
    
    for package_name, import_name in core_packages:
        try:
            importlib.import_module(import_name)
        except ImportError:
            missing_core.append(package_name)
    
    for package_name, import_name in optional_packages:
        try:
            importlib.import_module(import_name)
        except ImportError:
            missing_optional.append(package_name)
    
    return {
        'missing_core': missing_core,
        'missing_optional': missing_optional,
        'all_core_available': len(missing_core) == 0,
        'advanced_features_available': len(missing_optional) == 0
    }


def print_system_info():
    """Print system information and dependencies status"""
    import sys
    import platform
    
    print("=" * 60)
    print("Advanced Fingerprint Matcher - System Information")
    print("=" * 60)
    print(f"Package Version: {__version__}")
    print(f"Python Version: {sys.version}")
    print(f"Platform: {platform.platform()}")
    
    # Check PyTorch if available
    try:
        import torch
        print(f"PyTorch Version: {torch.__version__}")
        print(f"CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA Version: {torch.version.cuda}")
            print(f"GPU Count: {torch.cuda.device_count()}")
    except ImportError:
        print("PyTorch: Not installed")
    
    print("\nDependency Status:")
    deps = check_dependencies()
    
    if deps['all_core_available']:
        print("✓ All core dependencies are installed")
    else:
        print("✗ Missing core dependencies:")
        for dep in deps['missing_core']:
            print(f"  - {dep}")
    
    if deps['advanced_features_available']:
        print("✓ All advanced features available")
    else:
        print("⚠ Some advanced features unavailable (missing dependencies):")
        for dep in deps['missing_optional']:
            print(f"  - {dep}")
    
    print("=" * 60)


# Initialize logging safely
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Add handler if none exists
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# Startup message
logger.info(f"Advanced Fingerprint Matcher v{__version__} initialized")

# Check critical dependencies
deps_status = check_dependencies()
if not deps_status['all_core_available']:
    logger.warning(
        f"Missing core dependencies: {deps_status['missing_core']}. "
        "Some features may not work properly."
    )
elif not deps_status['advanced_features_available']:
    logger.info(
        "Core functionality available. Install optional dependencies for advanced features."
    )
