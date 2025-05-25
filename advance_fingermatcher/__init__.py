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

__version__ = "2.0.0"
__author__ = "Advanced Fingerprint Research Team"
__email__ = "research@fingermatcher.ai"

# Main imports for easy access
from .advanced_matcher import (
    AdvancedFingerprintMatcher,
    MatchingResult,
    SearchResultAdvanced,
    create_advanced_matcher
)

from .deep_learning.networks import (
    MinutiaNet,
    DescriptorNet,
    QualityNet,
    FusionNet,
    create_advanced_networks
)

from .deep_learning.graph_matching import (
    GraphMatchNet,
    AdvancedGraphMatcher,
    create_graph_matcher
)

from .search.ultra_fast_search import (
    UltraFastSearch,
    SearchConfig,
    SearchResult,
    create_ultra_fast_search,
    create_distributed_search
)

# Legacy imports for backward compatibility
from .algorithms.enhanced_bozorth3 import (
    EnhancedBozorth3Matcher,
    EnhancedMinutia,
    MinutiaPair
)

__all__ = [
    # Main interfaces
    'AdvancedFingerprintMatcher',
    'MatchingResult',
    'SearchResultAdvanced',
    'create_advanced_matcher',
    
    # Deep learning networks
    'MinutiaNet',
    'DescriptorNet', 
    'QualityNet',
    'FusionNet',
    'create_advanced_networks',
    
    # Graph matching
    'GraphMatchNet',
    'AdvancedGraphMatcher',
    'create_graph_matcher',
    
    # Search systems
    'UltraFastSearch',
    'SearchConfig',
    'SearchResult',
    'create_ultra_fast_search',
    'create_distributed_search',
    
    # Legacy components
    'EnhancedBozorth3Matcher',
    'EnhancedMinutia',
    'MinutiaPair',
]

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
    ],
    'classifiers': [
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Image Recognition',
        'Topic :: Software Development :: Libraries :: Python Modules',
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
    
    required_packages = [
        'torch',
        'torchvision', 
        'torch_geometric',
        'numpy',
        'opencv-python',
        'scikit-image',
        'scipy',
        'scikit-learn',
        'faiss',
        'matplotlib'
    ]
    
    optional_packages = [
        'redis',
        'plotly',
        'numba',
        'ray'
    ]
    
    missing_required = []
    missing_optional = []
    
    for package in required_packages:
        try:
            # Handle package names that differ from import names
            import_name = package
            if package == 'opencv-python':
                import_name = 'cv2'
            elif package == 'scikit-image':
                import_name = 'skimage'
            elif package == 'scikit-learn':
                import_name = 'sklearn'
            elif package == 'torch_geometric':
                import_name = 'torch_geometric'
                
            importlib.import_module(import_name)
        except ImportError:
            missing_required.append(package)
    
    for package in optional_packages:
        try:
            importlib.import_module(package)
        except ImportError:
            missing_optional.append(package)
    
    return {
        'missing_required': missing_required,
        'missing_optional': missing_optional,
        'all_required_available': len(missing_required) == 0
    }

def print_system_info():
    """Print system information and dependencies status"""
    import sys
    import platform
    import torch
    
    print("=" * 60)
    print("Advanced Fingerprint Matcher - System Information")
    print("=" * 60)
    print(f"Package Version: {__version__}")
    print(f"Python Version: {sys.version}")
    print(f"Platform: {platform.platform()}")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"GPU Count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    
    print("\nDependency Status:")
    deps = check_dependencies()
    
    if deps['all_required_available']:
        print("✓ All required dependencies are installed")
    else:
        print("✗ Missing required dependencies:")
        for dep in deps['missing_required']:
            print(f"  - {dep}")
    
    if deps['missing_optional']:
        print("\nMissing optional dependencies:")
        for dep in deps['missing_optional']:
            print(f"  - {dep} (optional)")
    
    print("=" * 60)

# Initialize logging
import logging

# Create package logger
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

# Welcome message
logger.info(f"Advanced Fingerprint Matcher v{__version__} initialized")

# Check dependencies on import
deps_status = check_dependencies()
if not deps_status['all_required_available']:
    logger.warning(
        f"Missing required dependencies: {deps_status['missing_required']}. "
        "Some features may not work properly."
    )
