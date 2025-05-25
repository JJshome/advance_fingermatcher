#!/usr/bin/env python3
"""
Setup script for advance_fingermatcher package.
"""

from setuptools import setup, find_packages
import os

# Read README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="advance_fingermatcher",
    version="1.0.2",
    author="JJshome",
    author_email="advance@fingermatcher.com",
    description="Advanced fingerprint matching library with Enhanced Bozorth3 algorithm",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/JJshome/advance_fingermatcher",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Topic :: Security",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "mypy>=0.900",
            "sphinx>=4.0",
            "sphinx-rtd-theme>=1.0",
        ],
        "viz": [
            "plotly>=5.0",
        ],
        "ml": [
            "tensorflow>=2.8.0",
            "scikit-learn>=1.0",
        ],
        "gpu": [
            "faiss-gpu>=1.7.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "fingermatcher=advance_fingermatcher.cli:main",
            "fingermatcher-demo=advance_fingermatcher.cli:demo",
            "enhanced-bozorth3-demo=examples.enhanced_bozorth3_demo:main",
            "comprehensive-demo=examples.comprehensive_demo:main",
        ],
    },
    include_package_data=True,
    package_data={
        "advance_fingermatcher": [
            "data/*.json",
            "models/*.h5",
            "configs/*.yaml",
        ],
    },
    keywords=[
        "fingerprint",
        "biometrics",
        "matching",
        "bozorth3",
        "enhanced-bozorth3",
        "minutiae",
        "recognition",
        "computer-vision",
        "security",
        "authentication",
        "quality-weighted",
        "adaptive-tolerance",
        "multi-stage-matching",
    ],
    project_urls={
        "Bug Reports": "https://github.com/JJshome/advance_fingermatcher/issues",
        "Source": "https://github.com/JJshome/advance_fingermatcher",
        "Documentation": "https://github.com/JJshome/advance_fingermatcher/blob/main/docs/enhanced_bozorth3_algorithm.md",
        "Enhanced Bozorth3": "https://github.com/JJshome/advance_fingermatcher/blob/main/docs/enhanced_bozorth3_algorithm.md",
    },
)
