#!/usr/bin/env python3
"""
Tests for the FingerprintMatcher class.
"""

import pytest
import numpy as np
import cv2
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from advance_fingermatcher.core.matcher import FingerprintMatcher, MatchResult


class TestFingerprintMatcher:
    """Test cases for FingerprintMatcher."""
    
    @pytest.fixture
    def matcher(self):
        """Create a FingerprintMatcher instance for testing."""
        return FingerprintMatcher()
    
    @pytest.fixture
    def sample_image(self):
        """Create a sample fingerprint image for testing."""
        # Create a simple synthetic fingerprint pattern
        size = (256, 256)
        image = np.zeros(size, dtype=np.uint8)
        
        # Add vertical ridges
        for i in range(0, size[1], 10):
            image[:, i:i+5] = 255
        
        # Add some noise
        noise = np.random.normal(0, 5, size).astype(np.uint8)
        image = cv2.add(image, noise)
        
        return image
    
    @pytest.fixture
    def different_image(self):
        """Create a different fingerprint image for testing."""
        # Create a simple synthetic fingerprint pattern
        size = (256, 256)
        image = np.zeros(size, dtype=np.uint8)
        
        # Add horizontal ridges
        for i in range(0, size[0], 10):
            image[i:i+5, :] = 255
        
        # Add some noise
        noise = np.random.normal(0, 5, size).astype(np.uint8)
        image = cv2.add(image, noise)
        
        return image
    
    def test_matcher_initialization(self, matcher):
        """Test that matcher initializes correctly."""
        assert matcher is not None
        assert hasattr(matcher, 'match_threshold')
        assert hasattr(matcher, 'quality_threshold')
        assert matcher.match_threshold == 0.85
    
    def test_preprocess_image(self, matcher, sample_image):
        """Test image preprocessing."""
        processed = matcher.preprocess_image(sample_image)
        
        assert processed is not None
        assert processed.shape == sample_image.shape
        assert processed.dtype == np.uint8
    
    def test_extract_features(self, matcher, sample_image):
        """Test feature extraction."""
        processed = matcher.preprocess_image(sample_image)
        features = matcher.extract_features(processed)
        
        assert features is not None
        assert isinstance(features, dict)
        assert 'sift' in features
        assert 'orb' in features
        assert 'minutiae' in features
    
    def test_sift_features(self, matcher, sample_image):
        """Test SIFT feature extraction."""
        processed = matcher.preprocess_image(sample_image)
        features = matcher.extract_features(processed)
        
        sift_features = features['sift']
        assert 'keypoints' in sift_features
        assert 'descriptors' in sift_features
        assert 'count' in sift_features
        assert sift_features['count'] >= 0
    
    def test_orb_features(self, matcher, sample_image):
        """Test ORB feature extraction."""
        processed = matcher.preprocess_image(sample_image)
        features = matcher.extract_features(processed)
        
        orb_features = features['orb']
        assert 'keypoints' in orb_features
        assert 'descriptors' in orb_features
        assert 'count' in orb_features
        assert orb_features['count'] >= 0
    
    def test_minutiae_features(self, matcher, sample_image):
        """Test minutiae extraction."""
        processed = matcher.preprocess_image(sample_image)
        features = matcher.extract_features(processed)
        
        minutiae = features['minutiae']
        assert isinstance(minutiae, list)
        
        # Check minutiae structure if any found
        if minutiae:
            for m in minutiae:
                assert 'x' in m
                assert 'y' in m
                assert 'type' in m
                assert 'angle' in m
                assert m['type'] in ['ending', 'bifurcation']
    
    def test_match_features_same_image(self, matcher, sample_image):
        """Test matching the same image to itself."""
        processed = matcher.preprocess_image(sample_image)
        features = matcher.extract_features(processed)
        
        result = matcher.match_features(features, features)
        
        assert isinstance(result, MatchResult)
        assert result.score >= 0.8  # Should have high similarity to itself
        assert result.confidence >= 0.0
        assert result.is_match is True
    
    def test_match_features_different_images(self, matcher, sample_image, different_image):
        """Test matching different images."""
        processed1 = matcher.preprocess_image(sample_image)
        processed2 = matcher.preprocess_image(different_image)
        
        features1 = matcher.extract_features(processed1)
        features2 = matcher.extract_features(processed2)
        
        result = matcher.match_features(features1, features2)
        
        assert isinstance(result, MatchResult)
        assert 0.0 <= result.score <= 1.0
        assert 0.0 <= result.confidence <= 1.0
        assert isinstance(result.is_match, bool)
        assert result.processing_time > 0
    
    def test_different_matching_methods(self, matcher, sample_image):
        """Test different matching methods."""
        processed = matcher.preprocess_image(sample_image)
        features = matcher.extract_features(processed)
        
        methods = ['sift', 'orb', 'hybrid']
        
        for method in methods:
            result = matcher.match_features(features, features, method=method)
            assert isinstance(result, MatchResult)
            assert result.method_used == method
            assert result.score >= 0.0
    
    def test_is_match_function(self, matcher):
        """Test the is_match function."""
        # Test with high score
        assert matcher.is_match(0.9) is True
        
        # Test with low score
        assert matcher.is_match(0.5) is False
        
        # Test with threshold score
        assert matcher.is_match(0.85) is True
        
        # Test with confidence
        assert matcher.is_match(0.9, 0.8) is True
        assert matcher.is_match(0.9, 0.3) is False  # Low confidence
    
    def test_invalid_matching_method(self, matcher, sample_image):
        """Test handling of invalid matching method."""
        processed = matcher.preprocess_image(sample_image)
        features = matcher.extract_features(processed)
        
        with pytest.raises(ValueError):
            matcher.match_features(features, features, method='invalid_method')
    
    def test_empty_features(self, matcher):
        """Test matching with empty features."""
        empty_features = {
            'sift': {'keypoints': [], 'descriptors': None, 'count': 0},
            'orb': {'keypoints': [], 'descriptors': None, 'count': 0},
            'minutiae': []
        }
        
        result = matcher.match_features(empty_features, empty_features)
        
        assert isinstance(result, MatchResult)
        assert result.score == 0.0
    
    def test_match_result_attributes(self, matcher, sample_image):
        """Test MatchResult attributes."""
        processed = matcher.preprocess_image(sample_image)
        features = matcher.extract_features(processed)
        
        result = matcher.match_features(features, features)
        
        # Check all required attributes
        assert hasattr(result, 'score')
        assert hasattr(result, 'confidence')
        assert hasattr(result, 'is_match')
        assert hasattr(result, 'method_used')
        assert hasattr(result, 'processing_time')
        assert hasattr(result, 'details')
        
        # Check data types
        assert isinstance(result.score, float)
        assert isinstance(result.confidence, float)
        assert isinstance(result.is_match, bool)
        assert isinstance(result.method_used, str)
        assert isinstance(result.processing_time, float)
        assert isinstance(result.details, dict)


if __name__ == '__main__':
    pytest.main([__file__])
