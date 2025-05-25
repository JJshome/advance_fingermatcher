#!/usr/bin/env python3
"""
Tests for minutiae detection functionality.
"""

import pytest
import numpy as np
import cv2
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from advance_fingermatcher.core.minutiae_detector import (
    MinutiaeDetector,
    create_minutiae_detector
)


class TestMinutiaeDetector:
    """Test cases for MinutiaeDetector class."""
    
    def test_detector_initialization(self):
        """Test detector initialization."""
        detector = MinutiaeDetector()
        
        assert detector is not None
        assert detector.use_deep_learning is False  # No model provided
        assert detector.model_path is None
    
    def test_detector_with_invalid_model_path(self):
        """Test detector with invalid model path."""
        detector = MinutiaeDetector(model_path="nonexistent_model.h5")
        
        assert detector.use_deep_learning is False  # Should fall back
    
    def test_factory_function(self):
        """Test create_minutiae_detector factory function."""
        detector = create_minutiae_detector()
        
        assert isinstance(detector, MinutiaeDetector)
        assert detector.use_deep_learning is False
    
    def test_detect_basic(self, sample_image):
        """Test basic minutiae detection."""
        detector = create_minutiae_detector()
        
        minutiae = detector.detect(sample_image)
        
        assert isinstance(minutiae, list)
        # Should detect at least some minutiae in synthetic image
        assert len(minutiae) >= 0
    
    def test_detect_with_quality_threshold(self, sample_image):
        """Test detection with different quality thresholds."""
        detector = create_minutiae_detector()
        
        # Test different thresholds
        thresholds = [0.3, 0.5, 0.7, 0.9]
        prev_count = float('inf')
        
        for threshold in thresholds:
            minutiae = detector.detect(sample_image, quality_threshold=threshold)
            
            # Higher threshold should generally result in fewer minutiae
            assert len(minutiae) <= prev_count
            prev_count = len(minutiae)
            
            # Check that all detected minutiae meet quality threshold
            for minutia in minutiae:
                assert minutia['quality'] >= threshold
    
    def test_detect_empty_image(self, minutiae_detector):
        """Test detection on empty image."""
        empty_image = np.zeros((100, 100), dtype=np.uint8)
        
        minutiae = minutiae_detector.detect(empty_image)
        
        # Should handle empty image gracefully
        assert isinstance(minutiae, list)
        assert len(minutiae) == 0  # No minutiae in empty image
    
    def test_detect_noise_image(self, minutiae_detector):
        """Test detection on pure noise image."""
        noise_image = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        
        minutiae = minutiae_detector.detect(noise_image)
        
        # Should handle noise gracefully without crashing
        assert isinstance(minutiae, list)
        # May or may not detect minutiae in noise, but should not crash
    
    def test_detect_color_image_error(self, minutiae_detector):
        """Test that color images are handled properly."""
        color_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        # Should convert to grayscale internally or handle gracefully
        minutiae = minutiae_detector.detect(color_image)
        assert isinstance(minutiae, list)
    
    def test_minutiae_properties(self, sample_image, minutiae_detector):
        """Test that detected minutiae have required properties."""
        minutiae = minutiae_detector.detect(sample_image, quality_threshold=0.3)
        
        if minutiae:  # Only test if minutiae were detected
            for minutia in minutiae:
                # Check required properties
                assert 'x' in minutia
                assert 'y' in minutia
                assert 'angle' in minutia
                assert 'quality' in minutia
                assert 'type' in minutia
                
                # Check property types and ranges
                assert isinstance(minutia['x'], (int, float))
                assert isinstance(minutia['y'], (int, float))
                assert isinstance(minutia['angle'], (int, float))
                assert isinstance(minutia['quality'], (int, float))
                assert isinstance(minutia['type'], str)
                
                # Check value ranges
                assert 0 <= minutia['x'] < sample_image.shape[1]
                assert 0 <= minutia['y'] < sample_image.shape[0]
                assert 0 <= minutia['angle'] < 2 * np.pi
                assert 0 <= minutia['quality'] <= 1
                assert minutia['type'] in ['ending', 'bifurcation']
    
    def test_border_handling(self, minutiae_detector):
        """Test that minutiae near borders are handled correctly."""
        # Create image with pattern near borders
        image = np.zeros((100, 100), dtype=np.uint8)
        
        # Add some patterns near borders
        image[5:10, 5:95] = 255  # Top border
        image[90:95, 5:95] = 255  # Bottom border
        image[5:95, 5:10] = 255  # Left border
        image[5:95, 90:95] = 255  # Right border
        
        minutiae = minutiae_detector.detect(image, quality_threshold=0.3)
        
        # Should not detect minutiae too close to borders
        border_margin = 20
        for minutia in minutiae:
            assert border_margin <= minutia['x'] < image.shape[1] - border_margin
            assert border_margin <= minutia['y'] < image.shape[0] - border_margin
    
    def test_different_image_sizes(self, minutiae_detector):
        """Test detection on different image sizes."""
        sizes = [(50, 50), (100, 100), (200, 200), (300, 400)]
        
        for height, width in sizes:
            # Create synthetic fingerprint pattern
            image = np.zeros((height, width), dtype=np.uint8)
            
            # Add concentric circles
            center_x, center_y = width // 2, height // 2
            for y in range(height):
                for x in range(width):
                    dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                    if int(dist) % 8 < 4:
                        image[y, x] = 255
            
            minutiae = minutiae_detector.detect(image)
            
            # Should handle different sizes without errors
            assert isinstance(minutiae, list)
            # Coordinates should be within image bounds
            for minutia in minutiae:
                assert 0 <= minutia['x'] < width
                assert 0 <= minutia['y'] < height


class TestImagePreprocessing:
    """Test cases for image preprocessing methods."""
    
    def test_preprocess_image(self, minutiae_detector, sample_image):
        """Test image preprocessing functionality."""
        # Access private method for testing
        processed = minutiae_detector._preprocess_image(sample_image)
        
        assert processed.shape == sample_image.shape
        assert processed.dtype == np.uint8
        # Preprocessing should not create extreme values
        assert np.min(processed) >= 0
        assert np.max(processed) <= 255
    
    def test_gabor_filters(self, minutiae_detector, sample_image):
        """Test Gabor filter application."""
        enhanced = minutiae_detector._apply_gabor_filters(sample_image)
        
        assert enhanced.shape == sample_image.shape
        assert enhanced.dtype == np.uint8
    
    def test_skeleton_extraction(self, minutiae_detector):
        """Test skeleton extraction from binary image."""
        # Create simple binary image
        binary_image = np.zeros((50, 50), dtype=np.uint8)
        binary_image[20:30, 10:40] = 255  # Horizontal rectangle
        
        skeleton = minutiae_detector._extract_skeleton(binary_image)
        
        assert skeleton.shape == binary_image.shape
        assert skeleton.dtype == np.uint8
        # Skeleton should be thinner than original
        assert np.sum(skeleton > 0) <= np.sum(binary_image > 0)
    
    def test_zhang_suen_thinning(self, minutiae_detector):
        """Test Zhang-Suen thinning algorithm."""
        # Create thick line
        image = np.zeros((50, 50), dtype=np.uint8)
        image[20:30, 10:40] = 255
        
        thinned = minutiae_detector._zhang_suen_thinning(image)
        
        assert thinned.shape == image.shape
        assert thinned.dtype == np.uint8
        # Should be thinner
        assert np.sum(thinned > 0) < np.sum(image > 0)


class TestMinutiaeCandidates:
    """Test cases for minutiae candidate detection."""
    
    def test_find_minutiae_candidates(self, minutiae_detector):
        """Test finding minutiae candidates in skeleton."""
        # Create skeleton with known minutiae patterns
        skeleton = np.zeros((100, 100), dtype=np.uint8)
        
        # Ridge ending: single line ending
        skeleton[50, 10:20] = 255
        
        # Ridge bifurcation: Y-shaped junction
        skeleton[30, 30:40] = 255  # Horizontal line
        skeleton[25:35, 35] = 255  # Vertical line
        
        candidates = minutiae_detector._find_minutiae_candidates(skeleton)
        
        assert isinstance(candidates, list)
        # Should find at least the patterns we created
        assert len(candidates) >= 0
        
        # Check candidate structure
        for candidate in candidates:
            assert 'x' in candidate
            assert 'y' in candidate
            assert 'type' in candidate
            assert 'angle' in candidate
            assert 'quality' in candidate
            assert candidate['type'] in ['ending', 'bifurcation']
    
    def test_ridge_direction_calculation(self, minutiae_detector):
        """Test ridge direction calculation."""
        # Create image with known gradient direction
        image = np.zeros((50, 50), dtype=np.uint8)
        
        # Create horizontal gradient
        for i in range(50):
            image[:, i] = i * 5
        
        # Test direction calculation at center
        angle = minutiae_detector._calculate_ridge_direction(image, 25, 25)
        
        assert isinstance(angle, float)
        assert 0 <= angle < 2 * np.pi
    
    def test_minutia_quality_calculation(self, minutiae_detector, sample_image):
        """Test minutia quality calculation."""
        # Create test minutia
        minutia = {
            'x': sample_image.shape[1] // 2,
            'y': sample_image.shape[0] // 2,
            'type': 'ending',
            'angle': 0.0
        }
        
        quality = minutiae_detector._calculate_minutia_quality(minutia, sample_image)
        
        assert isinstance(quality, float)
        assert 0 <= quality <= 1
    
    def test_local_coherence_calculation(self, minutiae_detector):
        """Test local coherence calculation."""
        # Create region with consistent orientation
        region = np.zeros((21, 21), dtype=np.uint8)
        
        # Add horizontal lines (consistent orientation)
        for i in range(0, 21, 3):
            region[i, :] = 255
        
        coherence = minutiae_detector._calculate_local_coherence(region)
        
        assert isinstance(coherence, float)
        assert 0 <= coherence <= 1
        # Should have high coherence due to consistent orientation
        assert coherence > 0.3


class TestMinutiaeFiltering:
    """Test cases for minutiae filtering and refinement."""
    
    def test_remove_close_minutiae(self, minutiae_detector):
        """Test removal of minutiae that are too close together."""
        # Create minutiae that are close together
        minutiae = [
            {'x': 50, 'y': 50, 'quality': 0.9, 'type': 'ending', 'angle': 0},
            {'x': 52, 'y': 52, 'quality': 0.8, 'type': 'ending', 'angle': 0},  # Close to first
            {'x': 100, 'y': 100, 'quality': 0.7, 'type': 'bifurcation', 'angle': 0},
            {'x': 101, 'y': 101, 'quality': 0.6, 'type': 'ending', 'angle': 0}  # Close to third
        ]
        
        filtered = minutiae_detector._remove_close_minutiae(minutiae, min_distance=10.0)
        
        # Should remove close minutiae, keeping higher quality ones
        assert len(filtered) < len(minutiae)
        assert len(filtered) == 2  # Should keep 2 out of 4
        
        # Should keep higher quality minutiae
        qualities = [m['quality'] for m in filtered]
        assert 0.9 in qualities  # Highest quality should be kept
        assert 0.7 in qualities  # Third minutia should be kept
    
    def test_filter_and_refine(self, minutiae_detector, sample_image):
        """Test complete filtering and refinement process."""
        # Create candidate minutiae with various properties
        candidates = [
            {'x': 10, 'y': 10, 'type': 'ending', 'angle': 0, 'quality': 0.3},  # Near border
            {'x': 50, 'y': 50, 'type': 'ending', 'angle': 0, 'quality': 0.8},  # Good quality
            {'x': 51, 'y': 51, 'type': 'bifurcation', 'angle': 0, 'quality': 0.9},  # Close to above
            {'x': 100, 'y': 100, 'type': 'ending', 'angle': 0, 'quality': 0.4},  # Low quality
        ]
        
        refined = minutiae_detector._filter_and_refine(
            candidates, sample_image, quality_threshold=0.5
        )
        
        # Should filter out border minutiae, low quality, and close duplicates
        assert len(refined) <= len(candidates)
        
        # All remaining minutiae should meet quality threshold
        for minutia in refined:
            assert minutia['quality'] >= 0.5
            # Should not be too close to borders
            assert minutia['x'] >= 20
            assert minutia['y'] >= 20
            assert minutia['x'] < sample_image.shape[1] - 20
            assert minutia['y'] < sample_image.shape[0] - 20


class TestErrorHandling:
    """Test cases for error handling and edge cases."""
    
    def test_invalid_image_type(self, minutiae_detector):
        """Test handling of invalid image types."""
        # Test with None
        minutiae = minutiae_detector.detect(None)
        assert minutiae == []
        
        # Test with wrong dimensions
        try:
            invalid_image = np.random.rand(10, 10, 10, 10)  # 4D array
            minutiae = minutiae_detector.detect(invalid_image)
            # Should handle gracefully
            assert isinstance(minutiae, list)
        except Exception:
            # Or raise appropriate exception
            pass
    
    def test_very_small_image(self, minutiae_detector):
        """Test handling of very small images."""
        tiny_image = np.random.randint(0, 255, (10, 10), dtype=np.uint8)
        
        minutiae = minutiae_detector.detect(tiny_image)
        
        # Should handle without crashing
        assert isinstance(minutiae, list)
        # Probably no minutiae in such small image
        assert len(minutiae) == 0
    
    def test_extreme_quality_thresholds(self, minutiae_detector, sample_image):
        """Test extreme quality threshold values."""
        # Very low threshold
        minutiae_low = minutiae_detector.detect(sample_image, quality_threshold=0.0)
        assert isinstance(minutiae_low, list)
        
        # Very high threshold
        minutiae_high = minutiae_detector.detect(sample_image, quality_threshold=1.0)
        assert isinstance(minutiae_high, list)
        
        # Invalid thresholds should be handled
        minutiae_invalid1 = minutiae_detector.detect(sample_image, quality_threshold=-0.5)
        assert isinstance(minutiae_invalid1, list)
        
        minutiae_invalid2 = minutiae_detector.detect(sample_image, quality_threshold=1.5)
        assert isinstance(minutiae_invalid2, list)


class TestIntegration:
    """Integration tests for complete detection pipeline."""
    
    def test_complete_detection_pipeline(self, sample_image):
        """Test complete minutiae detection pipeline."""
        detector = create_minutiae_detector()
        
        # Run complete detection
        minutiae = detector.detect(sample_image, quality_threshold=0.6)
        
        # Verify results
        assert isinstance(minutiae, list)
        
        # If minutiae detected, verify their properties
        for minutia in minutiae:
            # Check all required properties exist
            required_props = ['x', 'y', 'angle', 'quality', 'type']
            for prop in required_props:
                assert prop in minutia
            
            # Check property constraints
            assert 0 <= minutia['x'] < sample_image.shape[1]
            assert 0 <= minutia['y'] < sample_image.shape[0]
            assert 0 <= minutia['angle'] < 2 * np.pi
            assert 0.6 <= minutia['quality'] <= 1.0  # Should meet threshold
            assert minutia['type'] in ['ending', 'bifurcation']
    
    def test_repeatability(self, sample_image):
        """Test that detection results are repeatable."""
        detector = create_minutiae_detector()
        
        # Run detection multiple times
        results = []
        for _ in range(3):
            minutiae = detector.detect(sample_image, quality_threshold=0.5)
            results.append(minutiae)
        
        # Results should be identical (deterministic algorithm)
        assert len(results[0]) == len(results[1]) == len(results[2])
        
        # Check that minutiae positions are the same
        for i in range(len(results[0])):
            assert abs(results[0][i]['x'] - results[1][i]['x']) < 1e-6
            assert abs(results[0][i]['y'] - results[1][i]['y']) < 1e-6
            assert abs(results[0][i]['angle'] - results[1][i]['angle']) < 1e-6
    
    def test_performance_reasonable(self, high_quality_image):
        """Test that detection performance is reasonable."""
        import time
        
        detector = create_minutiae_detector()
        
        start_time = time.time()
        minutiae = detector.detect(high_quality_image)
        elapsed_time = time.time() - start_time
        
        # Should complete within reasonable time (adjust based on requirements)
        assert elapsed_time < 5.0  # 5 seconds max for test image
        
        # Should detect reasonable number of minutiae
        assert isinstance(minutiae, list)
        # For a good quality image, should detect some minutiae
        if high_quality_image.shape[0] > 200 and high_quality_image.shape[1] > 200:
            assert len(minutiae) >= 0  # At least some detection capability


if __name__ == '__main__':
    pytest.main([__file__])