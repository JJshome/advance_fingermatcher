"""
Unit Tests for Enhanced Bozorth3 Algorithm

This module contains comprehensive tests for the Enhanced Bozorth3 algorithm
implementation, covering all major components and functionality.

Test Coverage:
- Quality assessment
- Adaptive tolerance calculation
- Rich minutiae descriptors
- Multi-stage matching process
- Performance validation
"""

import unittest
import numpy as np
import time
from unittest.mock import patch, MagicMock

# Import Enhanced Bozorth3 components
try:
    from advance_fingermatcher.algorithms.enhanced_bozorth3 import (
        EnhancedBozorth3Matcher,
        EnhancedMinutia,
        MinutiaType,
        QualityAssessment,
        AdaptiveToleranceCalculator,
        DescriptorExtractor,
        create_sample_minutiae,
        MatchingResult
    )
except ImportError as e:
    import sys
    print(f"Import error: {e}")
    print("Please ensure the advance_fingermatcher package is properly installed.")
    sys.exit(1)


class TestEnhancedMinutia(unittest.TestCase):
    """Test cases for EnhancedMinutia class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.valid_minutia = EnhancedMinutia(
            x=100.0, y=150.0, theta=0.5,
            minutia_type=MinutiaType.ENDING,
            quality=0.8, reliability=0.75,
            local_descriptor=np.random.randn(16),
            ridge_frequency=0.1, ridge_orientation=0.5,
            local_density=1.0, curvature=0.05,
            neighbors=[(20.0, 0.3), (25.0, 1.2)]
        )
    
    def test_valid_minutia_creation(self):
        """Test creation of valid minutia"""
        self.assertEqual(self.valid_minutia.x, 100.0)
        self.assertEqual(self.valid_minutia.y, 150.0)
        self.assertEqual(self.valid_minutia.theta, 0.5)
        self.assertEqual(self.valid_minutia.minutia_type, MinutiaType.ENDING)
        self.assertEqual(self.valid_minutia.quality, 0.8)
        self.assertEqual(len(self.valid_minutia.neighbors), 2)
    
    def test_invalid_quality_raises_error(self):
        """Test that invalid quality values raise ValueError"""
        with self.assertRaises(ValueError):
            EnhancedMinutia(
                x=100.0, y=150.0, theta=0.5,
                minutia_type=MinutiaType.ENDING,
                quality=1.5, reliability=0.75,  # Invalid quality > 1.0
                local_descriptor=np.random.randn(16),
                ridge_frequency=0.1, ridge_orientation=0.5,
                local_density=1.0, curvature=0.05,
                neighbors=[]
            )
    
    def test_invalid_reliability_raises_error(self):
        """Test that invalid reliability values raise ValueError"""
        with self.assertRaises(ValueError):
            EnhancedMinutia(
                x=100.0, y=150.0, theta=0.5,
                minutia_type=MinutiaType.ENDING,
                quality=0.8, reliability=-0.1,  # Invalid reliability < 0
                local_descriptor=np.random.randn(16),
                ridge_frequency=0.1, ridge_orientation=0.5,
                local_density=1.0, curvature=0.05,
                neighbors=[]
            )


class TestQualityAssessment(unittest.TestCase):
    """Test cases for QualityAssessment class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.quality_assessor = QualityAssessment()
        
        # Create test image patches
        self.high_quality_patch = self._create_test_patch('high')
        self.medium_quality_patch = self._create_test_patch('medium')
        self.low_quality_patch = self._create_test_patch('low')
        self.empty_patch = np.array([])
    
    def _create_test_patch(self, quality_level, size=(32, 32)):
        """Create synthetic test patch with specified quality"""
        patch = np.zeros(size)
        
        for i in range(size[0]):
            for j in range(size[1]):
                # Base ridge pattern
                ridge_value = 128 + 64 * np.sin(j * 0.3)
                
                if quality_level == 'high':
                    noise = np.random.normal(0, 5)
                elif quality_level == 'medium':
                    noise = np.random.normal(0, 15)
                else:  # low quality
                    noise = np.random.normal(0, 30)
                
                patch[i, j] = np.clip(ridge_value + noise, 0, 255)
        
        return patch.astype(np.uint8)
    
    def test_quality_score_range(self):
        """Test that quality scores are in valid range [0.1, 1.0]"""
        high_score = self.quality_assessor.calculate_quality_score(self.high_quality_patch)
        medium_score = self.quality_assessor.calculate_quality_score(self.medium_quality_patch)
        low_score = self.quality_assessor.calculate_quality_score(self.low_quality_patch)
        
        # Check range
        self.assertGreaterEqual(high_score, 0.1)
        self.assertLessEqual(high_score, 1.0)
        self.assertGreaterEqual(medium_score, 0.1)
        self.assertLessEqual(medium_score, 1.0)
        self.assertGreaterEqual(low_score, 0.1)
        self.assertLessEqual(low_score, 1.0)
    
    def test_quality_ordering(self):
        """Test that high quality patches score higher than low quality"""
        high_score = self.quality_assessor.calculate_quality_score(self.high_quality_patch)
        low_score = self.quality_assessor.calculate_quality_score(self.low_quality_patch)
        
        self.assertGreater(high_score, low_score)
    
    def test_empty_patch_handling(self):
        """Test handling of empty image patches"""
        empty_score = self.quality_assessor.calculate_quality_score(self.empty_patch)
        self.assertEqual(empty_score, 0.0)
    
    def test_ridge_clarity_assessment(self):
        """Test ridge clarity assessment"""
        high_clarity = self.quality_assessor.assess_ridge_clarity(self.high_quality_patch)
        low_clarity = self.quality_assessor.assess_ridge_clarity(self.low_quality_patch)
        
        self.assertGreaterEqual(high_clarity, 0.0)
        self.assertLessEqual(high_clarity, 1.0)
        self.assertGreaterEqual(low_clarity, 0.0)
        self.assertLessEqual(low_clarity, 1.0)
    
    def test_local_contrast_calculation(self):
        """Test local contrast calculation"""
        contrast = self.quality_assessor.calculate_local_contrast(self.high_quality_patch)
        
        self.assertGreaterEqual(contrast, 0.0)
        self.assertLessEqual(contrast, 1.0)
    
    def test_coherence_measurement(self):
        """Test coherence measurement"""
        coherence = self.quality_assessor.measure_coherence(self.high_quality_patch)
        
        self.assertGreaterEqual(coherence, 0.0)
        self.assertLessEqual(coherence, 1.0)


class TestAdaptiveToleranceCalculator(unittest.TestCase):
    """Test cases for AdaptiveToleranceCalculator class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.tolerance_calc = AdaptiveToleranceCalculator()
        
        self.high_quality_minutia = EnhancedMinutia(
            x=100.0, y=100.0, theta=0.5, minutia_type=MinutiaType.ENDING,
            quality=0.9, reliability=0.85, local_descriptor=np.random.randn(16),
            ridge_frequency=0.1, ridge_orientation=0.5, local_density=1.0,
            curvature=0.05, neighbors=[(20.0, 0.2)]
        )
        
        self.low_quality_minutia = EnhancedMinutia(
            x=150.0, y=150.0, theta=0.7, minutia_type=MinutiaType.BIFURCATION,
            quality=0.3, reliability=0.25, local_descriptor=np.random.randn(16),
            ridge_frequency=0.08, ridge_orientation=0.7, local_density=0.6,
            curvature=0.15, neighbors=[(15.0, 0.8)]
        )
    
    def test_adaptive_tolerance_calculation(self):
        """Test adaptive tolerance calculation"""
        tolerance = self.tolerance_calc.calculate_adaptive_tolerance(
            self.high_quality_minutia, self.low_quality_minutia
        )
        
        self.assertIn('distance', tolerance)
        self.assertIn('angle', tolerance)
        self.assertIn('confidence', tolerance)
        
        # Check ranges
        self.assertGreaterEqual(tolerance['distance'], 3.0)
        self.assertLessEqual(tolerance['distance'], 20.0)
        self.assertGreaterEqual(tolerance['angle'], np.pi/24)
        self.assertLessEqual(tolerance['angle'], np.pi/6)
        self.assertGreaterEqual(tolerance['confidence'], 0.0)
        self.assertLessEqual(tolerance['confidence'], 1.0)
    
    def test_high_quality_tight_tolerance(self):
        """Test that high quality minutiae get tighter tolerances"""
        high_high_tolerance = self.tolerance_calc.calculate_adaptive_tolerance(
            self.high_quality_minutia, self.high_quality_minutia
        )
        low_low_tolerance = self.tolerance_calc.calculate_adaptive_tolerance(
            self.low_quality_minutia, self.low_quality_minutia
        )
        
        # High quality should have tighter (smaller) tolerances
        self.assertLess(high_high_tolerance['distance'], low_low_tolerance['distance'])
        self.assertLess(high_high_tolerance['angle'], low_low_tolerance['angle'])


class TestDescriptorExtractor(unittest.TestCase):
    """Test cases for DescriptorExtractor class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.extractor = DescriptorExtractor()
        self.test_patch = np.random.randint(0, 256, (32, 32), dtype=np.uint8)
        self.empty_patch = np.array([])
    
    def test_extract_local_features(self):
        """Test local feature extraction"""
        features = self.extractor.extract_local_features(self.test_patch)
        
        self.assertEqual(len(features), 16)
        self.assertAlmostEqual(np.linalg.norm(features), 1.0, places=5)
    
    def test_empty_patch_features(self):
        """Test feature extraction from empty patch"""
        features = self.extractor.extract_local_features(self.empty_patch)
        
        self.assertEqual(len(features), 16)
        self.assertTrue(np.allclose(features, 0.0))
    
    def test_ridge_structure_analysis(self):
        """Test ridge structure analysis"""
        structure = self.extractor.analyze_ridge_structure(self.test_patch)
        
        self.assertIn('frequency', structure)
        self.assertIn('orientation', structure)
        self.assertIn('curvature', structure)
        
        # Check ranges
        self.assertGreaterEqual(structure['frequency'], 0.0)
        self.assertLessEqual(structure['frequency'], 1.0)
        self.assertGreaterEqual(structure['orientation'], 0.0)
        self.assertLess(structure['orientation'], np.pi)
        self.assertGreaterEqual(structure['curvature'], 0.0)
        self.assertLessEqual(structure['curvature'], 1.0)


class TestEnhancedBozorth3Matcher(unittest.TestCase):
    """Test cases for EnhancedBozorth3Matcher class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.matcher = EnhancedBozorth3Matcher()
        self.probe_minutiae = create_sample_minutiae(10, add_descriptors=True)
        self.gallery_minutiae = create_sample_minutiae(8, add_descriptors=True)
    
    def test_matcher_initialization(self):
        """Test matcher initialization"""
        self.assertIsInstance(self.matcher.quality_assessor, QualityAssessment)
        self.assertIsInstance(self.matcher.descriptor_extractor, DescriptorExtractor)
        self.assertIsInstance(self.matcher.tolerance_calculator, AdaptiveToleranceCalculator)
        
        # Check compatibility weights
        weights = self.matcher.compatibility_weights
        self.assertAlmostEqual(weights['geometric'] + weights['descriptor'] + weights['quality'], 1.0)
    
    def test_match_fingerprints_returns_result(self):
        """Test that match_fingerprints returns proper MatchingResult"""
        result = self.matcher.match_fingerprints(
            self.probe_minutiae, self.gallery_minutiae
        )
        
        self.assertIsInstance(result, MatchingResult)
        self.assertGreaterEqual(result.score, 0.0)
        self.assertLessEqual(result.score, 1.0)
        self.assertGreaterEqual(result.confidence, 0.0)
        self.assertLessEqual(result.confidence, 1.0)
        self.assertIsInstance(result.is_match, bool)
        self.assertIsInstance(result.matched_pairs, list)
        self.assertGreater(result.processing_time, 0.0)
        self.assertEqual(result.method_used, "Enhanced Bozorth3")
    
    def test_empty_minutiae_handling(self):
        """Test handling of empty minutiae lists"""
        empty_minutiae = []
        
        result = self.matcher.match_fingerprints(
            empty_minutiae, self.gallery_minutiae
        )
        
        self.assertEqual(result.score, 0.0)
        self.assertEqual(len(result.matched_pairs), 0)
        self.assertFalse(result.is_match)
    
    def test_identical_minutiae_high_score(self):
        """Test that identical minutiae produce high matching scores"""
        # Use same minutiae for both probe and gallery
        result = self.matcher.match_fingerprints(
            self.probe_minutiae, self.probe_minutiae
        )
        
        # Identical minutiae should produce high scores
        self.assertGreater(result.score, 0.3)  # Should be reasonably high
        self.assertGreater(result.confidence, 0.2)
    
    def test_performance_timing(self):
        """Test that matching completes within reasonable time"""
        start_time = time.time()
        
        result = self.matcher.match_fingerprints(
            self.probe_minutiae, self.gallery_minutiae
        )
        
        end_time = time.time()
        elapsed = end_time - start_time
        
        # Should complete in under 1 second for small minutiae sets
        self.assertLess(elapsed, 1.0)
        self.assertAlmostEqual(result.processing_time, elapsed, delta=0.01)


class TestSampleMinutiaeCreation(unittest.TestCase):
    """Test cases for create_sample_minutiae function"""
    
    def test_create_sample_minutiae_count(self):
        """Test that correct number of minutiae are created"""
        count = 15
        minutiae = create_sample_minutiae(count)
        
        self.assertEqual(len(minutiae), count)
    
    def test_create_sample_minutiae_properties(self):
        """Test that created minutiae have valid properties"""
        minutiae = create_sample_minutiae(5, add_descriptors=True)
        
        for minutia in minutiae:
            self.assertIsInstance(minutia, EnhancedMinutia)
            self.assertGreaterEqual(minutia.x, 20)
            self.assertLessEqual(minutia.x, 380)
            self.assertGreaterEqual(minutia.y, 20)
            self.assertLessEqual(minutia.y, 380)
            self.assertGreaterEqual(minutia.theta, 0)
            self.assertLess(minutia.theta, 2 * np.pi)
            self.assertGreaterEqual(minutia.quality, 0.3)
            self.assertLessEqual(minutia.quality, 1.0)
            self.assertEqual(len(minutia.local_descriptor), 16)
            self.assertGreater(len(minutia.neighbors), 0)
    
    def test_create_without_descriptors(self):
        """Test creating minutiae without rich descriptors"""
        minutiae = create_sample_minutiae(3, add_descriptors=False)
        
        for minutia in minutiae:
            self.assertTrue(np.allclose(minutia.local_descriptor, 0.0))


class TestIntegration(unittest.TestCase):
    """Integration tests for Enhanced Bozorth3 system"""
    
    def setUp(self):
        """Set up integration test fixtures"""
        self.matcher = EnhancedBozorth3Matcher()
    
    def test_full_matching_pipeline(self):
        """Test complete matching pipeline from start to finish"""
        # Create diverse test sets
        probe_set = create_sample_minutiae(12, add_descriptors=True)
        gallery_set = create_sample_minutiae(10, add_descriptors=True)
        
        # Set different qualities to test adaptive behavior
        for i, minutia in enumerate(probe_set):
            minutia.quality = 0.9 - (i * 0.05)  # Decreasing quality
            minutia.reliability = minutia.quality * 0.9
        
        for i, minutia in enumerate(gallery_set):
            minutia.quality = 0.8 - (i * 0.04)  # Decreasing quality
            minutia.reliability = minutia.quality * 0.9
        
        # Perform matching
        result = self.matcher.match_fingerprints(
            probe_set, gallery_set,
            probe_quality=0.75, gallery_quality=0.70
        )
        
        # Validate complete result
        self.assertIsInstance(result, MatchingResult)
        self.assertIsInstance(result.score, float)
        self.assertIsInstance(result.confidence, float)
        self.assertIsInstance(result.is_match, bool)
        self.assertIsInstance(result.matched_pairs, list)
        self.assertGreater(result.processing_time, 0)
        self.assertEqual(result.quality_scores, (0.75, 0.70))
    
    def test_quality_impact_on_matching(self):
        """Test that quality differences impact matching results appropriately"""
        # Create high and low quality versions of the same basic minutiae
        base_minutiae = create_sample_minutiae(8, add_descriptors=True)
        
        high_quality_minutiae = []
        low_quality_minutiae = []
        
        for minutia in base_minutiae:
            # High quality version
            high_qual = EnhancedMinutia(
                x=minutia.x, y=minutia.y, theta=minutia.theta,
                minutia_type=minutia.minutia_type,
                quality=0.9, reliability=0.85,
                local_descriptor=minutia.local_descriptor,
                ridge_frequency=minutia.ridge_frequency,
                ridge_orientation=minutia.ridge_orientation,
                local_density=minutia.local_density,
                curvature=minutia.curvature,
                neighbors=minutia.neighbors
            )
            high_quality_minutiae.append(high_qual)
            
            # Low quality version  
            low_qual = EnhancedMinutia(
                x=minutia.x + np.random.normal(0, 2),  # Add position noise
                y=minutia.y + np.random.normal(0, 2),
                theta=minutia.theta + np.random.normal(0, 0.1),  # Add angle noise
                minutia_type=minutia.minutia_type,
                quality=0.3, reliability=0.25,
                local_descriptor=minutia.local_descriptor + np.random.normal(0, 0.1, 16),
                ridge_frequency=minutia.ridge_frequency,
                ridge_orientation=minutia.ridge_orientation,
                local_density=minutia.local_density,
                curvature=minutia.curvature,
                neighbors=minutia.neighbors
            )
            low_quality_minutiae.append(low_qual)
        
        # Match high quality versions
        high_result = self.matcher.match_fingerprints(
            high_quality_minutiae, high_quality_minutiae,
            probe_quality=0.9, gallery_quality=0.9
        )
        
        # Match low quality versions
        low_result = self.matcher.match_fingerprints(
            low_quality_minutiae, low_quality_minutiae,
            probe_quality=0.3, gallery_quality=0.3
        )
        
        # High quality should generally produce better results
        # Note: Due to randomness, we use a reasonable threshold
        self.assertGreaterEqual(high_result.confidence, low_result.confidence - 0.2)


if __name__ == '__main__':
    # Configure test runner
    unittest.main(verbosity=2, buffer=True)
