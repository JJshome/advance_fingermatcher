#!/usr/bin/env python3
"""
Enhanced Bozorth3 Algorithm Unit Tests

Comprehensive unit tests for the Enhanced Bozorth3 implementation,
covering all major components and edge cases.

Author: JJshome
Date: 2025
"""

import sys
import unittest
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from advance_fingermatcher.algorithms.enhanced_bozorth3 import EnhancedBozorth3Matcher
    from advance_fingermatcher.core.minutiae import MinutiaeTemplate
    FULL_TESTS = True
except ImportError:
    FULL_TESTS = False


class TestEnhancedBozorth3Matcher(unittest.TestCase):
    """Test cases for Enhanced Bozorth3 Matcher"""
    
    def setUp(self):
        """Set up test fixtures"""
        if not FULL_TESTS:
            self.skipTest("Full tests require complete installation")
            
        self.matcher = EnhancedBozorth3Matcher()
        
        # Sample minutiae templates for testing
        self.template1 = [
            {'x': 100, 'y': 100, 'angle': 0.0, 'type': 'ridge_ending', 'quality': 0.8},
            {'x': 150, 'y': 120, 'angle': 1.57, 'type': 'bifurcation', 'quality': 0.9},
            {'x': 200, 'y': 110, 'angle': 3.14, 'type': 'ridge_ending', 'quality': 0.7},
            {'x': 120, 'y': 180, 'angle': 0.78, 'type': 'bifurcation', 'quality': 0.6}
        ]
        
        self.template2 = [
            {'x': 105, 'y': 95, 'angle': 0.1, 'type': 'ridge_ending', 'quality': 0.8},
            {'x': 155, 'y': 125, 'angle': 1.67, 'type': 'bifurcation', 'quality': 0.85},
            {'x': 195, 'y': 115, 'angle': 3.04, 'type': 'ridge_ending', 'quality': 0.75},
            {'x': 125, 'y': 175, 'angle': 0.88, 'type': 'bifurcation', 'quality': 0.65}
        ]
        
        self.empty_template = []
        self.low_quality_template = [
            {'x': 100, 'y': 100, 'angle': 0.0, 'type': 'ridge_ending', 'quality': 0.1},
            {'x': 150, 'y': 120, 'angle': 1.57, 'type': 'bifurcation', 'quality': 0.2}
        ]
    
    def test_initialization_default(self):
        """Test default initialization"""
        matcher = EnhancedBozorth3Matcher()
        
        self.assertIsNotNone(matcher)
        self.assertTrue(matcher.quality_weighting)
        self.assertFalse(matcher.descriptor_matching)  # Default is False
        self.assertIn('distance', matcher.base_tolerances)
        self.assertIn('angle', matcher.base_tolerances)
    
    def test_initialization_custom(self):
        """Test custom initialization"""
        custom_tolerances = {'distance': 15.0, 'angle': 0.3}
        matcher = EnhancedBozorth3Matcher(
            base_tolerances=custom_tolerances,
            quality_weighting=False,
            descriptor_matching=True
        )
        
        self.assertFalse(matcher.quality_weighting)
        self.assertTrue(matcher.descriptor_matching)
        self.assertEqual(matcher.base_tolerances['distance'], 15.0)
        self.assertEqual(matcher.base_tolerances['angle'], 0.3)
    
    def test_basic_matching(self):
        """Test basic minutiae matching"""
        score = self.matcher.match_minutiae(self.template1, self.template2)
        
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)
        self.assertGreater(score, 0.0)  # Should have some similarity
    
    def test_identical_templates(self):
        """Test matching identical templates"""
        score = self.matcher.match_minutiae(self.template1, self.template1)
        
        self.assertGreater(score, 0.8)  # Should be very high similarity
    
    def test_empty_templates(self):
        """Test handling of empty templates"""
        score1 = self.matcher.match_minutiae(self.empty_template, self.template1)
        score2 = self.matcher.match_minutiae(self.template1, self.empty_template)
        score3 = self.matcher.match_minutiae(self.empty_template, self.empty_template)
        
        self.assertEqual(score1, 0.0)
        self.assertEqual(score2, 0.0)
        self.assertEqual(score3, 0.0)
    
    def test_quality_weighting_effect(self):
        """Test impact of quality weighting"""
        matcher_with_quality = EnhancedBozorth3Matcher(quality_weighting=True)
        matcher_without_quality = EnhancedBozorth3Matcher(quality_weighting=False)
        
        score_with = matcher_with_quality.match_minutiae(self.template1, self.low_quality_template)
        score_without = matcher_without_quality.match_minutiae(self.template1, self.low_quality_template)
        
        # Quality weighting should typically result in different scores
        # The exact relationship depends on the implementation
        self.assertIsInstance(score_with, float)
        self.assertIsInstance(score_without, float)
    
    def test_tolerance_validation(self):
        """Test tolerance parameter validation"""
        # Valid tolerances
        matcher = EnhancedBozorth3Matcher(
            base_tolerances={'distance': 10.0, 'angle': 0.26}
        )
        self.assertIsNotNone(matcher)
        
        # Test with extreme tolerances
        matcher_strict = EnhancedBozorth3Matcher(
            base_tolerances={'distance': 1.0, 'angle': 0.01}
        )
        matcher_relaxed = EnhancedBozorth3Matcher(
            base_tolerances={'distance': 50.0, 'angle': 1.0}
        )
        
        score_strict = matcher_strict.match_minutiae(self.template1, self.template2)
        score_relaxed = matcher_relaxed.match_minutiae(self.template1, self.template2)
        
        # Relaxed tolerances should generally give higher scores
        self.assertGreaterEqual(score_relaxed, score_strict)


class TestEnhancedBozorth3Performance(unittest.TestCase):
    """Performance-focused tests for Enhanced Bozorth3"""
    
    def setUp(self):
        """Set up performance test fixtures"""
        if not FULL_TESTS:
            self.skipTest("Full tests require complete installation")
            
        self.matcher = EnhancedBozorth3Matcher()
    
    def test_matching_speed(self):
        """Test matching speed with reasonable templates"""
        import time
        
        # Generate medium-sized templates
        template1 = []
        template2 = []
        
        np.random.seed(42)
        for i in range(50):
            minutia1 = {
                'x': np.random.randint(0, 500),
                'y': np.random.randint(0, 400),
                'angle': np.random.uniform(0, 2*np.pi),
                'type': np.random.choice(['ridge_ending', 'bifurcation']),
                'quality': np.random.uniform(0.3, 1.0)
            }
            minutia2 = {
                'x': np.random.randint(0, 500),
                'y': np.random.randint(0, 400),
                'angle': np.random.uniform(0, 2*np.pi),
                'type': np.random.choice(['ridge_ending', 'bifurcation']),
                'quality': np.random.uniform(0.3, 1.0)
            }
            template1.append(minutia1)
            template2.append(minutia2)
        
        # Measure performance
        times = []
        for _ in range(10):
            start_time = time.time()
            score = self.matcher.match_minutiae(template1, template2)
            elapsed = time.time() - start_time
            times.append(elapsed)
        
        avg_time = np.mean(times)
        std_time = np.std(times)
        
        # Performance assertions
        self.assertLess(avg_time, 1.0)  # Should complete within 1 second
        self.assertLess(std_time, 0.5)  # Should be reasonably consistent
        
        print(f"Average matching time: {avg_time*1000:.2f}ms ± {std_time*1000:.2f}ms")


class TestEnhancedBozorth3EdgeCases(unittest.TestCase):
    """Test edge cases and error handling"""
    
    def setUp(self):
        """Set up edge case test fixtures"""
        if not FULL_TESTS:
            self.skipTest("Full tests require complete installation")
            
        self.matcher = EnhancedBozorth3Matcher()
    
    def test_malformed_minutiae(self):
        """Test handling of malformed minutiae data"""
        malformed_templates = [
            # Missing required fields
            [{'x': 100, 'y': 100}],
            # Invalid data types
            [{'x': 'invalid', 'y': 100, 'angle': 0.0, 'type': 'ridge_ending', 'quality': 0.8}],
            # Out of range values
            [{'x': -100, 'y': 100, 'angle': 0.0, 'type': 'ridge_ending', 'quality': 1.5}],
        ]
        
        template = [
            {'x': 100, 'y': 100, 'angle': 0.0, 'type': 'ridge_ending', 'quality': 0.8}
        ]
        
        for malformed in malformed_templates:
            try:
                score = self.matcher.match_minutiae(template, malformed)
                # Should either return a valid score or handle gracefully
                if score is not None:
                    self.assertIsInstance(score, float)
                    self.assertGreaterEqual(score, 0.0)
                    self.assertLessEqual(score, 1.0)
            except Exception as e:
                # Should not crash - either handle gracefully or raise informative error
                self.assertIn('minutiae', str(e).lower())
    
    def test_extreme_coordinates(self):
        """Test with extreme coordinate values"""
        extreme_template = [
            {'x': 0, 'y': 0, 'angle': 0.0, 'type': 'ridge_ending', 'quality': 0.8},
            {'x': 10000, 'y': 10000, 'angle': 0.0, 'type': 'ridge_ending', 'quality': 0.8},
        ]
        
        normal_template = [
            {'x': 100, 'y': 100, 'angle': 0.0, 'type': 'ridge_ending', 'quality': 0.8}
        ]
        
        score = self.matcher.match_minutiae(extreme_template, normal_template)
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)


class TestEnhancedBozorth3Integration(unittest.TestCase):
    """Integration tests for Enhanced Bozorth3 with other components"""
    
    def setUp(self):
        """Set up integration test fixtures"""
        if not FULL_TESTS:
            self.skipTest("Full tests require complete installation")
    
    def test_with_different_image_sizes(self):
        """Test matching templates from different image sizes"""
        # Template from small image (300x200)
        small_template = [
            {'x': 50, 'y': 50, 'angle': 0.0, 'type': 'ridge_ending', 'quality': 0.8},
            {'x': 100, 'y': 80, 'angle': 1.57, 'type': 'bifurcation', 'quality': 0.9}
        ]
        
        # Template from large image (800x600) - scaled coordinates
        large_template = [
            {'x': 133, 'y': 150, 'angle': 0.0, 'type': 'ridge_ending', 'quality': 0.8},
            {'x': 267, 'y': 240, 'angle': 1.57, 'type': 'bifurcation', 'quality': 0.9}
        ]
        
        matcher = EnhancedBozorth3Matcher()
        score = matcher.match_minutiae(small_template, large_template)
        
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)
    
    def test_with_various_quality_distributions(self):
        """Test with various quality score distributions"""
        quality_distributions = [
            (0.9, 1.0),    # High quality
            (0.5, 0.7),    # Medium quality
            (0.1, 0.3),    # Low quality
            (0.1, 1.0),    # Mixed quality
        ]
        
        matcher = EnhancedBozorth3Matcher(quality_weighting=True)
        
        for min_qual, max_qual in quality_distributions:
            template = [
                {
                    'x': 100 + i*20, 
                    'y': 100 + i*15, 
                    'angle': i*0.5, 
                    'type': 'ridge_ending' if i%2==0 else 'bifurcation',
                    'quality': np.random.uniform(min_qual, max_qual)
                }
                for i in range(10)
            ]
            
            score = matcher.match_minutiae(template, template)
            self.assertGreater(score, 0.5)  # Self-match should be high


def create_test_suite():
    """Create comprehensive test suite"""
    suite = unittest.TestSuite()
    
    # Add all test classes
    test_classes = [
        TestEnhancedBozorth3Matcher,
        TestEnhancedBozorth3Performance,
        TestEnhancedBozorth3EdgeCases,
        TestEnhancedBozorth3Integration
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    return suite


def run_tests():
    """Run all tests with detailed output"""
    if not FULL_TESTS:
        print("⚠️  Enhanced Bozorth3 tests require complete installation")
        print("   Run: pip install -e \".[dev,ml]\"")
        return False
    
    print("🧪 Running Enhanced Bozorth3 Unit Tests")
    print("=" * 50)
    
    # Create test suite
    suite = create_test_suite()
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(
        verbosity=2,
        stream=sys.stdout,
        descriptions=True,
        failfast=False
    )
    
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped) if hasattr(result, 'skipped') else 0}")
    
    if result.failures:
        print(f"\n❌ FAILURES ({len(result.failures)}):")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback.split(chr(10))[-2]}")
    
    if result.errors:
        print(f"\n💥 ERRORS ({len(result.errors)}):")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback.split(chr(10))[-2]}")
    
    success = len(result.failures) == 0 and len(result.errors) == 0
    
    if success:
        print("\n✅ All Enhanced Bozorth3 tests passed!")
    else:
        print(f"\n❌ Some tests failed. See details above.")
    
    return success


if __name__ == "__main__":
    run_tests()
