#!/usr/bin/env python3
"""
Tests for the BatchProcessor class.
"""

import pytest
import numpy as np
import cv2
from pathlib import Path
import tempfile
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from advance_fingermatcher.utils.batch_processor import BatchProcessor


class TestBatchProcessor:
    """Test cases for BatchProcessor."""
    
    @pytest.fixture
    def processor(self):
        """Create a BatchProcessor instance for testing."""
        return BatchProcessor(max_workers=2)
    
    @pytest.fixture
    def sample_images_dir(self):
        """Create a temporary directory with sample images."""
        temp_dir = tempfile.mkdtemp()
        temp_path = Path(temp_dir)
        
        # Create sample images
        for i in range(3):
            # Create synthetic fingerprint
            size = (128, 128)
            image = np.zeros(size, dtype=np.uint8)
            
            # Add pattern based on index
            pattern_offset = i * 5
            for j in range(pattern_offset, size[1], 15):
                image[:, j:j+7] = 255
            
            # Add noise
            noise = np.random.normal(0, 5, size).astype(np.uint8)
            image = cv2.add(image, noise)
            
            # Save image
            filename = f'test_fingerprint_{i+1}.png'
            cv2.imwrite(str(temp_path / filename), image)
        
        yield str(temp_path)
        
        # Cleanup
        import shutil
        shutil.rmtree(temp_dir)
    
    def test_processor_initialization(self, processor):
        """Test that processor initializes correctly."""
        assert processor is not None
        assert processor.max_workers > 0
    
    def test_process_directory(self, processor, sample_images_dir):
        """Test processing a directory of images."""
        results = processor.process_directory(sample_images_dir)
        
        assert results is not None
        assert isinstance(results, dict)
        assert 'files' in results
        assert 'results' in results
        assert 'match_matrix' in results
        assert 'statistics' in results
        
        # Check that files were found
        assert len(results['files']) > 0
        
        # Check results structure
        if results['results']:
            for result in results['results']:
                assert 'index' in result
                assert 'file_path' in result
                assert 'file_name' in result
                assert 'features' in result
                assert 'quality' in result
    
    def test_match_matrix_generation(self, processor, sample_images_dir):
        """Test match matrix generation."""
        results = processor.process_directory(sample_images_dir)
        
        if len(results['results']) > 1:
            match_matrix = np.array(results['match_matrix'])
            
            # Check matrix properties
            n = len(results['results'])
            assert match_matrix.shape == (n, n)
            
            # Check diagonal is 1.0 (self-match)
            np.testing.assert_array_equal(np.diag(match_matrix), np.ones(n))
            
            # Check symmetry
            np.testing.assert_array_almost_equal(match_matrix, match_matrix.T)
            
            # Check values are in valid range
            assert np.all(match_matrix >= 0.0)
            assert np.all(match_matrix <= 1.0)
    
    def test_statistics_calculation(self, processor, sample_images_dir):
        """Test statistics calculation."""
        results = processor.process_directory(sample_images_dir)
        
        stats = results['statistics']
        assert isinstance(stats, dict)
        
        # Check required statistics
        expected_stats = [
            'total_images',
            'average_quality',
            'quality_std',
            'average_sift_features',
            'average_orb_features',
            'average_minutiae'
        ]
        
        for stat in expected_stats:
            assert stat in stats
            assert isinstance(stats[stat], (int, float))
    
    def test_find_best_matches(self, processor, sample_images_dir):
        """Test finding best matches."""
        results = processor.process_directory(sample_images_dir)
        
        if len(results['results']) > 1 and len(results['match_matrix']) > 0:
            match_matrix = np.array(results['match_matrix'])
            best_matches = processor.find_best_matches(
                match_matrix, 
                results['results'], 
                threshold=0.1  # Low threshold for testing
            )
            
            assert isinstance(best_matches, list)
            
            # Check match structure if any found
            if best_matches:
                for match in best_matches:
                    assert 'file1' in match
                    assert 'file2' in match
                    assert 'score' in match
                    assert 'quality1' in match
                    assert 'quality2' in match
                    assert isinstance(match['score'], float)
    
    def test_empty_directory(self, processor):
        """Test processing an empty directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            results = processor.process_directory(temp_dir)
            
            assert results['files'] == []
            assert results['results'] == []
            assert results['match_matrix'] == []
    
    def test_nonexistent_directory(self, processor):
        """Test processing a non-existent directory."""
        results = processor.process_directory('/nonexistent/directory')
        
        # Should handle gracefully
        assert isinstance(results, dict)
    
    def test_custom_file_extensions(self, processor, sample_images_dir):
        """Test processing with custom file extensions."""
        results = processor.process_directory(
            sample_images_dir,
            file_extensions=['.png']
        )
        
        # Should only find PNG files
        png_files = [f for f in results['files'] if f.endswith('.png')]
        assert len(png_files) == len(results['files'])
    
    def test_image_quality_assessment(self, processor):
        """Test image quality assessment."""
        # Create high quality image
        high_quality = np.random.randint(0, 255, (256, 256), dtype=np.uint8)
        high_quality = cv2.GaussianBlur(high_quality, (3, 3), 0)
        
        quality = processor._assess_image_quality(high_quality)
        
        assert isinstance(quality, float)
        assert 0.0 <= quality <= 1.0
        
        # Create low quality (uniform) image
        low_quality = np.ones((256, 256), dtype=np.uint8) * 128
        
        low_quality_score = processor._assess_image_quality(low_quality)
        
        # High quality should have higher score than uniform image
        assert quality >= low_quality_score


if __name__ == '__main__':
    pytest.main([__file__])
