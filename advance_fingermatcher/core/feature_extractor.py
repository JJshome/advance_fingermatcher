"""
Advanced feature extraction for fingerprint matching.

This module provides comprehensive feature extraction capabilities
using traditional computer vision and deep learning approaches.
"""

import numpy as np
import cv2
from typing import Dict, Any, Tuple, List, Optional
import logging
from sklearn.feature_extraction import image
from scipy import ndimage
import mahotas

logger = logging.getLogger(__name__)


class FeatureExtractor:
    """
    Advanced feature extraction for fingerprint images.
    
    Supports multiple feature types:
    - SIFT (Scale-Invariant Feature Transform)
    - ORB (Oriented FAST and Rotated BRIEF)
    - LBP (Local Binary Patterns)
    - Gabor filters
    - Ridge patterns
    - Texture features
    """
    
    def __init__(self, gpu_enabled: bool = True):
        """
        Initialize the feature extractor.
        
        Args:
            gpu_enabled: Whether to use GPU acceleration when available
        """
        self.gpu_enabled = gpu_enabled and cv2.cuda.getCudaEnabledDeviceCount() > 0
        
        # Initialize feature detectors
        self.sift = cv2.SIFT_create()
        self.orb = cv2.ORB_create(nfeatures=500)
        
        logger.info(f"FeatureExtractor initialized (GPU: {self.gpu_enabled})")
    
    def extract_sift(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Extract SIFT features from fingerprint image.
        
        Args:
            image: Input fingerprint image
            
        Returns:
            Dictionary with SIFT keypoints and descriptors
        """
        try:
            keypoints, descriptors = self.sift.detectAndCompute(image, None)
            
            # Convert keypoints to serializable format
            kp_data = []
            for kp in keypoints:
                kp_data.append({
                    'pt': kp.pt,
                    'angle': kp.angle,
                    'size': kp.size,
                    'response': kp.response,
                    'octave': kp.octave
                })
            
            return {
                'keypoints': keypoints,
                'keypoints_data': kp_data,
                'descriptors': descriptors,
                'count': len(keypoints) if keypoints else 0
            }
        except Exception as e:
            logger.error(f"SIFT extraction error: {e}")
            return {'keypoints': [], 'descriptors': None, 'count': 0}
    
    def extract_orb(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Extract ORB features from fingerprint image.
        
        Args:
            image: Input fingerprint image
            
        Returns:
            Dictionary with ORB keypoints and descriptors
        """
        try:
            keypoints, descriptors = self.orb.detectAndCompute(image, None)
            
            # Convert keypoints to serializable format
            kp_data = []
            for kp in keypoints:
                kp_data.append({
                    'pt': kp.pt,
                    'angle': kp.angle,
                    'size': kp.size,
                    'response': kp.response,
                    'octave': kp.octave
                })
            
            return {
                'keypoints': keypoints,
                'keypoints_data': kp_data,
                'descriptors': descriptors,
                'count': len(keypoints) if keypoints else 0
            }
        except Exception as e:
            logger.error(f"ORB extraction error: {e}")
            return {'keypoints': [], 'descriptors': None, 'count': 0}
    
    def extract_texture(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Extract texture features using multiple methods.
        
        Args:
            image: Input fingerprint image
            
        Returns:
            Dictionary with various texture features
        """
        try:
            features = {}
            
            # Local Binary Pattern
            lbp = self._calculate_lbp(image)
            lbp_hist, _ = np.histogram(lbp.ravel(), bins=256, range=(0, 256))
            lbp_hist = lbp_hist.astype(float)
            lbp_hist /= (lbp_hist.sum() + 1e-7)  # Normalize
            features['lbp_histogram'] = lbp_hist
            
            # Gray Level Co-occurrence Matrix features
            glcm_features = self._calculate_glcm_features(image)
            features.update(glcm_features)
            
            # Haralick texture features
            haralick_features = self._calculate_haralick_features(image)
            features['haralick'] = haralick_features
            
            # Basic statistical features
            features['mean'] = np.mean(image)
            features['std'] = np.std(image)
            features['variance'] = np.var(image)
            features['skewness'] = self._calculate_skewness(image)
            features['kurtosis'] = self._calculate_kurtosis(image)
            
            # Combine all features into a single histogram for matching
            combined_hist = np.concatenate([
                lbp_hist,
                [features['mean'], features['std'], features['variance']]
            ])
            features['histogram'] = combined_hist
            
            return features
        except Exception as e:
            logger.error(f"Texture extraction error: {e}")
            return {'histogram': np.zeros(259)}
    
    def extract_ridge_patterns(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Extract ridge pattern features using Gabor filters.
        
        Args:
            image: Input fingerprint image
            
        Returns:
            Dictionary with ridge pattern features
        """
        try:
            features = {}
            
            # Apply Gabor filter bank
            gabor_responses = self._apply_gabor_filters(image)
            features['gabor_responses'] = gabor_responses
            
            # Calculate ridge orientation
            orientation_map = self._calculate_orientation_map(image)
            features['orientation_map'] = orientation_map
            
            # Calculate ridge frequency
            frequency_map = self._calculate_frequency_map(image, orientation_map)
            features['frequency_map'] = frequency_map
            
            # Ridge quality assessment
            quality_map = self._assess_ridge_quality(image, orientation_map)
            features['quality_map'] = quality_map
            
            # Statistical features of ridge patterns
            features['avg_orientation'] = np.mean(orientation_map)
            features['avg_frequency'] = np.mean(frequency_map)
            features['avg_quality'] = np.mean(quality_map)
            
            return features
        except Exception as e:
            logger.error(f"Ridge pattern extraction error: {e}")
            return {}
    
    def _calculate_lbp(self, image: np.ndarray, radius: int = 1, n_points: int = 8) -> np.ndarray:
        """
        Calculate Local Binary Pattern.
        
        Args:
            image: Input image
            radius: Radius of the LBP
            n_points: Number of points to consider
            
        Returns:
            LBP image
        """
        def get_pixel(img, center, x, y):
            new_value = 0
            try:
                if img[x][y] >= center:
                    new_value = 1
            except IndexError:
                pass
            return new_value
        
        lbp = np.zeros_like(image)
        for i in range(radius, image.shape[0] - radius):
            for j in range(radius, image.shape[1] - radius):
                center = image[i, j]
                val = 0
                for k in range(n_points):
                    angle = 2 * np.pi * k / n_points
                    x = int(i + radius * np.cos(angle))
                    y = int(j - radius * np.sin(angle))
                    val += get_pixel(image, center, x, y) * (2 ** k)
                lbp[i, j] = val
        
        return lbp
    
    def _calculate_glcm_features(self, image: np.ndarray) -> Dict[str, float]:
        """
        Calculate Gray Level Co-occurrence Matrix features.
        
        Args:
            image: Input image
            
        Returns:
            Dictionary with GLCM features
        """
        try:
            # Quantize image to reduce computation
            quantized = (image // 4).astype(np.uint8)
            
            # Calculate GLCM for different directions
            glcm = np.zeros((64, 64, 4), dtype=np.float32)
            
            directions = [(0, 1), (1, 0), (1, 1), (1, -1)]  # 0°, 90°, 45°, 135°
            
            for d, (dy, dx) in enumerate(directions):
                for i in range(1, quantized.shape[0] - 1):
                    for j in range(1, quantized.shape[1] - 1):
                        if 0 <= i + dy < quantized.shape[0] and 0 <= j + dx < quantized.shape[1]:
                            val1 = quantized[i, j]
                            val2 = quantized[i + dy, j + dx]
                            glcm[val1, val2, d] += 1
            
            # Normalize GLCM
            for d in range(4):
                glcm[:, :, d] /= (np.sum(glcm[:, :, d]) + 1e-7)
            
            # Calculate texture features
            features = {}
            features['contrast'] = np.mean([np.sum(np.square(np.arange(64)[:, None] - np.arange(64)) * glcm[:, :, d]) for d in range(4)])
            features['homogeneity'] = np.mean([np.sum(glcm[:, :, d] / (1 + np.square(np.arange(64)[:, None] - np.arange(64)))) for d in range(4)])
            features['energy'] = np.mean([np.sum(np.square(glcm[:, :, d])) for d in range(4)])
            features['correlation'] = np.mean([self._calculate_correlation(glcm[:, :, d]) for d in range(4)])
            
            return features
        except Exception as e:
            logger.error(f"GLCM calculation error: {e}")
            return {'contrast': 0, 'homogeneity': 0, 'energy': 0, 'correlation': 0}
    
    def _calculate_correlation(self, glcm: np.ndarray) -> float:
        """
        Calculate correlation from GLCM.
        
        Args:
            glcm: Gray Level Co-occurrence Matrix
            
        Returns:
            Correlation value
        """
        try:
            i_indices, j_indices = np.mgrid[0:glcm.shape[0], 0:glcm.shape[1]]
            
            mean_i = np.sum(i_indices * glcm)
            mean_j = np.sum(j_indices * glcm)
            
            std_i = np.sqrt(np.sum(((i_indices - mean_i) ** 2) * glcm))
            std_j = np.sqrt(np.sum(((j_indices - mean_j) ** 2) * glcm))
            
            if std_i == 0 or std_j == 0:
                return 0
            
            correlation = np.sum(((i_indices - mean_i) * (j_indices - mean_j) * glcm)) / (std_i * std_j)
            return correlation
        except Exception:
            return 0
    
    def _calculate_haralick_features(self, image: np.ndarray) -> np.ndarray:
        """
        Calculate Haralick texture features.
        
        Args:
            image: Input image
            
        Returns:
            Haralick features array
        """
        try:
            # Use mahotas library for Haralick features
            features = mahotas.features.haralick(image).mean(axis=0)
            return features
        except Exception as e:
            logger.error(f"Haralick calculation error: {e}")
            return np.zeros(13)
    
    def _calculate_skewness(self, image: np.ndarray) -> float:
        """
        Calculate skewness of image intensity distribution.
        
        Args:
            image: Input image
            
        Returns:
            Skewness value
        """
        mean_val = np.mean(image)
        std_val = np.std(image)
        if std_val == 0:
            return 0
        return np.mean(((image - mean_val) / std_val) ** 3)
    
    def _calculate_kurtosis(self, image: np.ndarray) -> float:
        """
        Calculate kurtosis of image intensity distribution.
        
        Args:
            image: Input image
            
        Returns:
            Kurtosis value
        """
        mean_val = np.mean(image)
        std_val = np.std(image)
        if std_val == 0:
            return 0
        return np.mean(((image - mean_val) / std_val) ** 4) - 3
    
    def _apply_gabor_filters(self, image: np.ndarray) -> np.ndarray:
        """
        Apply bank of Gabor filters.
        
        Args:
            image: Input image
            
        Returns:
            Gabor filter responses
        """
        responses = []
        
        # Different orientations and frequencies
        orientations = np.arange(0, np.pi, np.pi/8)
        frequencies = [0.1, 0.3, 0.5]
        
        for theta in orientations:
            for freq in frequencies:
                kernel = cv2.getGaborKernel((21, 21), 5, theta, 2*np.pi/freq, 0.5, 0, ktype=cv2.CV_32F)
                filtered = cv2.filter2D(image, cv2.CV_8UC3, kernel)
                responses.append(filtered)
        
        return np.array(responses)
    
    def _calculate_orientation_map(self, image: np.ndarray, block_size: int = 16) -> np.ndarray:
        """
        Calculate local ridge orientation map.
        
        Args:
            image: Input image
            block_size: Size of local blocks
            
        Returns:
            Orientation map
        """
        # Calculate gradients
        gy, gx = np.gradient(image.astype(np.float32))
        
        # Initialize orientation map
        h, w = image.shape
        orientation_map = np.zeros((h // block_size, w // block_size))
        
        # Calculate orientation for each block
        for i in range(0, h - block_size, block_size):
            for j in range(0, w - block_size, block_size):
                # Get local gradients
                gx_block = gx[i:i+block_size, j:j+block_size]
                gy_block = gy[i:i+block_size, j:j+block_size]
                
                # Calculate local orientation
                Gxy = np.sum(gx_block * gy_block)
                Gxx = np.sum(gx_block * gx_block)
                Gyy = np.sum(gy_block * gy_block)
                
                orientation = 0.5 * np.arctan2(2 * Gxy, Gxx - Gyy)
                orientation_map[i // block_size, j // block_size] = orientation
        
        return orientation_map
    
    def _calculate_frequency_map(self, image: np.ndarray, orientation_map: np.ndarray, block_size: int = 16) -> np.ndarray:
        """
        Calculate local ridge frequency map.
        
        Args:
            image: Input image
            orientation_map: Ridge orientation map
            block_size: Size of local blocks
            
        Returns:
            Frequency map
        """
        h, w = image.shape
        frequency_map = np.zeros((h // block_size, w // block_size))
        
        for i in range(orientation_map.shape[0]):
            for j in range(orientation_map.shape[1]):
                # Get local block
                block = image[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size]
                
                if block.size == 0:
                    continue
                
                # Project along ridge direction
                orientation = orientation_map[i, j]
                
                # Simple frequency estimation using FFT
                projection = np.mean(block, axis=0)
                fft_proj = np.abs(np.fft.fft(projection))
                
                # Find dominant frequency
                freq_idx = np.argmax(fft_proj[1:len(fft_proj)//2]) + 1
                frequency = freq_idx / len(projection)
                
                frequency_map[i, j] = frequency
        
        return frequency_map
    
    def _assess_ridge_quality(self, image: np.ndarray, orientation_map: np.ndarray, block_size: int = 16) -> np.ndarray:
        """
        Assess ridge quality for each local block.
        
        Args:
            image: Input image
            orientation_map: Ridge orientation map
            block_size: Size of local blocks
            
        Returns:
            Quality map
        """
        h, w = image.shape
        quality_map = np.zeros((h // block_size, w // block_size))
        
        for i in range(orientation_map.shape[0]):
            for j in range(orientation_map.shape[1]):
                # Get local block
                block = image[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size]
                
                if block.size == 0:
                    continue
                
                # Calculate local variance as quality measure
                quality = np.var(block.astype(np.float32))
                
                # Normalize quality
                quality = min(quality / 1000.0, 1.0)
                
                quality_map[i, j] = quality
        
        return quality_map
