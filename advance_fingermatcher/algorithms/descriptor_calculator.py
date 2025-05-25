"""
Descriptor calculation for enhanced minutiae representation.

This module provides various methods to calculate rich descriptors
for minutiae points, enhancing their discriminative power.
"""

import numpy as np
import cv2
from typing import Tuple, Optional, Dict, Any
from scipy import ndimage
from skimage.feature import local_binary_pattern
import logging

logger = logging.getLogger(__name__)


class MinutiaeDescriptorCalculator:
    """
    Calculator for various types of minutiae descriptors.
    
    This class provides methods to calculate rich descriptors around
    minutiae points, including LBP, HOG-like features, and ridge-based features.
    """
    
    def __init__(self, patch_size: int = 32):
        """
        Initialize the descriptor calculator.

        Args:
            patch_size: Size of the patch around each minutia for descriptor calculation
        """
        self.patch_size = patch_size
        self.half_size = patch_size // 2
        
        # LBP parameters
        self.lbp_radius = 3
        self.lbp_points = 8 * self.lbp_radius
        
        logger.info(f"MinutiaeDescriptorCalculator initialized with patch_size={patch_size}")
    
    def extract_patch(self, 
                     image: np.ndarray, 
                     x: float, 
                     y: float, 
                     orientation: float = 0.0) -> Optional[np.ndarray]:
        """
        Extract a patch around a minutia point.
        
        Args:
            image: Input fingerprint image
            x, y: Minutia coordinates
            orientation: Minutia orientation for rotation normalization
            
        Returns:
            Extracted patch or None if extraction fails
        """
        try:
            h, w = image.shape
            
            # Check bounds
            if (x - self.half_size < 0 or x + self.half_size >= w or
                y - self.half_size < 0 or y + self.half_size >= h):
                logger.debug(f"Patch extraction failed: coordinates ({x}, {y}) out of bounds")
                return None
            
            # Extract basic patch
            patch = image[int(y - self.half_size):int(y + self.half_size),
                         int(x - self.half_size):int(x + self.half_size)]
            
            # Normalize orientation if specified
            if orientation != 0.0:
                patch = self._rotate_patch(patch, -orientation)  # Rotate to canonical orientation
            
            return patch.astype(np.float32)
            
        except Exception as e:
            logger.error(f"Error extracting patch at ({x}, {y}): {e}")
            return None
    
    def _rotate_patch(self, patch: np.ndarray, angle: float) -> np.ndarray:
        """
        Rotate a patch by the given angle.
        
        Args:
            patch: Input patch
            angle: Rotation angle in radians
            
        Returns:
            Rotated patch
        """
        center = (patch.shape[1] // 2, patch.shape[0] // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, np.degrees(angle), 1.0)
        
        rotated = cv2.warpAffine(patch, rotation_matrix, patch.shape[::-1], 
                                flags=cv2.INTER_LINEAR, 
                                borderMode=cv2.BORDER_REFLECT)
        return rotated
    
    def calculate_lbp_descriptor(self, 
                               image: np.ndarray, 
                               x: float, 
                               y: float, 
                               orientation: float = 0.0) -> Optional[np.ndarray]:
        """
        Calculate Local Binary Pattern descriptor around a minutia.
        
        Args:
            image: Input fingerprint image
            x, y: Minutia coordinates
            orientation: Minutia orientation
            
        Returns:
            LBP histogram descriptor (normalized)
        """
        patch = self.extract_patch(image, x, y, orientation)
        
        if patch is None:
            return None
        
        try:
            # Calculate LBP
            lbp = local_binary_pattern(patch, self.lbp_points, self.lbp_radius, method='uniform')
            
            # Calculate histogram
            hist, _ = np.histogram(lbp.ravel(), bins=self.lbp_points + 2, 
                                 range=(0, self.lbp_points + 2))
            
            # Normalize histogram
            hist = hist.astype(np.float32)
            hist_sum = np.sum(hist)
            if hist_sum > 0:
                hist = hist / hist_sum
            
            return hist
            
        except Exception as e:
            logger.error(f"Error calculating LBP descriptor: {e}")
            return None
    
    def calculate_gradient_descriptor(self, 
                                    image: np.ndarray, 
                                    x: float, 
                                    y: float, 
                                    orientation: float = 0.0,
                                    n_bins: int = 8) -> Optional[np.ndarray]:
        """
        Calculate gradient-based descriptor (HOG-like) around a minutia.
        
        Args:
            image: Input fingerprint image
            x, y: Minutia coordinates
            orientation: Minutia orientation
            n_bins: Number of orientation bins
            
        Returns:
            Gradient histogram descriptor
        """
        patch = self.extract_patch(image, x, y, orientation)
        
        if patch is None:
            return None
        
        try:
            # Calculate gradients
            gy, gx = np.gradient(patch.astype(np.float32))
            
            # Calculate magnitude and angle
            magnitude = np.sqrt(gx**2 + gy**2)
            angle = np.arctan2(gy, gx)
            
            # Convert angles to [0, 2*pi] range
            angle = (angle + 2 * np.pi) % (2 * np.pi)
            
            # Create histogram
            hist = np.zeros(n_bins)
            bin_size = 2 * np.pi / n_bins
            
            for i in range(patch.shape[0]):
                for j in range(patch.shape[1]):
                    if magnitude[i, j] > 0:  # Only consider significant gradients
                        bin_idx = int(angle[i, j] / bin_size) % n_bins
                        hist[bin_idx] += magnitude[i, j]
            
            # Normalize histogram
            hist_sum = np.sum(hist)
            if hist_sum > 0:
                hist = hist / hist_sum
            
            return hist.astype(np.float32)
            
        except Exception as e:
            logger.error(f"Error calculating gradient descriptor: {e}")
            return None
    
    def calculate_ridge_descriptor(self, 
                                 image: np.ndarray, 
                                 x: float, 
                                 y: float, 
                                 orientation: float = 0.0) -> Optional[np.ndarray]:
        """
        Calculate ridge-pattern based descriptor around a minutia.
        
        Args:
            image: Input fingerprint image
            x, y: Minutia coordinates
            orientation: Minutia orientation
            
        Returns:
            Ridge pattern descriptor
        """
        patch = self.extract_patch(image, x, y, orientation)
        
        if patch is None:
            return None
        
        try:
            features = []
            
            # 1. Ridge density in different regions
            # Divide patch into 4 quadrants
            h, w = patch.shape
            mid_h, mid_w = h // 2, w // 2
            
            quadrants = [
                patch[:mid_h, :mid_w],      # Top-left
                patch[:mid_h, mid_w:],      # Top-right
                patch[mid_h:, :mid_w],      # Bottom-left
                patch[mid_h:, mid_w:]       # Bottom-right
            ]
            
            for quad in quadrants:
                # Ridge density (proportion of pixels above threshold)
                threshold = np.mean(quad)
                ridge_density = np.sum(quad > threshold) / quad.size
                features.append(ridge_density)
            
            # 2. Ridge orientation consistency
            gy, gx = np.gradient(patch)
            local_orientations = np.arctan2(gy, gx)
            
            # Calculate orientation variance (lower = more consistent)
            # Use circular statistics for angles
            cos_sum = np.sum(np.cos(2 * local_orientations))
            sin_sum = np.sum(np.sin(2 * local_orientations))
            orientation_consistency = np.sqrt(cos_sum**2 + sin_sum**2) / local_orientations.size
            features.append(orientation_consistency)
            
            # 3. Ridge frequency estimation
            # Project along the dominant orientation
            center_line = patch[mid_h, :]
            
            # Simple frequency estimation using zero crossings
            mean_val = np.mean(center_line)
            crossings = np.sum(np.diff(np.signbit(center_line - mean_val)))
            ridge_frequency = crossings / len(center_line)
            features.append(ridge_frequency)
            
            # 4. Local contrast
            local_contrast = np.std(patch) / (np.mean(patch) + 1e-6)
            features.append(local_contrast)
            
            # 5. Ridge curvature (simplified)
            # Calculate second derivatives
            gyy, gyx = np.gradient(gy)
            gxy, gxx = np.gradient(gx)
            
            # Mean curvature approximation
            curvature = np.mean(np.abs(gxx + gyy))
            features.append(curvature)
            
            return np.array(features, dtype=np.float32)
            
        except Exception as e:
            logger.error(f"Error calculating ridge descriptor: {e}")
            return None
    
    def calculate_combined_descriptor(self, 
                                    image: np.ndarray, 
                                    x: float, 
                                    y: float, 
                                    orientation: float = 0.0) -> Optional[np.ndarray]:
        """
        Calculate combined descriptor using multiple methods.
        
        Args:
            image: Input fingerprint image
            x, y: Minutia coordinates
            orientation: Minutia orientation
            
        Returns:
            Combined descriptor vector
        """
        try:
            descriptors = []
            
            # 1. LBP descriptor (texture)
            lbp_desc = self.calculate_lbp_descriptor(image, x, y, orientation)
            if lbp_desc is not None:
                descriptors.append(lbp_desc)
            
            # 2. Gradient descriptor (shape)
            grad_desc = self.calculate_gradient_descriptor(image, x, y, orientation)
            if grad_desc is not None:
                descriptors.append(grad_desc)
            
            # 3. Ridge descriptor (domain-specific)
            ridge_desc = self.calculate_ridge_descriptor(image, x, y, orientation)
            if ridge_desc is not None:
                descriptors.append(ridge_desc)
            
            if not descriptors:
                logger.warning(f"No descriptors calculated for minutia at ({x}, {y})")
                return None
            
            # Combine all descriptors
            combined = np.concatenate(descriptors)
            
            # Normalize the combined descriptor
            norm = np.linalg.norm(combined)
            if norm > 0:
                combined = combined / norm
            
            return combined
            
        except Exception as e:
            logger.error(f"Error calculating combined descriptor: {e}")
            return None
    
    def calculate_quality_metrics(self, 
                                image: np.ndarray, 
                                x: float, 
                                y: float) -> Dict[str, float]:
        """
        Calculate quality metrics for a minutia point.
        
        Args:
            image: Input fingerprint image
            x, y: Minutia coordinates
            
        Returns:
            Dictionary of quality metrics
        """
        patch = self.extract_patch(image, x, y)
        
        if patch is None:
            return {'overall_quality': 0.0}
        
        try:
            metrics = {}
            
            # 1. Local contrast
            metrics['contrast'] = np.std(patch) / (np.mean(patch) + 1e-6)
            
            # 2. Edge strength
            gy, gx = np.gradient(patch)
            edge_strength = np.mean(np.sqrt(gx**2 + gy**2))
            metrics['edge_strength'] = edge_strength
            
            # 3. Ridge clarity (using Gabor response)
            # Simple Gabor-like filter
            kernel_size = min(15, self.patch_size // 2)
            if kernel_size % 2 == 0:
                kernel_size += 1
            
            gabor_kernel = cv2.getGaborKernel((kernel_size, kernel_size), 3, 0, 
                                            2*np.pi/3, 0.5, 0, ktype=cv2.CV_32F)
            gabor_response = cv2.filter2D(patch, cv2.CV_32F, gabor_kernel)
            ridge_clarity = np.mean(np.abs(gabor_response))
            metrics['ridge_clarity'] = ridge_clarity
            
            # 4. Noise level (high frequency content)
            # Apply high-pass filter
            blurred = cv2.GaussianBlur(patch, (5, 5), 1.0)
            high_freq = patch - blurred
            noise_level = np.std(high_freq)
            metrics['noise_level'] = noise_level
            
            # 5. Overall quality (weighted combination)
            # Normalize metrics to [0, 1] range and combine
            contrast_norm = min(metrics['contrast'] / 50.0, 1.0)
            edge_norm = min(edge_strength / 100.0, 1.0)
            clarity_norm = min(ridge_clarity / 50.0, 1.0)
            noise_norm = max(0.0, 1.0 - noise_level / 30.0)  # Lower noise = higher quality
            
            overall_quality = (
                0.3 * contrast_norm +
                0.3 * edge_norm +
                0.3 * clarity_norm +
                0.1 * noise_norm
            )
            
            metrics['overall_quality'] = overall_quality
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating quality metrics: {e}")
            return {'overall_quality': 0.0}


# Integration with Enhanced Minutia class
def enhance_minutia_with_descriptors(image: np.ndarray, 
                                   minutiae: list,
                                   calculator: Optional[MinutiaeDescriptorCalculator] = None) -> list:
    """
    Enhance a list of basic minutiae with rich descriptors.
    
    Args:
        image: Input fingerprint image
        minutiae: List of basic minutiae (with x, y, theta, quality attributes)
        calculator: Descriptor calculator instance
        
    Returns:
        List of EnhancedMinutia objects with descriptors
    """
    if calculator is None:
        calculator = MinutiaeDescriptorCalculator()
    
    enhanced_minutiae = []
    
    for minutia in minutiae:
        try:
            # Extract basic attributes
            x = getattr(minutia, 'x', 0)
            y = getattr(minutia, 'y', 0)
            theta = getattr(minutia, 'theta', 0)
            quality = getattr(minutia, 'quality', 0.5)
            minutia_type = getattr(minutia, 'type', 'unknown')
            
            # Calculate descriptor
            descriptor = calculator.calculate_combined_descriptor(image, x, y, theta)
            
            # Calculate quality metrics
            quality_metrics = calculator.calculate_quality_metrics(image, x, y)
            
            # Update quality with calculated metrics
            calculated_quality = quality_metrics.get('overall_quality', quality)
            final_quality = (quality + calculated_quality) / 2.0  # Average with original
            
            # Create enhanced minutia
            from .enhanced_bozorth3 import EnhancedMinutia
            
            enhanced_minutia = EnhancedMinutia(
                x=x, y=y, theta=theta, quality=final_quality,
                descriptor=descriptor, minutia_type=minutia_type,
                local_features=quality_metrics
            )
            
            enhanced_minutiae.append(enhanced_minutia)
            
        except Exception as e:
            logger.error(f"Error enhancing minutia: {e}")
            continue
    
    logger.info(f"Enhanced {len(enhanced_minutiae)} minutiae with descriptors")
    return enhanced_minutiae


if __name__ == "__main__":
    # Example usage and testing
    logging.basicConfig(level=logging.INFO)
    
    print("Minutiae Descriptor Calculator Demo")
    print("=" * 40)
    
    # Create a sample fingerprint-like image
    print("Creating sample image...")
    image_size = (256, 256)
    sample_image = np.zeros(image_size, dtype=np.uint8)
    
    # Add some ridge-like patterns
    for i in range(0, image_size[1], 12):
        sample_image[:, i:i+6] = 255
    
    # Add some noise
    noise = np.random.normal(0, 10, image_size).astype(np.int16)
    sample_image = np.clip(sample_image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    # Apply some blur
    sample_image = cv2.GaussianBlur(sample_image, (3, 3), 0)
    
    # Initialize calculator
    calculator = MinutiaeDescriptorCalculator(patch_size=32)
    
    # Test descriptor calculation at various points
    test_points = [
        (64, 64, 0.0),
        (128, 128, np.pi/4),
        (192, 64, np.pi/2)
    ]
    
    print(f"\nTesting descriptor calculation at {len(test_points)} points...")
    
    for i, (x, y, orientation) in enumerate(test_points):
        print(f"\nPoint {i+1}: ({x}, {y}), orientation={orientation:.2f}")
        
        # Calculate different types of descriptors
        lbp_desc = calculator.calculate_lbp_descriptor(sample_image, x, y, orientation)
        grad_desc = calculator.calculate_gradient_descriptor(sample_image, x, y, orientation)
        ridge_desc = calculator.calculate_ridge_descriptor(sample_image, x, y, orientation)
        combined_desc = calculator.calculate_combined_descriptor(sample_image, x, y, orientation)
        
        # Calculate quality metrics
        quality_metrics = calculator.calculate_quality_metrics(sample_image, x, y)
        
        print(f"  LBP descriptor: {len(lbp_desc) if lbp_desc is not None else 0} dimensions")
        print(f"  Gradient descriptor: {len(grad_desc) if grad_desc is not None else 0} dimensions")
        print(f"  Ridge descriptor: {len(ridge_desc) if ridge_desc is not None else 0} dimensions")
        print(f"  Combined descriptor: {len(combined_desc) if combined_desc is not None else 0} dimensions")
        print(f"  Overall quality: {quality_metrics.get('overall_quality', 0):.3f}")
        print(f"  Contrast: {quality_metrics.get('contrast', 0):.3f}")
        print(f"  Ridge clarity: {quality_metrics.get('ridge_clarity', 0):.3f}")
    
    print("\nDemo completed successfully!")
