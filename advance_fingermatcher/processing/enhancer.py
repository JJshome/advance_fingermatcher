"""
Advanced fingerprint image enhancement.

This module provides sophisticated ridge enhancement techniques
using Gabor filters, directional filtering, and frequency domain methods.
"""

import numpy as np
import cv2
from typing import Tuple, Optional
import logging
from scipy import ndimage, signal
from skimage import filters, morphology

logger = logging.getLogger(__name__)


class ImageEnhancer:
    """
    Advanced fingerprint image enhancement system.
    
    Provides sophisticated enhancement techniques including:
    - Gabor filter enhancement
    - Contextual filtering
    - Ridge-valley structure enhancement
    - Frequency domain enhancement
    """
    
    def __init__(self):
        """
        Initialize the image enhancer.
        """
        logger.info("ImageEnhancer initialized")
    
    def enhance_ridges(self, image: np.ndarray, method: str = 'gabor') -> np.ndarray:
        """
        Enhance ridge structures in fingerprint image.
        
        Args:
            image: Input fingerprint image
            method: Enhancement method ('gabor', 'contextual', 'frequency', 'hybrid')
            
        Returns:
            Enhanced image
        """
        try:
            if method == 'gabor':
                return self._gabor_enhancement(image)
            elif method == 'contextual':
                return self._contextual_enhancement(image)
            elif method == 'frequency':
                return self._frequency_enhancement(image)
            elif method == 'hybrid':
                return self._hybrid_enhancement(image)
            else:
                logger.warning(f"Unknown enhancement method: {method}")
                return image
        except Exception as e:
            logger.error(f"Ridge enhancement error: {e}")
            return image
    
    def _gabor_enhancement(self, image: np.ndarray) -> np.ndarray:
        """
        Enhance image using Gabor filter bank.
        
        Args:
            image: Input image
            
        Returns:
            Gabor-enhanced image
        """
        # Calculate local ridge orientation
        orientation_map = self._calculate_orientation_field(image)
        
        # Calculate local ridge frequency
        frequency_map = self._calculate_frequency_field(image, orientation_map)
        
        # Apply oriented Gabor filters
        enhanced = self._apply_oriented_gabor(image, orientation_map, frequency_map)
        
        return enhanced
    
    def _contextual_enhancement(self, image: np.ndarray) -> np.ndarray:
        """
        Enhance image using contextual filtering.
        
        Args:
            image: Input image
            
        Returns:
            Contextually enhanced image
        """
        # Calculate local statistics
        mean_map = self._calculate_local_mean(image)
        std_map = self._calculate_local_std(image)
        
        # Apply contextual enhancement
        enhanced = np.zeros_like(image, dtype=np.float32)
        
        # Target statistics
        target_mean = 128.0
        target_std = 50.0
        
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                local_mean = mean_map[i, j]
                local_std = std_map[i, j]
                
                if local_std > 10:  # Avoid division by zero
                    # Normalize and scale
                    normalized = (image[i, j] - local_mean) / local_std
                    enhanced[i, j] = normalized * target_std + target_mean
                else:
                    enhanced[i, j] = target_mean
        
        # Clip to valid range
        enhanced = np.clip(enhanced, 0, 255)
        
        return enhanced.astype(np.uint8)
    
    def _frequency_enhancement(self, image: np.ndarray) -> np.ndarray:
        """
        Enhance image in frequency domain.
        
        Args:
            image: Input image
            
        Returns:
            Frequency-enhanced image
        """
        # Apply FFT
        f_transform = np.fft.fft2(image)
        f_shift = np.fft.fftshift(f_transform)
        
        # Create enhancement filter
        rows, cols = image.shape
        crow, ccol = rows // 2, cols // 2
        
        # Create bandpass filter for ridge frequencies
        mask = np.zeros((rows, cols), np.uint8)
        r_outer = 30
        r_inner = 10
        
        y, x = np.ogrid[:rows, :cols]
        mask_area = (x - ccol) ** 2 + (y - crow) ** 2
        mask[(mask_area <= r_outer**2) & (mask_area >= r_inner**2)] = 1
        
        # Apply filter
        f_shift_filtered = f_shift * mask
        
        # Inverse FFT
        f_ishift = np.fft.ifftshift(f_shift_filtered)
        enhanced = np.fft.ifft2(f_ishift)
        enhanced = np.abs(enhanced)
        
        # Normalize
        enhanced = (enhanced - np.min(enhanced)) / (np.max(enhanced) - np.min(enhanced)) * 255
        
        return enhanced.astype(np.uint8)
    
    def _hybrid_enhancement(self, image: np.ndarray) -> np.ndarray:
        """
        Hybrid enhancement combining multiple methods.
        
        Args:
            image: Input image
            
        Returns:
            Hybrid-enhanced image
        """
        # Apply different enhancement methods
        gabor_enhanced = self._gabor_enhancement(image)
        contextual_enhanced = self._contextual_enhancement(image)
        
        # Combine results
        combined = cv2.addWeighted(gabor_enhanced, 0.6, contextual_enhanced, 0.4, 0)
        
        return combined
    
    def _calculate_orientation_field(self, image: np.ndarray, block_size: int = 16) -> np.ndarray:
        """
        Calculate local ridge orientation field.
        
        Args:
            image: Input image
            block_size: Size of local blocks
            
        Returns:
            Orientation field
        """
        # Calculate gradients
        gy, gx = np.gradient(image.astype(np.float32))
        
        # Initialize orientation field
        h, w = image.shape
        orientation_field = np.zeros((h // block_size, w // block_size))
        
        # Calculate orientation for each block
        for i in range(0, h - block_size, block_size):
            for j in range(0, w - block_size, block_size):
                # Get local gradients
                gx_block = gx[i:i+block_size, j:j+block_size]
                gy_block = gy[i:i+block_size, j:j+block_size]
                
                # Calculate local orientation
                Gxy = np.sum(2 * gx_block * gy_block)
                Gxx = np.sum(gx_block**2 - gy_block**2)
                
                if Gxx != 0 or Gxy != 0:
                    orientation = 0.5 * np.arctan2(Gxy, Gxx)
                else:
                    orientation = 0
                
                orientation_field[i // block_size, j // block_size] = orientation
        
        # Smooth orientation field
        orientation_field = ndimage.gaussian_filter(orientation_field, sigma=1.0)
        
        return orientation_field
    
    def _calculate_frequency_field(self, image: np.ndarray, orientation_field: np.ndarray, 
                                 block_size: int = 16) -> np.ndarray:
        """
        Calculate local ridge frequency field.
        
        Args:
            image: Input image
            orientation_field: Ridge orientation field
            block_size: Size of local blocks
            
        Returns:
            Frequency field
        """
        h, w = image.shape
        frequency_field = np.zeros((h // block_size, w // block_size))
        
        for i in range(orientation_field.shape[0]):
            for j in range(orientation_field.shape[1]):
                # Get local block
                block = image[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size]
                
                if block.size == 0:
                    continue
                
                # Get local orientation
                orientation = orientation_field[i, j]
                
                # Create projection along ridge direction
                cos_o = np.cos(orientation)
                sin_o = np.sin(orientation)
                
                # Project block along orientation
                projection = []
                for k in range(block_size):
                    line_sum = 0
                    count = 0
                    for l in range(block_size):
                        # Calculate rotated coordinates
                        x_rot = int(l * cos_o + k * sin_o)
                        y_rot = int(-l * sin_o + k * cos_o)
                        
                        if 0 <= x_rot < block_size and 0 <= y_rot < block_size:
                            line_sum += block[y_rot, x_rot]
                            count += 1
                    
                    if count > 0:
                        projection.append(line_sum / count)
                
                if len(projection) > 2:
                    # Find peaks in projection to estimate frequency
                    projection = np.array(projection)
                    peaks = self._find_peaks(projection)
                    
                    if len(peaks) > 1:
                        # Calculate average distance between peaks
                        peak_distances = np.diff(peaks)
                        avg_distance = np.mean(peak_distances)
                        frequency = 1.0 / avg_distance if avg_distance > 0 else 0.1
                    else:
                        frequency = 0.1  # Default frequency
                else:
                    frequency = 0.1
                
                frequency_field[i, j] = frequency
        
        # Smooth frequency field
        frequency_field = ndimage.gaussian_filter(frequency_field, sigma=1.0)
        
        return frequency_field
    
    def _find_peaks(self, signal_1d: np.ndarray) -> np.ndarray:
        """
        Find peaks in 1D signal.
        
        Args:
            signal_1d: Input 1D signal
            
        Returns:
            Peak indices
        """
        peaks = []
        for i in range(1, len(signal_1d) - 1):
            if signal_1d[i] > signal_1d[i-1] and signal_1d[i] > signal_1d[i+1]:
                peaks.append(i)
        return np.array(peaks)
    
    def _apply_oriented_gabor(self, image: np.ndarray, orientation_field: np.ndarray, 
                            frequency_field: np.ndarray, block_size: int = 16) -> np.ndarray:
        """
        Apply oriented Gabor filters based on local orientation and frequency.
        
        Args:
            image: Input image
            orientation_field: Local orientation field
            frequency_field: Local frequency field
            block_size: Size of local blocks
            
        Returns:
            Gabor-filtered image
        """
        enhanced = np.zeros_like(image, dtype=np.float32)
        h, w = image.shape
        
        for i in range(orientation_field.shape[0]):
            for j in range(orientation_field.shape[1]):
                # Get local parameters
                orientation = orientation_field[i, j]
                frequency = frequency_field[i, j]
                
                # Create Gabor kernel
                kernel_size = min(block_size, 21)
                if kernel_size % 2 == 0:
                    kernel_size += 1
                
                gabor_kernel = cv2.getGaborKernel(
                    (kernel_size, kernel_size),
                    sigma=4,
                    theta=orientation,
                    lambd=1.0/frequency if frequency > 0 else 10,
                    gamma=0.5,
                    psi=0,
                    ktype=cv2.CV_32F
                )
                
                # Apply kernel to local block
                start_i = i * block_size
                end_i = min((i + 1) * block_size, h)
                start_j = j * block_size
                end_j = min((j + 1) * block_size, w)
                
                block = image[start_i:end_i, start_j:end_j].astype(np.float32)
                
                # Apply Gabor filter
                filtered_block = cv2.filter2D(block, cv2.CV_32F, gabor_kernel)
                
                # Store result
                enhanced[start_i:end_i, start_j:end_j] = filtered_block
        
        # Normalize result
        enhanced = np.clip(enhanced, 0, 255)
        
        return enhanced.astype(np.uint8)
    
    def _calculate_local_mean(self, image: np.ndarray, window_size: int = 15) -> np.ndarray:
        """
        Calculate local mean using sliding window.
        
        Args:
            image: Input image
            window_size: Size of sliding window
            
        Returns:
            Local mean map
        """
        kernel = np.ones((window_size, window_size)) / (window_size ** 2)
        return cv2.filter2D(image.astype(np.float32), -1, kernel)
    
    def _calculate_local_std(self, image: np.ndarray, window_size: int = 15) -> np.ndarray:
        """
        Calculate local standard deviation using sliding window.
        
        Args:
            image: Input image
            window_size: Size of sliding window
            
        Returns:
            Local standard deviation map
        """
        # Calculate local mean
        local_mean = self._calculate_local_mean(image, window_size)
        
        # Calculate local variance
        image_sq = image.astype(np.float32) ** 2
        kernel = np.ones((window_size, window_size)) / (window_size ** 2)
        local_mean_sq = cv2.filter2D(image_sq, -1, kernel)
        local_variance = local_mean_sq - local_mean ** 2
        
        # Ensure non-negative variance
        local_variance = np.maximum(local_variance, 0)
        
        # Calculate standard deviation
        local_std = np.sqrt(local_variance)
        
        return local_std
    
    def enhance_for_minutiae(self, image: np.ndarray) -> np.ndarray:
        """
        Enhance image specifically for minutiae detection.
        
        Args:
            image: Input fingerprint image
            
        Returns:
            Enhanced image optimized for minutiae detection
        """
        try:
            # Step 1: Ridge enhancement
            ridge_enhanced = self.enhance_ridges(image, method='gabor')
            
            # Step 2: Contrast enhancement
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            contrast_enhanced = clahe.apply(ridge_enhanced)
            
            # Step 3: Morphological enhancement
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            morph_enhanced = cv2.morphologyEx(contrast_enhanced, cv2.MORPH_CLOSE, kernel)
            
            return morph_enhanced
        except Exception as e:
            logger.error(f"Minutiae enhancement error: {e}")
            return image
