"""
Image processing utilities for fingerprint analysis.

This module provides various image processing functions
specifically designed for fingerprint image enhancement and analysis.
"""

import numpy as np
import cv2
from typing import Tuple, Optional, Union
import math
import logging

logger = logging.getLogger(__name__)


def normalize_image(image: np.ndarray, target_mean: float = 100, target_std: float = 50) -> np.ndarray:
    """
    Normalize fingerprint image to desired mean and standard deviation.
    
    Args:
        image: Input grayscale image
        target_mean: Target mean value
        target_std: Target standard deviation
        
    Returns:
        Normalized image
    """
    if len(image.shape) != 2:
        raise ValueError("Input must be a grayscale image")
    
    # Calculate current statistics
    current_mean = np.mean(image)
    current_std = np.std(image)
    
    if current_std == 0:
        return np.full_like(image, target_mean, dtype=np.uint8)
    
    # Normalize
    normalized = (image - current_mean) * (target_std / current_std) + target_mean
    
    # Clip to valid range
    normalized = np.clip(normalized, 0, 255)
    
    return normalized.astype(np.uint8)


def enhance_contrast(image: np.ndarray, method: str = 'clahe') -> np.ndarray:
    """
    Enhance contrast of fingerprint image.
    
    Args:
        image: Input grayscale image
        method: Enhancement method ('clahe', 'histogram_eq', 'adaptive')
        
    Returns:
        Contrast-enhanced image
    """
    if len(image.shape) != 2:
        raise ValueError("Input must be a grayscale image")
    
    if method == 'clahe':
        # Contrast Limited Adaptive Histogram Equalization
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(image)
    
    elif method == 'histogram_eq':
        # Standard histogram equalization
        return cv2.equalizeHist(image)
    
    elif method == 'adaptive':
        # Adaptive histogram equalization
        return cv2.adaptiveThreshold(
            image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
    
    else:
        raise ValueError(f"Unknown enhancement method: {method}")


def apply_gabor_filter_bank(image: np.ndarray, 
                           num_orientations: int = 8,
                           frequency: float = 0.1,
                           sigma_x: float = 4,
                           sigma_y: float = 4) -> np.ndarray:
    """
    Apply bank of Gabor filters with different orientations.
    
    Args:
        image: Input grayscale image
        num_orientations: Number of filter orientations
        frequency: Spatial frequency of the filters
        sigma_x: Standard deviation in x direction
        sigma_y: Standard deviation in y direction
        
    Returns:
        Enhanced image (maximum response across orientations)
    """
    if len(image.shape) != 2:
        raise ValueError("Input must be a grayscale image")
    
    responses = []
    
    for i in range(num_orientations):
        theta = i * np.pi / num_orientations
        
        # Create Gabor kernel
        kernel = cv2.getGaborKernel(
            (21, 21), sigma_x, theta, 2*np.pi*frequency, 0.5, 0, ktype=cv2.CV_32F
        )
        
        # Apply filter
        filtered = cv2.filter2D(image, cv2.CV_8UC3, kernel)
        responses.append(filtered)
    
    # Take maximum response
    enhanced = np.maximum.reduce(responses)
    
    return enhanced


def calculate_ridge_orientation(image: np.ndarray, 
                               block_size: int = 16,
                               overlap: int = 8) -> np.ndarray:
    """
    Calculate ridge orientation map using gradient-based method.
    
    Args:
        image: Input grayscale image
        block_size: Size of analysis blocks
        overlap: Overlap between blocks
        
    Returns:
        Orientation map (angles in radians)
    """
    if len(image.shape) != 2:
        raise ValueError("Input must be a grayscale image")
    
    height, width = image.shape
    
    # Calculate gradients
    grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    
    # Initialize orientation map
    orientation_map = np.zeros((height, width), dtype=np.float32)
    
    # Process blocks
    step = block_size - overlap
    
    for y in range(0, height - block_size + 1, step):
        for x in range(0, width - block_size + 1, step):
            # Extract block
            block_gx = grad_x[y:y+block_size, x:x+block_size]
            block_gy = grad_y[y:y+block_size, x:x+block_size]
            
            # Calculate structure tensor components
            gxx = np.sum(block_gx * block_gx)
            gyy = np.sum(block_gy * block_gy)
            gxy = np.sum(block_gx * block_gy)
            
            # Calculate orientation
            if gxx != gyy:
                orientation = 0.5 * np.arctan2(2 * gxy, gxx - gyy)
            else:
                orientation = 0.0
            
            # Normalize to [0, π)
            if orientation < 0:
                orientation += np.pi
            
            # Fill block in orientation map
            orientation_map[y:y+block_size, x:x+block_size] = orientation
    
    return orientation_map


def calculate_ridge_frequency(image: np.ndarray,
                             orientation_map: np.ndarray,
                             block_size: int = 32) -> np.ndarray:
    """
    Calculate ridge frequency map.
    
    Args:
        image: Input grayscale image
        orientation_map: Ridge orientation map
        block_size: Size of analysis blocks
        
    Returns:
        Frequency map (cycles per pixel)
    """
    if len(image.shape) != 2:
        raise ValueError("Input must be a grayscale image")
    
    height, width = image.shape
    frequency_map = np.zeros((height, width), dtype=np.float32)
    
    for y in range(0, height - block_size + 1, block_size):
        for x in range(0, width - block_size + 1, block_size):
            # Extract block
            block = image[y:y+block_size, x:x+block_size]
            orientation = orientation_map[y + block_size//2, x + block_size//2]
            
            # Calculate frequency in this block
            frequency = _estimate_ridge_frequency_in_block(block, orientation)
            
            # Fill block in frequency map
            frequency_map[y:y+block_size, x:x+block_size] = frequency
    
    return frequency_map


def _estimate_ridge_frequency_in_block(block: np.ndarray, orientation: float) -> float:
    """
    Estimate ridge frequency in a single block.
    
    Args:
        block: Image block
        orientation: Ridge orientation in the block
        
    Returns:
        Estimated frequency (cycles per pixel)
    """
    block_size = block.shape[0]
    
    # Create a signature by projecting along the orientation
    signature = np.zeros(block_size)
    
    for i in range(block_size):
        # Calculate projection line
        x_start = block_size // 2
        y_start = i
        
        # Project along perpendicular to ridge orientation
        dx = int(round(math.cos(orientation + math.pi/2)))
        dy = int(round(math.sin(orientation + math.pi/2)))
        
        # Collect pixel values along the line
        values = []
        x, y = x_start, y_start
        
        while 0 <= x < block_size and 0 <= y < block_size:
            values.append(block[y, x])
            x += dx
            y += dy
        
        if values:
            signature[i] = np.mean(values)
    
    # Find peaks and valleys in signature
    if len(signature) < 4:
        return 0.1  # Default frequency
    
    # Smooth signature
    signature = cv2.GaussianBlur(signature.reshape(-1, 1), (1, 5), 1).flatten()
    
    # Find peaks
    peaks = []
    for i in range(1, len(signature) - 1):
        if signature[i] > signature[i-1] and signature[i] > signature[i+1]:
            peaks.append(i)
    
    # Estimate frequency from peak spacing
    if len(peaks) >= 2:
        spacings = []
        for i in range(len(peaks) - 1):
            spacings.append(peaks[i+1] - peaks[i])
        
        if spacings:
            avg_spacing = np.mean(spacings)
            frequency = 1.0 / avg_spacing if avg_spacing > 0 else 0.1
            return min(max(frequency, 0.05), 0.25)  # Clamp to reasonable range
    
    return 0.1  # Default frequency


def create_ridge_mask(image: np.ndarray, 
                     block_size: int = 16,
                     threshold: float = 0.1) -> np.ndarray:
    """
    Create a mask identifying ridge regions vs background.
    
    Args:
        image: Input grayscale image
        block_size: Size of analysis blocks
        threshold: Threshold for ridge detection
        
    Returns:
        Binary mask (255 for ridge regions, 0 for background)
    """
    if len(image.shape) != 2:
        raise ValueError("Input must be a grayscale image")
    
    height, width = image.shape
    mask = np.zeros((height, width), dtype=np.uint8)
    
    # Calculate local statistics
    for y in range(0, height - block_size + 1, block_size):
        for x in range(0, width - block_size + 1, block_size):
            block = image[y:y+block_size, x:x+block_size]
            
            # Calculate local variance
            variance = np.var(block)
            
            # High variance indicates ridge region
            if variance > threshold * 10000:  # Scale threshold
                mask[y:y+block_size, x:x+block_size] = 255
    
    # Morphological operations to clean up mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    return mask


def enhance_fingerprint_image(image: np.ndarray, 
                             enhancement_steps: Optional[list] = None) -> np.ndarray:
    """
    Apply comprehensive fingerprint image enhancement.
    
    Args:
        image: Input fingerprint image
        enhancement_steps: List of enhancement steps to apply
                          Default: ['normalize', 'contrast', 'gabor']
        
    Returns:
        Enhanced fingerprint image
    """
    if enhancement_steps is None:
        enhancement_steps = ['normalize', 'contrast', 'gabor']
    
    enhanced = image.copy()
    
    for step in enhancement_steps:
        if step == 'normalize':
            enhanced = normalize_image(enhanced)
            logger.debug("Applied normalization")
        
        elif step == 'contrast':
            enhanced = enhance_contrast(enhanced, method='clahe')
            logger.debug("Applied contrast enhancement")
        
        elif step == 'gabor':
            enhanced = apply_gabor_filter_bank(enhanced)
            logger.debug("Applied Gabor filter bank")
        
        elif step == 'bilateral':
            enhanced = cv2.bilateralFilter(enhanced, 9, 75, 75)
            logger.debug("Applied bilateral filter")
        
        elif step == 'median':
            enhanced = cv2.medianBlur(enhanced, 5)
            logger.debug("Applied median filter")
        
        else:
            logger.warning(f"Unknown enhancement step: {step}")
    
    return enhanced


def resize_image_maintain_aspect(image: np.ndarray, 
                               target_size: Tuple[int, int],
                               interpolation: int = cv2.INTER_LINEAR) -> np.ndarray:
    """
    Resize image while maintaining aspect ratio.
    
    Args:
        image: Input image
        target_size: Target size (width, height)
        interpolation: Interpolation method
        
    Returns:
        Resized image
    """
    height, width = image.shape[:2]
    target_width, target_height = target_size
    
    # Calculate scaling factor
    scale_x = target_width / width
    scale_y = target_height / height
    scale = min(scale_x, scale_y)
    
    # Calculate new dimensions
    new_width = int(width * scale)
    new_height = int(height * scale)
    
    # Resize image
    resized = cv2.resize(image, (new_width, new_height), interpolation=interpolation)
    
    # Create padded image if needed
    if new_width != target_width or new_height != target_height:
        # Calculate padding
        pad_x = (target_width - new_width) // 2
        pad_y = (target_height - new_height) // 2
        
        # Create padded image
        if len(image.shape) == 2:
            padded = np.zeros((target_height, target_width), dtype=image.dtype)
            padded[pad_y:pad_y+new_height, pad_x:pad_x+new_width] = resized
        else:
            padded = np.zeros((target_height, target_width, image.shape[2]), dtype=image.dtype)
            padded[pad_y:pad_y+new_height, pad_x:pad_x+new_width] = resized
        
        return padded
    
    return resized


def calculate_image_quality_metrics(image: np.ndarray) -> dict:
    """
    Calculate various quality metrics for fingerprint image.
    
    Args:
        image: Input grayscale image
        
    Returns:
        Dictionary of quality metrics
    """
    if len(image.shape) != 2:
        raise ValueError("Input must be a grayscale image")
    
    metrics = {}
    
    # Basic statistics
    metrics['mean'] = float(np.mean(image))
    metrics['std'] = float(np.std(image))
    metrics['variance'] = float(np.var(image))
    
    # Gradient-based sharpness
    grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    metrics['sharpness'] = float(np.mean(gradient_magnitude))
    
    # Local contrast
    kernel = np.ones((3, 3)) / 9
    local_mean = cv2.filter2D(image.astype(np.float32), -1, kernel)
    local_contrast = np.abs(image.astype(np.float32) - local_mean)
    metrics['local_contrast'] = float(np.mean(local_contrast))
    
    # Edge density
    edges = cv2.Canny(image, 50, 150)
    metrics['edge_density'] = float(np.sum(edges > 0) / edges.size)
    
    # Ridge coherence
    orientation_map = calculate_ridge_orientation(image)
    coherence = _calculate_coherence_from_orientation(orientation_map)
    metrics['ridge_coherence'] = float(coherence)
    
    return metrics


def _calculate_coherence_from_orientation(orientation_map: np.ndarray) -> float:
    """
    Calculate coherence measure from orientation map.
    
    Args:
        orientation_map: Ridge orientation map
        
    Returns:
        Coherence measure (0-1, higher is more coherent)
    """
    # Convert orientations to unit vectors
    cos_2theta = np.cos(2 * orientation_map)
    sin_2theta = np.sin(2 * orientation_map)
    
    # Calculate local coherence
    coherence = np.sqrt(np.mean(cos_2theta)**2 + np.mean(sin_2theta)**2)
    
    return coherence


if __name__ == "__main__":
    # Example usage
    # Create a sample fingerprint-like image
    sample_image = np.random.randint(0, 255, (400, 400), dtype=np.uint8)
    
    # Apply enhancement
    enhanced = enhance_fingerprint_image(sample_image)
    
    # Calculate quality metrics
    metrics = calculate_image_quality_metrics(enhanced)
    
    print("Image Quality Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")