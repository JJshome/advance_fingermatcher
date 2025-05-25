"""
Minutiae detection using deep learning and traditional methods.

This module provides advanced minutiae detection capabilities
using both traditional computer vision and deep learning approaches.
"""

import numpy as np
import cv2
from typing import List, Dict, Any, Optional, Tuple
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class MinutiaeDetector:
    """
    Advanced minutiae detection system.
    
    Combines traditional image processing techniques with deep learning
    for accurate minutiae detection in fingerprint images.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the minutiae detector.
        
        Args:
            model_path: Path to pre-trained deep learning models
        """
        self.model_path = model_path
        self.use_deep_learning = False
        
        # Try to load deep learning model if available
        if model_path and Path(model_path).exists():
            try:
                self._load_deep_model(model_path)
                self.use_deep_learning = True
                logger.info("Deep learning model loaded successfully")
            except Exception as e:
                logger.warning(f"Could not load deep learning model: {e}")
                logger.info("Falling back to traditional methods")
        
        logger.info(f"MinutiaeDetector initialized (Deep Learning: {self.use_deep_learning})")
    
    def _load_deep_model(self, model_path: str):
        """
        Load pre-trained deep learning model.
        
        Args:
            model_path: Path to the model file
        """
        # Placeholder for deep learning model loading
        # In a real implementation, this would load TensorFlow/PyTorch models
        pass
    
    def detect(self, image: np.ndarray, quality_threshold: float = 0.5) -> List[Dict[str, Any]]:
        """
        Detect minutiae points in fingerprint image.
        
        Args:
            image: Input fingerprint image
            quality_threshold: Minimum quality threshold for minutiae
            
        Returns:
            List of minutiae points with their properties
        """
        try:
            if self.use_deep_learning:
                return self._detect_deep_learning(image, quality_threshold)
            else:
                return self._detect_traditional(image, quality_threshold)
        except Exception as e:
            logger.error(f"Minutiae detection error: {e}")
            return []
    
    def _detect_deep_learning(self, image: np.ndarray, quality_threshold: float) -> List[Dict[str, Any]]:
        """
        Detect minutiae using deep learning model.
        
        Args:
            image: Input fingerprint image
            quality_threshold: Minimum quality threshold
            
        Returns:
            List of detected minutiae
        """
        # Placeholder for deep learning minutiae detection
        # In a real implementation, this would use a trained CNN model
        
        # For now, fall back to traditional method
        return self._detect_traditional(image, quality_threshold)
    
    def _detect_traditional(self, image: np.ndarray, quality_threshold: float) -> List[Dict[str, Any]]:
        """
        Detect minutiae using traditional image processing methods.
        
        Args:
            image: Input fingerprint image
            quality_threshold: Minimum quality threshold
            
        Returns:
            List of detected minutiae
        """
        # Enhance the image
        enhanced = self._enhance_image(image)
        
        # Binarize the image
        binary = self._binarize_image(enhanced)
        
        # Thin the ridges
        thinned = self._thin_ridges(binary)
        
        # Find minutiae points
        minutiae = self._find_minutiae_points(thinned)
        
        # Filter minutiae by quality
        filtered_minutiae = self._filter_minutiae_by_quality(minutiae, image, quality_threshold)
        
        # Remove spurious minutiae
        cleaned_minutiae = self._remove_spurious_minutiae(filtered_minutiae)
        
        return cleaned_minutiae
    
    def _enhance_image(self, image: np.ndarray) -> np.ndarray:
        """
        Enhance fingerprint image for better ridge structure.
        
        Args:
            image: Input image
            
        Returns:
            Enhanced image
        """
        # Apply histogram equalization
        enhanced = cv2.equalizeHist(image)
        
        # Apply Gabor filter enhancement
        gabor_enhanced = self._apply_gabor_enhancement(enhanced)
        
        # Combine original and Gabor-enhanced images
        combined = cv2.addWeighted(enhanced, 0.5, gabor_enhanced, 0.5, 0)
        
        return combined
    
    def _apply_gabor_enhancement(self, image: np.ndarray) -> np.ndarray:
        """
        Apply Gabor filter bank for ridge enhancement.
        
        Args:
            image: Input image
            
        Returns:
            Gabor-enhanced image
        """
        enhanced = np.zeros_like(image, dtype=np.float32)
        
        # Apply Gabor filters with different orientations
        for theta in np.arange(0, np.pi, np.pi/8):
            kernel = cv2.getGaborKernel((21, 21), 5, theta, 2*np.pi/3, 0.5, 0, ktype=cv2.CV_32F)
            filtered = cv2.filter2D(image, cv2.CV_32F, kernel)
            enhanced = np.maximum(enhanced, filtered)
        
        return enhanced.astype(np.uint8)
    
    def _binarize_image(self, image: np.ndarray) -> np.ndarray:
        """
        Binarize the enhanced image.
        
        Args:
            image: Enhanced input image
            
        Returns:
            Binary image
        """
        # Apply adaptive thresholding
        binary = cv2.adaptiveThreshold(
            image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        
        # Apply morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        return binary
    
    def _thin_ridges(self, binary: np.ndarray) -> np.ndarray:
        """
        Thin ridge lines to single pixel width.
        
        Args:
            binary: Binary fingerprint image
            
        Returns:
            Thinned image
        """
        # Skeletonization using morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        
        # Initialize skeleton
        skeleton = np.zeros(binary.shape, np.uint8)
        eroded = binary.copy()
        
        while True:
            # Open the image
            opened = cv2.morphologyEx(eroded, cv2.MORPH_OPEN, kernel)
            
            # Subtract opened from eroded
            temp = cv2.subtract(eroded, opened)
            
            # Add to skeleton
            skeleton = cv2.bitwise_or(skeleton, temp)
            
            # Erode the image
            eroded = cv2.erode(eroded, kernel)
            
            # If erosion results in empty image, break
            if cv2.countNonZero(eroded) == 0:
                break
        
        return skeleton
    
    def _find_minutiae_points(self, thinned: np.ndarray) -> List[Dict[str, Any]]:
        """
        Find minutiae points (ridge endings and bifurcations).
        
        Args:
            thinned: Thinned ridge image
            
        Returns:
            List of minutiae points
        """
        minutiae = []
        h, w = thinned.shape
        
        # Define 3x3 neighborhood directions
        directions = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]
        
        # Scan the image for ridge pixels
        for i in range(1, h - 1):
            for j in range(1, w - 1):
                if thinned[i, j] == 255:  # Ridge pixel
                    # Count ridge neighbors
                    neighbors = []
                    for di, dj in directions:
                        neighbors.append(1 if thinned[i + di, j + dj] == 255 else 0)
                    
                    # Count crossing number (transitions from 0 to 1)
                    crossing_number = 0
                    for k in range(8):
                        crossing_number += abs(neighbors[k] - neighbors[(k + 1) % 8])
                    crossing_number //= 2
                    
                    # Classify minutiae
                    if crossing_number == 1:
                        # Ridge ending
                        minutiae.append({
                            'x': j,
                            'y': i,
                            'type': 'ending',
                            'angle': self._calculate_ridge_direction(thinned, i, j),
                            'quality': 1.0  # Will be updated later
                        })
                    elif crossing_number == 3:
                        # Ridge bifurcation
                        minutiae.append({
                            'x': j,
                            'y': i,
                            'type': 'bifurcation',
                            'angle': self._calculate_ridge_direction(thinned, i, j),
                            'quality': 1.0  # Will be updated later
                        })
        
        return minutiae
    
    def _calculate_ridge_direction(self, thinned: np.ndarray, y: int, x: int, window_size: int = 7) -> float:
        """
        Calculate ridge direction at given point.
        
        Args:
            thinned: Thinned ridge image
            y: Y coordinate
            x: X coordinate
            window_size: Size of analysis window
            
        Returns:
            Ridge direction in degrees
        """
        half_size = window_size // 2
        
        # Extract local window
        y1, y2 = max(0, y - half_size), min(thinned.shape[0], y + half_size + 1)
        x1, x2 = max(0, x - half_size), min(thinned.shape[1], x + half_size + 1)
        
        window = thinned[y1:y2, x1:x2].astype(np.float32)
        
        if window.size == 0:
            return 0.0
        
        # Calculate gradients
        gy, gx = np.gradient(window)
        
        # Calculate dominant orientation using gradient covariance
        gx_flat = gx.flatten()
        gy_flat = gy.flatten()
        
        # Calculate covariance matrix
        Gxx = np.sum(gx_flat * gx_flat)
        Gyy = np.sum(gy_flat * gy_flat)
        Gxy = np.sum(gx_flat * gy_flat)
        
        # Calculate orientation
        if Gxx != Gyy:
            angle = 0.5 * np.arctan2(2 * Gxy, Gxx - Gyy)
        else:
            angle = 0.0
        
        # Convert to degrees and normalize
        angle_degrees = np.degrees(angle) % 360
        
        return angle_degrees
    
    def _filter_minutiae_by_quality(self, minutiae: List[Dict[str, Any]], 
                                   original_image: np.ndarray, 
                                   quality_threshold: float) -> List[Dict[str, Any]]:
        """
        Filter minutiae based on local image quality.
        
        Args:
            minutiae: List of detected minutiae
            original_image: Original fingerprint image
            quality_threshold: Minimum quality threshold
            
        Returns:
            Filtered minutiae list
        """
        filtered_minutiae = []
        
        for minutia in minutiae:
            quality = self._assess_minutia_quality(minutia, original_image)
            minutia['quality'] = quality
            
            if quality >= quality_threshold:
                filtered_minutiae.append(minutia)
        
        return filtered_minutiae
    
    def _assess_minutia_quality(self, minutia: Dict[str, Any], image: np.ndarray, window_size: int = 15) -> float:
        """
        Assess the quality of a minutia point.
        
        Args:
            minutia: Minutia point information
            image: Original image
            window_size: Size of analysis window
            
        Returns:
            Quality score (0.0 to 1.0)
        """
        x, y = minutia['x'], minutia['y']
        half_size = window_size // 2
        
        # Extract local window
        y1, y2 = max(0, y - half_size), min(image.shape[0], y + half_size + 1)
        x1, x2 = max(0, x - half_size), min(image.shape[1], x + half_size + 1)
        
        window = image[y1:y2, x1:x2].astype(np.float32)
        
        if window.size == 0:
            return 0.0
        
        # Calculate local variance as quality measure
        variance = np.var(window)
        
        # Calculate local gradient magnitude
        gy, gx = np.gradient(window)
        gradient_magnitude = np.sqrt(gx**2 + gy**2)
        avg_gradient = np.mean(gradient_magnitude)
        
        # Combine variance and gradient for quality score
        quality = min((variance / 1000.0) * (avg_gradient / 50.0), 1.0)
        
        return max(0.0, quality)
    
    def _remove_spurious_minutiae(self, minutiae: List[Dict[str, Any]], 
                                 min_distance: int = 10) -> List[Dict[str, Any]]:
        """
        Remove spurious minutiae that are too close to each other.
        
        Args:
            minutiae: List of minutiae
            min_distance: Minimum distance between minutiae
            
        Returns:
            Cleaned minutiae list
        """
        if len(minutiae) <= 1:
            return minutiae
        
        # Sort minutiae by quality (descending)
        sorted_minutiae = sorted(minutiae, key=lambda m: m['quality'], reverse=True)
        
        cleaned_minutiae = []
        
        for minutia in sorted_minutiae:
            # Check if this minutia is too close to any already accepted minutia
            too_close = False
            
            for accepted in cleaned_minutiae:
                distance = np.sqrt((minutia['x'] - accepted['x'])**2 + 
                                 (minutia['y'] - accepted['y'])**2)
                
                if distance < min_distance:
                    too_close = True
                    break
            
            if not too_close:
                cleaned_minutiae.append(minutia)
        
        return cleaned_minutiae
    
    def visualize_minutiae(self, image: np.ndarray, minutiae: List[Dict[str, Any]]) -> np.ndarray:
        """
        Visualize detected minutiae on the fingerprint image.
        
        Args:
            image: Original fingerprint image
            minutiae: List of detected minutiae
            
        Returns:
            Image with minutiae overlaid
        """
        # Convert to color image
        if len(image.shape) == 2:
            vis_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            vis_image = image.copy()
        
        # Draw minutiae points
        for minutia in minutiae:
            x, y = int(minutia['x']), int(minutia['y'])
            
            # Color based on type
            if minutia['type'] == 'ending':
                color = (0, 255, 0)  # Green for endings
                marker = cv2.MARKER_CROSS
            else:  # bifurcation
                color = (0, 0, 255)  # Red for bifurcations
                marker = cv2.MARKER_TRIANGLE_UP
            
            # Draw marker
            cv2.drawMarker(vis_image, (x, y), color, marker, 10, 2)
            
            # Draw orientation line
            angle_rad = np.radians(minutia['angle'])
            end_x = int(x + 15 * np.cos(angle_rad))
            end_y = int(y + 15 * np.sin(angle_rad))
            cv2.line(vis_image, (x, y), (end_x, end_y), color, 1)
        
        return vis_image
