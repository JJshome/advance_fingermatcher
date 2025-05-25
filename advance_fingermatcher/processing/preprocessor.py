"""
Image preprocessing for fingerprint matching.

This module provides comprehensive preprocessing capabilities
for fingerprint images including normalization, noise removal,
and basic enhancement operations.
"""

import numpy as np
import cv2
from typing import Tuple, Optional
import logging
from scipy import ndimage
from skimage import restoration, filters

logger = logging.getLogger(__name__)


class ImagePreprocessor:
    """
    Comprehensive image preprocessing for fingerprint images.
    
    Provides various preprocessing operations including:
    - Image normalization
    - Noise removal
    - Contrast enhancement
    - Size standardization
    - Quality assessment
    """
    
    def __init__(self):
        """
        Initialize the image preprocessor.
        """
        logger.info("ImagePreprocessor initialized")
    
    def normalize(self, image: np.ndarray, target_mean: float = 128.0, target_std: float = 50.0) -> np.ndarray:
        """
        Normalize image to have specified mean and standard deviation.
        
        Args:
            image: Input image
            target_mean: Target mean value
            target_std: Target standard deviation
            
        Returns:
            Normalized image
        """
        try:
            # Convert to float
            image_float = image.astype(np.float32)
            
            # Calculate current statistics
            current_mean = np.mean(image_float)
            current_std = np.std(image_float)
            
            # Avoid division by zero
            if current_std == 0:
                return np.full_like(image, target_mean, dtype=np.uint8)
            
            # Normalize
            normalized = (image_float - current_mean) / current_std
            normalized = normalized * target_std + target_mean
            
            # Clip to valid range
            normalized = np.clip(normalized, 0, 255)
            
            return normalized.astype(np.uint8)
        except Exception as e:
            logger.error(f"Normalization error: {e}")
            return image
    
    def remove_noise(self, image: np.ndarray, method: str = 'gaussian') -> np.ndarray:
        """
        Remove noise from fingerprint image.
        
        Args:
            image: Input image
            method: Noise removal method ('gaussian', 'median', 'bilateral', 'wiener')
            
        Returns:
            Denoised image
        """
        try:
            if method == 'gaussian':
                return cv2.GaussianBlur(image, (3, 3), 0)
            elif method == 'median':
                return cv2.medianBlur(image, 3)
            elif method == 'bilateral':
                return cv2.bilateralFilter(image, 9, 75, 75)
            elif method == 'wiener':
                # Wiener filtering using scikit-image
                noise_var = np.var(image) * 0.1
                return restoration.wiener(image, noise=noise_var).astype(np.uint8)
            else:
                logger.warning(f"Unknown noise removal method: {method}")
                return image
        except Exception as e:
            logger.error(f"Noise removal error: {e}")
            return image
    
    def enhance_contrast(self, image: np.ndarray, method: str = 'clahe') -> np.ndarray:
        """
        Enhance image contrast.
        
        Args:
            image: Input image
            method: Enhancement method ('clahe', 'histogram_eq', 'adaptive')
            
        Returns:
            Enhanced image
        """
        try:
            if method == 'clahe':
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                return clahe.apply(image)
            elif method == 'histogram_eq':
                return cv2.equalizeHist(image)
            elif method == 'adaptive':
                return cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                           cv2.THRESH_BINARY, 11, 2)
            else:
                logger.warning(f"Unknown contrast enhancement method: {method}")
                return image
        except Exception as e:
            logger.error(f"Contrast enhancement error: {e}")
            return image
    
    def resize_image(self, image: np.ndarray, target_size: Tuple[int, int] = (512, 512), 
                    maintain_aspect: bool = True) -> np.ndarray:
        """
        Resize image to target size.
        
        Args:
            image: Input image
            target_size: Target size (width, height)
            maintain_aspect: Whether to maintain aspect ratio
            
        Returns:
            Resized image
        """
        try:
            if maintain_aspect:
                # Calculate scaling factor
                h, w = image.shape[:2]
                target_w, target_h = target_size
                
                scale = min(target_w / w, target_h / h)
                new_w, new_h = int(w * scale), int(h * scale)
                
                # Resize image
                resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
                
                # Create padded image
                padded = np.ones((target_h, target_w), dtype=image.dtype) * 128
                
                # Center the resized image
                start_y = (target_h - new_h) // 2
                start_x = (target_w - new_w) // 2
                padded[start_y:start_y + new_h, start_x:start_x + new_w] = resized
                
                return padded
            else:
                return cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
        except Exception as e:
            logger.error(f"Image resize error: {e}")
            return image
    
    def crop_roi(self, image: np.ndarray, roi_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """
        Crop region of interest from fingerprint image.
        
        Args:
            image: Input image
            roi_size: Size of ROI (width, height). If None, auto-detect
            
        Returns:
            Cropped image
        """
        try:
            if roi_size is None:
                # Auto-detect ROI based on image content
                return self._auto_crop(image)
            else:
                # Center crop
                h, w = image.shape[:2]
                roi_w, roi_h = roi_size
                
                start_x = max(0, (w - roi_w) // 2)
                start_y = max(0, (h - roi_h) // 2)
                end_x = min(w, start_x + roi_w)
                end_y = min(h, start_y + roi_h)
                
                return image[start_y:end_y, start_x:end_x]
        except Exception as e:
            logger.error(f"ROI cropping error: {e}")
            return image
    
    def _auto_crop(self, image: np.ndarray) -> np.ndarray:
        """
        Automatically crop fingerprint based on foreground detection.
        
        Args:
            image: Input image
            
        Returns:
            Auto-cropped image
        """
        try:
            # Apply threshold to separate foreground from background
            _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Find contours
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                return image
            
            # Find largest contour (assumed to be fingerprint)
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(largest_contour)
            
            # Add some padding
            padding = 20
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(image.shape[1] - x, w + 2 * padding)
            h = min(image.shape[0] - y, h + 2 * padding)
            
            return image[y:y+h, x:x+w]
        except Exception as e:
            logger.error(f"Auto-crop error: {e}")
            return image
    
    def assess_quality(self, image: np.ndarray) -> float:
        """
        Assess fingerprint image quality.
        
        Args:
            image: Input fingerprint image
            
        Returns:
            Quality score (0.0 to 1.0)
        """
        try:
            # Calculate various quality metrics
            
            # 1. Variance (higher is better for fingerprints)
            variance = np.var(image.astype(np.float32))
            variance_score = min(variance / 1000.0, 1.0)
            
            # 2. Gradient magnitude (edge strength)
            gy, gx = np.gradient(image.astype(np.float32))
            gradient_magnitude = np.sqrt(gx**2 + gy**2)
            gradient_score = min(np.mean(gradient_magnitude) / 50.0, 1.0)
            
            # 3. Local contrast
            contrast_score = self._calculate_local_contrast(image)
            
            # 4. Ridge clarity (using Gabor filters)
            clarity_score = self._calculate_ridge_clarity(image)
            
            # Combine scores
            quality = (variance_score * 0.3 + gradient_score * 0.3 + 
                      contrast_score * 0.2 + clarity_score * 0.2)
            
            return max(0.0, min(1.0, quality))
        except Exception as e:
            logger.error(f"Quality assessment error: {e}")
            return 0.5
    
    def _calculate_local_contrast(self, image: np.ndarray, block_size: int = 16) -> float:
        """
        Calculate local contrast score.
        
        Args:
            image: Input image
            block_size: Size of local blocks
            
        Returns:
            Contrast score
        """
        try:
            h, w = image.shape
            contrasts = []
            
            for i in range(0, h - block_size, block_size):
                for j in range(0, w - block_size, block_size):
                    block = image[i:i+block_size, j:j+block_size].astype(np.float32)
                    if block.size > 0:
                        contrast = np.std(block)
                        contrasts.append(contrast)
            
            if contrasts:
                return min(np.mean(contrasts) / 30.0, 1.0)
            else:
                return 0.0
        except Exception:
            return 0.0
    
    def _calculate_ridge_clarity(self, image: np.ndarray) -> float:
        """
        Calculate ridge clarity using Gabor filters.
        
        Args:
            image: Input image
            
        Returns:
            Clarity score
        """
        try:
            # Apply Gabor filter
            kernel = cv2.getGaborKernel((21, 21), 5, 0, 2*np.pi/3, 0.5, 0, ktype=cv2.CV_32F)
            gabor_response = cv2.filter2D(image, cv2.CV_32F, kernel)
            
            # Calculate energy of Gabor response
            energy = np.mean(np.square(gabor_response))
            
            return min(energy / 10000.0, 1.0)
        except Exception:
            return 0.0
    
    def preprocess_pipeline(self, image: np.ndarray, 
                          target_size: Tuple[int, int] = (512, 512)) -> Tuple[np.ndarray, float]:
        """
        Complete preprocessing pipeline.
        
        Args:
            image: Input fingerprint image
            target_size: Target image size
            
        Returns:
            Tuple of (preprocessed_image, quality_score)
        """
        try:
            # Step 1: Assess initial quality
            initial_quality = self.assess_quality(image)
            
            # Step 2: Normalize image
            normalized = self.normalize(image)
            
            # Step 3: Remove noise
            denoised = self.remove_noise(normalized, method='gaussian')
            
            # Step 4: Enhance contrast
            enhanced = self.enhance_contrast(denoised, method='clahe')
            
            # Step 5: Crop ROI
            cropped = self.crop_roi(enhanced)
            
            # Step 6: Resize to target size
            resized = self.resize_image(cropped, target_size)
            
            # Step 7: Final quality assessment
            final_quality = self.assess_quality(resized)
            
            # Return average quality
            avg_quality = (initial_quality + final_quality) / 2
            
            return resized, avg_quality
        except Exception as e:
            logger.error(f"Preprocessing pipeline error: {e}")
            return image, 0.0
