"""
Main fingerprint matcher class.

This module provides the core fingerprint matching functionality,
combining various algorithms for optimal performance.
"""

import numpy as np
import cv2
from typing import Tuple, Dict, Any, Optional, List
from pathlib import Path
import logging
from dataclasses import dataclass
import time

logger = logging.getLogger(__name__)


@dataclass
class MatchResult:
    """Result of fingerprint matching."""
    score: float
    confidence: float
    is_match: bool
    method_used: str
    processing_time: float
    details: Dict[str, Any]


class FingerprintMatcher:
    """
    Advanced fingerprint matching system.
    
    This class provides high-level interface for fingerprint matching
    using multiple algorithms and deep learning techniques.
    """
    
    def __init__(self, 
                 config: Optional[Dict[str, Any]] = None,
                 model_path: Optional[str] = None,
                 gpu_enabled: bool = True):
        """
        Initialize the fingerprint matcher.
        
        Args:
            config: Configuration dictionary
            model_path: Path to pre-trained models
            gpu_enabled: Whether to use GPU acceleration
        """
        self.config = config or {}
        self.gpu_enabled = gpu_enabled and cv2.cuda.getCudaEnabledDeviceCount() > 0
        
        # Matching parameters
        self.match_threshold = self.config.get('match_threshold', 0.85)
        self.quality_threshold = self.config.get('quality_threshold', 0.7)
        
        logger.info(f"FingerprintMatcher initialized (GPU: {self.gpu_enabled})")
    
    def load_image(self, image_path: str) -> np.ndarray:
        """
        Load and validate fingerprint image.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Loaded image as numpy array
        """
        try:
            image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            # Validate image properties
            if image.shape[0] < 100 or image.shape[1] < 100:
                logger.warning(f"Image {image_path} is very small: {image.shape}")
            
            return image
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {e}")
            raise
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess fingerprint image for optimal matching.
        
        Args:
            image: Input fingerprint image
            
        Returns:
            Preprocessed image
        """
        # Basic preprocessing
        processed = cv2.equalizeHist(image)
        processed = cv2.GaussianBlur(processed, (3, 3), 0)
        
        # Enhance contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        processed = clahe.apply(processed)
        
        return processed
    
    def extract_features(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Extract comprehensive features from fingerprint image.
        
        Args:
            image: Preprocessed fingerprint image
            
        Returns:
            Dictionary containing various feature types
        """
        features = {}
        
        try:
            # Extract SIFT features
            features['sift'] = self._extract_sift_basic(image)
            
            # Extract ORB features
            features['orb'] = self._extract_orb_basic(image)
            
            # Extract minutiae using traditional methods
            features['minutiae'] = self._extract_minutiae_basic(image)
            
        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            features = {'error': str(e)}
        
        return features
    
    def _extract_sift_basic(self, image: np.ndarray) -> Dict[str, Any]:
        """Basic SIFT feature extraction."""
        sift = cv2.SIFT_create()
        keypoints, descriptors = sift.detectAndCompute(image, None)
        return {
            'keypoints': keypoints,
            'descriptors': descriptors,
            'count': len(keypoints) if keypoints else 0
        }
    
    def _extract_orb_basic(self, image: np.ndarray) -> Dict[str, Any]:
        """Basic ORB feature extraction."""
        orb = cv2.ORB_create()
        keypoints, descriptors = orb.detectAndCompute(image, None)
        return {
            'keypoints': keypoints,
            'descriptors': descriptors,
            'count': len(keypoints) if keypoints else 0
        }
    
    def _extract_minutiae_basic(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Basic minutiae extraction using traditional methods."""
        # Enhance the image
        enhanced = self._enhance_ridges(image)
        
        # Binarize the image
        _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Thin the ridges
        thinned = self._thin_ridges(binary)
        
        # Find minutiae points
        minutiae = self._find_minutiae(thinned)
        
        return minutiae
    
    def _enhance_ridges(self, image: np.ndarray) -> np.ndarray:
        """Enhance ridge patterns using Gabor filters."""
        # Apply Gabor filter bank
        enhanced = np.zeros_like(image, dtype=np.float32)
        
        for theta in np.arange(0, np.pi, np.pi/8):
            kernel = cv2.getGaborKernel((21, 21), 5, theta, 2*np.pi/3, 0.5, 0, ktype=cv2.CV_32F)
            filtered = cv2.filter2D(image, cv2.CV_8UC3, kernel)
            enhanced = np.maximum(enhanced, filtered)
        
        return enhanced.astype(np.uint8)
    
    def _thin_ridges(self, binary: np.ndarray) -> np.ndarray:
        """Thin ridge lines using morphological operations."""
        # Simple thinning using morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        thinned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        # Apply skeletonization
        skel = np.zeros(binary.shape, np.uint8)
        eroded = binary.copy()
        
        while True:
            opened = cv2.morphologyEx(eroded, cv2.MORPH_OPEN, kernel)
            temp = cv2.subtract(eroded, opened)
            skel = cv2.bitwise_or(skel, temp)
            eroded = cv2.erode(eroded, kernel)
            
            if cv2.countNonZero(eroded) == 0:
                break
        
        return skel
    
    def _find_minutiae(self, thinned: np.ndarray) -> List[Dict[str, Any]]:
        """Find minutiae points (ridge endings and bifurcations)."""
        minutiae = []
        h, w = thinned.shape
        
        # Create 3x3 kernel for checking neighbors
        for i in range(1, h-1):
            for j in range(1, w-1):
                if thinned[i, j] == 255:  # Ridge pixel
                    # Count neighbors
                    neighbors = [
                        thinned[i-1, j-1], thinned[i-1, j], thinned[i-1, j+1],
                        thinned[i, j-1],                    thinned[i, j+1],
                        thinned[i+1, j-1], thinned[i+1, j], thinned[i+1, j+1]
                    ]
                    
                    neighbor_count = sum(1 for n in neighbors if n == 255)
                    
                    # Ridge ending (1 neighbor) or bifurcation (3+ neighbors)
                    if neighbor_count == 1:
                        minutiae.append({
                            'x': j, 'y': i, 'type': 'ending',
                            'angle': self._calculate_ridge_angle(thinned, i, j)
                        })
                    elif neighbor_count >= 3:
                        minutiae.append({
                            'x': j, 'y': i, 'type': 'bifurcation',
                            'angle': self._calculate_ridge_angle(thinned, i, j)
                        })
        
        return minutiae
    
    def _calculate_ridge_angle(self, image: np.ndarray, y: int, x: int) -> float:
        """Calculate ridge angle at given point."""
        # Simple gradient-based angle calculation
        if y > 0 and y < image.shape[0]-1 and x > 0 and x < image.shape[1]-1:
            dy = float(image[y+1, x]) - float(image[y-1, x])
            dx = float(image[y, x+1]) - float(image[y, x-1])
            angle = np.arctan2(dy, dx) * 180 / np.pi
            return angle % 360
        return 0.0
    
    def match_features(self, 
                      features1: Dict[str, Any], 
                      features2: Dict[str, Any],
                      method: str = 'hybrid') -> MatchResult:
        """
        Match two sets of fingerprint features.
        
        Args:
            features1: Features from first fingerprint
            features2: Features from second fingerprint
            method: Matching method ('minutiae', 'sift', 'orb', 'hybrid')
            
        Returns:
            MatchResult object with matching details
        """
        start_time = time.time()
        
        try:
            if method == 'hybrid':
                result = self._hybrid_matching(features1, features2)
            elif method == 'minutiae':
                result = self._minutiae_matching(features1, features2)
            elif method == 'sift':
                result = self._sift_matching(features1, features2)
            elif method == 'orb':
                result = self._orb_matching(features1, features2)
            else:
                raise ValueError(f"Unknown matching method: {method}")
        except Exception as e:
            logger.error(f"Error in matching: {e}")
            result = {'score': 0.0, 'confidence': 0.0, 'error': str(e)}
        
        processing_time = time.time() - start_time
        
        return MatchResult(
            score=result.get('score', 0.0),
            confidence=result.get('confidence', 0.0),
            is_match=result.get('score', 0.0) > self.match_threshold,
            method_used=method,
            processing_time=processing_time,
            details=result
        )
    
    def _hybrid_matching(self, features1: Dict[str, Any], features2: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform hybrid matching using multiple algorithms.
        """
        # Get individual matching scores
        sift_result = self._sift_matching(features1, features2)
        orb_result = self._orb_matching(features1, features2)
        minutiae_result = self._minutiae_matching(features1, features2)
        
        # Weighted combination of scores
        weights = {
            'sift': 0.4,
            'orb': 0.3,
            'minutiae': 0.3
        }
        
        combined_score = (
            weights['sift'] * sift_result['score'] +
            weights['orb'] * orb_result['score'] +
            weights['minutiae'] * minutiae_result['score']
        )
        
        # Calculate confidence based on agreement between methods
        scores = [sift_result['score'], orb_result['score'], minutiae_result['score']]
        confidence = 1.0 - np.std(scores) / np.mean(scores) if np.mean(scores) > 0 else 0.0
        confidence = max(0.0, min(1.0, confidence))
        
        return {
            'score': combined_score,
            'confidence': confidence,
            'individual_scores': {
                'sift': sift_result['score'],
                'orb': orb_result['score'],
                'minutiae': minutiae_result['score']
            },
            'weights': weights
        }
    
    def _sift_matching(self, features1: Dict[str, Any], features2: Dict[str, Any]) -> Dict[str, Any]:
        """Match fingerprints using SIFT features."""
        sift1 = features1.get('sift', {})
        sift2 = features2.get('sift', {})
        
        if sift1.get('descriptors') is None or sift2.get('descriptors') is None:
            return {'score': 0.0, 'confidence': 0.0, 'matches': 0}
        
        try:
            # Use FLANN matcher for SIFT features
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            search_params = dict(checks=50)
            flann = cv2.FlannBasedMatcher(index_params, search_params)
            
            matches = flann.knnMatch(sift1['descriptors'], sift2['descriptors'], k=2)
            
            # Apply ratio test
            good_matches = []
            for match_pair in matches:
                if len(match_pair) == 2:
                    m, n = match_pair
                    if m.distance < 0.7 * n.distance:
                        good_matches.append(m)
            
            # Calculate score
            total_features = min(len(sift1.get('keypoints', [])), len(sift2.get('keypoints', [])))
            score = len(good_matches) / total_features if total_features > 0 else 0.0
            score = min(1.0, score)  # Cap at 1.0
            
            return {
                'score': score,
                'confidence': min(1.0, len(good_matches) / 20),
                'matches': len(good_matches)
            }
        except Exception as e:
            logger.error(f"SIFT matching error: {e}")
            return {'score': 0.0, 'confidence': 0.0, 'matches': 0}
    
    def _orb_matching(self, features1: Dict[str, Any], features2: Dict[str, Any]) -> Dict[str, Any]:
        """Match fingerprints using ORB features."""
        orb1 = features1.get('orb', {})
        orb2 = features2.get('orb', {})
        
        if orb1.get('descriptors') is None or orb2.get('descriptors') is None:
            return {'score': 0.0, 'confidence': 0.0, 'matches': 0}
        
        try:
            # Use BFMatcher for ORB features
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(orb1['descriptors'], orb2['descriptors'])
            
            # Sort matches by distance
            matches = sorted(matches, key=lambda x: x.distance)
            
            # Filter good matches
            good_matches = [m for m in matches if m.distance < 50]
            
            # Calculate score
            total_features = min(len(orb1.get('keypoints', [])), len(orb2.get('keypoints', [])))
            score = len(good_matches) / total_features if total_features > 0 else 0.0
            score = min(1.0, score)  # Cap at 1.0
            
            return {
                'score': score,
                'confidence': min(1.0, len(good_matches) / 15),
                'matches': len(good_matches)
            }
        except Exception as e:
            logger.error(f"ORB matching error: {e}")
            return {'score': 0.0, 'confidence': 0.0, 'matches': 0}
    
    def _minutiae_matching(self, features1: Dict[str, Any], features2: Dict[str, Any]) -> Dict[str, Any]:
        """Match fingerprints based on minutiae points."""
        minutiae1 = features1.get('minutiae', [])
        minutiae2 = features2.get('minutiae', [])
        
        if not minutiae1 or not minutiae2:
            return {'score': 0.0, 'confidence': 0.0, 'matched_points': 0}
        
        # Implement minutiae matching algorithm
        matched_points = self._match_minutiae_points(minutiae1, minutiae2)
        
        # Calculate score based on matched points and total points
        total_points = min(len(minutiae1), len(minutiae2))
        score = matched_points / total_points if total_points > 0 else 0.0
        
        return {
            'score': score,
            'confidence': min(1.0, matched_points / 12),  # ISO standard requires 12 minutiae
            'matched_points': matched_points,
            'total_points1': len(minutiae1),
            'total_points2': len(minutiae2)
        }
    
    def _match_minutiae_points(self, minutiae1: List[Dict], minutiae2: List[Dict]) -> int:
        """Match minutiae points between two fingerprints."""
        matched_count = 0
        distance_threshold = 20  # pixels
        angle_threshold = 15  # degrees
        
        for m1 in minutiae1:
            for m2 in minutiae2:
                # Calculate distance
                dist = np.sqrt((m1['x'] - m2['x'])**2 + (m1['y'] - m2['y'])**2)
                
                # Calculate angle difference
                angle_diff = abs(m1['angle'] - m2['angle'])
                angle_diff = min(angle_diff, 360 - angle_diff)  # Handle circular nature
                
                # Check if points match
                if dist < distance_threshold and angle_diff < angle_threshold:
                    matched_count += 1
                    break  # Each minutiae can only match once
        
        return matched_count
    
    def is_match(self, score: float, confidence: float = None) -> bool:
        """
        Determine if fingerprints match based on score and confidence.
        """
        if confidence is not None and confidence < 0.5:
            return False  # Low confidence, reject match
        
        return score >= self.match_threshold
    
    def match_fingerprints(self, 
                          image1: str, 
                          image2: str,
                          method: str = 'hybrid') -> MatchResult:
        """
        Complete fingerprint matching pipeline.
        
        Args:
            image1: Path to first fingerprint image
            image2: Path to second fingerprint image
            method: Matching method to use
            
        Returns:
            MatchResult object
        """
        # Load images
        img1 = self.load_image(image1)
        img2 = self.load_image(image2)
        
        # Preprocess images
        img1_processed = self.preprocess_image(img1)
        img2_processed = self.preprocess_image(img2)
        
        # Extract features
        features1 = self.extract_features(img1_processed)
        features2 = self.extract_features(img2_processed)
        
        # Match features
        result = self.match_features(features1, features2, method=method)
        
        return result
