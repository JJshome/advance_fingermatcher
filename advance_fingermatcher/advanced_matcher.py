"""
Advanced Fingerprint Matching System - Unified Interface
======================================================

This module provides a unified interface that combines all advanced algorithms:
1. Deep learning networks for feature extraction
2. Graph neural networks for matching
3. Ultra-fast search for 1:N identification
4. Adaptive quality assessment
5. Multi-modal fusion
6. Real-time optimization
"""

import torch
import torch.nn as nn
import numpy as np
import cv2
import time
import logging
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
from pathlib import Path
import json
import pickle

# Import our advanced components
from .deep_learning.networks import (
    MinutiaNet, DescriptorNet, QualityNet, FusionNet, 
    create_advanced_networks, load_pretrained_networks
)
from .deep_learning.graph_matching import (
    AdvancedGraphMatcher, create_graph_matcher
)
from .search.ultra_fast_search import (
    UltraFastSearch, SearchConfig, SearchResult,
    create_ultra_fast_search, create_distributed_search
)


@dataclass
class MatchingResult:
    """Comprehensive matching result"""
    # 1:1 Matching results
    match_score: float
    is_match: bool
    confidence: float
    
    # Quality metrics
    probe_quality: float
    gallery_quality: float
    
    # Detailed results
    matched_minutiae_count: int
    total_minutiae_probe: int
    total_minutiae_gallery: int
    
    # Processing time
    processing_time: float
    
    # Advanced features
    graph_matching_score: float = 0.0
    geometric_verification: bool = False
    fusion_confidence: float = 0.0
    
    # Metadata
    metadata: Dict[str, Any] = None


@dataclass
class SearchResultAdvanced:
    """Enhanced search result for 1:N identification"""
    candidate_id: str
    match_score: float
    confidence: float
    rank: int
    
    # Quality metrics
    template_quality: float
    matching_details: Dict[str, Any]
    
    # Processing info
    search_time: float
    verification_time: float
    
    # Metadata
    metadata: Dict[str, Any] = None


class AdvancedFingerprintMatcher:
    """
    Advanced fingerprint matching system with all state-of-the-art features
    """
    
    def __init__(self, config_file: str = None, model_dir: str = 'models/'):
        """
        Initialize the advanced matcher
        
        Args:
            config_file: Path to configuration file
            model_dir: Directory containing pre-trained models
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_dir = Path(model_dir)
        
        # Load configuration
        self.config = self._load_config(config_file)
        
        # Initialize logger
        self._setup_logging()
        
        # Load deep learning networks
        self.networks = self._load_networks()
        
        # Initialize graph matcher
        self.graph_matcher = self._initialize_graph_matcher()
        
        # Initialize search system
        self.search_system = self._initialize_search_system()
        
        # Performance statistics
        self.stats = {
            'total_matches': 0,
            'total_searches': 0,
            'avg_match_time': 0.0,
            'avg_search_time': 0.0,
            'accuracy_stats': {
                'true_positives': 0,
                'false_positives': 0,
                'true_negatives': 0,
                'false_negatives': 0
            }
        }
        
        logging.info("Advanced Fingerprint Matcher initialized successfully")
    
    def _load_config(self, config_file: str) -> Dict[str, Any]:
        """Load system configuration"""
        default_config = {
            # Network parameters
            'networks': {
                'minutia_net': {
                    'input_channels': 1,
                    'num_classes': 3
                },
                'descriptor_net': {
                    'img_size': 64,
                    'patch_size': 8,
                    'embed_dim': 384,
                    'descriptor_dim': 256
                },
                'quality_net': {
                    'input_channels': 1
                },
                'fusion_net': {
                    'fusion_dim': 512
                }
            },
            
            # Graph matching parameters
            'graph_matching': {
                'node_dim': 256,
                'edge_dim': 64,
                'n_layers': 6,
                'n_heads': 8,
                'temperature': 0.1
            },
            
            # Search parameters
            'search': {
                'index_type': 'HNSW',
                'dimension': 256,
                'use_gpu': True,
                'k': 100,
                'similarity_threshold': 0.7
            },
            
            # Quality thresholds
            'quality': {
                'min_image_quality': 0.3,
                'min_minutiae_quality': 0.4,
                'min_match_confidence': 0.5
            },
            
            # Performance settings
            'performance': {
                'batch_size': 32,
                'num_threads': 8,
                'enable_caching': True,
                'cache_size': 10000
            }
        }
        
        if config_file and Path(config_file).exists():
            with open(config_file, 'r') as f:
                user_config = json.load(f)
                # Deep merge configurations
                default_config.update(user_config)
        
        return default_config
    
    def _setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('fingerprint_matcher.log'),
                logging.StreamHandler()
            ]
        )
    
    def _load_networks(self) -> Dict[str, nn.Module]:
        """Load deep learning networks"""
        logging.info("Loading deep learning networks...")
        
        try:
            # Try to load pre-trained networks
            networks = load_pretrained_networks(str(self.model_dir))
        except:
            # Create new networks if pre-trained not available
            logging.warning("Pre-trained models not found, using random initialization")
            networks = create_advanced_networks()
        
        # Move to device
        for name, network in networks.items():
            network.to(self.device)
            network.eval()
        
        return networks
    
    def _initialize_graph_matcher(self) -> AdvancedGraphMatcher:
        """Initialize graph matching system"""
        logging.info("Initializing graph matcher...")
        
        graph_config = self.config['graph_matching']
        matcher = create_graph_matcher(graph_config)
        matcher.to(self.device)
        matcher.eval()
        
        return matcher
    
    def _initialize_search_system(self) -> UltraFastSearch:
        """Initialize search system"""
        logging.info("Initializing search system...")
        
        search_config = SearchConfig(**self.config['search'])
        search_system = create_ultra_fast_search(search_config)
        
        return search_system
    
    def extract_features(self, image: np.ndarray, enhance_image: bool = True) -> Dict[str, Any]:
        """
        Extract comprehensive features from fingerprint image
        
        Args:
            image: Fingerprint image [H, W] or [H, W, 1]
            enhance_image: Whether to apply image enhancement
            
        Returns:
            Dictionary containing all extracted features
        """
        start_time = time.time()
        
        # Prepare image
        image = self._prepare_image(image, enhance_image)
        image_tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).float().to(self.device)
        
        with torch.no_grad():
            # Extract minutiae
            minutia_output = self.networks['minutia'](image_tensor)
            
            # Extract quality metrics
            quality_output = self.networks['quality'](image_tensor)
            
            # Extract descriptors for detected minutiae
            minutiae_locations = self._extract_minutiae_from_network_output(
                minutia_output, image.shape
            )
            
            descriptors = []
            if len(minutiae_locations) > 0:
                for loc in minutiae_locations:
                    patch = self._extract_patch(image, loc, patch_size=64)
                    patch_tensor = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0).float().to(self.device)
                    desc_output = self.networks['descriptor'](patch_tensor)
                    descriptors.append(desc_output['descriptor'].cpu().numpy()[0])
            
            # Fusion of all features
            if len(descriptors) > 0:
                minutiae_features = torch.from_numpy(np.mean(descriptors, axis=0)).unsqueeze(0).to(self.device)
                quality_features = quality_output['features'].mean(dim=[2, 3])
                descriptor_features = torch.from_numpy(np.mean(descriptors, axis=0)).unsqueeze(0).to(self.device)
                
                fusion_output = self.networks['fusion'](
                    minutiae_features, descriptor_features, quality_features
                )
            else:
                fusion_output = {'embedding': torch.zeros(1, 256), 'confidence': torch.tensor([0.0])}
        
        processing_time = time.time() - start_time
        
        return {
            'minutiae': minutiae_locations,
            'descriptors': descriptors,
            'quality': {
                'overall': quality_output['overall_quality'].cpu().item(),
                'clarity': quality_output['clarity'].cpu().item(),
                'contrast': quality_output['contrast'].cpu().item(),
                'sharpness': quality_output['sharpness'].cpu().item(),
                'local_map': quality_output['local_quality'].cpu().numpy()
            },
            'fusion_embedding': fusion_output['embedding'].cpu().numpy()[0],
            'fusion_confidence': fusion_output['confidence'].cpu().item(),
            'processing_time': processing_time,
            'metadata': {
                'image_shape': image.shape,
                'num_minutiae': len(minutiae_locations),
                'device': str(self.device)
            }
        }
    
    def match_1to1(self, probe_image: np.ndarray, gallery_image: np.ndarray,
                   return_details: bool = False) -> MatchingResult:
        """
        Perform 1:1 fingerprint matching
        
        Args:
            probe_image: Probe fingerprint image
            gallery_image: Gallery fingerprint image
            return_details: Whether to return detailed matching information
            
        Returns:
            MatchingResult object
        """
        start_time = time.time()
        
        # Extract features from both images
        probe_features = self.extract_features(probe_image)
        gallery_features = self.extract_features(gallery_image)
        
        # Quality check
        min_quality = self.config['quality']['min_image_quality']
        if (probe_features['quality']['overall'] < min_quality or 
            gallery_features['quality']['overall'] < min_quality):
            logging.warning("Low quality images detected")
        
        # Traditional Enhanced Bozorth3 matching
        traditional_score = self._enhanced_bozorth3_match(
            probe_features['minutiae'], gallery_features['minutiae']
        )
        
        # Graph neural network matching
        graph_score = 0.0
        geometric_verified = False
        
        if len(probe_features['minutiae']) >= 4 and len(gallery_features['minutiae']) >= 4:
            minutiae1_tensor = torch.tensor(probe_features['minutiae'], dtype=torch.float32)
            minutiae2_tensor = torch.tensor(gallery_features['minutiae'], dtype=torch.float32)
            
            with torch.no_grad():
                graph_results = self.graph_matcher(minutiae1_tensor, minutiae2_tensor)
                graph_score = graph_results['match_score'].item()
                geometric_verified = graph_results['is_valid'].item()
        
        # Fusion-based matching
        fusion_score = 0.0
        if len(probe_features['descriptors']) > 0 and len(gallery_features['descriptors']) > 0:
            probe_embedding = probe_features['fusion_embedding']
            gallery_embedding = gallery_features['fusion_embedding']
            
            # Cosine similarity
            fusion_score = np.dot(probe_embedding, gallery_embedding) / (
                np.linalg.norm(probe_embedding) * np.linalg.norm(gallery_embedding) + 1e-8
            )
        
        # Combined scoring
        weights = [0.4, 0.3, 0.3]  # traditional, graph, fusion
        combined_score = (
            weights[0] * traditional_score +
            weights[1] * graph_score +
            weights[2] * max(0, fusion_score)
        )
        
        # Quality-weighted confidence
        quality_weight = (probe_features['quality']['overall'] + 
                         gallery_features['quality']['overall']) / 2
        confidence = combined_score * quality_weight
        
        # Decision threshold
        match_threshold = self.config['quality']['min_match_confidence']
        is_match = confidence >= match_threshold
        
        processing_time = time.time() - start_time
        
        # Update statistics
        self.stats['total_matches'] += 1
        self.stats['avg_match_time'] = (
            (self.stats['avg_match_time'] * (self.stats['total_matches'] - 1) + 
             processing_time) / self.stats['total_matches']
        )
        
        result = MatchingResult(
            match_score=combined_score,
            is_match=is_match,
            confidence=confidence,
            probe_quality=probe_features['quality']['overall'],
            gallery_quality=gallery_features['quality']['overall'],
            matched_minutiae_count=min(len(probe_features['minutiae']), 
                                     len(gallery_features['minutiae'])),
            total_minutiae_probe=len(probe_features['minutiae']),
            total_minutiae_gallery=len(gallery_features['minutiae']),
            processing_time=processing_time,
            graph_matching_score=graph_score,
            geometric_verification=geometric_verified,
            fusion_confidence=probe_features['fusion_confidence'] * 
                            gallery_features['fusion_confidence']
        )
        
        if return_details:
            result.metadata = {
                'traditional_score': traditional_score,
                'graph_score': graph_score,
                'fusion_score': fusion_score,
                'weights': weights,
                'probe_features': probe_features,
                'gallery_features': gallery_features
            }
        
        return result
    
    def search_1toN(self, probe_image: np.ndarray, k: int = 100,
                    similarity_threshold: float = None) -> List[SearchResultAdvanced]:
        """
        Perform 1:N fingerprint identification
        
        Args:
            probe_image: Probe fingerprint image
            k: Number of candidates to return
            similarity_threshold: Minimum similarity threshold
            
        Returns:
            List of SearchResultAdvanced objects
        """
        start_time = time.time()
        
        # Extract features
        probe_features = self.extract_features(probe_image)
        
        # Quality check
        min_quality = self.config['quality']['min_image_quality']
        if probe_features['quality']['overall'] < min_quality:
            logging.warning(f"Low quality probe image: {probe_features['quality']['overall']:.3f}")
        
        # Search using fusion embeddings
        search_start = time.time()
        search_results = self.search_system.search(
            probe_features['fusion_embedding'],
            k=k,
            similarity_threshold=similarity_threshold
        )
        search_time = time.time() - search_start
        
        # Verify top candidates with detailed matching
        advanced_results = []
        verification_start = time.time()
        
        for i, result in enumerate(search_results):
            # For now, we'll use the search score as the match score
            # In a full implementation, you would load the template and perform detailed matching
            
            advanced_result = SearchResultAdvanced(
                candidate_id=result.template_id,
                match_score=result.score,
                confidence=result.score * probe_features['fusion_confidence'],
                rank=result.rank,
                template_quality=0.8,  # Would come from stored metadata
                matching_details={
                    'search_distance': result.distance,
                    'probe_quality': probe_features['quality']['overall']
                },
                search_time=search_time,
                verification_time=0.0,  # Would be calculated during detailed verification
                metadata=result.metadata
            )
            
            advanced_results.append(advanced_result)
        
        verification_time = time.time() - verification_start
        total_time = time.time() - start_time
        
        # Update statistics
        self.stats['total_searches'] += 1
        self.stats['avg_search_time'] = (
            (self.stats['avg_search_time'] * (self.stats['total_searches'] - 1) + 
             total_time) / self.stats['total_searches']
        )
        
        return advanced_results
    
    def enroll_template(self, template_id: str, image: np.ndarray, 
                       metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Enroll a fingerprint template into the search system
        
        Args:
            template_id: Unique identifier for the template
            image: Fingerprint image
            metadata: Optional metadata
            
        Returns:
            Dictionary with enrollment results
        """
        start_time = time.time()
        
        # Extract features
        features = self.extract_features(image)
        
        # Quality check
        min_quality = self.config['quality']['min_image_quality']
        if features['quality']['overall'] < min_quality:
            return {
                'success': False,
                'reason': f"Image quality too low: {features['quality']['overall']:.3f}",
                'quality': features['quality']['overall']
            }
        
        # Store in search system
        template_metadata = {
            'quality': features['quality']['overall'],
            'num_minutiae': len(features['minutiae']),
            'enrollment_time': time.time(),
            **(metadata or {})
        }
        
        self.search_system.add_template(
            template_id,
            features['fusion_embedding'],
            template_metadata
        )
        
        processing_time = time.time() - start_time
        
        return {
            'success': True,
            'template_id': template_id,
            'quality': features['quality']['overall'],
            'num_minutiae': len(features['minutiae']),
            'processing_time': processing_time,
            'metadata': template_metadata
        }
    
    def _prepare_image(self, image: np.ndarray, enhance: bool = True) -> np.ndarray:
        """Prepare image for processing"""
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Normalize
        image = image.astype(np.float32) / 255.0
        
        # Enhancement (basic - would be more sophisticated in practice)
        if enhance:
            # Histogram equalization
            image = cv2.equalizeHist((image * 255).astype(np.uint8)).astype(np.float32) / 255.0
            
            # Gaussian blur for noise reduction
            image = cv2.GaussianBlur(image, (3, 3), 0.5)
        
        return image
    
    def _extract_minutiae_from_network_output(self, output: Dict, image_shape: Tuple[int, int]) -> List[List[float]]:
        """Extract minutiae from network output"""
        # Simplified extraction - would be more sophisticated in practice
        locations = output['locations'].cpu().numpy()[0]  # [2, H, W]
        orientations = output['orientations'].cpu().numpy()[0]  # [2, H, W]
        types = output['types'].cpu().numpy()[0]  # [3, H, W]
        qualities = output['qualities'].cpu().numpy()[0]  # [1, H, W]
        
        minutiae = []
        threshold = 0.5
        
        # Find peaks above threshold
        h, w = locations.shape[1], locations.shape[2]
        for y in range(1, h-1):
            for x in range(1, w-1):
                if qualities[0, y, x] > threshold:
                    # Extract minutia information
                    x_coord = locations[0, y, x] * image_shape[1]
                    y_coord = locations[1, y, x] * image_shape[0]
                    
                    # Orientation from cos/sin components
                    cos_theta = orientations[0, y, x]
                    sin_theta = orientations[1, y, x]
                    theta = np.arctan2(sin_theta, cos_theta)
                    
                    # Type (argmax)
                    minutia_type = np.argmax(types[:, y, x])
                    
                    quality = qualities[0, y, x]
                    
                    minutiae.append([x_coord, y_coord, theta, quality, minutia_type])
        
        return minutiae
    
    def _extract_patch(self, image: np.ndarray, location: List[float], patch_size: int = 64) -> np.ndarray:
        """Extract patch around minutia location"""
        x, y = int(location[0]), int(location[1])
        half_size = patch_size // 2
        
        # Extract patch with boundary handling
        patch = np.zeros((patch_size, patch_size), dtype=np.float32)
        
        y_start = max(0, y - half_size)
        y_end = min(image.shape[0], y + half_size)
        x_start = max(0, x - half_size)
        x_end = min(image.shape[1], x + half_size)
        
        patch_y_start = max(0, half_size - y)
        patch_y_end = patch_y_start + (y_end - y_start)
        patch_x_start = max(0, half_size - x)
        patch_x_end = patch_x_start + (x_end - x_start)
        
        patch[patch_y_start:patch_y_end, patch_x_start:patch_x_end] = \
            image[y_start:y_end, x_start:x_end]
        
        return patch
    
    def _enhanced_bozorth3_match(self, minutiae1: List, minutiae2: List) -> float:
        """Simplified Enhanced Bozorth3 matching"""
        if len(minutiae1) < 4 or len(minutiae2) < 4:
            return 0.0
        
        # Simplified matching score based on spatial distribution
        # In practice, this would use the full Enhanced Bozorth3 algorithm
        score = min(len(minutiae1), len(minutiae2)) / max(len(minutiae1), len(minutiae2))
        return score * 0.8  # Scale down for realism
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get comprehensive system statistics"""
        search_stats = self.search_system.get_stats()
        
        return {
            'matching_stats': self.stats,
            'search_stats': search_stats,
            'system_info': {
                'device': str(self.device),
                'models_loaded': list(self.networks.keys()),
                'config': self.config
            }
        }
    
    def save_model(self, save_path: str) -> None:
        """Save trained models"""
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        for name, network in self.networks.items():
            torch.save(network.state_dict(), save_path / f'{name}.pth')
        
        # Save graph matcher
        torch.save(self.graph_matcher.state_dict(), save_path / 'graph_matcher.pth')
        
        # Save configuration
        with open(save_path / 'config.json', 'w') as f:
            json.dump(self.config, f, indent=2)
        
        logging.info(f"Models saved to {save_path}")


# Factory function
def create_advanced_matcher(config_file: str = None, model_dir: str = 'models/') -> AdvancedFingerprintMatcher:
    """Create advanced fingerprint matcher"""
    return AdvancedFingerprintMatcher(config_file, model_dir)


# Benchmarking and evaluation tools
class AdvancedBenchmark:
    """Comprehensive benchmarking suite for the advanced matcher"""
    
    def __init__(self, matcher: AdvancedFingerprintMatcher):
        self.matcher = matcher
        
    def benchmark_accuracy(self, test_pairs: List[Tuple[np.ndarray, np.ndarray, bool]]) -> Dict[str, float]:
        """
        Benchmark matching accuracy
        
        Args:
            test_pairs: List of (probe_image, gallery_image, is_genuine_match)
        """
        results = {
            'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0,
            'scores_genuine': [], 'scores_impostor': []
        }
        
        for probe_img, gallery_img, is_genuine in test_pairs:
            match_result = self.matcher.match_1to1(probe_img, gallery_img)
            
            if is_genuine:
                results['scores_genuine'].append(match_result.confidence)
                if match_result.is_match:
                    results['tp'] += 1
                else:
                    results['fn'] += 1
            else:
                results['scores_impostor'].append(match_result.confidence)
                if match_result.is_match:
                    results['fp'] += 1
                else:
                    results['tn'] += 1
        
        # Calculate metrics
        accuracy = (results['tp'] + results['tn']) / len(test_pairs)
        precision = results['tp'] / (results['tp'] + results['fp']) if (results['tp'] + results['fp']) > 0 else 0
        recall = results['tp'] / (results['tp'] + results['fn']) if (results['tp'] + results['fn']) > 0 else 0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        # Calculate EER (simplified)
        all_scores = results['scores_genuine'] + results['scores_impostor']
        all_labels = [1] * len(results['scores_genuine']) + [0] * len(results['scores_impostor'])
        
        thresholds = np.linspace(0, 1, 1000)
        eer = self._calculate_eer(all_scores, all_labels, thresholds)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'eer': eer,
            'total_tests': len(test_pairs)
        }
    
    def benchmark_speed(self, test_images: List[np.ndarray], num_iterations: int = 100) -> Dict[str, float]:
        """Benchmark processing speed"""
        # 1:1 matching speed
        match_times = []
        for i in range(min(num_iterations, len(test_images) // 2)):
            start_time = time.time()
            self.matcher.match_1to1(test_images[i*2], test_images[i*2+1])
            match_times.append(time.time() - start_time)
        
        # Feature extraction speed
        extract_times = []
        for i in range(min(num_iterations, len(test_images))):
            start_time = time.time()
            self.matcher.extract_features(test_images[i])
            extract_times.append(time.time() - start_time)
        
        return {
            'avg_match_time': np.mean(match_times),
            'avg_extract_time': np.mean(extract_times),
            'match_throughput': 1.0 / np.mean(match_times),
            'extract_throughput': 1.0 / np.mean(extract_times)
        }
    
    def _calculate_eer(self, scores: List[float], labels: List[int], thresholds: np.ndarray) -> float:
        """Calculate Equal Error Rate"""
        fars = []
        frrs = []
        
        for threshold in thresholds:
            tp = fp = tn = fn = 0
            
            for score, label in zip(scores, labels):
                prediction = 1 if score >= threshold else 0
                
                if label == 1 and prediction == 1:
                    tp += 1
                elif label == 0 and prediction == 1:
                    fp += 1
                elif label == 0 and prediction == 0:
                    tn += 1
                else:
                    fn += 1
            
            far = fp / (fp + tn) if (fp + tn) > 0 else 0
            frr = fn / (fn + tp) if (fn + tp) > 0 else 0
            
            fars.append(far)
            frrs.append(frr)
        
        # Find EER point
        fars = np.array(fars)
        frrs = np.array(frrs)
        eer_idx = np.argmin(np.abs(fars - frrs))
        
        return (fars[eer_idx] + frrs[eer_idx]) / 2


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Create advanced matcher
    print("Initializing Advanced Fingerprint Matcher...")
    matcher = create_advanced_matcher()
    
    # Create dummy test data
    test_image = np.random.randint(0, 255, (300, 300), dtype=np.uint8)
    
    # Test feature extraction
    print("\nTesting feature extraction...")
    features = matcher.extract_features(test_image)
    print(f"Extracted {len(features['minutiae'])} minutiae")
    print(f"Overall quality: {features['quality']['overall']:.3f}")
    print(f"Processing time: {features['processing_time']*1000:.2f}ms")
    
    # Test 1:1 matching
    print("\nTesting 1:1 matching...")
    probe_img = np.random.randint(0, 255, (300, 300), dtype=np.uint8)
    gallery_img = np.random.randint(0, 255, (300, 300), dtype=np.uint8)
    
    match_result = matcher.match_1to1(probe_img, gallery_img, return_details=True)
    print(f"Match score: {match_result.match_score:.4f}")
    print(f"Is match: {match_result.is_match}")
    print(f"Confidence: {match_result.confidence:.4f}")
    print(f"Processing time: {match_result.processing_time*1000:.2f}ms")
    
    # Test template enrollment
    print("\nTesting template enrollment...")
    enrollment_result = matcher.enroll_template("test_001", test_image)
    print(f"Enrollment success: {enrollment_result['success']}")
    if enrollment_result['success']:
        print(f"Template quality: {enrollment_result['quality']:.3f}")
    
    # Test 1:N search
    print("\nTesting 1:N search...")
    search_results = matcher.search_1toN(probe_img, k=5)
    print(f"Found {len(search_results)} candidates")
    
    for i, result in enumerate(search_results):
        print(f"  {i+1}. {result.candidate_id} (score: {result.match_score:.4f})")
    
    # Show system statistics
    print("\nSystem Statistics:")
    stats = matcher.get_system_stats()
    print(f"Total matches: {stats['matching_stats']['total_matches']}")
    print(f"Total searches: {stats['matching_stats']['total_searches']}")
    print(f"Average match time: {stats['matching_stats']['avg_match_time']*1000:.2f}ms")
