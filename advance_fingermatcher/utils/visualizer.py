"""
Visualization utilities for fingerprint matching results.

This module provides comprehensive visualization capabilities
for fingerprint matching results, features, and statistics.
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any, Optional, Tuple
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class Visualizer:
    """
    Comprehensive visualization system for fingerprint matching.
    
    Provides visualization for:
    - Fingerprint images with features
    - Match results and matrices
    - Statistical analysis
    - Quality assessment
    """
    
    def __init__(self):
        """
        Initialize the visualizer.
        """
        # Set up matplotlib style
        plt.style.use('default')
        try:
            sns.set_palette("husl")
        except:
            pass  # Skip if seaborn not available
        logger.info("Visualizer initialized")
    
    def visualize_fingerprint(self, image: np.ndarray, 
                            features: Optional[Dict[str, Any]] = None,
                            title: str = "Fingerprint") -> plt.Figure:
        """
        Visualize fingerprint image with optional features overlay.
        
        Args:
            image: Fingerprint image
            features: Optional features to overlay
            title: Plot title
            
        Returns:
            Matplotlib figure
        """
        try:
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))
            
            # Original image
            axes[0].imshow(image, cmap='gray')
            axes[0].set_title(f"{title} - Original")
            axes[0].axis('off')
            
            # Image with features
            if features:
                enhanced_img = self._overlay_features(image, features)
                axes[1].imshow(enhanced_img)
                axes[1].set_title(f"{title} - With Features")
            else:
                axes[1].imshow(image, cmap='gray')
                axes[1].set_title(f"{title} - No Features")
            axes[1].axis('off')
            
            plt.tight_layout()
            return fig
        except Exception as e:
            logger.error(f"Fingerprint visualization error: {e}")
            return plt.figure()
    
    def _overlay_features(self, image: np.ndarray, features: Dict[str, Any]) -> np.ndarray:
        """
        Overlay features on fingerprint image.
        
        Args:
            image: Original fingerprint image
            features: Feature dictionary
            
        Returns:
            Image with features overlaid
        """
        # Convert to color image
        if len(image.shape) == 2:
            overlay = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            overlay = image.copy()
        
        # Draw SIFT keypoints
        sift_features = features.get('sift', {})
        if sift_features.get('keypoints'):
            for kp in sift_features['keypoints']:
                x, y = int(kp.pt[0]), int(kp.pt[1])
                cv2.circle(overlay, (x, y), 3, (0, 255, 0), 1)  # Green circles
        
        # Draw ORB keypoints
        orb_features = features.get('orb', {})
        if orb_features.get('keypoints'):
            for kp in orb_features['keypoints']:
                x, y = int(kp.pt[0]), int(kp.pt[1])
                cv2.circle(overlay, (x, y), 2, (255, 0, 0), 1)  # Blue circles
        
        # Draw minutiae
        minutiae = features.get('minutiae', [])
        for minutia in minutiae:
            x, y = int(minutia['x']), int(minutia['y'])
            if minutia['type'] == 'ending':
                cv2.drawMarker(overlay, (x, y), (0, 0, 255), cv2.MARKER_CROSS, 8, 2)  # Red cross
            else:  # bifurcation
                cv2.drawMarker(overlay, (x, y), (255, 0, 255), cv2.MARKER_TRIANGLE_UP, 8, 2)  # Magenta triangle
        
        return overlay
    
    def visualize_match_matrix(self, match_matrix: np.ndarray, 
                             file_names: Optional[List[str]] = None,
                             title: str = "Match Matrix") -> plt.Figure:
        """
        Visualize match matrix as heatmap.
        
        Args:
            match_matrix: Match matrix
            file_names: Optional list of file names for labels
            title: Plot title
            
        Returns:
            Matplotlib figure
        """
        try:
            fig, ax = plt.subplots(figsize=(12, 10))
            
            # Create heatmap
            im = ax.imshow(match_matrix, cmap='viridis', aspect='auto')
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('Match Score')
            
            # Set labels if provided
            if file_names:
                # Show only every nth label to avoid overcrowding
                n = max(1, len(file_names) // 20)
                ax.set_xticks(range(0, len(file_names), n))
                ax.set_yticks(range(0, len(file_names), n))
                ax.set_xticklabels([file_names[i] for i in range(0, len(file_names), n)], rotation=45)
                ax.set_yticklabels([file_names[i] for i in range(0, len(file_names), n)])
            
            ax.set_title(title)
            plt.tight_layout()
            return fig
        except Exception as e:
            logger.error(f"Match matrix visualization error: {e}")
            return plt.figure()
    
    def visualize_statistics(self, statistics: Dict[str, Any], 
                           title: str = "Processing Statistics") -> plt.Figure:
        """
        Visualize processing statistics.
        
        Args:
            statistics: Statistics dictionary
            title: Plot title
            
        Returns:
            Matplotlib figure
        """
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # Quality distribution
            if 'average_quality' in statistics:
                axes[0, 0].bar(['Average Quality'], [statistics['average_quality']])
                axes[0, 0].set_ylim(0, 1)
                axes[0, 0].set_title('Image Quality')
                axes[0, 0].set_ylabel('Quality Score')
            
            # Match score distribution
            if all(key in statistics for key in ['high_matches', 'medium_matches', 'low_matches']):
                match_counts = [statistics['high_matches'], statistics['medium_matches'], statistics['low_matches']]
                match_labels = ['High (>0.8)', 'Medium (0.5-0.8)', 'Low (<0.5)']
                
                axes[0, 1].pie(match_counts, labels=match_labels, autopct='%1.1f%%')
                axes[0, 1].set_title('Match Score Distribution')
            
            # Feature counts
            if all(key in statistics for key in ['average_sift_features', 'average_orb_features', 'average_minutiae']):
                feature_types = ['SIFT', 'ORB', 'Minutiae']
                feature_counts = [statistics['average_sift_features'], 
                                statistics['average_orb_features'], 
                                statistics['average_minutiae']]
                
                axes[1, 0].bar(feature_types, feature_counts)
                axes[1, 0].set_title('Average Feature Counts')
                axes[1, 0].set_ylabel('Count')
            
            # Summary text
            summary_text = f"Total Images: {statistics.get('total_images', 0)}\n"
            summary_text += f"Average Quality: {statistics.get('average_quality', 0):.3f}\n"
            summary_text += f"Average Match Score: {statistics.get('average_match_score', 0):.3f}\n"
            summary_text += f"Max Match Score: {statistics.get('max_match_score', 0):.3f}\n"
            summary_text += f"Min Match Score: {statistics.get('min_match_score', 0):.3f}"
            
            axes[1, 1].text(0.1, 0.5, summary_text, fontsize=12, verticalalignment='center')
            axes[1, 1].set_xlim(0, 1)
            axes[1, 1].set_ylim(0, 1)
            axes[1, 1].axis('off')
            axes[1, 1].set_title('Summary')
            
            plt.suptitle(title, fontsize=16)
            plt.tight_layout()
            return fig
        except Exception as e:
            logger.error(f"Statistics visualization error: {e}")
            return plt.figure()
    
    def save_visualization(self, fig: plt.Figure, output_path: str, dpi: int = 300):
        """
        Save visualization to file.
        
        Args:
            fig: Matplotlib figure
            output_path: Output file path
            dpi: Resolution in DPI
        """
        try:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            fig.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white')
            logger.info(f"Visualization saved to {output_path}")
        except Exception as e:
            logger.error(f"Error saving visualization: {e}")
    
    def create_comparison_plot(self, images: List[np.ndarray], 
                             titles: List[str],
                             suptitle: str = "Fingerprint Comparison") -> plt.Figure:
        """
        Create comparison plot of multiple fingerprints.
        
        Args:
            images: List of fingerprint images
            titles: List of titles for each image
            suptitle: Super title for the plot
            
        Returns:
            Matplotlib figure
        """
        try:
            n_images = len(images)
            cols = min(4, n_images)
            rows = (n_images + cols - 1) // cols
            
            fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
            
            if n_images == 1:
                axes = [axes]
            elif rows == 1:
                axes = axes.reshape(1, -1)
            
            for i, (image, title) in enumerate(zip(images, titles)):
                row = i // cols
                col = i % cols
                
                if rows > 1:
                    ax = axes[row, col]
                else:
                    ax = axes[col] if cols > 1 else axes[0]
                
                ax.imshow(image, cmap='gray')
                ax.set_title(title)
                ax.axis('off')
            
            # Hide unused subplots
            for i in range(n_images, rows * cols):
                row = i // cols
                col = i % cols
                if rows > 1:
                    axes[row, col].axis('off')
                else:
                    if cols > 1:
                        axes[col].axis('off')
            
            plt.suptitle(suptitle, fontsize=16)
            plt.tight_layout()
            return fig
        except Exception as e:
            logger.error(f"Comparison plot error: {e}")
            return plt.figure()
