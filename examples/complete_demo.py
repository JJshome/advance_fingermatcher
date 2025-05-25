"""
Complete Example: Advanced Fingerprint Matching System
=====================================================

This example demonstrates all the advanced features of the fingerprint matching system:
1. Deep learning-based feature extraction
2. Graph neural network matching
3. Ultra-fast 1:N search
4. Quality assessment and adaptive processing
5. Multi-modal fusion
"""

import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
import time
from pathlib import Path
import logging

# Import our advanced fingerprint matching system
from advance_fingermatcher import (
    create_advanced_matcher,
    AdvancedFingerprintMatcher,
    MatchingResult,
    SearchResultAdvanced
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FingerprintMatchingDemo:
    """Complete demonstration of advanced fingerprint matching"""
    
    def __init__(self):
        """Initialize the demo"""
        print("🔬 Initializing Advanced Fingerprint Matching Demo")
        print("=" * 60)
        
        # Create the advanced matcher
        self.matcher = create_advanced_matcher()
        
        # Demo data
        self.demo_images = self._generate_demo_data()
        
        print(f"✅ System initialized successfully!")
        print(f"📊 Device: {self.matcher.device}")
        print(f"🧠 Networks loaded: {list(self.matcher.networks.keys())}")
        print()
    
    def _generate_demo_data(self) -> dict:
        """Generate realistic demo fingerprint images"""
        print("🎨 Generating demo fingerprint images...")
        
        images = {}
        
        # Generate synthetic fingerprint patterns
        for i in range(10):
            # Create a synthetic fingerprint-like pattern
            image = self._create_synthetic_fingerprint(i)
            images[f"fp_{i:03d}"] = image
        
        return images
    
    def _create_synthetic_fingerprint(self, seed: int) -> np.ndarray:
        """Create a synthetic fingerprint image"""
        np.random.seed(seed)
        
        # Create base image
        img = np.zeros((300, 300), dtype=np.uint8)
        
        # Add ridge patterns using sine waves
        x = np.arange(300)
        y = np.arange(300)
        X, Y = np.meshgrid(x, y)
        
        # Create ridge-like patterns
        frequency = 0.1 + np.random.random() * 0.05
        angle = np.random.random() * np.pi
        
        # Rotate coordinates
        X_rot = X * np.cos(angle) - Y * np.sin(angle)
        Y_rot = X * np.sin(angle) + Y * np.cos(angle)
        
        # Create ridge pattern
        ridge_pattern = np.sin(X_rot * frequency) * np.sin(Y_rot * frequency * 0.8)
        ridge_pattern = ((ridge_pattern + 1) * 127.5).astype(np.uint8)
        
        # Add noise and texture
        noise = np.random.normal(0, 20, (300, 300))
        img = np.clip(ridge_pattern + noise, 0, 255).astype(np.uint8)
        
        # Apply Gaussian blur to simulate skin texture
        img = cv2.GaussianBlur(img, (3, 3), 1.0)
        
        # Add some minutiae-like features
        for _ in range(np.random.randint(8, 15)):
            cx, cy = np.random.randint(50, 250, 2)
            # Add ending or bifurcation-like structures
            cv2.circle(img, (cx, cy), 2, 0, -1)
            
        return img
    
    def run_complete_demo(self):
        """Run the complete demonstration"""
        print("🚀 Starting Complete Advanced Fingerprint Matching Demo")
        print("=" * 60)
        
        # Demo 1: Feature Extraction
        self.demo_feature_extraction()
        
        # Demo 2: 1:1 Matching
        self.demo_1to1_matching()
        
        # Demo 3: Template Enrollment
        self.demo_template_enrollment()
        
        # Demo 4: 1:N Search
        self.demo_1toN_search()
        
        # Demo 5: Performance Benchmarking
        self.demo_performance_benchmarking()
        
        # Demo 6: System Statistics
        self.demo_system_statistics()
        
        print("\n🎉 Demo completed successfully!")
        print("=" * 60)
    
    def demo_feature_extraction(self):
        """Demonstrate advanced feature extraction"""
        print("\n📊 Demo 1: Advanced Feature Extraction")
        print("-" * 40)
        
        # Select a demo image
        image_id = "fp_001"
        image = self.demo_images[image_id]
        
        print(f"🖼️  Processing image: {image_id}")
        print(f"📏 Image shape: {image.shape}")
        
        # Extract comprehensive features
        start_time = time.time()
        features = self.matcher.extract_features(image)
        processing_time = time.time() - start_time
        
        # Display results
        print(f"⏱️  Processing time: {processing_time*1000:.2f}ms")
        print(f"🔍 Minutiae detected: {len(features['minutiae'])}")
        print(f"🎯 Overall quality: {features['quality']['overall']:.3f}")
        print(f"✨ Clarity: {features['quality']['clarity']:.3f}")
        print(f"🌈 Contrast: {features['quality']['contrast']:.3f}")
        print(f"📐 Sharpness: {features['quality']['sharpness']:.3f}")
        print(f"🧠 Fusion confidence: {features['fusion_confidence']:.3f}")
        
        # Show some minutiae details
        if features['minutiae']:
            print(f"\n📍 Sample minutiae:")
            for i, minutia in enumerate(features['minutiae'][:3]):
                x, y, theta, quality, m_type = minutia
                type_name = "Ending" if m_type == 0 else "Bifurcation"
                print(f"  {i+1}. ({x:.1f}, {y:.1f}) θ={theta:.2f} Q={quality:.3f} [{type_name}]")
    
    def demo_1to1_matching(self):
        """Demonstrate 1:1 fingerprint matching"""
        print("\n🤝 Demo 2: 1:1 Fingerprint Matching")
        print("-" * 40)
        
        # Test genuine match (same person, different impressions)
        probe_img = self.demo_images["fp_001"]
        gallery_img = self.demo_images["fp_002"]  # Similar but different
        
        print("🔍 Testing genuine match...")
        start_time = time.time()
        result = self.matcher.match_1to1(probe_img, gallery_img, return_details=True)
        matching_time = time.time() - start_time
        
        print(f"⏱️  Matching time: {matching_time*1000:.2f}ms")
        print(f"📊 Match score: {result.match_score:.4f}")
        print(f"✅ Is match: {result.is_match}")
        print(f"🎯 Confidence: {result.confidence:.4f}")
        print(f"🔍 Probe quality: {result.probe_quality:.3f}")
        print(f"🎨 Gallery quality: {result.gallery_quality:.3f}")
        print(f"📍 Matched minutiae: {result.matched_minutiae_count}")
        print(f"🧠 Graph matching score: {result.graph_matching_score:.4f}")
        print(f"📐 Geometric verification: {result.geometric_verification}")
        
        # Test impostor match (different people)
        print("\n🚫 Testing impostor match...")
        impostor_img = self.demo_images["fp_007"]  # Very different
        
        start_time = time.time()
        result_impostor = self.matcher.match_1to1(probe_img, impostor_img)
        matching_time = time.time() - start_time
        
        print(f"⏱️  Matching time: {matching_time*1000:.2f}ms")
        print(f"📊 Match score: {result_impostor.match_score:.4f}")
        print(f"❌ Is match: {result_impostor.is_match}")
        print(f"🎯 Confidence: {result_impostor.confidence:.4f}")
    
    def demo_template_enrollment(self):
        """Demonstrate template enrollment"""
        print("\n📝 Demo 3: Template Enrollment")
        print("-" * 40)
        
        enrolled_count = 0
        
        # Enroll several templates
        for i, (image_id, image) in enumerate(self.demo_images.items()):
            print(f"📋 Enrolling template: {image_id}")
            
            metadata = {
                'person_id': f"person_{i//2:03d}",  # Each person has 2 impressions
                'finger_id': i % 10,  # Finger number
                'enrollment_date': time.time(),
                'image_id': image_id
            }
            
            start_time = time.time()
            result = self.matcher.enroll_template(image_id, image, metadata)
            enrollment_time = time.time() - start_time
            
            if result['success']:
                enrolled_count += 1
                print(f"  ✅ Success! Quality: {result['quality']:.3f}, "
                      f"Minutiae: {result['num_minutiae']}, "
                      f"Time: {enrollment_time*1000:.1f}ms")
            else:
                print(f"  ❌ Failed: {result['reason']}")
        
        print(f"\n📊 Enrollment Summary:")
        print(f"  ✅ Successfully enrolled: {enrolled_count}/{len(self.demo_images)}")
        print(f"  🏆 Success rate: {enrolled_count/len(self.demo_images)*100:.1f}%")
    
    def demo_1toN_search(self):
        """Demonstrate 1:N fingerprint identification"""
        print("\n🔍 Demo 4: 1:N Fingerprint Identification")
        print("-" * 40)
        
        # Use a probe image
        probe_img = self.demo_images["fp_003"]
        
        print("🔎 Performing 1:N search...")
        start_time = time.time()
        search_results = self.matcher.search_1toN(probe_img, k=5, similarity_threshold=0.1)
        search_time = time.time() - start_time
        
        print(f"⏱️  Search time: {search_time*1000:.2f}ms")
        print(f"📊 Candidates found: {len(search_results)}")
        
        if search_results:
            print(f"\n🏆 Top candidates:")
            for i, result in enumerate(search_results[:5]):
                print(f"  {result.rank}. {result.candidate_id}")
                print(f"     Score: {result.match_score:.4f}")
                print(f"     Confidence: {result.confidence:.4f}")
                print(f"     Template Quality: {result.template_quality:.3f}")
                
                if result.metadata:
                    person_id = result.metadata.get('person_id', 'Unknown')
                    finger_id = result.metadata.get('finger_id', 'Unknown')
                    print(f"     Person: {person_id}, Finger: {finger_id}")
                print()
        else:
            print("❌ No candidates found above threshold")
    
    def demo_performance_benchmarking(self):
        """Demonstrate performance benchmarking"""
        print("\n⚡ Demo 5: Performance Benchmarking")
        print("-" * 40)
        
        # Benchmark feature extraction
        print("🏃 Benchmarking feature extraction...")
        images = list(self.demo_images.values())[:5]
        
        extraction_times = []
        for img in images:
            start_time = time.time()
            self.matcher.extract_features(img)
            extraction_times.append(time.time() - start_time)
        
        avg_extraction_time = np.mean(extraction_times)
        extraction_fps = 1.0 / avg_extraction_time
        
        print(f"  📊 Average extraction time: {avg_extraction_time*1000:.2f}ms")
        print(f"  🚀 Extraction throughput: {extraction_fps:.1f} FPS")
        
        # Benchmark 1:1 matching
        print("\n🤝 Benchmarking 1:1 matching...")
        matching_times = []
        
        for i in range(min(10, len(images)//2)):
            start_time = time.time()
            self.matcher.match_1to1(images[i*2], images[i*2+1])
            matching_times.append(time.time() - start_time)
        
        avg_matching_time = np.mean(matching_times)
        matching_fps = 1.0 / avg_matching_time
        
        print(f"  📊 Average matching time: {avg_matching_time*1000:.2f}ms")
        print(f"  🚀 Matching throughput: {matching_fps:.1f} FPS")
        
        # Benchmark 1:N search
        print("\n🔍 Benchmarking 1:N search...")
        search_times = []
        
        for img in images[:3]:
            start_time = time.time()
            self.matcher.search_1toN(img, k=10)
            search_times.append(time.time() - start_time)
        
        avg_search_time = np.mean(search_times)
        search_fps = 1.0 / avg_search_time
        
        print(f"  📊 Average search time: {avg_search_time*1000:.2f}ms")
        print(f"  🚀 Search throughput: {search_fps:.1f} QPS")
    
    def demo_system_statistics(self):
        """Demonstrate system statistics"""
        print("\n📈 Demo 6: System Statistics")
        print("-" * 40)
        
        stats = self.matcher.get_system_stats()
        
        # Matching statistics
        matching_stats = stats['matching_stats']
        print("🤝 Matching Statistics:")
        print(f"  Total matches: {matching_stats['total_matches']}")
        print(f"  Average match time: {matching_stats['avg_match_time']*1000:.2f}ms")
        
        # Search statistics
        search_stats = stats['search_stats']['search_stats']
        print(f"\n🔍 Search Statistics:")
        print(f"  Total searches: {search_stats['total_searches']}")
        print(f"  Total templates: {search_stats['total_templates']}")
        print(f"  Average search time: {search_stats['avg_search_time']*1000:.2f}ms")
        
        # Cache statistics
        cache_stats = stats['search_stats']['cache_stats']
        print(f"\n💾 Cache Statistics:")
        print(f"  Cache hits: {cache_stats['hits']}")
        print(f"  Cache misses: {cache_stats['misses']}")
        print(f"  Hit rate: {cache_stats['hit_rate']*100:.1f}%")
        print(f"  Memory cache size: {cache_stats['memory_cache_size']}")
        
        # System information
        system_info = stats['system_info']
        print(f"\n🖥️  System Information:")
        print(f"  Device: {system_info['device']}")
        print(f"  Models loaded: {', '.join(system_info['models_loaded'])}")
        
        # Index information
        index_stats = stats['search_stats']['index_stats']
        print(f"\n📚 Index Information:")
        print(f"  Index type: {index_stats['index_type']}")
        print(f"  Total vectors: {index_stats['total_vectors']}")
        print(f"  Dimension: {index_stats['dimension']}")
    
    def visualize_features(self, image_id: str = "fp_001"):
        """Visualize extracted features (requires matplotlib)"""
        try:
            import matplotlib.pyplot as plt
            
            print(f"\n🎨 Visualizing features for {image_id}")
            print("-" * 40)
            
            image = self.demo_images[image_id]
            features = self.matcher.extract_features(image)
            
            # Create visualization
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            
            # Original image
            axes[0, 0].imshow(image, cmap='gray')
            axes[0, 0].set_title('Original Image')
            axes[0, 0].axis('off')
            
            # Image with minutiae
            axes[0, 1].imshow(image, cmap='gray')
            if features['minutiae']:
                minutiae = np.array(features['minutiae'])
                axes[0, 1].scatter(minutiae[:, 0], minutiae[:, 1], 
                                 c='red', s=30, marker='x')
            axes[0, 1].set_title(f'Detected Minutiae ({len(features["minutiae"])})')
            axes[0, 1].axis('off')
            
            # Quality map
            quality_map = features['quality']['local_map']
            im = axes[1, 0].imshow(quality_map, cmap='jet')
            axes[1, 0].set_title('Quality Map')
            axes[1, 0].axis('off')
            plt.colorbar(im, ax=axes[1, 0])
            
            # Quality metrics
            quality_metrics = [
                features['quality']['overall'],
                features['quality']['clarity'],
                features['quality']['contrast'],
                features['quality']['sharpness']
            ]
            metric_names = ['Overall', 'Clarity', 'Contrast', 'Sharpness']
            
            axes[1, 1].bar(metric_names, quality_metrics)
            axes[1, 1].set_title('Quality Metrics')
            axes[1, 1].set_ylim(0, 1)
            axes[1, 1].tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            plt.savefig(f'fingerprint_analysis_{image_id}.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            print(f"📁 Visualization saved as 'fingerprint_analysis_{image_id}.png'")
            
        except ImportError:
            print("❌ Matplotlib not available for visualization")
        except Exception as e:
            print(f"❌ Visualization error: {e}")


def main():
    """Main demo function"""
    print("🎯 Advanced Fingerprint Matching System - Complete Demo")
    print("=" * 60)
    print("This demo showcases all the advanced features including:")
    print("• Deep learning-based feature extraction")
    print("• Graph neural network matching")
    print("• Ultra-fast 1:N search with learned indexing")
    print("• Quality assessment and adaptive processing")
    print("• Multi-modal fusion techniques")
    print("• Real-time performance optimization")
    print()
    
    try:
        # Create and run demo
        demo = FingerprintMatchingDemo()
        demo.run_complete_demo()
        
        # Optional: Create visualization
        response = input("\n🎨 Would you like to create feature visualizations? (y/n): ")
        if response.lower() == 'y':
            demo.visualize_features()
        
        print("\n🎉 Thank you for trying the Advanced Fingerprint Matching System!")
        print("📚 For more information, check out the documentation:")
        print("   https://github.com/JJshome/advance_fingermatcher")
        
    except KeyboardInterrupt:
        print("\n👋 Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
