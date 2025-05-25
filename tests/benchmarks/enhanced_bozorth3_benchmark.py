#!/usr/bin/env python3
"""
Enhanced Bozorth3 Benchmark Suite

Comprehensive benchmarking and performance analysis for the Enhanced Bozorth3
algorithm with comparison against traditional implementations.

Author: JJshome
Date: 2025
"""

import sys
import time
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from advance_fingermatcher.algorithms.enhanced_bozorth3 import EnhancedBozorth3Matcher
    from advance_fingermatcher.utils.logger import get_logger
    FULL_BENCHMARK = True
except ImportError:
    FULL_BENCHMARK = False


@dataclass
class BenchmarkResult:
    """Store benchmark results"""
    algorithm_name: str
    configuration: Dict[str, Any]
    match_scores: List[float]
    processing_times: List[float]
    memory_usage: List[float]
    accuracy_metrics: Dict[str, float]
    template_count: int
    minutiae_count_avg: float


class EnhancedBozorth3Benchmark:
    """Comprehensive benchmark suite for Enhanced Bozorth3"""
    
    def __init__(self, output_dir: str = "benchmark_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.logger = get_logger("benchmark") if FULL_BENCHMARK else None
        self.results = []
        
    def generate_synthetic_templates(self, count: int, minutiae_range: Tuple[int, int] = (20, 80),
                                   quality_range: Tuple[float, float] = (0.2, 1.0)) -> List[List[Dict]]:
        """Generate synthetic minutiae templates for testing"""
        np.random.seed(42)  # Reproducible results
        templates = []
        
        for i in range(count):
            minutiae_count = np.random.randint(*minutiae_range)
            template = []
            
            for j in range(minutiae_count):
                minutia = {
                    'x': np.random.randint(10, 490),
                    'y': np.random.randint(10, 390),
                    'angle': np.random.uniform(0, 2*np.pi),
                    'type': np.random.choice(['ridge_ending', 'bifurcation']),
                    'quality': np.random.uniform(*quality_range),
                    'descriptor': np.random.rand(64).tolist()  # Mock descriptor
                }
                template.append(minutia)
            
            templates.append(template)
        
        return templates
    
    def benchmark_matching_speed(self, configurations: List[Dict], 
                                template_pairs: List[Tuple]) -> Dict[str, BenchmarkResult]:
        """Benchmark matching speed across different configurations"""
        if not FULL_BENCHMARK:
            return {}
            
        print("🚀 Running matching speed benchmark...")
        results = {}
        
        for config in configurations:
            config_name = config['name']
            matcher = EnhancedBozorth3Matcher(**config.get('params', {}))
            
            match_scores = []
            processing_times = []
            
            print(f"  Testing configuration: {config_name}")
            
            for template1, template2 in template_pairs:
                start_time = time.perf_counter()
                score = matcher.match_minutiae(template1, template2)
                elapsed = time.perf_counter() - start_time
                
                match_scores.append(score)
                processing_times.append(elapsed * 1000)  # Convert to ms
            
            avg_minutiae = np.mean([len(t1) + len(t2) for t1, t2 in template_pairs]) / 2
            
            result = BenchmarkResult(
                algorithm_name=f"Enhanced Bozorth3 ({config_name})",
                configuration=config.get('params', {}),
                match_scores=match_scores,
                processing_times=processing_times,
                memory_usage=[],  # TODO: Implement memory profiling
                accuracy_metrics={},
                template_count=len(template_pairs),
                minutiae_count_avg=avg_minutiae
            )
            
            results[config_name] = result
            
            print(f"    Avg Score: {np.mean(match_scores):.4f}")
            print(f"    Avg Time: {np.mean(processing_times):.2f}ms")
            print(f"    Std Time: {np.std(processing_times):.2f}ms")
        
        return results
    
    def benchmark_scalability(self, base_config: Dict) -> BenchmarkResult:
        """Test scalability with increasing template sizes"""
        if not FULL_BENCHMARK:
            return None
            
        print("📈 Running scalability benchmark...")
        
        matcher = EnhancedBozorth3Matcher(**base_config)
        minutiae_counts = [10, 20, 50, 100, 150, 200, 300]
        
        processing_times = []
        memory_usage = []
        
        for count in minutiae_counts:
            print(f"  Testing with {count} minutiae...")
            
            # Generate templates with specific minutiae count
            template1 = self.generate_synthetic_templates(1, (count, count))[0]
            template2 = self.generate_synthetic_templates(1, (count, count))[0]
            
            # Multiple runs for accuracy
            times = []
            for _ in range(10):
                start_time = time.perf_counter()
                matcher.match_minutiae(template1, template2)
                elapsed = time.perf_counter() - start_time
                times.append(elapsed * 1000)
            
            processing_times.append({
                'minutiae_count': count,
                'avg_time': np.mean(times),
                'std_time': np.std(times)
            })
        
        return BenchmarkResult(
            algorithm_name="Enhanced Bozorth3 (Scalability)",
            configuration=base_config,
            match_scores=[],
            processing_times=processing_times,
            memory_usage=memory_usage,
            accuracy_metrics={},
            template_count=len(minutiae_counts),
            minutiae_count_avg=np.mean(minutiae_counts)
        )
    
    def benchmark_quality_impact(self) -> Dict[str, BenchmarkResult]:
        """Analyze impact of minutiae quality on matching performance"""
        if not FULL_BENCHMARK:
            return {}
            
        print("🎯 Running quality impact benchmark...")
        
        quality_ranges = [
            ('High Quality', (0.8, 1.0)),
            ('Medium Quality', (0.5, 0.8)),
            ('Low Quality', (0.2, 0.5)),
            ('Mixed Quality', (0.2, 1.0))
        ]
        
        results = {}
        
        for quality_name, quality_range in quality_ranges:
            print(f"  Testing {quality_name}...")
            
            # Generate templates with specific quality range
            templates = self.generate_synthetic_templates(50, (30, 60), quality_range)
            template_pairs = [(templates[i], templates[i+1]) for i in range(0, len(templates)-1, 2)]
            
            # Test with quality weighting enabled
            matcher_with_quality = EnhancedBozorth3Matcher(quality_weighting=True)
            scores_with_quality = []
            times_with_quality = []
            
            for template1, template2 in template_pairs:
                start_time = time.perf_counter()
                score = matcher_with_quality.match_minutiae(template1, template2)
                elapsed = time.perf_counter() - start_time
                
                scores_with_quality.append(score)
                times_with_quality.append(elapsed * 1000)
            
            # Test without quality weighting
            matcher_without_quality = EnhancedBozorth3Matcher(quality_weighting=False)
            scores_without_quality = []
            times_without_quality = []
            
            for template1, template2 in template_pairs:
                start_time = time.perf_counter()
                score = matcher_without_quality.match_minutiae(template1, template2)
                elapsed = time.perf_counter() - start_time
                
                scores_without_quality.append(score)
                times_without_quality.append(elapsed * 1000)
            
            # Calculate improvement metrics
            score_improvement = (np.mean(scores_with_quality) - np.mean(scores_without_quality)) / np.mean(scores_without_quality) * 100
            
            result = BenchmarkResult(
                algorithm_name=f"Enhanced Bozorth3 ({quality_name})",
                configuration={'quality_range': quality_range},
                match_scores=scores_with_quality,
                processing_times=times_with_quality,
                memory_usage=[],
                accuracy_metrics={
                    'score_with_quality': np.mean(scores_with_quality),
                    'score_without_quality': np.mean(scores_without_quality),
                    'improvement_percent': score_improvement,
                    'score_std_with_quality': np.std(scores_with_quality),
                    'score_std_without_quality': np.std(scores_without_quality)
                },
                template_count=len(template_pairs),
                minutiae_count_avg=45.0
            )
            
            results[quality_name] = result
            
            print(f"    Avg Score (with quality): {np.mean(scores_with_quality):.4f}")
            print(f"    Avg Score (without quality): {np.mean(scores_without_quality):.4f}")
            print(f"    Improvement: {score_improvement:+.1f}%")
        
        return results
    
    def benchmark_parallel_processing(self, base_config: Dict) -> BenchmarkResult:
        """Test parallel processing capabilities"""
        if not FULL_BENCHMARK:
            return None
            
        print("⚡ Running parallel processing benchmark...")
        
        # Generate large number of template pairs
        templates = self.generate_synthetic_templates(200, (40, 80))
        template_pairs = [(templates[i], templates[i+100]) for i in range(100)]
        
        matcher = EnhancedBozorth3Matcher(**base_config)
        
        def match_pair(pair):
            template1, template2 = pair
            start_time = time.perf_counter()
            score = matcher.match_minutiae(template1, template2)
            elapsed = time.perf_counter() - start_time
            return score, elapsed * 1000
        
        # Sequential processing
        print("  Testing sequential processing...")
        sequential_start = time.perf_counter()
        sequential_results = [match_pair(pair) for pair in template_pairs]
        sequential_time = time.perf_counter() - sequential_start
        
        # Parallel processing
        print("  Testing parallel processing...")
        parallel_start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=4) as executor:
            parallel_results = list(executor.map(match_pair, template_pairs))
        parallel_time = time.perf_counter() - parallel_start
        
        speedup = sequential_time / parallel_time
        
        return BenchmarkResult(
            algorithm_name="Enhanced Bozorth3 (Parallel)",
            configuration=base_config,
            match_scores=[r[0] for r in parallel_results],
            processing_times=[r[1] for r in parallel_results],
            memory_usage=[],
            accuracy_metrics={
                'sequential_total_time': sequential_time,
                'parallel_total_time': parallel_time,
                'speedup': speedup,
                'efficiency': speedup / 4  # 4 workers
            },
            template_count=len(template_pairs),
            minutiae_count_avg=60.0
        )
    
    def generate_performance_report(self, results: Dict[str, BenchmarkResult]):
        """Generate comprehensive performance report"""
        if not results:
            print("⚠️  No results to generate report")
            return
            
        print("📊 Generating performance report...")
        
        # Create performance summary
        summary = {
            'benchmark_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'total_configurations': len(results),
            'results_summary': {}
        }
        
        for name, result in results.items():
            if result and result.processing_times:
                avg_time = np.mean(result.processing_times) if isinstance(result.processing_times[0], (int, float)) else None
                avg_score = np.mean(result.match_scores) if result.match_scores else None
                
                summary['results_summary'][name] = {
                    'average_processing_time_ms': avg_time,
                    'average_match_score': avg_score,
                    'template_count': result.template_count,
                    'minutiae_count_avg': result.minutiae_count_avg,
                    'accuracy_metrics': result.accuracy_metrics
                }
        
        # Save JSON report
        report_file = self.output_dir / 'performance_report.json'
        with open(report_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        print(f"📄 Performance report saved to: {report_file}")
        
        # Generate visualizations if matplotlib is available
        try:
            self.generate_visualizations(results)
        except ImportError:
            print("⚠️  Matplotlib not available for visualizations")
    
    def generate_visualizations(self, results: Dict[str, BenchmarkResult]):
        """Generate performance visualization charts"""
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Enhanced Bozorth3 Performance Analysis', fontsize=16, fontweight='bold')
        
        # Processing time comparison
        config_names = []
        avg_times = []
        
        for name, result in results.items():
            if result and result.processing_times:
                if isinstance(result.processing_times[0], (int, float)):
                    config_names.append(name)
                    avg_times.append(np.mean(result.processing_times))
        
        if config_names:
            axes[0, 0].bar(config_names, avg_times, color='skyblue', alpha=0.7)
            axes[0, 0].set_title('Average Processing Time by Configuration')
            axes[0, 0].set_ylabel('Time (ms)')
            axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Match score distribution
        all_scores = []
        score_labels = []
        
        for name, result in results.items():
            if result and result.match_scores:
                all_scores.extend(result.match_scores)
                score_labels.extend([name] * len(result.match_scores))
        
        if all_scores:
            axes[0, 1].hist(all_scores, bins=20, alpha=0.7, color='lightgreen')
            axes[0, 1].set_title('Match Score Distribution')
            axes[0, 1].set_xlabel('Match Score')
            axes[0, 1].set_ylabel('Frequency')
        
        # Quality impact visualization
        quality_results = {k: v for k, v in results.items() if 'Quality' in k}
        if quality_results:
            quality_names = list(quality_results.keys())
            improvements = [r.accuracy_metrics.get('improvement_percent', 0) for r in quality_results.values()]
            
            axes[1, 0].bar(quality_names, improvements, color='orange', alpha=0.7)
            axes[1, 0].set_title('Quality Weighting Impact')
            axes[1, 0].set_ylabel('Improvement (%)')
            axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Performance vs accuracy scatter
        times = []
        scores = []
        names = []
        
        for name, result in results.items():
            if result and result.processing_times and result.match_scores:
                if isinstance(result.processing_times[0], (int, float)):
                    times.append(np.mean(result.processing_times))
                    scores.append(np.mean(result.match_scores))
                    names.append(name)
        
        if times and scores:
            scatter = axes[1, 1].scatter(times, scores, alpha=0.7, s=100, c=range(len(times)), cmap='viridis')
            axes[1, 1].set_title('Processing Time vs Match Score')
            axes[1, 1].set_xlabel('Average Processing Time (ms)')
            axes[1, 1].set_ylabel('Average Match Score')
            
            # Add labels
            for i, name in enumerate(names):
                axes[1, 1].annotate(name, (times[i], scores[i]), xytext=(5, 5), 
                                  textcoords='offset points', fontsize=8)
        
        plt.tight_layout()
        
        # Save visualization
        viz_file = self.output_dir / 'performance_visualization.png'
        plt.savefig(viz_file, dpi=300, bbox_inches='tight')
        print(f"📊 Visualization saved to: {viz_file}")
        
        plt.show()
    
    def run_comprehensive_benchmark(self):
        """Run all benchmark tests"""
        print("🚀 Starting Enhanced Bozorth3 Comprehensive Benchmark")
        print("=" * 60)
        
        if not FULL_BENCHMARK:
            print("⚠️  Full benchmark requires complete installation")
            print("   Run: pip install -e \".[dev,ml,viz]\"")
            return
        
        # Generate test data
        print("📋 Generating test templates...")
        templates = self.generate_synthetic_templates(100, (30, 80))
        template_pairs = [(templates[i], templates[i+50]) for i in range(50)]
        
        # Define configurations to test
        configurations = [
            {
                'name': 'Basic',
                'params': {
                    'quality_weighting': False,
                    'descriptor_matching': False
                }
            },
            {
                'name': 'Quality Weighted',
                'params': {
                    'quality_weighting': True,
                    'descriptor_matching': False
                }
            },
            {
                'name': 'Full Enhanced',
                'params': {
                    'quality_weighting': True,
                    'descriptor_matching': True,
                    'base_tolerances': {'distance': 10.0, 'angle': 0.26}
                }
            }
        ]
        
        all_results = {}
        
        # Run benchmarks
        try:
            # Speed benchmark
            speed_results = self.benchmark_matching_speed(configurations, template_pairs)
            all_results.update(speed_results)
            
            # Scalability benchmark
            scalability_result = self.benchmark_scalability(configurations[2]['params'])
            if scalability_result:
                all_results['Scalability'] = scalability_result
            
            # Quality impact benchmark
            quality_results = self.benchmark_quality_impact()
            all_results.update(quality_results)
            
            # Parallel processing benchmark
            parallel_result = self.benchmark_parallel_processing(configurations[2]['params'])
            if parallel_result:
                all_results['Parallel Processing'] = parallel_result
            
            # Generate report
            self.generate_performance_report(all_results)
            
            print("\n✅ Comprehensive benchmark completed successfully!")
            print(f"📁 Results saved to: {self.output_dir}")
            
        except Exception as e:
            print(f"\n❌ Benchmark failed: {e}")
            import traceback
            traceback.print_exc()


def main():
    """Main benchmark execution"""
    benchmark = EnhancedBozorth3Benchmark()
    benchmark.run_comprehensive_benchmark()


if __name__ == "__main__":
    main()
