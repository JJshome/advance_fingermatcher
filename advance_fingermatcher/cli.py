"""
Command line interface for advance fingerprint matcher.

This module provides CLI commands for fingerprint matching operations.
"""

import click
import numpy as np
from pathlib import Path
import json
from typing import Optional
import sys
import os

# Safe import with fallbacks
try:
    from .core.matcher import FingerprintMatcher
except ImportError:
    FingerprintMatcher = None

try:
    from .utils.batch_processor import BatchProcessor
except ImportError:
    BatchProcessor = None

try:
    from .utils.visualizer import Visualizer
except ImportError:
    Visualizer = None

try:
    from .utils.logger import setup_logging, get_logger
except ImportError:
    import logging
    def setup_logging(level, log_dir=None):
        logging.basicConfig(level=getattr(logging, level.upper()))
    
    def get_logger(name):
        return logging.getLogger(name)


@click.group()
@click.option('--log-level', default='INFO', help='Logging level')
@click.option('--log-dir', help='Directory for log files')
def main(log_level: str, log_dir: Optional[str]):
    """Advance Fingerprint Matcher CLI."""
    setup_logging(log_level, log_dir)


@main.command()
@click.option('--help', is_flag=True, help='Show this help message')
def demo(help):
    """Run a demonstration of the fingerprint matcher."""
    if help:
        click.echo("Demo command shows basic usage of the fingerprint matcher")
        return
    
    click.echo("🔍 Advanced Fingerprint Matcher Demo")
    click.echo("=====================================")
    click.echo()
    click.echo("This is a demonstration of the Advanced Fingerprint Matcher.")
    click.echo("Key Features:")
    click.echo("  • Enhanced Bozorth3 algorithm")
    click.echo("  • Deep learning integration")
    click.echo("  • High-performance matching")
    click.echo("  • Batch processing capabilities")
    click.echo()
    click.echo("Available commands:")
    click.echo("  fingermatcher match <image1> <image2>  - Match two fingerprints")
    click.echo("  fingermatcher batch <directory>        - Process multiple images")
    click.echo("  fingermatcher visualize <image>        - Show fingerprint features")
    click.echo("  fingermatcher serve                    - Start API server")
    click.echo()
    click.echo("Example usage:")
    click.echo("  fingermatcher match finger1.png finger2.png")
    click.echo("  fingermatcher batch ./fingerprints/")
    click.echo()
    click.echo("For more information, visit:")
    click.echo("  https://github.com/JJshome/advance_fingermatcher")


@main.command()
@click.argument('image1', type=click.Path(exists=True))
@click.argument('image2', type=click.Path(exists=True))
@click.option('--method', default='hybrid', help='Matching method')
@click.option('--output', help='Output file for results')
def match(image1: str, image2: str, method: str, output: Optional[str]):
    """Match two fingerprint images."""
    logger = get_logger(__name__)
    
    if FingerprintMatcher is None:
        click.echo("Fingerprint matcher module not available.", err=True)
        click.echo("This is a demo version. Full functionality requires complete installation.")
        return
    
    try:
        # Initialize matcher
        matcher = FingerprintMatcher()
        
        # Match fingerprints
        result = matcher.match_fingerprints(image1, image2, method=method)
        
        # Print results
        click.echo(f"Match Score: {result.score:.3f}")
        click.echo(f"Confidence: {result.confidence:.3f}")
        click.echo(f"Is Match: {'Yes' if result.is_match else 'No'}")
        click.echo(f"Method: {result.method_used}")
        click.echo(f"Processing Time: {result.processing_time:.3f}s")
        
        # Save results if output specified
        if output:
            result_dict = {
                'image1': str(image1),
                'image2': str(image2),
                'score': result.score,
                'confidence': result.confidence,
                'is_match': result.is_match,
                'method': result.method_used,
                'processing_time': result.processing_time
            }
            
            with open(output, 'w') as f:
                json.dump(result_dict, f, indent=2)
            
            click.echo(f"Results saved to {output}")
    
    except Exception as e:
        logger.error(f"Match error: {e}")
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@main.command()
@click.argument('directory', type=click.Path(exists=True))
@click.option('--output-dir', help='Output directory for results')
@click.option('--extensions', default='png,jpg,jpeg', help='File extensions to process')
def batch(directory: str, output_dir: Optional[str], extensions: str):
    """Process directory of fingerprint images."""
    logger = get_logger(__name__)
    
    if BatchProcessor is None:
        click.echo("Batch processor module not available.", err=True)
        click.echo("This is a demo version. Full functionality requires complete installation.")
        return
    
    try:
        # Parse extensions
        ext_list = [f'.{ext.strip()}' for ext in extensions.split(',')]
        
        # Initialize batch processor
        processor = BatchProcessor()
        
        # Process directory
        click.echo(f"Processing directory: {directory}")
        results = processor.process_directory(directory, ext_list, output_dir)
        
        # Print summary
        stats = results.get('statistics', {})
        click.echo(f"\nProcessing Summary:")
        click.echo(f"Total Images: {stats.get('total_images', 0)}")
        click.echo(f"Average Quality: {stats.get('average_quality', 0):.3f}")
        click.echo(f"Average Match Score: {stats.get('average_match_score', 0):.3f}")
        click.echo(f"High Matches (>0.8): {stats.get('high_matches', 0)}")
        
        if output_dir:
            click.echo(f"Results saved to {output_dir}")
    
    except Exception as e:
        logger.error(f"Batch processing error: {e}")
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@main.command()
@click.argument('image', type=click.Path(exists=True))
@click.option('--output', help='Output file for visualization')
def visualize(image: str, output: Optional[str]):
    """Visualize fingerprint with features."""
    logger = get_logger(__name__)
    
    if FingerprintMatcher is None or Visualizer is None:
        click.echo("Visualizer modules not available.", err=True)
        click.echo("This is a demo version. Full functionality requires complete installation.")
        return
    
    try:
        # Initialize components
        matcher = FingerprintMatcher()
        visualizer = Visualizer()
        
        # Load and process image
        img = matcher.load_image(image)
        processed_img = matcher.preprocess_image(img)
        features = matcher.extract_features(processed_img)
        
        # Create visualization
        fig = visualizer.visualize_fingerprint(img, features, Path(image).name)
        
        if output:
            visualizer.save_visualization(fig, output)
            click.echo(f"Visualization saved to {output}")
        else:
            import matplotlib.pyplot as plt
            plt.show()
    
    except Exception as e:
        logger.error(f"Visualization error: {e}")
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@main.command()
@click.option('--host', default='0.0.0.0', help='Host address')
@click.option('--port', default=8000, help='Port number')
@click.option('--reload', is_flag=True, help='Enable auto-reload')
def serve(host: str, port: int, reload: bool):
    """Start the API server."""
    try:
        import uvicorn
        from .api.server import create_app
        
        app = create_app()
        
        click.echo(f"Starting server on {host}:{port}")
        click.echo(f"API docs available at http://{host}:{port}/docs")
        
        uvicorn.run(app, host=host, port=port, reload=reload)
    
    except ImportError:
        click.echo("FastAPI and uvicorn required for API server", err=True)
        click.echo("Install with: pip install fastapi uvicorn", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"Server error: {e}", err=True)
        sys.exit(1)


@main.command()
def version():
    """Show version information."""
    click.echo("Advanced Fingerprint Matcher v1.0.1")
    click.echo("Enhanced Bozorth3 Algorithm with Deep Learning")
    click.echo("Copyright (c) 2025 JJshome")


if __name__ == '__main__':
    main()
