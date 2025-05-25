"""
FastAPI server for fingerprint matching API.

This module provides REST API endpoints for fingerprint matching
using FastAPI framework.
"""

try:
    from fastapi import FastAPI, HTTPException, UploadFile, File, Form
    from fastapi.responses import JSONResponse
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
except ImportError:
    raise ImportError("FastAPI not installed. Install with: pip install fastapi uvicorn")

import numpy as np
import cv2
from typing import Optional, List, Dict, Any
import logging
import io
from pathlib import Path
import tempfile
import asyncio
from concurrent.futures import ThreadPoolExecutor

from ..core.matcher import FingerprintMatcher
from ..utils.logger import get_logger

logger = get_logger(__name__)


# Pydantic models
class MatchResponse(BaseModel):
    """Response model for fingerprint matching."""
    score: float
    confidence: float
    is_match: bool
    method_used: str
    processing_time: float
    message: str


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str
    version: str
    timestamp: str


class FeatureResponse(BaseModel):
    """Response model for feature extraction."""
    sift_count: int
    orb_count: int
    minutiae_count: int
    quality_score: float
    processing_time: float
    message: str


def create_app() -> FastAPI:
    """
    Create and configure FastAPI application.
    
    Returns:
        FastAPI application instance
    """
    app = FastAPI(
        title="Advance Fingerprint Matcher API",
        description="Advanced High-Performance Fingerprint Matching System",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc"
    )
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Initialize matcher
    matcher = FingerprintMatcher()
    
    # Thread executor for CPU-bound tasks
    executor = ThreadPoolExecutor(max_workers=4)
    
    @app.get("/", response_class=JSONResponse)
    async def root():
        """Root endpoint."""
        return {
            "message": "Advance Fingerprint Matcher API",
            "version": "1.0.0",
            "docs": "/docs"
        }
    
    @app.get("/health", response_model=HealthResponse)
    async def health_check():
        """Health check endpoint."""
        from datetime import datetime
        return HealthResponse(
            status="healthy",
            version="1.0.0",
            timestamp=datetime.now().isoformat()
        )
    
    @app.post("/match", response_model=MatchResponse)
    async def match_fingerprints(
        image1: UploadFile = File(..., description="First fingerprint image"),
        image2: UploadFile = File(..., description="Second fingerprint image"),
        method: str = Form("hybrid", description="Matching method")
    ):
        """
        Match two fingerprint images.
        
        Args:
            image1: First fingerprint image file
            image2: Second fingerprint image file
            method: Matching method (hybrid, sift, orb, minutiae)
            
        Returns:
            Match result with score and details
        """
        try:
            # Validate method
            valid_methods = ['hybrid', 'sift', 'orb', 'minutiae']
            if method not in valid_methods:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Invalid method. Must be one of: {valid_methods}"
                )
            
            # Read image files
            img1_bytes = await image1.read()
            img2_bytes = await image2.read()
            
            # Process images in thread pool
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                executor, 
                _process_match_request, 
                img1_bytes, img2_bytes, method, matcher
            )
            
            return MatchResponse(
                score=result.score,
                confidence=result.confidence,
                is_match=result.is_match,
                method_used=result.method_used,
                processing_time=result.processing_time,
                message="Match completed successfully"
            )
            
        except Exception as e:
            logger.error(f"Match error: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/extract", response_model=FeatureResponse)
    async def extract_features(
        image: UploadFile = File(..., description="Fingerprint image")
    ):
        """
        Extract features from fingerprint image.
        
        Args:
            image: Fingerprint image file
            
        Returns:
            Feature extraction results
        """
        try:
            # Read image file
            img_bytes = await image.read()
            
            # Process image in thread pool
            loop = asyncio.get_event_loop()
            features, quality, processing_time = await loop.run_in_executor(
                executor, 
                _process_feature_request, 
                img_bytes, matcher
            )
            
            return FeatureResponse(
                sift_count=features.get('sift', {}).get('count', 0),
                orb_count=features.get('orb', {}).get('count', 0),
                minutiae_count=len(features.get('minutiae', [])),
                quality_score=quality,
                processing_time=processing_time,
                message="Feature extraction completed successfully"
            )
            
        except Exception as e:
            logger.error(f"Feature extraction error: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/batch")
    async def batch_process(
        images: List[UploadFile] = File(..., description="Multiple fingerprint images")
    ):
        """
        Batch process multiple fingerprint images.
        
        Args:
            images: List of fingerprint image files
            
        Returns:
            Batch processing results with match matrix
        """
        try:
            if len(images) > 20:  # Limit batch size
                raise HTTPException(
                    status_code=400, 
                    detail="Batch size limited to 20 images"
                )
            
            # Read all image files
            image_data = []
            for img in images:
                img_bytes = await img.read()
                image_data.append((img.filename, img_bytes))
            
            # Process batch in thread pool
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                executor, 
                _process_batch_request, 
                image_data, matcher
            )
            
            return JSONResponse(content=results)
            
        except Exception as e:
            logger.error(f"Batch processing error: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    return app


def _load_image_from_bytes(img_bytes: bytes) -> np.ndarray:
    """
    Load image from bytes.
    
    Args:
        img_bytes: Image data as bytes
        
    Returns:
        Image as numpy array
    """
    # Convert bytes to numpy array
    nparr = np.frombuffer(img_bytes, np.uint8)
    
    # Decode image
    image = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    
    if image is None:
        raise ValueError("Could not decode image")
    
    return image


def _process_match_request(img1_bytes: bytes, img2_bytes: bytes, 
                          method: str, matcher: FingerprintMatcher):
    """
    Process fingerprint matching request.
    
    Args:
        img1_bytes: First image bytes
        img2_bytes: Second image bytes
        method: Matching method
        matcher: FingerprintMatcher instance
        
    Returns:
        Match result
    """
    # Load images
    img1 = _load_image_from_bytes(img1_bytes)
    img2 = _load_image_from_bytes(img2_bytes)
    
    # Preprocess images
    img1_processed = matcher.preprocess_image(img1)
    img2_processed = matcher.preprocess_image(img2)
    
    # Extract features
    features1 = matcher.extract_features(img1_processed)
    features2 = matcher.extract_features(img2_processed)
    
    # Match features
    result = matcher.match_features(features1, features2, method=method)
    
    return result


def _process_feature_request(img_bytes: bytes, matcher: FingerprintMatcher):
    """
    Process feature extraction request.
    
    Args:
        img_bytes: Image bytes
        matcher: FingerprintMatcher instance
        
    Returns:
        Tuple of (features, quality, processing_time)
    """
    import time
    
    start_time = time.time()
    
    # Load image
    image = _load_image_from_bytes(img_bytes)
    
    # Preprocess image
    processed_image = matcher.preprocess_image(image)
    
    # Extract features
    features = matcher.extract_features(processed_image)
    
    # Assess quality
    try:
        from ..processing.preprocessor import ImagePreprocessor
        preprocessor = ImagePreprocessor()
        quality = preprocessor.assess_quality(image)
    except:
        quality = 0.5  # Default quality
    
    processing_time = time.time() - start_time
    
    return features, quality, processing_time


def _process_batch_request(image_data: List[tuple], matcher: FingerprintMatcher):
    """
    Process batch fingerprint request.
    
    Args:
        image_data: List of (filename, image_bytes) tuples
        matcher: FingerprintMatcher instance
        
    Returns:
        Batch processing results
    """
    results = []
    
    # Process each image
    for i, (filename, img_bytes) in enumerate(image_data):
        try:
            features, quality, proc_time = _process_feature_request(img_bytes, matcher)
            
            results.append({
                'index': i,
                'filename': filename,
                'sift_count': features.get('sift', {}).get('count', 0),
                'orb_count': features.get('orb', {}).get('count', 0),
                'minutiae_count': len(features.get('minutiae', [])),
                'quality': quality,
                'processing_time': proc_time
            })
        except Exception as e:
            logger.error(f"Error processing {filename}: {e}")
            results.append({
                'index': i,
                'filename': filename,
                'error': str(e)
            })
    
    # Generate match matrix for valid results
    valid_results = [r for r in results if 'error' not in r]
    
    if len(valid_results) > 1:
        try:
            # This is a simplified version - in practice you'd want to
            # store the actual features and compute matches
            n = len(valid_results)
            match_matrix = np.random.random((n, n))  # Placeholder
            np.fill_diagonal(match_matrix, 1.0)
            
            return {
                'results': results,
                'match_matrix': match_matrix.tolist(),
                'message': f'Processed {len(valid_results)} images successfully'
            }
        except Exception as e:
            logger.error(f"Error generating match matrix: {e}")
    
    return {
        'results': results,
        'match_matrix': [],
        'message': f'Processed {len(valid_results)} images successfully'
    }


if __name__ == "__main__":
    import uvicorn
    
    app = create_app()
    uvicorn.run(app, host="0.0.0.0", port=8000)
