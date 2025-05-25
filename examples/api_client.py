#!/usr/bin/env python3
"""
API client example for Advance Fingerprint Matcher.

This example demonstrates how to interact with the API server.
"""

import requests
import json
from pathlib import Path
import cv2
import numpy as np
import tempfile
import sys


def create_sample_image(filename, pattern='vertical'):
    """Create a sample fingerprint image."""
    size = (256, 256)
    image = np.zeros(size, dtype=np.uint8)
    
    if pattern == 'vertical':
        for i in range(0, size[1], 8):
            image[:, i:i+4] = 255
    elif pattern == 'horizontal':
        for i in range(0, size[0], 8):
            image[i:i+4, :] = 255
    
    # Add noise
    noise = np.random.normal(0, 10, size).astype(np.uint8)
    image = cv2.add(image, noise)
    image = cv2.GaussianBlur(image, (3, 3), 0)
    
    cv2.imwrite(filename, image)
    return filename


def test_api_server(base_url="http://localhost:8000"):
    """Test the API server endpoints."""
    print("Advance Fingerprint Matcher - API Client Example")
    print("=" * 52)
    
    # Test health endpoint
    print("1. Testing health endpoint...")
    try:
        response = requests.get(f"{base_url}/health")
        if response.status_code == 200:
            health_data = response.json()
            print(f"   ✓ Server is healthy")
            print(f"   - Status: {health_data['status']}")
            print(f"   - Version: {health_data['version']}")
        else:
            print(f"   ✗ Health check failed: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print(f"   ✗ Cannot connect to server at {base_url}")
        print("   Please start the server with: fingermatcher serve")
        return False
    
    # Create sample images
    print("\n2. Creating sample fingerprint images...")
    with tempfile.TemporaryDirectory() as temp_dir:
        img1_path = Path(temp_dir) / "fingerprint1.png"
        img2_path = Path(temp_dir) / "fingerprint2.png"
        img3_path = Path(temp_dir) / "fingerprint3.png"
        
        create_sample_image(str(img1_path), 'vertical')
        create_sample_image(str(img2_path), 'vertical')  # Similar to img1
        create_sample_image(str(img3_path), 'horizontal')  # Different from img1
        
        print(f"   - Created {img1_path.name}")
        print(f"   - Created {img2_path.name}")
        print(f"   - Created {img3_path.name}")
        
        # Test feature extraction
        print("\n3. Testing feature extraction...")
        try:
            with open(img1_path, 'rb') as f:
                files = {'image': ('fingerprint1.png', f, 'image/png')}
                response = requests.post(f"{base_url}/extract", files=files)
            
            if response.status_code == 200:
                features = response.json()
                print(f"   ✓ Feature extraction successful")
                print(f"   - SIFT features: {features['sift_count']}")
                print(f"   - ORB features: {features['orb_count']}")
                print(f"   - Minutiae: {features['minutiae_count']}")
                print(f"   - Quality: {features['quality_score']:.3f}")
                print(f"   - Processing time: {features['processing_time']:.3f}s")
            else:
                print(f"   ✗ Feature extraction failed: {response.status_code}")
                print(f"   Error: {response.text}")
        except Exception as e:
            print(f"   ✗ Feature extraction error: {e}")
        
        # Test fingerprint matching - similar images
        print("\n4. Testing fingerprint matching (similar images)...")
        try:
            with open(img1_path, 'rb') as f1, open(img2_path, 'rb') as f2:
                files = {
                    'image1': ('fingerprint1.png', f1, 'image/png'),
                    'image2': ('fingerprint2.png', f2, 'image/png')
                }
                data = {'method': 'hybrid'}
                response = requests.post(f"{base_url}/match", files=files, data=data)
            
            if response.status_code == 200:
                match_result = response.json()
                print(f"   ✓ Matching successful")
                print(f"   - Match score: {match_result['score']:.3f}")
                print(f"   - Confidence: {match_result['confidence']:.3f}")
                print(f"   - Is match: {match_result['is_match']}")
                print(f"   - Method: {match_result['method_used']}")
                print(f"   - Processing time: {match_result['processing_time']:.3f}s")
            else:
                print(f"   ✗ Matching failed: {response.status_code}")
                print(f"   Error: {response.text}")
        except Exception as e:
            print(f"   ✗ Matching error: {e}")
        
        # Test fingerprint matching - different images
        print("\n5. Testing fingerprint matching (different images)...")
        try:
            with open(img1_path, 'rb') as f1, open(img3_path, 'rb') as f3:
                files = {
                    'image1': ('fingerprint1.png', f1, 'image/png'),
                    'image2': ('fingerprint3.png', f3, 'image/png')
                }
                data = {'method': 'hybrid'}
                response = requests.post(f"{base_url}/match", files=files, data=data)
            
            if response.status_code == 200:
                match_result = response.json()
                print(f"   ✓ Matching successful")
                print(f"   - Match score: {match_result['score']:.3f}")
                print(f"   - Confidence: {match_result['confidence']:.3f}")
                print(f"   - Is match: {match_result['is_match']}")
                print(f"   - Method: {match_result['method_used']}")
                print(f"   - Processing time: {match_result['processing_time']:.3f}s")
            else:
                print(f"   ✗ Matching failed: {response.status_code}")
                print(f"   Error: {response.text}")
        except Exception as e:
            print(f"   ✗ Matching error: {e}")
        
        # Test batch processing
        print("\n6. Testing batch processing...")
        try:
            files = []
            with open(img1_path, 'rb') as f1, open(img2_path, 'rb') as f2, open(img3_path, 'rb') as f3:
                files_data = [
                    ('images', ('fingerprint1.png', f1.read(), 'image/png')),
                    ('images', ('fingerprint2.png', f2.read(), 'image/png')),
                    ('images', ('fingerprint3.png', f3.read(), 'image/png'))
                ]
                response = requests.post(f"{base_url}/batch", files=files_data)
            
            if response.status_code == 200:
                batch_result = response.json()
                print(f"   ✓ Batch processing successful")
                print(f"   - Processed images: {len(batch_result['results'])}")
                print(f"   - Message: {batch_result['message']}")
                
                # Show individual results
                for i, result in enumerate(batch_result['results']):
                    if 'error' not in result:
                        print(f"   - Image {i+1}: SIFT={result['sift_count']}, "
                              f"ORB={result['orb_count']}, Quality={result['quality']:.3f}")
                    else:
                        print(f"   - Image {i+1}: Error - {result['error']}")
            else:
                print(f"   ✗ Batch processing failed: {response.status_code}")
                print(f"   Error: {response.text}")
        except Exception as e:
            print(f"   ✗ Batch processing error: {e}")
    
    print("\n✓ API client example completed successfully!")
    print("\nAPI Endpoints tested:")
    print("- GET /health - Server health check")
    print("- POST /extract - Feature extraction")
    print("- POST /match - Fingerprint matching")
    print("- POST /batch - Batch processing")
    print(f"\nAPI documentation available at: {base_url}/docs")
    
    return True


def main():
    """Main function."""
    # Default server URL
    server_url = "http://localhost:8000"
    
    # Allow custom server URL from command line
    if len(sys.argv) > 1:
        server_url = sys.argv[1]
    
    print(f"Testing API server at: {server_url}")
    print("(Start the server with: fingermatcher serve)\n")
    
    # Test the API
    success = test_api_server(server_url)
    
    if not success:
        print("\nTroubleshooting:")
        print("1. Make sure the server is running: fingermatcher serve")
        print("2. Check if the server URL is correct")
        print("3. Verify that all dependencies are installed")
        sys.exit(1)


if __name__ == "__main__":
    main()
