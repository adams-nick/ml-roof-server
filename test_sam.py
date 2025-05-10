import cv2
import numpy as np
import requests
import base64
import json
import sys
import os

def test_endpoint(image_path, building_box):
    """Test the prediction endpoint with a local image file"""
    # Check if image exists
    if not os.path.exists(image_path):
        print(f"Error: Image file not found at {image_path}")
        return
    
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image from {image_path}")
        return
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Print image information
    print(f"Image shape: {img.shape}")
    
    # Encode image to base64
    _, buffer = cv2.imencode('.png', img)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    
    # Prepare request
    request_data = {
        "building_id": "test_building",
        "rgb_image": img_base64,
        "building_box": building_box
    }
    
    print(f"Sending request to predict endpoint...")
    
    # Send request
    try:
        response = requests.post(
            "http://localhost:8000/api/predict",
            json=request_data
        )
        
        # Print response
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"Number of roof segments: {result['num_segments']}")
            print(f"Processing time: {result['processing_time_seconds']:.2f} seconds")
            
            if result['num_segments'] > 0:
                print(f"First segment area: {result['roof_segments'][0]['area']:.2f} pixels")
                print(f"First segment confidence: {result['roof_segments'][0]['confidence']:.4f}")
            
            print("\nResponse summary:", json.dumps(result, indent=2))
        else:
            print("Error:", response.text)
    except Exception as e:
        print(f"Error sending request: {e}")

if __name__ == "__main__":
    # Check if image path was provided
    if len(sys.argv) < 2:
        print("Usage: python test_sam.py <image_path>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    # Example building box - adjust based on your image
    # For a simple test, use a box that covers most of the image
    building_box = {
        "min_x": 50,
        "min_y": 50,
        "max_x": 500,
        "max_y": 500
    }
    
    test_endpoint(image_path, building_box)
